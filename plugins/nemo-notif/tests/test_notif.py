# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for notif event fan-out: the broker, the SSE framing, and the publish route."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest
from fake_entities import FakeEntitiesClient
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_helix_plugin.entity_client import get_entity_client
from nemo_notif_plugin import broker as broker_module
from nemo_notif_plugin import store
from nemo_notif_plugin.broker import NotifBroker
from nemo_notif_plugin.entities import NotifEventEntity
from nemo_notif_plugin.service import build_events_router, format_sse, stream_events
from nemo_notif_plugin.types import CreateNotifEventRequest, NotifEvent


def _event(workspace: str = "default", type_: str = "jobs.job.failed", event_id: str = "e1") -> NotifEvent:
    return NotifEvent(
        id=event_id,
        workspace=workspace,
        type=type_,
        title="Job failed",
        created_at=datetime.now(timezone.utc),
    )


async def _never_disconnected() -> bool:
    return False


async def test_publish_reaches_only_matching_listeners_in_workspace():
    notif_broker = NotifBroker()
    async with (
        notif_broker.listen("default") as everything,
        notif_broker.listen("default", frozenset({"deployments.deployment.ready"})) as filtered,
        notif_broker.listen("other") as other_workspace,
    ):
        assert notif_broker.publish(_event()) == 1
        assert everything.queue.get_nowait().id == "e1"
        assert filtered.queue.empty()
        assert other_workspace.queue.empty()


async def test_listener_is_removed_when_context_exits():
    notif_broker = NotifBroker()
    async with notif_broker.listen("default"):
        assert notif_broker.listener_count("default") == 1
    assert notif_broker.listener_count("default") == 0
    assert notif_broker.publish(_event()) == 0


async def test_slow_listener_is_cut_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(broker_module, "DEFAULT_QUEUE_SIZE", 2)
    notif_broker = NotifBroker()
    async with notif_broker.listen("default") as listener:
        assert notif_broker.publish(_event(event_id="a")) == 1
        assert notif_broker.publish(_event(event_id="b")) == 1
        assert notif_broker.publish(_event(event_id="c")) == 0
        assert notif_broker.listener_count("default") == 0
        assert listener.queue.get_nowait() is None


def test_format_sse_carries_type_inside_data():
    frame = format_sse(_event())
    assert frame.startswith("id: e1\nevent: jobs.job.failed\ndata: ")
    assert frame.endswith("\n\n")
    data = json.loads(frame.split("data: ", 1)[1])
    assert data["type"] == "jobs.job.failed"
    assert data["workspace"] == "default"


async def test_stream_yields_events_and_keepalives_until_cut_off():
    notif_broker = NotifBroker()
    async with notif_broker.listen("default") as listener:
        stream = stream_events(listener, _never_disconnected, keepalive_seconds=0.01)
        assert await anext(stream) == ": connected\n\n"
        assert await anext(stream) == ": keepalive\n\n"
        notif_broker.publish(_event())
        assert (await anext(stream)).startswith("id: e1\n")
        listener.queue.put_nowait(None)
        with pytest.raises(StopAsyncIteration):
            await anext(stream)


async def test_stream_stops_when_client_disconnects():
    async def disconnected() -> bool:
        return True

    async with NotifBroker().listen("default") as listener:
        stream = stream_events(listener, disconnected, keepalive_seconds=0.01)
        await anext(stream)
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), timeout=1)


def _app(notif_broker: NotifBroker, entities: FakeEntitiesClient, principal: dict[str, str]) -> FastAPI:
    app = FastAPI()
    app.include_router(build_events_router(notif_broker), prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entities
    app.dependency_overrides[get_effective_principal_id] = lambda: principal["id"]
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_publish_route_stores_and_fans_out():
    notif_broker, entities = NotifBroker(), FakeEntitiesClient()
    app = _app(notif_broker, entities, {"id": "alice"})

    async with notif_broker.listen("default") as listener, _client(app) as client:
        response = await client.post(
            "/v2/workspaces/default/events",
            json={"type": "jobs.job.failed", "title": "Job failed", "severity": "error"},
        )
        received = listener.queue.get_nowait()

    assert response.status_code == 201
    body = response.json()
    assert body["delivered"] == 1
    assert body["event"]["id"].startswith("evt-")
    assert body["event"]["read"] is False
    assert body["event"]["created_by"] == "alice"
    assert received is not None and received.id == body["event"]["id"]
    assert ("default", body["event"]["id"]) in entities.rows


async def test_read_state_is_per_principal_and_drives_unread_count():
    entities = FakeEntitiesClient()
    principal = {"id": "alice"}
    app = _app(NotifBroker(), entities, principal)
    url = "/v2/workspaces/default/events"

    async with _client(app) as client:
        ids = [(await client.post(url, json={"type": "t.a", "title": f"e{i}"})).json()["event"]["id"] for i in range(3)]

        listed = (await client.get(url)).json()
        assert [e["id"] for e in listed["data"]] == ids[::-1]  # newest first
        assert listed["unread_count"] == 3

        marked = await client.post(f"{url}/{ids[0]}/read")
        assert marked.status_code == 200 and marked.json()["read"] is True

        listed = (await client.get(url, params={"unread_only": True})).json()
        assert listed["unread_count"] == 2
        assert [e["id"] for e in listed["data"]] == [ids[2], ids[1]]

        principal["id"] = "bob"  # bob has read nothing
        assert (await client.get(url)).json()["unread_count"] == 3
        assert (await client.post(f"{url}/read-all")).json() == {"marked": 3}
        assert (await client.get(url)).json()["unread_count"] == 0

        principal["id"] = "alice"
        assert (await client.post(f"{url}/{ids[0]}/unread")).json()["read"] is False
        assert (await client.get(url)).json()["unread_count"] == 3


async def test_mark_read_unknown_event_is_404():
    app = _app(NotifBroker(), FakeEntitiesClient(), {"id": "alice"})
    async with _client(app) as client:
        response = await client.post("/v2/workspaces/default/events/evt-missing/read")
    assert response.status_code == 404


async def test_mark_read_retries_on_conflict():
    entities = FakeEntitiesClient()
    entity = await store.create_event(entities, "default", CreateNotifEventRequest(type="t.a", title="x"))
    entities.conflicts_to_raise = store.UPDATE_ATTEMPTS - 1
    updated = await store.set_read(entities, "default", entity.name, "alice", read=True)
    assert updated.read_by == ["alice"]


async def test_anonymous_callers_share_read_state():
    entities = FakeEntitiesClient()
    entity = await store.create_event(entities, "default", CreateNotifEventRequest(type="t.a", title="x"))
    await store.set_read(entities, "default", entity.name, "", read=True)
    stored = await entities.get(NotifEventEntity, name=entity.name, workspace="default")
    assert store.to_event(stored, None).read is True


def test_publish_route_rejects_bad_type():
    app = _app(NotifBroker(), FakeEntitiesClient(), {"id": ""})
    with TestClient(app) as client:
        response = client.post("/v2/workspaces/default/events", json={"type": "Not Valid", "title": "x"})
    assert response.status_code == 422
