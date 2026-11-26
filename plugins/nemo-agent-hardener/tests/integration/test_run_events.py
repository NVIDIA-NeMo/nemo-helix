# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run events against a real (SQLite) Entity Store: ingest, ordered reads, resends, legacy runs, deletion."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.api.v2 import events, runs
from nemo_agent_hardener_plugin.entities import AGENT_HARDENER_RUN_TYPE
from nemo_helix_plugin.dependencies import get_entity_client
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.entities.types import EntityCreateInput
from nhx.testing import ClientContext, create_test_client

BASE = "/v2/workspaces/default/runs"


@pytest.fixture
def platform() -> Iterator[ClientContext]:
    with create_test_client(client_type=ClientContext) as ctx:
        yield ctx


@pytest.fixture
def api(platform: ClientContext, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(events, "_run_ids", {})
    app = FastAPI()
    for router in (events.router, runs.router):
        app.include_router(router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: platform.entity_client
    with TestClient(app) as client:
        yield client


def _create_run(platform: ClientContext, name: str, **data: Any) -> None:
    EntitiesClient.from_client(platform.client).create_entity(
        entity_type=AGENT_HARDENER_RUN_TYPE, workspace="default", body=EntityCreateInput(name=name, data=data)
    )


def _post(api: TestClient, run: str, event: str, **sender: Any) -> None:
    response = api.post(f"{BASE}/{run}/events", json={"event": event, "payload": {"n": event}, **sender})
    assert response.status_code == 204, response.text


def _events(api: TestClient, run: str, after: int = 0) -> list[dict[str, Any]]:
    response = api.get(f"{BASE}/{run}/events", params={"after": after})
    assert response.status_code == 200, response.text
    return response.json()["events"]


def test_events_are_numbered_in_arrival_order_and_read_after_a_cursor(
    api: TestClient, platform: ClientContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(events, "_EVENT_PAGE_SIZE", 2)  # the read must follow more than one page
    _create_run(platform, "run-a")
    _create_run(platform, "run-b")
    for name in ("first", "second", "third", "fourth", "fifth"):
        _post(api, "run-a", name)
    _post(api, "run-b", "other-run")

    assert [(e["id"], e["event"]) for e in _events(api, "run-a")] == [
        (1, "first"),
        (2, "second"),
        (3, "third"),
        (4, "fourth"),
        (5, "fifth"),
    ]
    assert [e["id"] for e in _events(api, "run-a", after=3)] == [4, 5]
    assert _events(api, "run-a", after=5) == []
    assert _events(api, "run-b")[0] == {"id": 1, "event": "other-run", "payload": {"n": "other-run"}}


def test_a_resent_event_is_stored_once_but_another_senders_same_seq_is_not_a_resend(
    api: TestClient, platform: ClientContext
) -> None:
    # agent-hardener's `serve` and `run` processes each count from 1 against the same run.
    _create_run(platform, "run-a")
    _post(api, "run-a", "serve-1", sink_id="serve", seq=1)
    _post(api, "run-a", "serve-1", sink_id="serve", seq=1)
    _post(api, "run-a", "run-1", sink_id="run", seq=1)
    _post(api, "run-a", "legacy")  # an older agent-hardener sends neither field

    assert [(e["id"], e["event"]) for e in _events(api, "run-a")] == [(1, "serve-1"), (2, "run-1"), (3, "legacy")]


def test_ingest_for_an_unknown_run_is_404(api: TestClient) -> None:
    response = api.post(f"{BASE}/missing/events", json={"event": "x"})
    assert response.status_code == 404
    assert _events(api, "missing") == []


def test_deleting_a_run_deletes_its_events(api: TestClient, platform: ClientContext) -> None:
    _create_run(platform, "run-a")
    _post(api, "run-a", "first")
    assert api.delete(f"{BASE}/run-a").status_code == 204

    # A new run that reuses the name starts empty: the old events went with the old run.
    _create_run(platform, "run-a")
    assert _events(api, "run-a") == []
    _post(api, "run-a", "fresh")
    assert [(e["id"], e["event"]) for e in _events(api, "run-a")] == [(1, "fresh")]


def test_a_run_without_event_children_is_served_from_its_legacy_fileset(
    api: TestClient, platform: ClientContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_run(platform, "old-run", events_fileset="default/old-events")
    downloads: list[str] = []

    def fake_download(_client: object, ref: str, dest: Path) -> Path:
        downloads.append(ref)
        (dest / "old-run.jsonl").write_text('{"event": "a", "payload": {}}\n\n{"event": "b", "payload": {}}\n')
        return dest

    monkeypatch.setattr(events, "_get_client", lambda: None)
    monkeypatch.setattr(events, "download_fileset", fake_download)

    assert [(e["id"], e["event"]) for e in _events(api, "old-run")] == [(1, "a"), (3, "b")]
    assert [e["id"] for e in _events(api, "old-run", after=1)] == [3]
    assert downloads == ["default/old-events", "default/old-events"]


def test_a_run_with_stored_events_never_falls_back_to_its_fileset(
    api: TestClient, platform: ClientContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_run(platform, "run-a", events_fileset="default/stale")
    monkeypatch.setattr(events, "download_fileset", lambda *_: pytest.fail("fileset read for a stored run"))
    _post(api, "run-a", "first")

    assert [e["event"] for e in _events(api, "run-a")] == ["first"]
    assert _events(api, "run-a", after=1) == []
