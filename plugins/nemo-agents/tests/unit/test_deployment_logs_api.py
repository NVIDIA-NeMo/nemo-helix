# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for /apis/agents/v2/workspaces/{ws}/deployments/{name}/logs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import deployment_logs as module


@pytest.fixture
def fake_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_path = tmp_path / "test-deployment.log"
    log_path.write_text(
        "2026-05-19 21:00:00 - INFO - first line\n2026-05-19 21:00:01 - WARNING - second line\nplain trailing line\n",
        encoding="utf-8",
    )
    # Stub the runner backend so the route resolves the log path through the
    # ABC rather than reaching into ``runner.in_memory`` directly. The stub
    # records the workspace it's queried with so the cross-workspace
    # regression test can assert on it.
    from nemo_agents_plugin.runner.backend import LocalLog

    backend = type(
        "_StubBackend",
        (),
        {
            "get_log_location": staticmethod(
                lambda _workspace, _name: LocalLog(path=log_path),
            ),
        },
    )()
    monkeypatch.setattr(module, "get_runner_backend", lambda: backend)
    return log_path


@pytest.fixture
def client(fake_log: Path) -> Iterator[TestClient]:  # noqa: ARG001 — fixture wires monkeypatch
    app = FastAPI()
    app.include_router(module.router, prefix="/apis/agents/v2/workspaces/{workspace}")
    # Workspace-scoped authorization now runs an entity-client lookup before
    # touching the log file. Override the dependency with an AsyncMock that
    # silently accepts every (workspace, deployment) — these tests focus on
    # the path-resolution and tail behavior. A separate test asserts the
    # 404 from a missing entity.
    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=object())
    app.dependency_overrides[module.get_entity_client] = lambda: fake_client
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_logs_returns_parsed_lines(client: TestClient, fake_log: Path) -> None:
    resp = client.get("/apis/agents/v2/workspaces/default/deployments/test/logs")
    assert resp.status_code == 200
    body = resp.json()
    # ``path`` was dropped from the response — absolute on-disk paths
    # shouldn't leak through the public API. ``total_lines`` and the
    # parsed entries are the contract.
    assert "path" not in body
    assert body["total_lines"] == 3
    assert "first line" in body["data"][0]["message"]
    assert body["data"][-1]["timestamp"] == ""
    assert body["data"][-1]["message"] == "plain trailing line"
    # next_offset is the end-of-tail cursor used to resume the SSE stream.
    assert body["next_offset"] > 0


def test_logs_respects_tail(client: TestClient) -> None:
    resp = client.get("/apis/agents/v2/workspaces/default/deployments/test/logs?tail=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_lines"] == 1
    assert body["data"][0]["message"] == "plain trailing line"


def test_logs_returns_404_when_log_not_yet_available(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Local backend with no file yet maps to 404 with no path leak."""
    from nemo_agents_plugin.runner.backend import NotYetAvailable

    backend = type(
        "_NotYetBackend",
        (),
        {"get_log_location": staticmethod(lambda _workspace, _name: NotYetAvailable())},
    )()
    monkeypatch.setattr(module, "get_runner_backend", lambda: backend)
    resp = client.get("/apis/agents/v2/workspaces/default/deployments/missing/logs")
    assert resp.status_code == 404


def test_logs_returns_404_with_hint_for_external_log_backend(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote backends (Docker/K8s) surface their fetch hint in the detail."""
    from nemo_agents_plugin.runner.backend import ExternalLog

    backend = type(
        "_RemoteBackend",
        (),
        {
            "get_log_location": staticmethod(lambda _workspace, _name: ExternalLog(hint="Run: docker logs abc123")),
        },
    )()
    monkeypatch.setattr(module, "get_runner_backend", lambda: backend)
    resp = client.get("/apis/agents/v2/workspaces/default/deployments/missing/logs")
    assert resp.status_code == 404
    assert "docker logs abc123" in resp.json()["detail"]
    assert ".log" not in resp.json()["detail"]


def test_logs_rejects_negative_tail(client: TestClient) -> None:
    # ge=0 on the Query param → FastAPI validation rejects with 422.
    resp = client.get("/apis/agents/v2/workspaces/default/deployments/test/logs?tail=-3")
    assert resp.status_code == 422


def test_logs_rejects_tail_over_cap(client: TestClient) -> None:
    # le=_TAIL_LINE_CAP (10_000) → over-cap requests are rejected, not silently clamped.
    resp = client.get("/apis/agents/v2/workspaces/default/deployments/test/logs?tail=10001")
    assert resp.status_code == 422


def test_logs_tail_larger_than_file_returns_all_lines(client: TestClient) -> None:
    # Reverse-seek tail must still return every line when n exceeds the line count.
    resp = client.get("/apis/agents/v2/workspaces/default/deployments/test/logs?tail=100")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_lines"] == 3
    assert "first line" in body["data"][0]["message"]
    assert body["data"][-1]["message"] == "plain trailing line"


def test_logs_workspace_namespacing_separates_same_named_deployments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two workspaces with deployment ``foo`` must read distinct log files.

    Regression for the cross-workspace leak Codex flagged: even after auth
    scoped the lookup to ``(workspace, name)``, ``get_log_path(name)`` was
    workspace-agnostic and returned a shared ``foo.log`` for both. With the
    workspace-namespaced backend hook + path layout, asking the same backend
    for ``("ws-a", "foo")`` and ``("ws-b", "foo")`` resolves to different
    files.
    """
    log_a = tmp_path / "ws-a" / "foo.log"
    log_a.parent.mkdir()
    log_a.write_text("from workspace a\n", encoding="utf-8")
    log_b = tmp_path / "ws-b" / "foo.log"
    log_b.parent.mkdir()
    log_b.write_text("from workspace b\n", encoding="utf-8")

    from nemo_agents_plugin.runner.backend import LocalLog, NotYetAvailable

    def _resolve(workspace: str, _name: str):  # noqa: ANN202
        target = {"ws-a": log_a, "ws-b": log_b}.get(workspace)
        return LocalLog(path=target) if target is not None else NotYetAvailable()

    backend = type("_PerWorkspaceBackend", (), {"get_log_location": staticmethod(_resolve)})()
    monkeypatch.setattr(module, "get_runner_backend", lambda: backend)

    app = FastAPI()
    app.include_router(module.router, prefix="/apis/agents/v2/workspaces/{workspace}")
    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=object())
    app.dependency_overrides[module.get_entity_client] = lambda: fake_client

    with TestClient(app, raise_server_exceptions=False) as c:
        resp_a = c.get("/apis/agents/v2/workspaces/ws-a/deployments/foo/logs")
        resp_b = c.get("/apis/agents/v2/workspaces/ws-b/deployments/foo/logs")

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    assert "from workspace a" in resp_a.json()["data"][0]["message"]
    assert "from workspace b" in resp_b.json()["data"][0]["message"]


def test_logs_404_when_deployment_not_in_workspace(fake_log: Path) -> None:  # noqa: ARG001
    """Cross-workspace requests must 404 instead of returning logs."""
    from nemo_helix_plugin.entity_client import NemoEntityNotFoundError

    app = FastAPI()
    app.include_router(module.router, prefix="/apis/agents/v2/workspaces/{workspace}")
    fake_client = AsyncMock()
    fake_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("not found"))
    app.dependency_overrides[module.get_entity_client] = lambda: fake_client
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/apis/agents/v2/workspaces/other-ws/deployments/test/logs")
    assert resp.status_code == 404
    # Absolute path leak guard: the error must not contain the on-disk path.
    detail = resp.json()["detail"]
    assert "/" not in detail or not detail.endswith(".log")


# --- SSE streaming (id cursor / Last-Event-ID resume / termination) ---------


def _parse_event(raw: str) -> tuple[int | None, dict | None]:
    """Split a raw SSE event into (id, parsed-data) — keepalives return (None, None)."""
    event_id: int | None = None
    payload: dict | None = None
    for field in raw.strip().split("\n"):
        if field.startswith("id:"):
            event_id = int(field[len("id:") :].strip())
        elif field.startswith("data:"):
            payload = json.loads(field[len("data:") :].strip())
    return event_id, payload


async def _collect(gen: AsyncIterator[str], n: int, timeout: float = 2.0) -> list[str]:
    out: list[str] = []
    try:
        for _ in range(n):
            out.append(await asyncio.wait_for(gen.__anext__(), timeout))
    except (TimeoutError, StopAsyncIteration):
        pass  # expected exit: collected what we could within the limit
    finally:
        aclose = getattr(gen, "aclose", None)
        if aclose is not None:
            await aclose()
    return out


def test_parse_last_event_id_handles_absent_and_invalid() -> None:
    assert module._parse_last_event_id(None) is None
    assert module._parse_last_event_id("") is None
    assert module._parse_last_event_id("not-an-int") is None
    assert module._parse_last_event_id("-5") is None
    assert module._parse_last_event_id("42") == 42


async def test_stream_emits_byte_offset_ids_from_start(tmp_path: Path) -> None:
    """start_offset=0 replays the whole file, each event carrying its end-offset id."""
    log = tmp_path / "d.log"
    log.write_text("alpha\nbeta\n", encoding="utf-8")

    events = await _collect(module._stream_log_lines(log, start_offset=0), n=2)
    data_events = [(eid, pl) for eid, pl in (_parse_event(e) for e in events) if pl is not None]

    assert [pl["message"] for _eid, pl in data_events] == ["alpha", "beta"]
    # id is the byte offset after each line: len("alpha\n")=6, +len("beta\n")=11.
    assert [eid for eid, _pl in data_events] == [6, 11]


async def test_stream_resumes_from_last_event_id_without_gaps(tmp_path: Path) -> None:
    """Reconnecting with the previous id (byte offset) skips delivered lines, not new ones."""
    log = tmp_path / "d.log"
    log.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    # Resume right after "alpha\n" (offset 6) — must get beta, then gamma.
    events = await _collect(module._stream_log_lines(log, start_offset=6), n=2)
    messages = [pl["message"] for _eid, pl in (_parse_event(e) for e in events) if pl is not None]
    assert messages == ["beta", "gamma"]


async def test_stream_terminates_when_log_file_removed(tmp_path: Path) -> None:
    """A deleted log file ends the stream instead of polling a dead file forever."""
    log = tmp_path / "d.log"
    log.write_text("alpha\n", encoding="utf-8")

    # First event establishes the stream (opens the file, delivers "alpha").
    gen = module._stream_log_lines(log, start_offset=0)
    first = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
    _eid, payload = _parse_event(first)
    assert payload is not None and payload["message"] == "alpha"

    # Now the deployment is cleaned up out from under the live stream.
    log.unlink()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(gen.__anext__(), timeout=2.0)


async def test_stream_stops_when_client_disconnects(tmp_path: Path) -> None:
    """The stream ends promptly when the client disconnects, even with data pending."""
    log = tmp_path / "d.log"
    log.write_text("alpha\nbeta\n", encoding="utf-8")

    async def _disconnected() -> bool:
        return True

    gen = module._stream_log_lines(log, start_offset=0, is_disconnected=_disconnected)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(gen.__anext__(), timeout=2.0)


async def _next_message(gen: AsyncIterator[str]) -> str:
    _eid, payload = _parse_event(await asyncio.wait_for(gen.__anext__(), timeout=2.0))
    assert payload is not None
    return payload["message"]


async def test_stream_replays_new_file_after_truncation(tmp_path: Path) -> None:
    """A truncated/rotated file is replayed from the top instead of dropping lines."""
    log = tmp_path / "d.log"
    log.write_text("alpha\nbeta\n", encoding="utf-8")

    gen = module._stream_log_lines(log, start_offset=0)
    assert await _next_message(gen) == "alpha"
    assert await _next_message(gen) == "beta"

    # Rotate: truncate and write a fresh shorter file. The stream should detect
    # size < position and replay from the start rather than sit past EOF.
    log.write_text("gamma\n", encoding="utf-8")
    assert await _next_message(gen) == "gamma"
