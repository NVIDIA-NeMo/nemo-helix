# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the event relay: durable ``events.jsonl`` history, ingest, and polling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from _doubles import make_job_context
from fastapi import FastAPI
from nemo_agent_hardener_plugin.api.v2 import events
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.jobs import execution
from nemo_agents_plugin.jobs import gateway_proxy
from nemo_helix_plugin.client.auth import StaticToken
from pytest_httpserver import HTTPServer
from starlette.testclient import TestClient


def _write_events(path: Path, event_list: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in event_list:
            f.write(json.dumps(e) + "\n")


def test_run_events_use_parent_workload_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, httpserver: HTTPServer
) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: victim\n  secrets: []\n", encoding="utf-8")
    config = AgentHardenerConfig(operator_env_file=tmp_path / "operator.env")
    monkeypatch.setenv("NHX_BASE_URL", httpserver.url_for(""))
    monkeypatch.setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", "/job/subject-token")
    monkeypatch.setattr(
        gateway_proxy, "resolve_workload_exchange_provider", lambda **_kwargs: StaticToken("job-access-token")
    )
    event_path = "/apis/agent-hardener/v2/workspaces/default/runs/run-x/events"
    event_body = {"event": "round_started", "payload": {"round": 1}}
    httpserver.expect_request(
        event_path, method="POST", json=event_body, headers={"Authorization": "Bearer job-access-token"}
    ).respond_with_data(status=204)

    def drive(_manifest, _env_file, env, *_args, **_kwargs):
        assert "NHX_WORKLOAD_IDENTITY_TOKEN_FILE" not in env
        assert "AGENT_HARDENER_EVENT_SINK_HEADERS" not in env
        assert "job-access-token" not in env.values()
        response = httpx.post(env["AGENT_HARDENER_EVENT_SINK_URL"], json=event_body, timeout=5)
        assert response.status_code == 204
        return execution.RunOutcome("completed", 0, record_name="run-x")

    monkeypatch.setattr(execution, "_drive_service_run", drive)
    outcome = execution._run_service_driven(
        str(manifest),
        None,
        config,
        make_job_context(tmp_path),
        None,
        "victim",
        8000,
        prepared_run_name="run-x",
    )
    assert outcome.status == "completed"
    httpserver.check_assertions()


def test_history_filters_by_last_seen_id(tmp_path: Path) -> None:
    stream = events._RunStream(tmp_path / "events.jsonl")
    for i in range(3):
        stream.publish({"event": f"e{i}", "payload": {}})
    assert [seq for seq, _ in stream.history(after_id=0)] == [1, 2, 3]
    assert [seq for seq, _ in stream.history(after_id=2)] == [3]


def test_publish_persists_and_a_fresh_stream_replays(tmp_path: Path) -> None:
    # Durability: events written by one stream survive on disk and replay from a brand-new stream on the
    # same file (as after a service restart), with ids continuing monotonically from the file's line count.
    path = tmp_path / "events.jsonl"
    first = events._RunStream(path)
    first.publish({"event": "agent_exchange", "payload": {"agent_name": "Direct Prompt Attacker"}})
    first.publish({"event": "agent_completed", "payload": {"agent_name": "Direct Prompt Attacker"}})

    assert path.exists()
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == [
        "agent_exchange",
        "agent_completed",
    ]

    reopened = events._RunStream(path)
    assert [seq for seq, _ in reopened.history(after_id=0)] == [1, 2]
    reopened.publish({"event": "round_completed", "payload": {}})  # id continues from the persisted count
    assert reopened.history(after_id=2)[0][0] == 3


def test_ingest_endpoint_publishes_to_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(events, "hub", events.EventHub())
    monkeypatch.setattr(events, "_events_path", lambda ws, name: tmp_path / f"{ws}-{name}.jsonl")
    app = FastAPI()
    app.include_router(events.router, prefix="/v2/workspaces/{workspace}")
    with TestClient(app) as client:
        resp = client.post(
            "/v2/workspaces/default/runs/run-42/events",
            json={"event": "FINAL_VERDICT", "payload": {"passed": True}},
        )
    assert resp.status_code == 204
    persisted = events.hub.stream("default", "run-42").history(after_id=0)
    assert persisted[-1][1] == {"event": "FINAL_VERDICT", "payload": {"passed": True}}


def test_get_events_endpoint_returns_events_after_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(events, "hub", events.EventHub())
    monkeypatch.setattr(events, "_events_path", lambda ws, name: tmp_path / f"{ws}-{name}.jsonl")
    app = FastAPI()
    app.include_router(events.router, prefix="/v2/workspaces/{workspace}")
    with TestClient(app) as client:
        for i in range(3):
            client.post(
                "/v2/workspaces/default/runs/run-1/events",
                json={"event": f"e{i}", "payload": {"i": i}},
            )
        resp = client.get("/v2/workspaces/default/runs/run-1/events?after=1")
    assert resp.status_code == 200
    body = resp.json()
    assert [e["id"] for e in body["events"]] == [2, 3]
    assert body["events"][0]["event"] == "e1"
    assert body["events"][1]["event"] == "e2"


def test_get_events_falls_back_to_fileset_when_local_missing(tmp_path: Path) -> None:
    """When the local log is absent but the run entity has events_fileset, download and serve.

    The log is named per run (`_events_path` -> ``<safe-run-name>.jsonl``), and the fileset member
    carries that same basename because `_save_events_fileset` uploads that exact file. Naming both
    ``events.jsonl`` here would pass even if the fallback wrote somewhere the reader never looks.
    """
    events_file = tmp_path / "source" / "my-run.jsonl"
    _write_events(events_file, [{"event": "run_started", "payload": {}}])

    missing_path = tmp_path / "missing" / "my-run.jsonl"

    mock_platform = MagicMock()
    # Shaped like the real entity-store record: get_entity_by_name returns a generic Entity whose
    # domain fields live under `.data` (a bare MagicMock would falsely expose `.events_fileset`).
    mock_run = SimpleNamespace(name="my-run", data={"events_fileset": "default/my-events-fs"})
    mock_platform.entities.get_entity_by_name.return_value = mock_run

    def fake_download(sdk, ref, dest):
        # download_fileset preserves the member basename, so the run-named file lands in `dest`.
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "my-run.jsonl").write_text(events_file.read_text())
        return dest

    with (
        patch.object(events, "hub", events.EventHub()),
        patch("nemo_agent_hardener_plugin.api.v2.events._events_path", return_value=missing_path),
        patch("nemo_agent_hardener_plugin.api.v2.events._get_client", return_value=mock_platform),
        patch("nemo_agent_hardener_plugin.api.v2.events.download_fileset", side_effect=fake_download),
    ):
        app = FastAPI()
        app.include_router(events.router, prefix="/v2/workspaces/{workspace}")
        client = TestClient(app)
        resp = client.get("/v2/workspaces/default/runs/my-run/events?after=0")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["events"]) == 1
    assert data["events"][0]["event"] == "run_started"


def test_get_events_returns_empty_when_no_local_and_no_fileset(tmp_path: Path) -> None:
    """When local file is missing and no fileset ref exists, return empty list."""
    missing_path = tmp_path / "missing" / "events.jsonl"

    mock_platform = MagicMock()
    mock_run = SimpleNamespace(name="my-run", data={"events_fileset": ""})
    mock_platform.entities.get_entity_by_name.return_value = mock_run

    with (
        patch.object(events, "hub", events.EventHub()),
        patch("nemo_agent_hardener_plugin.api.v2.events._events_path", return_value=missing_path),
        patch("nemo_agent_hardener_plugin.api.v2.events._get_client", return_value=mock_platform),
    ):
        app = FastAPI()
        app.include_router(events.router, prefix="/v2/workspaces/{workspace}")
        client = TestClient(app)
        resp = client.get("/v2/workspaces/default/runs/my-run/events?after=0")

    assert resp.status_code == 200
    assert resp.json() == {"events": []}
