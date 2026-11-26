# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the event relay: the job's sink identity and the legacy fileset read.

Ingest and reads against a real Entity Store are in ``tests/integration/test_run_events.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from _doubles import make_job_context
from fastapi import HTTPException
from nemo_agent_hardener_plugin.api.v2 import events
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.jobs import execution
from nemo_agents_plugin.jobs import gateway_proxy
from nemo_helix_plugin.client.auth import StaticToken
from nemo_helix_plugin.client.errors import NotFoundError
from pytest_httpserver import HTTPServer


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


def _legacy_events_failing_with(failure: Exception) -> list[dict[str, Any]]:
    with (
        patch.object(events, "_get_client", return_value=MagicMock()),
        patch.object(events, "download_fileset", side_effect=failure),
    ):
        return events._legacy_events("my-run", "default/my-events-fs", after=0)


def test_legacy_events_are_empty_when_the_events_fileset_is_gone() -> None:
    gone = NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://files")))
    assert _legacy_events_failing_with(gone) == []


def test_a_failed_legacy_download_is_reported_instead_of_an_empty_run() -> None:
    """An empty list would read as a run that emitted nothing; the caller needs to know the load failed."""
    with pytest.raises(HTTPException) as raised:
        _legacy_events_failing_with(RuntimeError("files service down"))
    assert raised.value.status_code == 502
    assert "files service down" in str(raised.value.detail)
