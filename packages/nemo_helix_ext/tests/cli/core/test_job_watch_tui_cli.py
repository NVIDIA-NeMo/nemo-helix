# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from unittest.mock import patch

import httpx
import pytest
from nemo_helix_ext.cli.commands.jobs import app
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.job_watch_renderer import JobWatchRenderResult
from nemo_helix_ext.cli.core.job_watch_tui import DagWatchApp
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.jobs.watch_types import JobWatchEvent
from typer.testing import CliRunner


def status_body(*, workflow: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "id": "job-1",
        "name": "pipeline",
        "status": "completed",
        "status_details": {},
        "error_details": None,
        "steps": [],
        "created_at": "2026-11-19T12:00:00Z",
        "updated_at": "2026-11-19T12:01:00Z",
        "attempt_id": "attempt-1",
    }
    if workflow:
        result["workflow"] = {
            "mode": "continuous",
            "pass_number": 1,
            "control": "running",
            "waiting_for_changes": False,
            "degraded": False,
            "nodes": [],
        }
    return result


@pytest.mark.parametrize(
    "interactive,tui_flag,workflow,expected_tui",
    [
        (False, [], True, False),
        (True, ["--no-tui"], True, False),
        (True, [], False, False),
        (True, [], True, True),
        (True, ["--tui"], True, True),
    ],
)
def test_cli_selects_tui_only_for_interactive_workflows(
    interactive: bool, tui_flag: list[str], workflow: bool, expected_tui: bool
) -> None:
    requests: list[httpx.Request] = []
    used_tui: list[bool] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/logs"):
            return httpx.Response(200, json={"data": [], "total": 0, "next_page": None, "prev_page": None})
        return httpx.Response(200, json=status_body(workflow=workflow))

    async def run_tui(
        self: DagWatchApp, status_events: AsyncIterator[JobWatchEvent], log_events: AsyncIterator[JobWatchEvent]
    ) -> JobWatchRenderResult:
        used_tui.append(True)
        assert not self.model.emoji
        assert self.model.attempt_id == "attempt-1"
        assert self.model.job_id == "job-1"
        await self._consume_status(status_events)
        await self._consume_logs(log_events)
        return self.result

    with httpx.Client(transport=httpx.MockTransport(respond)) as sync_transport:
        async_transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        sync_client = NemoClient(base_url="http://test", workspace="ws", http_client=sync_transport)
        async_client = AsyncNemoClient(
            base_url="http://test", workspace="ws", http_client=async_transport, owns_http_client=True
        )
        state = CLIContext(_client=sync_client, _async_client=async_client)
        with (
            patch("nemo_helix_ext.cli.core.job_watch_tui.interactive_watch_available", return_value=interactive),
            patch.object(DagWatchApp, "run", run_tui),
        ):
            result = CliRunner().invoke(
                app, ["watch", "pipeline", "--workspace", "ws", "--no-emoji", *tui_flag], obj=state
            )
        assert result.exit_code == 0, result.output + str(result.exception)
    assert bool(used_tui) == expected_tui
    if expected_tui:
        assert all(request.url.params.get("attempt_id") == "attempt-1" for request in requests[1:])
        assert all("/job-1/" in request.url.path for request in requests[1:])
        assert async_transport.is_closed


@pytest.mark.parametrize(
    "interactive,workflow,message",
    [(False, True, "requires interactive"), (True, False, "requires a job with a workflow")],
)
def test_forced_tui_reports_actionable_error(interactive: bool, workflow: bool, message: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=status_body(workflow=workflow))

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = NemoClient(base_url="http://test", workspace="ws", http_client=transport)
        with patch("nemo_helix_ext.cli.core.job_watch_tui.interactive_watch_available", return_value=interactive):
            result = CliRunner().invoke(
                app, ["watch", "pipeline", "--workspace", "ws", "--tui"], obj=CLIContext(_client=client)
            )
    assert result.exit_code != 0
    assert message in result.output
