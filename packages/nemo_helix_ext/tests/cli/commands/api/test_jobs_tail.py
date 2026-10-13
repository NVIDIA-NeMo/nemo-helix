# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer
from nemo_helix_ext.cli.commands.jobs import tail_platform_job
from nemo_helix_plugin.client.client import NemoClient


def _client(workspace: str | None) -> NemoClient:
    """A real NemoClient so the production workspace accessor is exercised.

    Using a real client (rather than a stub of the private helper) is what lets
    these tests catch a regression where the command calls an accessor the
    typed client does not have.
    """
    return NemoClient(base_url="http://nemo.test", workspace=workspace)


def _ctx(client: NemoClient) -> SimpleNamespace:
    state = MagicMock()
    state.get_client.return_value = client
    return SimpleNamespace(obj=state)


def test_jobs_tail_command_fetches_tail_page() -> None:
    client = _client(workspace="default")
    logs = [object(), object()]
    response = MagicMock()
    response.page.return_value = SimpleNamespace(items=logs)
    jobs_client = MagicMock()
    jobs_client.list_job_logs.return_value = response

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.render_job_logs") as render_logs,
    ):
        tail_platform_job(
            _ctx(client),
            name="job-a",
            lines=25,
            workspace=None,
            attempt_id=1,
            step_id="step-a",
            task_id="task-a",
        )

    jobs_client.list_job_logs.assert_called_once_with(
        workspace="default",
        name="job-a",
        query_params={"tail": 25, "attempt_id": 1, "step_id": "step-a", "task_id": "task-a"},
    )
    render_logs.assert_called_once_with(logs)


def test_jobs_tail_command_uses_explicit_workspace() -> None:
    client = _client(workspace="client-default")
    response = MagicMock()
    response.page.return_value = SimpleNamespace(items=[])
    jobs_client = MagicMock()
    jobs_client.list_job_logs.return_value = response

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.render_job_logs"),
    ):
        tail_platform_job(
            _ctx(client),
            name="job-a",
            lines=100,
            workspace="custom",
            attempt_id=None,
            step_id=None,
            task_id=None,
        )

    jobs_client.list_job_logs.assert_called_once_with(
        workspace="custom",
        name="job-a",
        query_params={"tail": 100},
    )


def test_jobs_tail_command_raises_when_no_workspace() -> None:
    """With no explicit and no client-default workspace, the command must surface
    a clean missing-workspace error (matching the removed generated-SDK helper's
    raise-on-missing behavior) rather than crash with an AttributeError.

    The accessor's ValueError is translated by the CLI error handler into a
    typer.Exit, so no job-logs request is made."""
    client = _client(workspace=None)
    jobs_client = MagicMock()

    with (
        patch("nemo_helix_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_helix_ext.cli.commands.jobs.render_job_logs"),
        pytest.raises(typer.Exit) as exc_info,
    ):
        tail_platform_job(
            _ctx(client),
            name="job-a",
            lines=100,
            workspace=None,
            attempt_id=None,
            step_id=None,
            task_id=None,
        )

    # Exit code 2 is the clean missing-workspace path; the original AttributeError
    # regression would exit 1, so this pins the intended behavior.
    assert exc_info.value.exit_code == 2
    jobs_client.list_job_logs.assert_not_called()
