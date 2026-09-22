# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nemo_platform_ext.cli.commands.jobs import tail_platform_job


class _PlatformClient:
    _get_workspace_path_param: MagicMock

    def __init__(self) -> None:
        self._get_workspace_path_param = MagicMock(return_value="default")


def _ctx(client: _PlatformClient) -> SimpleNamespace:
    state = MagicMock()
    state.get_client.return_value = client
    return SimpleNamespace(obj=state)


def test_jobs_tail_command_fetches_tail_page() -> None:
    client = _PlatformClient()
    logs = [object(), object()]
    response = MagicMock()
    response.page.return_value = SimpleNamespace(items=logs)
    jobs_client = MagicMock()
    jobs_client.list_job_logs.return_value = response

    with (
        patch("nemo_platform_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_platform_ext.cli.commands.jobs.render_job_logs") as render_logs,
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
    client = _PlatformClient()
    response = MagicMock()
    response.page.return_value = SimpleNamespace(items=[])
    jobs_client = MagicMock()
    jobs_client.list_job_logs.return_value = response

    with (
        patch("nemo_platform_ext.cli.commands.jobs.JobsClient.from_client", return_value=jobs_client),
        patch("nemo_platform_ext.cli.commands.jobs.render_job_logs"),
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

    client._get_workspace_path_param.assert_not_called()
    jobs_client.list_job_logs.assert_called_once_with(
        workspace="custom",
        name="job-a",
        query_params={"tail": 100},
    )
