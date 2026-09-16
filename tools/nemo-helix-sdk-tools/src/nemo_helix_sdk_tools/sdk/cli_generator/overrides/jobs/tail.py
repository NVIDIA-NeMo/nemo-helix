# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

import typer
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.help_formatter import collect_warnings
from nemo_helix_ext.cli.core.job_log_renderer import render_job_logs
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import JobLogsQueryParams

app = typer.Typer()  # override-skip: provided by generated file


@app.command("tail")
@collect_warnings
@handle_errors
def tail_platform_job(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Name of the platform job whose logs to tail")],
    lines: Annotated[int, typer.Option("-n", "--lines", min=1, max=10_000, help="Number of lines to show")] = 100,
    workspace: Annotated[str | None, typer.Option("--workspace", help="Workspace containing the job")] = None,
    attempt_id: Annotated[int | None, typer.Option("--attempt-id", help="Filter logs to an attempt ID")] = None,
    step_id: Annotated[str | None, typer.Option("--step-id", help="Filter logs to a step ID")] = None,
    task_id: Annotated[str | None, typer.Option("--task-id", help="Filter logs to a task ID")] = None,
) -> None:
    """Print the newest lines from a platform job log."""
    state: CLIContext = ctx.obj
    client = state.get_client()
    jobs_client = client_from_platform(client, JobsClient)
    if workspace is None:
        workspace = client._get_workspace_path_param()

    query_params: JobLogsQueryParams = {"tail": lines}
    if attempt_id is not None:
        query_params["attempt_id"] = attempt_id
    if step_id is not None:
        query_params["step_id"] = step_id
    if task_id is not None:
        query_params["task_id"] = task_id

    response = jobs_client.list_job_logs(workspace=workspace, name=name, query_params=query_params)
    render_job_logs(response.page().items)
