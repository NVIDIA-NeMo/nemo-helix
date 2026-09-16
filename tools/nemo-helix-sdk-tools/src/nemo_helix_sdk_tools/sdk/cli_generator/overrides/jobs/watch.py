# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Any, cast

import typer
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.help_formatter import collect_warnings
from nemo_helix_ext.cli.core.job_watch_renderer import JobWatchRenderResult, render_job_watch_events
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.jobs.client import JobsClient

app = cast(Any, None)  # override-skip: provided by generated file


@app.command("watch")
@collect_warnings
@handle_errors
def watch_platform_job(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Name of the platform job to watch")],
    workspace: Annotated[str | None, typer.Option("--workspace", help="Workspace containing the job")] = None,
    attempt_id: Annotated[int | None, typer.Option("--attempt-id", help="Filter logs to an attempt ID")] = None,
    step_id: Annotated[str | None, typer.Option("--step-id", help="Filter logs to a step ID")] = None,
    task_id: Annotated[str | None, typer.Option("--task-id", help="Filter logs to a task ID")] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Maximum logs to fetch per page")] = None,
    timeout: Annotated[int | None, typer.Option("--timeout", min=1, help="Maximum watch time in seconds")] = None,
    poll_interval: Annotated[
        int,
        typer.Option("--poll-interval", min=1, help="Seconds between status checks"),
    ] = 3,
    include_history: Annotated[
        bool,
        typer.Option("--history/--no-history", help="Include logs already present before watching"),
    ] = True,
) -> None:
    """Watch a platform job until it reaches a terminal status."""
    state: CLIContext = ctx.obj
    client = state.get_client()
    jobs_client = client_from_platform(client, JobsClient)
    if workspace is None:
        workspace = client._get_workspace_path_param()

    events = jobs_client.watch_job(
        name,
        workspace=workspace,
        attempt_id=attempt_id,
        step_id=step_id,
        task_id=task_id,
        limit=limit,
        timeout=timeout,
        poll_interval=poll_interval,
        include_history=include_history,
    )
    watch_result = render_job_watch_events(events, resource_label="job")
    if watch_result is JobWatchRenderResult.INTERRUPTED:
        raise typer.Exit(130)
    if watch_result is not JobWatchRenderResult.SUCCEEDED:
        raise typer.Exit(1)
