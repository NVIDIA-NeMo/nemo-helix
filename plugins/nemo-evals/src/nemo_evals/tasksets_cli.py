# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo evals tasksets``: publish a taskset and build its images apart from submitting a run.

``runs submit --harbor-*`` does all three steps and waits for the image builds, which can take an hour for a
large dataset. These commands split it: ``tasksets create`` publishes, ``tasksets build-images`` starts the builds
and returns (``--status`` checks them, ``--wait`` follows them), and ``runs submit --taskset`` submits once every
image is built.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import typer
from nemo_evals.image_builds.models import ImageBuild
from nemo_evals.runs_cli import (
    SelectionError,
    _build_state,
    _duration,
    _print_tasks,
    _workspace_flag,
    builds_summary,
    check_git_source,
    create_taskset,
    dataset_flag,
    failed_builds,
    qualify_ref,
    read_builds,
    request_builds,
    taskset_members,
    wait_for_builds,
)
from nemo_evals.sdk.resources import Evaluator
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoClientError

STATUS_BUILDING = 3
"""``build-images --status`` exit code while some build is in progress; 2 is the usage-error code."""


def add_tasksets_commands(app: typer.Typer, client_factory: Callable[[], NemoClient]) -> None:
    """Register ``tasksets create`` and ``tasksets build-images`` under ``app``."""
    tasksets = typer.Typer(
        name="tasksets", help="Harbor tasksets for Kubernetes runs: publish, build images.", no_args_is_help=True
    )
    app.add_typer(tasksets)

    @tasksets.command("create")
    def create(
        harbor_dir: Path | None = typer.Option(None, "--harbor-dir", help="Local Harbor dataset directory."),
        harbor_hub: str | None = typer.Option(None, "--harbor-hub", help="Harbor Hub dataset, org/name@ref."),
        harbor_git: str | None = typer.Option(
            None,
            "--harbor-git",
            help="Public git repo of Harbor tasks, pinned to a tag or commit: `<url>@<ref>` reads `tasks/` at that ref. "
            "For another directory, use a GitHub tree URL: `https://github.com/<org>/<repo>/tree/<ref>/<dir>`.",
        ),
        tasks: str | None = typer.Option(None, "--tasks", help="Indexes or names, comma-separated; default all."),
        list_tasks: bool = typer.Option(False, "--list-tasks", help="Print the selectable tasks and exit."),
        workspace: str | None = typer.Option(
            None, "--workspace", help="Workspace for the tasks and the taskset; default: the configured workspace."
        ),
        output_json: bool = typer.Option(False, "--json", help="Print the taskset as JSON."),
    ) -> None:
        """Publish the selected tasks and a taskset of exactly those; without --json the last line printed is its ref."""
        sources = [
            flag
            for flag, value in (
                ("--harbor-dir", harbor_dir),
                ("--harbor-hub", harbor_hub),
                ("--harbor-git", harbor_git),
            )
            if value is not None
        ]
        if len(sources) != 1:
            raise typer.BadParameter("pass one of --harbor-dir, --harbor-hub or --harbor-git")
        if harbor_git is not None:
            try:
                check_git_source(harbor_git)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--harbor-git") from exc
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        echo = (lambda _: None) if output_json else typer.echo
        try:
            if list_tasks:
                _print_tasks(client_factory, harbor_dir, harbor_hub, harbor_git, None, workspace)
                return
            created = create_taskset(
                Evaluator.from_client(client),
                harbor_dir=harbor_dir,
                harbor_hub=harbor_hub,
                harbor_git=harbor_git,
                tasks=tasks,
                workspace=workspace,
                echo=echo,
            )
        except (SelectionError, RuntimeError, NemoClientError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        dataset = dataset_flag(created.dataset) if created.dataset is not None else None
        if output_json:
            body = {"taskset_ref": created.ref, "tasks": created.refs, "total": created.total, "dataset": dataset}
            typer.echo(json.dumps(body, indent=2))
            return
        if dataset is not None:
            typer.echo(f"Dataset for Intake: --dataset {dataset}")
        typer.echo(created.ref)

    @tasksets.command("build-images")
    def build_images(
        ref: str = typer.Argument(
            ..., help="Taskset, workspace/name#digest; the workspace defaults to the configured."
        ),
        wait: bool = typer.Option(False, "--wait", help="Follow the builds until each is ready or failed."),
        status: bool = typer.Option(
            False,
            "--status",
            help=f"Only read the builds; start none. Exit 0 when all are ready, 1 when one failed or was never "
            f"requested, {STATUS_BUILDING} while some are building.",
        ),
        poll_seconds: float = typer.Option(10.0, "--poll-seconds", help="Polling interval with --wait."),
    ) -> None:
        """Request every task's image build and return; rerunning it retries failed builds and repairs stuck ones."""
        if wait and status:
            raise typer.BadParameter("--wait and --status don't combine: --status only reads")
        client = client_factory()
        configured = client.resolve_workspace()
        ref = qualify_ref(ref, configured)
        flag = _workspace_flag(ref.partition("/")[0], configured)
        evaluator = Evaluator.from_client(client)
        try:
            code = _build_images(evaluator, ref, wait=wait, status=status, poll_seconds=poll_seconds, flag=flag)
        except KeyboardInterrupt:
            typer.echo("Stopped waiting; the builds keep going. Check with --status, or follow with --wait.")
            raise typer.Exit(code=130) from None
        except (NemoClientError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if code:
            raise typer.Exit(code=code)


def _build_images(evaluator: Evaluator, ref: str, *, wait: bool, status: bool, poll_seconds: float, flag: str) -> int:
    """``build-images``' work; returns its exit code."""
    refs = taskset_members(evaluator, ref)
    if status:
        return _print_status(read_builds(evaluator, refs))
    if not wait:
        typer.echo(builds_summary(request_builds(evaluator, refs)))
        typer.echo("Check with --status, follow with --wait, or submit once all are ready.")
        return 0
    builds = wait_for_builds(evaluator, refs, poll_seconds=poll_seconds, echo=typer.echo)
    if failed_builds(builds) is not None:
        typer.echo("Rerun this command to retry the failed builds.")
        return 1
    typer.echo(f"Submit with: nemo evals runs submit --taskset {ref}{flag}")
    return 0


def _print_status(builds: dict[str, ImageBuild | None], now: datetime | None = None) -> int:
    """Print the summary and each task that isn't ready; return the exit code."""
    now = now or datetime.now(UTC)
    typer.echo(builds_summary(builds))
    states = {ref: _build_state(build) for ref, build in builds.items()}
    for ref, build in builds.items():
        state = states[ref]
        if state == "ready":
            continue
        if build is None:
            detail = ""
        elif state == "failed":
            detail = f"  {build.error or 'its Job ended'}"
        else:
            changed = build.updated_at or build.created_at
            detail = f"  (no change for {_duration((now - changed).total_seconds())})" if changed else ""
        typer.echo(f"  {state:<13} {ref}{detail}")
    if any(state in ("failed", "not requested") for state in states.values()):
        typer.echo("Rerun without --status to retry failed builds, request missing ones and repair stuck ones.")
        return 1
    if "building" in states.values():
        typer.echo("A build with no change for long may have lost its Job; rerun without --status to repair it.")
        return STATUS_BUILDING
    return 0
