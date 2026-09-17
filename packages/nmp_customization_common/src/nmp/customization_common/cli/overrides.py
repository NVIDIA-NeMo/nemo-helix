# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI override machinery for customization contributor plugins.

After the platform's ``_add_submit_command`` registers the default submit verb,
Customizer backends swap in the same shape:

- ``submit`` → positional ``JOB_JSON`` argument plus standard submit flags;
  loads + validates the JSON (via the backend's ``load_job_json``), then
  delegates to the original ``submit`` callback with ``--spec`` set.
- any pre-existing generated ``run`` command is removed; Customizer jobs are
  submitted to the platform, not executed through local CLI scheduling.
- ``explain`` → unchanged.

After a successful submit, the wrapper reports the created job and the commands
that track it, and follows the job to a terminal state for ``--wait`` and
``--watch``. That reporting lives in
``nmp.customization_common.cli.tracking``.

Only the backend's name, ``load_job_json``, ``JOB_JSON`` help text and ``submit``
help text differ; everything else is shared here.
"""

from collections.abc import Callable
from pathlib import Path

import typer
from nemo_platform_plugin.cli_options import WorkspaceOption
from nemo_platform_plugin.cli_state import resolve_cli_workspace
from nemo_platform_plugin.commands import SubmittedJob
from nmp.customization_common.cli.tracking import FollowResult, follow_job, print_tracking_instructions

LoadJobJson = Callable[[Path], str]

_LIFECYCLE_PANEL = "Lifecycle Options"


def apply_job_cli_overrides(
    group: typer.Typer,
    *,
    backend: str,
    load_job_json: LoadJobJson,
    job_json_help: str,
    submit_help: str | None = None,
) -> None:
    """Drop generated ``run``/``submit`` verbs, then re-register submit.

    Order matters: drop first, then re-register. Typer iterates
    ``registered_commands`` in insertion order, so stale entries would route
    users back to the auto-generated shapes.
    """
    _drop_command(group, "run")
    _replace_job_submit(group, backend, load_job_json, job_json_help, submit_help)


def _pluck_callback(group: typer.Typer, verb: str) -> Callable[..., SubmittedJob | None]:
    command = next((c for c in group.registered_commands if c.name == verb), None)
    if command is None or command.callback is None:
        raise RuntimeError(f"missing {verb!r} callback to override")
    return command.callback


def _drop_command(group: typer.Typer, name: str) -> None:
    group.registered_commands = [c for c in group.registered_commands if c.name != name]


def _replace_job_submit(
    group: typer.Typer,
    backend: str,
    load_job_json: LoadJobJson,
    job_json_help: str,
    submit_help: str | None = None,
) -> None:
    """Replace ``submit`` with a ``JOB_JSON`` positional + standard submit flags."""
    original = _pluck_callback(group, "submit")
    # Drop the original before re-registering so we don't leave a duplicate
    # ``submit`` entry (Typer would otherwise keep both and dispatch the last).
    _drop_command(group, "submit")

    @group.command("submit", help=submit_help)
    def submit(
        typer_ctx: typer.Context,
        job_json: Path = typer.Argument(..., metavar="JOB_JSON", help=job_json_help),
        workspace: WorkspaceOption = None,
        profile: str | None = typer.Option(
            None,
            "--profile",
            help="Execution profile to run the job on. Uses the backend default when omitted.",
        ),
        cluster: str | None = typer.Option(
            None,
            "--cluster",
            help="Name of a cluster in the CLI config. Submits to that cluster's base URL.",
        ),
        base_url: str | None = typer.Option(
            None,
            "--base-url",
            help=(
                "Override platform API host. If omitted: --cluster, then CLI context, "
                "then $NMP_BASE_URL, then http://localhost:8080."
            ),
        ),
        options: list[str] = typer.Option([], "-o", help="Backend option override, 'backend.key=value'."),
        options_file: Path | None = typer.Option(
            None,
            "--options-file",
            help="JSON or YAML file of backend option overrides. Any -o flag wins over it.",
        ),
        wait: bool = typer.Option(
            False,
            "--wait",
            help="Wait for the job to finish, showing its status.",
            rich_help_panel=_LIFECYCLE_PANEL,
        ),
        watch: bool = typer.Option(
            False,
            "--watch",
            help="Wait for the job to finish, showing its status and logs.",
            rich_help_panel=_LIFECYCLE_PANEL,
        ),
        timeout: int | None = typer.Option(
            None,
            "--timeout",
            min=1,
            help="Give up waiting after this many seconds. Waits indefinitely when omitted.",
            rich_help_panel=_LIFECYCLE_PANEL,
        ),
        poll_interval: int = typer.Option(
            3,
            "--poll-interval",
            min=1,
            help="Seconds between status checks while waiting.",
            rich_help_panel=_LIFECYCLE_PANEL,
        ),
    ) -> None:
        workspace = resolve_cli_workspace(typer_ctx, workspace)

        if wait and watch:
            raise typer.BadParameter("Use either --wait or --watch, not both.")

        spec_json = load_job_json(job_json)
        submitted = original(
            typer_ctx,
            spec=spec_json,
            spec_file=None,
            options=options,
            options_file=options_file,
            profile=profile,
            cluster=cluster,
            base_url=base_url,
            workspace=workspace,
            config=None,
            config_file=None,
        )
        _report_submitted(
            submitted, backend=backend, wait=wait, watch=watch, timeout=timeout, poll_interval=poll_interval
        )


def _report_submitted(
    submitted: SubmittedJob | None,
    *,
    backend: str,
    wait: bool,
    watch: bool,
    timeout: int | None,
    poll_interval: int,
) -> None:
    """Print tracking instructions, then follow the job when asked to."""
    # ``submitted`` is None only when the platform response carried no job, which
    # the submit callback has already reported. There is nothing left to track.
    if submitted is None:
        return
    job_name = submitted.name
    if job_name is None:
        return

    print_tracking_instructions(
        backend=backend,
        job_name=job_name,
        workspace=submitted.workspace,
    )
    if not (wait or watch):
        return

    result = follow_job(
        base_url=submitted.base_url,
        job_name=job_name,
        workspace=submitted.workspace,
        headers=submitted.headers,
        include_logs=watch,
        timeout=timeout,
        poll_interval=poll_interval,
    )
    if result is FollowResult.INTERRUPTED:
        raise typer.Exit(code=130)
    if result is not FollowResult.SUCCEEDED:
        raise typer.Exit(code=1)
