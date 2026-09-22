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

Only the backend's ``load_job_json`` and the ``JOB_JSON`` help text differ;
everything else is shared here.
"""

from collections.abc import Callable
from pathlib import Path

import typer
from nemo_platform_plugin.cli_options import WorkspaceOption
from nemo_platform_plugin.cli_state import resolve_cli_workspace

LoadJobJson = Callable[[Path], str]


def apply_job_cli_overrides(
    group: typer.Typer,
    *,
    load_job_json: LoadJobJson,
    job_json_help: str,
) -> None:
    """Drop generated ``run``/``submit`` verbs, then re-register submit.

    Order matters: drop first, then re-register. Typer iterates
    ``registered_commands`` in insertion order, so stale entries would route
    users back to the auto-generated shapes.
    """
    _drop_command(group, "run")
    _replace_job_submit(group, load_job_json, job_json_help)


def _pluck_callback(group: typer.Typer, verb: str) -> Callable[..., None]:
    command = next((c for c in group.registered_commands if c.name == verb), None)
    if command is None or command.callback is None:
        raise RuntimeError(f"missing {verb!r} callback to override")
    return command.callback


def _drop_command(group: typer.Typer, name: str) -> None:
    group.registered_commands = [c for c in group.registered_commands if c.name != name]


def _replace_job_submit(group: typer.Typer, load_job_json: LoadJobJson, job_json_help: str) -> None:
    """Replace ``submit`` with a ``JOB_JSON`` positional + standard submit flags."""
    original = _pluck_callback(group, "submit")
    # Drop the original before re-registering so we don't leave a duplicate
    # ``submit`` entry (Typer would otherwise keep both and dispatch the last).
    _drop_command(group, "submit")

    @group.command("submit")
    def submit(
        typer_ctx: typer.Context,
        job_json: Path = typer.Argument(..., metavar="JOB_JSON", help=job_json_help),
        workspace: WorkspaceOption = None,
        profile: str | None = typer.Option(None, "--profile"),
        cluster: str | None = typer.Option(None, "--cluster"),
        base_url: str | None = typer.Option(
            None,
            "--base-url",
            help=(
                "Override platform API host. If omitted: --cluster, then CLI context, "
                "then $NMP_BASE_URL, then http://localhost:8080."
            ),
        ),
        options: list[str] = typer.Option([], "-o", help="Backend option override, 'backend.key=value'."),
        options_file: Path | None = typer.Option(None, "--options-file"),
    ) -> None:
        workspace = resolve_cli_workspace(typer_ctx, workspace)
        spec_json = load_job_json(job_json)
        original(
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
