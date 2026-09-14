# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents optimize prepare-fileset`` — stage an optimize bundle for remote submission.

The remote optimize command takes a fileset ref and nothing else, because the job runs on
the platform and cannot see the submitting client's filesystem. Something has to put the bundle
there first, and that is this command. It is a separate job-specific command rather than an
implicit upload inside the remote submission so the expensive, stateful half (validate a tree,
create a fileset, push bytes) stays explicit and re-runnable, and so a bundle can be staged once
and submitted many times.

Lives in the agents plugin because agents owns the ``optimize`` CLI surface and the fileset
helpers; the validation itself is library code in :mod:`nemo_optimization.bundle`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from nemo_agents_plugin.cli_context import BaseUrlOption, resolve_base_url, resolve_context_headers
from nemo_agents_plugin.jobs.fileset_io import split_fileset_ref, upload_to_fileset

logger = logging.getLogger(__name__)

PREPARE_FILESET_COMMAND = "prepare-fileset"


def register_prepare_fileset_command(group: typer.Typer) -> None:
    """Add ``prepare-fileset`` to the generated ``optimize`` job command."""

    @group.command(
        name=PREPARE_FILESET_COMMAND,
        help="Validate an optimize bundle and upload it to a fileset for `optimize`.",
    )
    def prepare_fileset(
        source: Annotated[
            Path,
            typer.Option(
                "--source",
                help="Directory holding the optimize config and every asset it references.",
                exists=True,
                file_okay=False,
                dir_okay=True,
            ),
        ],
        optimize_config: Annotated[
            str,
            typer.Option(
                "--optimize-config",
                help="Optimize YAML, as a path relative to --source. This is the value to pass to "
                "`optimize --optimize-config`.",
            ),
        ],
        fileset: Annotated[
            str,
            typer.Option("--fileset", help="Fileset to upload into ('name' or 'workspace/name'). Created if missing."),
        ],
        workspace: Annotated[
            str,
            typer.Option("--workspace", help="Workspace for the fileset and for agent / model preflight."),
        ] = "default",
        agent: Annotated[
            Optional[str],
            typer.Option(
                "--agent",
                help="Platform agent supplying the Agent under Test, for configs that carry only "
                "the optimizer and eval overlay.",
            ),
        ] = None,
        check_models: Annotated[
            bool,
            typer.Option(
                "--check-models/--no-check-models",
                help="Also resolve the config's models against the platform before uploading.",
            ),
        ] = True,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Run preflight and print the result without uploading."),
        ] = False,
        base_url: BaseUrlOption = None,
    ) -> None:
        from nemo_optimization.bundle import BundlePreflightError, preflight_bundle

        try:
            config = preflight_bundle(source, optimize_config, agent=agent)
        except BundlePreflightError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

        ws, name = split_fileset_ref(fileset, workspace)
        if dry_run and not check_models:
            typer.echo(f"Preflight passed. Would upload {source}/ to fileset {ws}/{name}.")
            return

        sdk = _platform_sdk(resolve_base_url(base_url))
        if check_models:
            _preflight_models(config, workspace=workspace, agent=agent, sdk=sdk)
        if dry_run:
            typer.echo(f"Preflight passed. Would upload {source}/ to fileset {ws}/{name}.")
            return

        try:
            upload_to_fileset(source, fileset=name, workspace=ws, sdk=sdk)
        except Exception as exc:
            typer.echo(f"Error: failed to upload {source} to fileset {ws}/{name}: {exc}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Staged {source}/ to fileset {ws}/{name}.\n")
        typer.echo("Submit the study with:\n")
        typer.echo(
            f"  nemo agents optimize \\\n"
            f"    --optimize-config-fileset {ws}/{name} \\\n"
            f"    --optimize-config {optimize_config} \\\n"
            + (f"    --agent {agent} \\\n" if agent else "")
            + f"    --workspace {workspace}"
        )


def _preflight_models(config: dict[str, Any], *, workspace: str, agent: str | None, sdk: Any) -> None:
    """Resolve the config's models against the platform, warning rather than failing.

    A model that does not resolve is worth knowing about before a study burns trials on it, but it
    is not a reason to refuse to stage the bundle: the model may be created between staging and
    submit, and ``--no-check-models`` should not be the price of an offline `prepare-fileset`.
    """
    from nemo_optimization.agents import resolve_agent_config
    from nemo_optimization.preflight import preflight_validate_llm_models

    try:
        agent_config = resolve_agent_config(agent, workspace=workspace, sdk=sdk)
        preflight_validate_llm_models(config, workspace=workspace, sdk=sdk, agent_config=agent_config)
    except Exception as exc:
        typer.echo(f"Warning: model preflight did not pass: {exc}", err=True)


def _platform_sdk(base_url: str) -> Any:
    """An auth-aware platform SDK client for the fileset upload."""
    from nemo_platform import NeMoPlatform

    headers = resolve_context_headers()
    if headers:
        return NeMoPlatform(base_url=base_url, default_headers=headers)
    return NeMoPlatform(base_url=base_url)
