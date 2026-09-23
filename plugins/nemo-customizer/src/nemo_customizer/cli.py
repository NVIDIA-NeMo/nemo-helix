# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI router for customization — mounts contributor subgroups."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import ClassVar

import typer
from nemo_helix_plugin.cli import NemoCLI
from nemo_helix_plugin.cli_options import WorkspaceOption
from nemo_helix_plugin.cli_state import resolve_cli_workspace
from nemo_helix_plugin.customization_contributor import (
    CustomizationCLISummaryProvider,
    CustomizationContributorDiscoveryError,
)
from nemo_helix_plugin.discovery import (
    CUSTOMIZATION_CONTRIBUTORS_GROUP,
    discover_customization_contributors,
)
from nhx.customization_common.cli.uploads import UploadReport

# The router is deliberately backend-neutral: it never names automodel, unsloth or
# rl. Backend-specific text comes from each contributor's get_cli_summary().
_OVERVIEW = """Train a model on your own data.

Choose a backend, write a job JSON for it, and submit it. The platform
creates the job and runs the training on a GPU execution profile. Each backend
trains a different way, and the schema of the job JSON depends on the backend
you choose."""

_UPLOAD_PANEL = "Resource Creation Options"

_NEXT_STEPS = """Run 'nemo customization <backend> --help' for the full description of a
backend, or 'nemo customization <backend> explain' to print its job JSON
schema."""


class CustomizationCLIError(CustomizationContributorDiscoveryError):
    """Raised when the customization CLI cannot start."""


class CustomizationCLI(NemoCLI):
    """``nemo customization`` root command."""

    name: ClassVar[str] = "customization"
    description: ClassVar[str] = "Train a model on your own data with an installed training backend."

    def __init__(self) -> None:
        self._contributors = discover_customization_contributors()
        if not self._contributors:
            raise CustomizationCLIError(
                "Customization CLI is enabled but no contributors were discovered. "
                "Install a backend plugin (e.g. nemo-automodel) and ensure "
                f"'{CUSTOMIZATION_CONTRIBUTORS_GROUP}' entry points are registered.",
            )

    def get_cli(self) -> typer.Typer:
        # A contributor may return no CLI. Collect the subgroups first so the help
        # lists only backends that have a command to run.
        subgroups = {
            key: subgroup
            for key in sorted(self._contributors.keys())
            if (subgroup := self._contributors[key].get_cli()) is not None
        }
        app = typer.Typer(
            name=self.name,
            help=self._compose_help(subgroups.keys()),
            no_args_is_help=True,
        )

        _add_upload_callback(app)

        for key, subgroup in subgroups.items():
            app.add_typer(subgroup, name=key)

        return app

    def _compose_help(self, mounted: Iterable[str]) -> str:
        """Overview, one summary per mounted backend in name order, then next steps."""
        blocks = [_OVERVIEW, "Installed backends:"]

        for key in mounted:
            # Contributing a summary is optional, so a backend without one is
            # listed by name instead of breaking --help for the others.
            contributor = self._contributors[key]
            summary = (
                contributor.get_cli_summary() if isinstance(contributor, CustomizationCLISummaryProvider) else None
            )
            blocks.append(summary.render(key) if summary is not None else key)

        blocks.append(_NEXT_STEPS)
        return "\n\n".join(blocks)


def _add_upload_callback(app: typer.Typer) -> None:
    """Let ``nemo customization`` create a model and a dataset on its own.

    Creating these needs no backend: it is the same fileset, upload and model
    entity that ``nemo files`` and ``nemo models`` create today. Only ``submit``
    has to know where the reference belongs in a backend's job JSON.
    """

    @app.callback(invoke_without_command=True)
    def customization(
        typer_ctx: typer.Context,
        upload_model: str | None = typer.Option(
            None,
            "--upload-model",
            metavar="SOURCE",
            help="Local path to model weights, or a HuggingFace repo id. Creates the fileset and model entity.",
            rich_help_panel=_UPLOAD_PANEL,
        ),
        upload_dataset: str | None = typer.Option(
            None,
            "--upload-dataset",
            metavar="SOURCE",
            help=(
                "Local file or directory to upload as the dataset. Pass a directory to send "
                "several files, the same as 'nemo files upload'."
            ),
            rich_help_panel=_UPLOAD_PANEL,
        ),
        upload_environment: str | None = typer.Option(
            None,
            "--upload-environment",
            metavar="SOURCE",
            help="Local directory to upload as a NeMo Gym environment, for GRPO.",
            rich_help_panel=_UPLOAD_PANEL,
        ),
        exist_ok: bool = typer.Option(
            False,
            "--exist-ok",
            help="Reuse whichever of the above already exists. Its files are left as they are.",
            rich_help_panel=_UPLOAD_PANEL,
        ),
        workspace: WorkspaceOption = None,
        base_url: str | None = typer.Option(None, "--base-url", help="Override the platform API host."),
        cluster: str | None = typer.Option(None, "--cluster", help="Name of a cluster in the CLI config."),
        hf_token_secret: str | None = typer.Option(
            None,
            "--hf-token-secret",
            help="Platform secret holding a HuggingFace token, for a gated or private repo.",
            rich_help_panel=_UPLOAD_PANEL,
        ),
    ) -> None:
        # The callback also runs on the way to a backend subcommand, where these
        # flags do not apply and the subcommand owns the work. Refuse them there
        # rather than drop them: `nemo customization --upload-dataset x automodel
        # submit` would otherwise submit without uploading anything. Every option
        # here defaults to None or False and none reads an env var, so a set value
        # means the user typed the flag.
        if typer_ctx.invoked_subcommand is not None:
            given = {
                "--upload-model": upload_model is not None,
                "--upload-dataset": upload_dataset is not None,
                "--upload-environment": upload_environment is not None,
                "--exist-ok": exist_ok,
                "--workspace": workspace is not None,
                "--base-url": base_url is not None,
                "--cluster": cluster is not None,
                "--hf-token-secret": hf_token_secret is not None,
            }
            misplaced = [flag for flag, is_set in given.items() if is_set]
            if misplaced:
                typer_ctx.fail(
                    f"{', '.join(misplaced)} must come after the subcommand: "
                    f"'nemo customization {typer_ctx.invoked_subcommand} submit [OPTIONS] JOB_JSON'."
                )
            return
        if upload_model is None and upload_dataset is None and upload_environment is None:
            typer.echo(typer_ctx.get_help())
            raise typer.Exit()

        workspace = resolve_cli_workspace(typer_ctx, workspace)
        report = _create_customization_resources(
            typer_ctx,
            model_source=upload_model,
            dataset_source=upload_dataset,
            environment_source=upload_environment,
            workspace=workspace,
            base_url=base_url,
            cluster=cluster,
            exist_ok=exist_ok,
            hf_token_secret=hf_token_secret,
        )
        _print_refs(report)


def _create_customization_resources(
    typer_ctx: typer.Context,
    *,
    model_source: str | None,
    dataset_source: str | None,
    environment_source: str | None,
    workspace: str,
    base_url: str | None,
    cluster: str | None,
    exist_ok: bool,
    hf_token_secret: str | None,
) -> UploadReport:
    from nemo_helix_plugin.commands import resolve_submit_auth_headers, resolve_submit_base_url
    from nemo_helix_plugin.files.client import FilesClient
    from nemo_helix_plugin.models.client import ModelsClient
    from nhx.customization_common.cli.overrides import run_uploads

    resolved_base_url = resolve_submit_base_url(typer_ctx, base_url=base_url, cluster=cluster)
    headers = resolve_submit_auth_headers(typer_ctx) or None
    return run_uploads(
        model_source=model_source,
        dataset_source=dataset_source,
        environment_source=environment_source,
        files=FilesClient(base_url=resolved_base_url, workspace=workspace, default_headers=headers),
        models=ModelsClient(base_url=resolved_base_url, workspace=workspace, default_headers=headers),
        workspace=workspace,
        exist_ok=exist_ok,
        hf_token_secret=hf_token_secret,
    )


def _print_refs(report: UploadReport) -> None:
    """Print the references to put in a job JSON, on stdout so they can be piped."""
    refs = {}
    if report.model_ref is not None:
        refs["model"] = report.model_ref
    if report.dataset_ref is not None:
        refs["dataset"] = report.dataset_ref
    if report.environment_ref is not None:
        refs["environment"] = report.environment_ref
    typer.echo(json.dumps(refs, indent=2))
