# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution context-aware CLI validation of config sources"""

from __future__ import annotations

from typing import Annotated

import typer
from data_designer.cli.ui import print_error, print_header, print_success
from nemo_data_designer_plugin.cli._context import (
    OutputFormat,
    load_builder_or_exit,
    resolve_sdks_or_exit,
    resolve_workspace,
)
from nemo_data_designer_plugin.sdk.validation import ValidationReport, validate_config_sync


def validate_command(
    typer_ctx: typer.Context,
    config_source: Annotated[
        str,
        typer.Argument(
            help=(
                "Path or URL to a config file (.yaml/.yml/.json), or a local Python module (.py)"
                " that defines a load_config_builder() function."
            ),
        ),
    ],
    workspace: Annotated[
        str | None,
        typer.Option(
            "--workspace",
            help=(
                "Workspace used to resolve provider references and seed sources (remote pass). "
                "Defaults to the SDK's configured workspace, or 'default'."
            ),
        ),
    ] = None,
    output: Annotated[
        OutputFormat,
        typer.Option(
            "--output",
            help="Output format. 'json' suppresses the human-formatted blocks.",
        ),
    ] = "text",
) -> None:
    """Validate a Data Designer configuration.

    Checks that the configuration is well-formed and that the platform
    resources it names resolve: Inference Gateway providers, Files service seed
    sources, and Nemotron Personas filesets.

    Does not check whether those models actually respond — a provider can
    resolve while still refusing to serve the model named alongside it. Run
    `nemo data-designer check-models` for that.
    """
    config_builder = load_builder_or_exit(config_source)
    sdk, async_sdk = resolve_sdks_or_exit(typer_ctx)
    resolved_workspace = resolve_workspace(workspace, sdk=sdk, async_sdk=async_sdk)

    report = validate_config_sync(
        config_builder,
        sdk=sdk,
        async_sdk=async_sdk,
        workspace=resolved_workspace,
        config_source=config_source,
    )

    if output == "json":
        typer.echo(report.model_dump_json())
    else:
        _render_text_report(report, config_source=config_source)

    if not report.ok:
        raise typer.Exit(code=1)


def _render_text_report(report: ValidationReport, *, config_source: str) -> None:
    print_header("Data Designer Validate")
    typer.echo(f"  Config: {config_source}")
    typer.echo("")

    if report.ok:
        print_success("  ✔ Configuration is valid")
        return

    for err in report.errors:
        print_error(f"  ✘ {err.message}")
