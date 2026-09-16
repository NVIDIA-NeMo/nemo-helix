# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution context-aware CLI validation of config sources"""

from __future__ import annotations

from typing import Annotated, Literal, cast

import typer
from data_designer.cli.ui import print_error, print_header, print_success
from data_designer.cli.utils.config_loader import ConfigLoadError, load_config_builder
from nemo_data_designer_plugin.sdk.validation import (
    ValidationReport,
    validate_config_sync,
)
from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix_plugin.cli_state import resolve_local_cli_sdks

OutputFormat = Literal["text", "json"]


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
    """Validate a Data Designer configuration."""
    try:
        config_builder = load_config_builder(config_source)
    except ConfigLoadError as e:
        print_error(f"Could not load config: {e}")
        raise typer.Exit(code=1) from e

    sdk, async_sdk = resolve_local_cli_sdks(typer_ctx)
    sdk = cast("NeMoHelix | None", sdk)
    async_sdk = cast("AsyncNeMoHelix | None", async_sdk)

    if sdk is None and async_sdk is None:
        print_error(
            "No NeMo Helix SDK is available. Run `nemo` from a configured environment "
            "or supply credentials via the top-level CLI."
        )
        raise typer.Exit(code=1)

    resolved_workspace = (
        workspace
        or (getattr(sdk, "workspace", None) if sdk is not None else None)
        or (getattr(async_sdk, "workspace", None) if async_sdk is not None else None)
        or "default"
    )

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
