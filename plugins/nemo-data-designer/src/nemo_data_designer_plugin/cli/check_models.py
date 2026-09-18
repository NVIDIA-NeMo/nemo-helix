# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution context-aware CLI model health checks for config sources"""

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
from nemo_data_designer_plugin.sdk.check_models import CheckModelsReport, check_models_config_sync


def check_models_command(
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
                "Workspace used to resolve provider references and seed sources. "
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
    """Check that every model referenced by the configuration is reachable.

    Sends a tiny generation request to each referenced model alias, routed
    through the Inference Gateway. Models with skip_health_check=True are
    skipped.

    Complements `validate`: `validate` checks the configuration is well-formed
    and that the platform resources it names resolve; `check-models` checks
    that those models actually respond. A provider can resolve while still
    refusing to serve the model named alongside it, so a green `validate` is
    not a promise that a preview will run.

    Unlike `validate`, this stops at the first model that fails rather than
    reporting every problem at once.
    """
    config_builder = load_builder_or_exit(config_source)
    sdk, async_sdk = resolve_sdks_or_exit(typer_ctx)
    resolved_workspace = resolve_workspace(workspace, sdk=sdk, async_sdk=async_sdk)

    if output == "text":
        print_header("Data Designer Check Models")
        typer.echo(f"  Config: {config_source}")
        typer.echo("")

    report = check_models_config_sync(
        config_builder,
        sdk=sdk,
        async_sdk=async_sdk,
        workspace=resolved_workspace,
        config_source=config_source,
        # The engine names each alias as it probes it, which is the only place
        # that identity appears — the error it raises on failure does not carry
        # it. Suppressed for json so the document stays the only thing on stdout.
        on_log=_echo_engine_log if output == "text" else None,
    )

    if output == "json":
        typer.echo(report.model_dump_json())
    else:
        _render_text_report(report)

    if not report.ok:
        raise typer.Exit(code=1)


def _echo_engine_log(message: str) -> None:
    typer.echo(f"  {message}")


def _render_text_report(report: CheckModelsReport) -> None:
    typer.echo("")

    if report.ok:
        print_success("  ✔ All models responded successfully")
        return

    for err in report.errors:
        print_error(f"  ✘ Model health check failed ({err.error_type}): {err.message}")
