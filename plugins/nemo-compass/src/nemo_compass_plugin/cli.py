# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compass scheduling CLI."""

from importlib.metadata import entry_points
from typing import Any, ClassVar

import click
import typer
from nemo_compass_plugin.client import CompassClient
from nemo_compass_plugin.model_guidance import model_warning
from nemo_compass_plugin.schema import EnableAnalysisConfigRequest
from nemo_compass_plugin.sdk_resources.analysis_configs import _list_params as _config_list_params
from nemo_helix_plugin.cli import NemoCLI, create_typer_app
from nemo_helix_plugin.cli_codegen import handle_code_generation
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_options import (
    WORKSPACE_FLAGS,
    AllPagesOption,
    EntityOutputFormatOption,
    ListOutputFormat,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    workspace_help,
)
from nemo_helix_plugin.cli_output import Column, check_output_columns_with_format, format_output
from nemo_helix_plugin.cli_pagination import PaginationType, collect_offset_pages, warn_if_more_pages
from nemo_helix_plugin.cli_state import CLIState, cli_state, resolve_cli_workspace, resolve_output_format
from nemo_helix_plugin.cli_warnings import collect_warnings
from nemo_helix_plugin.nooa_model_client import configured_fast_model, configured_model_refs

_CONFIG_COLUMNS = [
    Column("agent"),
    Column("enabled"),
    Column("default_model"),
    Column("fast_model"),
    Column("updated_at"),
]


class CompassCLI(NemoCLI):
    """``nemo compass ...`` subcommands."""

    name: ClassVar[str] = "compass"
    description: ClassVar[str] = "Analyze agent telemetry and act on insights."

    def get_cli(self) -> typer.Typer:
        app = create_typer_app(help=self.description)

        @app.callback()
        def _root() -> None:
            """Force subcommand dispatch even when only one verb is registered."""

        analysis_app = create_typer_app(help="Manage periodic agent analysis opt-in state.")
        app.add_typer(analysis_app, name="analysis")

        @analysis_app.command("enable")
        @collect_warnings
        @handle_errors
        def enable_analysis(
            ctx: typer.Context,
            agent: str = typer.Option(
                ...,
                "--agent",
                help="Name of the agent to opt in to periodic analysis.",
            ),
            workspace: str | None = typer.Option(
                None,
                *WORKSPACE_FLAGS,
                help=workspace_help("Workspace the agent belongs to."),
            ),
            default_model: str | None = typer.Option(
                None,
                "--default-model",
                help="Model Entity ref for analysis work. Default: the configured default model.",
            ),
            fast_model: str | None = typer.Option(
                None,
                "--fast-model",
                help="Faster Model Entity ref Compass uses for part of its analysis. Default: the configured fast model.",
            ),
            output_format: EntityOutputFormatOption = None,
        ) -> None:
            """Enable periodic analysis for an agent."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            default_model, fast_model = _resolve_model_refs(default_model, fast_model)
            kwargs = {
                "workspace": resolve_cli_workspace(ctx, workspace),
                "agent": agent,
                "body": EnableAnalysisConfigRequest(default_model=default_model, fast_model=fast_model),
            }
            if handle_code_generation(CompassClient, "enable_analysis_config", kwargs, resolved_output_format, state):
                return
            response = state.typed_client(CompassClient).enable_analysis_config(**kwargs)
            format_output(response, output_format=resolved_output_format)

        @analysis_app.command("disable")
        @collect_warnings
        @handle_errors
        def disable_analysis(
            ctx: typer.Context,
            agent: str = typer.Option(
                ...,
                "--agent",
                help="Name of the agent to opt out of periodic analysis.",
            ),
            workspace: str | None = typer.Option(
                None,
                *WORKSPACE_FLAGS,
                help=workspace_help("Workspace the agent belongs to."),
            ),
            output_format: EntityOutputFormatOption = None,
        ) -> None:
            """Disable periodic analysis for an agent."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            kwargs = {"workspace": resolve_cli_workspace(ctx, workspace), "agent": agent}
            if handle_code_generation(CompassClient, "disable_analysis_config", kwargs, resolved_output_format, state):
                return
            response = state.typed_client(CompassClient).disable_analysis_config(**kwargs)
            format_output(response, output_format=resolved_output_format)

        @analysis_app.command("status")
        @collect_warnings
        @handle_errors
        def analysis_status(
            ctx: typer.Context,
            agent: str | None = typer.Option(
                None,
                "--agent",
                help="Optional agent name. Omit to list all analysis configs.",
            ),
            workspace: str | None = typer.Option(
                None,
                *WORKSPACE_FLAGS,
                help=workspace_help("Workspace to inspect."),
            ),
            page: int = typer.Option(1, "--page", help="Page number (1-indexed). Ignored with --agent."),
            page_size: int = typer.Option(100, "--page-size", help="Items per page. Ignored with --agent."),
            all_pages: AllPagesOption = False,
            output_format: ListOutputFormatOption = None,
            no_truncate: NoTruncateOption = None,
            columns: OutputColumnsOption = None,
        ) -> None:
            """Show periodic analysis opt-in state."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            workspace = resolve_cli_workspace(ctx, workspace)
            if agent:
                kwargs: dict[str, Any] = {"workspace": workspace, "agent": agent}
                if handle_code_generation(CompassClient, "get_analysis_config", kwargs, resolved_output_format, state):
                    return
                response = state.typed_client(CompassClient).get_analysis_config(**kwargs)
                format_output(response, output_format=resolved_output_format)
                return

            check_output_columns_with_format(columns, resolved_output_format)
            kwargs = {
                "workspace": workspace,
                "query_params": _config_list_params(page=page, page_size=page_size, sort="-created_at", enabled=None),
            }
            if handle_code_generation(
                CompassClient,
                "list_analysis_configs",
                kwargs,
                resolved_output_format,
                state,
                result="all-pages" if all_pages else "list",
            ):
                return
            response = state.typed_client(CompassClient).list_analysis_configs(**kwargs)
            _print_list(
                state,
                collect_offset_pages(response, all_pages=all_pages),
                output_format=resolved_output_format,
                columns=columns,
                default_columns=_CONFIG_COLUMNS,
                no_truncate=no_truncate,
                all_pages=all_pages,
            )

        for entry_point in sorted(entry_points(group="nemo.compass.commands"), key=lambda item: item.name):
            app.add_typer(entry_point.load()(), name=entry_point.name)
        return app


def _print_list(
    state: CLIState,
    result: Any,
    *,
    output_format: ListOutputFormat,
    columns: str | None,
    default_columns: list[Column],
    no_truncate: bool | None,
    all_pages: bool,
) -> None:
    """Render one page (or all pages) of a list command, then warn if more pages remain."""
    output_columns: str | list[Column] = default_columns
    if columns is not None and columns.strip() != "default":
        output_columns = columns
    format_output(
        result,
        is_list=True,
        output_format=output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
    )
    if not all_pages:
        warn_if_more_pages(result, PaginationType.PAGE_NUMBER)


def _resolve_model_refs(default_model: str | None, fast_model: str | None) -> tuple[str, str]:
    pair = _configured_model_pair(default_model, fast_model)
    for model in dict.fromkeys(pair):
        if warning := model_warning(model):
            typer.echo(f"Warning: {warning}", err=True)
    return pair


def _configured_model_pair(default_model: str | None, fast_model: str | None) -> tuple[str, str]:
    """Fill either model ref from the operator's CLI config when not given.

    The Helix process cannot read that config, so the request has to carry
    the pair; the CLI is where it is known.
    """
    if default_model and fast_model:
        return default_model, fast_model
    if default_model:
        return default_model, configured_fast_model() or default_model
    try:
        configured = configured_model_refs()
    except ValueError as exc:
        raise click.ClickException(f"{exc} Or pass --default-model and --fast-model.") from None
    return default_model or configured.default, fast_model or configured.fast
