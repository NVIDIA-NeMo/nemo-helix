# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for the example plugin: ``nemo example ...``.

This is the reference pattern for plugin CLI commands. Everything comes from
``nemo_helix_plugin``:

- groups from :func:`~nemo_helix_plugin.cli.create_typer_app`;
- the typed client from the ``nemo`` CLI state, so the global ``--base-url``,
  ``--context``, and auth apply (``cli_state(ctx).typed_client(ExampleClient)``);
- the shared ``--workspace`` and ``--output-format`` options, rendered with
  :func:`~nemo_helix_plugin.cli_output.format_output`;
- ``-f code`` via :func:`~nemo_helix_plugin.cli_codegen.handle_code_generation`;
- ``@collect_warnings`` / ``@handle_errors`` for warnings and exit codes.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

import typer
from nemo_example_plugin.client import ExampleClient
from nemo_example_plugin.core import say_hello
from nemo_example_plugin.types.endpoints import ListMiddlewareConfigsQueryParams
from nemo_example_plugin.types.payloads import (
    CreateExampleMiddlewareConfigRequest,
    UpdateExampleMiddlewareConfigRequest,
)
from nemo_helix_plugin.cli import NemoCLI, create_typer_app
from nemo_helix_plugin.cli_codegen import handle_code_generation
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_options import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    WorkspaceOption,
)
from nemo_helix_plugin.cli_output import Column, check_output_columns_with_format, format_output
from nemo_helix_plugin.cli_state import cli_state, resolve_cli_workspace, resolve_output_format
from nemo_helix_plugin.cli_warnings import collect_warnings

_MIDDLEWARE_CONFIG_COLUMNS = [
    Column("name"),
    Column("workspace"),
    Column("blocked_keywords"),
    Column("updated_at"),
]

NameArgument = Annotated[str, typer.Argument(help="Config name.")]


def _keywords(value: str) -> list[str]:
    return [keyword.strip() for keyword in value.split(",")]


class ExampleCLI(NemoCLI):
    """Exposes plugin commands as ``nemo example ...``."""

    name = "example"
    description = "Example plugin commands."

    def get_cli(self) -> typer.Typer:
        app = create_typer_app(help="Example plugin commands.")

        # ── hello ─────────────────────────────────────────────────────

        @app.command()
        def hello(name: str = typer.Option(default="world", help="Name to greet.")) -> None:
            """Greet a name."""
            typer.echo(say_hello(name))

        # ── middleware-configs subgroup ────────────────────────────────

        mw = create_typer_app(help="Manage ExampleMiddlewareConfig entities.")
        app.add_typer(mw, name="middleware-configs")

        @mw.command("create")
        @collect_warnings
        @handle_errors
        def create_middleware_config(
            ctx: typer.Context,
            name: NameArgument,
            blocked_keywords: Optional[str] = typer.Option(None, help="Comma-separated list of keywords to block."),
            block_message: Optional[str] = typer.Option(
                None, help="Refusal message returned when a request is blocked."
            ),
            workspace: WorkspaceOption = None,
            output_format: EntityOutputFormatOption = None,
        ) -> None:
            """Create a new middleware config entity."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            fields: dict[str, Any] = {"name": name}
            if blocked_keywords:
                fields["blocked_keywords"] = _keywords(blocked_keywords)
            if block_message:
                fields["block_message"] = block_message
            kwargs = {
                "workspace": resolve_cli_workspace(ctx, workspace),
                "body": CreateExampleMiddlewareConfigRequest(**fields),
            }
            if handle_code_generation(ExampleClient, "create_middleware_config", kwargs, resolved_output_format, state):
                return
            response = state.typed_client(ExampleClient).create_middleware_config(**kwargs)
            format_output(response, output_format=resolved_output_format)

        @mw.command("list")
        @collect_warnings
        @handle_errors
        def list_middleware_configs(
            ctx: typer.Context,
            workspace: WorkspaceOption = None,
            page: Optional[int] = typer.Option(None, "--page", help="Page number (1-indexed)."),
            page_size: Optional[int] = typer.Option(None, "--page-size", help="Items per page (max 100)."),
            sort: Optional[str] = typer.Option(None, "--sort", help="Sort field; prefix with '-' for descending."),
            output_format: ListOutputFormatOption = None,
            no_truncate: NoTruncateOption = None,
            columns: OutputColumnsOption = None,
        ) -> None:
            """List middleware configs in a workspace."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            check_output_columns_with_format(columns, resolved_output_format)
            query_params: ListMiddlewareConfigsQueryParams = {}
            if page is not None:
                query_params["page"] = page
            if page_size is not None:
                query_params["page_size"] = page_size
            if sort is not None:
                query_params["sort"] = sort
            kwargs = {"workspace": resolve_cli_workspace(ctx, workspace), "query_params": query_params or None}
            # The route returns a plain list rather than a paginated envelope, so the
            # snippet reads the whole response instead of iterating pages.
            if handle_code_generation(
                ExampleClient, "list_middleware_configs", kwargs, resolved_output_format, state, result="entity"
            ):
                return
            response = state.typed_client(ExampleClient).list_middleware_configs(**kwargs)
            format_output(
                response,
                is_list=True,
                output_format=resolved_output_format,
                output_columns=columns if columns and columns.strip() != "default" else _MIDDLEWARE_CONFIG_COLUMNS,
                no_truncate=state.get_no_truncate(no_truncate),
                timestamp_format=state.get_timestamp_format(),
            )

        @mw.command("get")
        @collect_warnings
        @handle_errors
        def get_middleware_config(
            ctx: typer.Context,
            name: NameArgument,
            workspace: WorkspaceOption = None,
            output_format: EntityOutputFormatOption = None,
        ) -> None:
            """Get a single middleware config by name."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            kwargs = {"workspace": resolve_cli_workspace(ctx, workspace), "name": name}
            if handle_code_generation(ExampleClient, "get_middleware_config", kwargs, resolved_output_format, state):
                return
            response = state.typed_client(ExampleClient).get_middleware_config(**kwargs)
            format_output(response, output_format=resolved_output_format)

        @mw.command("update")
        @collect_warnings
        @handle_errors
        def update_middleware_config(
            ctx: typer.Context,
            name: NameArgument,
            blocked_keywords: Optional[str] = typer.Option(
                None, help="Comma-separated keywords (replaces existing list)."
            ),
            block_message: Optional[str] = typer.Option(None, help="New refusal message."),
            workspace: WorkspaceOption = None,
            output_format: EntityOutputFormatOption = None,
        ) -> None:
            """Partially update a middleware config (omitted fields unchanged)."""
            state = cli_state(ctx)
            resolved_output_format = resolve_output_format(ctx, output_format)
            fields: dict[str, Any] = {}
            if blocked_keywords is not None:
                fields["blocked_keywords"] = _keywords(blocked_keywords)
            if block_message is not None:
                fields["block_message"] = block_message
            kwargs = {
                "workspace": resolve_cli_workspace(ctx, workspace),
                "name": name,
                "body": UpdateExampleMiddlewareConfigRequest(**fields),
            }
            if handle_code_generation(ExampleClient, "update_middleware_config", kwargs, resolved_output_format, state):
                return
            response = state.typed_client(ExampleClient).update_middleware_config(**kwargs)
            format_output(response, output_format=resolved_output_format)

        @mw.command("delete")
        @collect_warnings
        @handle_errors
        def delete_middleware_config(
            ctx: typer.Context,
            name: NameArgument,
            workspace: WorkspaceOption = None,
            yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
        ) -> None:
            """Delete a middleware config."""
            workspace = resolve_cli_workspace(ctx, workspace)
            if not yes:
                typer.confirm(f"Delete middleware config '{workspace}/{name}'?", abort=True)
            cli_state(ctx).typed_client(ExampleClient).delete_middleware_config(workspace=workspace, name=name)
            typer.echo(f"Deleted '{workspace}/{name}'.")

        return app
