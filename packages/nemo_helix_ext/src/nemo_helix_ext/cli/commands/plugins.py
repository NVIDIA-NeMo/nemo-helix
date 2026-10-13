# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for plugin discovery."""

from __future__ import annotations

import typer

from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.types import (
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(
    name="plugins",
    help="""\
Commands for plugin discovery.

Examples:
# List installed plugins.
nemo plugins list""",
)


@app.callback(invoke_without_command=True)
def plugins_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("list")
@collect_warnings
@handle_errors
def list_plugins(
    ctx: typer.Context,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """List installed plugins.

    Discovers installed plugins from registered NeMo plugin entry points.

    Examples:
      nemo plugins list
      nemo plugins list -f json
    """
    state: CLIContext = ctx.obj
    columns_explicit = columns is not None and str(columns).strip() != "default"
    output_format = state.get_output_format(output_format, apply_non_tty_default=not columns_explicit)
    validate_stream_output_format(output_format, stream)

    check_output_columns_with_format(columns, output_format)

    default_columns = [
        Column("name", None),
        Column("version", None),
        Column("description", None),
    ]
    output_columns: str | list[Column] | None = columns
    if not columns_explicit:
        output_columns = default_columns

    try:
        from nemo_helix_plugin.discovery import discover_manifests

        manifests = discover_manifests()
    except ImportError:
        manifests = {}

    items = [
        {
            "name": manifest.name,
            "version": manifest.version,
            "description": manifest.description,
        }
        for manifest in sorted(manifests.values(), key=lambda manifest: manifest.name)
    ]

    format_output(
        items,
        is_list=True,
        output_format=output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
