# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from typing import Annotated

import typer

from nemo_helix_ext.cli.core.api import build_kwargs
from nemo_helix_ext.cli.core.code_generator import handle_code_generation
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import format_output
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.stdin_utils import read_data_input_with_flags
from nemo_helix_ext.cli.core.types import EntityOutputFormatOption

app = create_typer_app(name="logs", help="Manage logs")


@app.command("create")
@collect_warnings
@handle_errors
def create_logs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    artifact_base_path: Annotated[
        str | None, typer.Option("--artifact-base-path", help="Folder inside the fileset to nest logs under")
    ] = None,
    input_file: Annotated[
        str | None,
        typer.Option("--input-file", help="Path to JSON file (use '-' for stdin)", rich_help_panel="Input Options"),
    ] = None,
    input_data: Annotated[
        str | None,
        typer.Option("--input-data", help="Input data for the request (JSON or YAML)", rich_help_panel="Input Options"),
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Upload OTLP logs to a specified fileset in JSON or Protobuf format.

    Supports both application/json and application/x-protobuf content types.

        [green]Examples:[/]
        nemo files otlp logs create <name> --input-file config.json
        nemo files otlp logs create <name> --input-data '{"field": "value"}'
        echo '{"json": "data"}' | nemo files otlp logs create <name> --input-file -
        nemo files otlp logs create <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if artifact_base_path is not None:
        input_payload["artifact_base_path"] = artifact_base_path

    all_kwargs = {"name": name, **input_payload}
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["files", "otlp", "logs"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.files.otlp.logs.create(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("query")
@collect_warnings
@handle_errors
def query_logs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    artifact_base_path: Annotated[
        str | None,
        typer.Option(
            "--artifact-base-path",
            help="Folder inside the fileset the logs were nested under (must match the value used on write)",
        ),
    ] = None,
    filters: Annotated[str | None, typer.Option("--filters", help="Key-value filters to apply to the query")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Maximum number of results to return")] = None,
    page_cursor: Annotated[str | None, typer.Option("--page-cursor", help="Cursor for pagination")] = None,
    tail: Annotated[int | None, typer.Option("--tail", help="Number of newest log lines to return")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Query logs from parquet files in a fileset.

    This is an internal endpoint that runs DuckDB queries with direct storage
    access."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
        artifact_base_path=artifact_base_path,
        filters=filters,
        limit=limit,
        page_cursor=page_cursor,
        tail=tail,
    )
    if handle_code_generation(["files", "otlp", "logs"], "query", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.files.otlp.logs.query(name, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
