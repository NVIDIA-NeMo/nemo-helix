# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from typing import Annotated

import typer

from nemo_helix_ext.cli.core.code_generator import handle_code_generation
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import format_output
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.stdin_utils import read_data_input_with_flags, read_payload, validate_required_fields
from nemo_helix_ext.cli.core.types import EntityOutputFormatOption

app = create_typer_app(name="spans", help="Manage spans")


@app.command("create")
@collect_warnings
@handle_errors
def create_spans(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source", help="Stable name for the source trace store, such as `langsmith` or `mlflow`. (required)"
        ),
    ] = None,
    spans: Annotated[str | None, typer.Option("--spans", help="JSON string (required)")] = None,
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
    """Ingest Spans

    [bold red]Required fields:[/] source, spans

    [green]Examples:[/]
    nemo intake ingest spans create --input-file spans.json
    nemo intake ingest spans create --input-data '{"source":"langsmith","spans":[{"span_id":"span-1","trace_id":"trace-1","started_at":"2026-08-14T00:00:00Z"}]}'
    echo '{"source":"langsmith","spans":[{"span_id":"span-1","trace_id":"trace-1","started_at":"2026-08-14T00:00:00Z"}]}' | nemo intake ingest spans create --input-file -
    nemo intake ingest spans create --source langsmith --spans '[{"span_id":"span-1","trace_id":"trace-1","started_at":"2026-08-14T00:00:00Z"}]'
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if source is not None:
        input_payload["source"] = source
    if spans is not None:
        input_payload["spans"] = read_payload("spans", spans)
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["source", "spans"],
        "intake ingest spans create",
        {
            "source": "Stable name for the source trace store, such as `langsmith` or `mlflow`. (required)",
            "spans": "JSON string (required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["intake", "ingest", "spans"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.intake.ingest.spans.create(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
