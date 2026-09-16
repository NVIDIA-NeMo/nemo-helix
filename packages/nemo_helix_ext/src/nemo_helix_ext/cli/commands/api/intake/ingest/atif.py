# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from typing import Annotated, Literal

import typer

from nemo_helix_ext.cli.core.code_generator import handle_code_generation
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import format_output
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.stdin_utils import read_data_input_with_flags, read_payload, validate_required_fields
from nemo_helix_ext.cli.core.types import EntityOutputFormatOption

app = create_typer_app(name="atif", help="Manage atif")


@app.command("create")
@collect_warnings
@handle_errors
def create_atif(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    agent: Annotated[str | None, typer.Option("--agent", help="JSON string (required)")] = None,
    schema_version: Annotated[
        Literal["ATIF-v1.0", "ATIF-v1.1", "ATIF-v1.2", "ATIF-v1.3", "ATIF-v1.4", "ATIF-v1.5", "ATIF-v1.6", "ATIF-v1.7"]
        | None,
        typer.Option("--schema-version", help="(required)"),
    ] = None,
    continued_trajectory_ref: Annotated[str | None, typer.Option("--continued-trajectory-ref")] = None,
    evaluation_context: Annotated[
        str | None,
        typer.Option(
            "--evaluation-context",
            help="Identifies the Evaluation and optional test case associated with ingested telemetry. (JSON string)",
        ),
    ] = None,
    extra: Annotated[str | None, typer.Option("--extra", help="JSON string")] = None,
    final_metrics: Annotated[str | None, typer.Option("--final-metrics", help="JSON string")] = None,
    notes: Annotated[str | None, typer.Option("--notes")] = None,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    steps: Annotated[str | None, typer.Option("--steps", help="JSON string")] = None,
    subagent_trajectories: Annotated[str | None, typer.Option("--subagent-trajectories", help="JSON string")] = None,
    trajectory_id: Annotated[str | None, typer.Option("--trajectory-id")] = None,
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
    """Ingest Atif

    [bold red]Required fields:[/] agent, schema_version

    [green]Examples:[/]
    nemo intake ingest atif create --input-file config.json
    nemo intake ingest atif create --input-data '{"agent": {}, "schema_version": "value"}'
    echo '{"json": "data"}' | nemo intake ingest atif create --input-file -
    nemo intake ingest atif create --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if agent is not None:
        input_payload["agent"] = read_payload("agent", agent)
    if schema_version is not None:
        input_payload["schema_version"] = schema_version
    if continued_trajectory_ref is not None:
        input_payload["continued_trajectory_ref"] = continued_trajectory_ref
    if evaluation_context is not None:
        input_payload["evaluation_context"] = read_payload("evaluation_context", evaluation_context)
    if extra is not None:
        input_payload["extra"] = read_payload("extra", extra)
    if final_metrics is not None:
        input_payload["final_metrics"] = read_payload("final_metrics", final_metrics)
    if notes is not None:
        input_payload["notes"] = notes
    if session_id is not None:
        input_payload["session_id"] = session_id
    if steps is not None:
        input_payload["steps"] = read_payload("steps", steps)
    if subagent_trajectories is not None:
        input_payload["subagent_trajectories"] = read_payload("subagent_trajectories", subagent_trajectories)
    if trajectory_id is not None:
        input_payload["trajectory_id"] = trajectory_id
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["agent", "schema_version"],
        "intake ingest atif create",
        {
            "agent": "JSON string (required)",
            "schema_version": "(required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["intake", "ingest", "atif"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.intake.ingest.atif.create(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
