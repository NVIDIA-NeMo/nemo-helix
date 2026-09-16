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

app = create_typer_app(name="chat_completions", help="Manage chat_completions")


@app.command("create")
@collect_warnings
@handle_errors
def create_chat_completions(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    request: Annotated[
        str | None,
        typer.Option("--request", help="Flexible captured chat-completions request. (JSON string) (required)"),
    ] = None,
    response: Annotated[
        str | None,
        typer.Option("--response", help="Flexible captured chat-completions response. (JSON string) (required)"),
    ] = None,
    cost_details: Annotated[
        str | None,
        typer.Option("--cost-details", help="Additional estimated cost breakdown fields in USD. (JSON string)"),
    ] = None,
    cost_input_usd: Annotated[
        float | None, typer.Option("--cost-input-usd", help="Estimated input-token cost of this model call in USD.")
    ] = None,
    cost_output_usd: Annotated[
        float | None, typer.Option("--cost-output-usd", help="Estimated output-token cost of this model call in USD.")
    ] = None,
    cost_usd: Annotated[
        float | None,
        typer.Option(
            "--cost-usd",
            help="Total estimated cost of this model call in USD. This matches ATIF step metrics; Intake stores it as semantic cost_total_usd on spans.",
        ),
    ] = None,
    evaluation_context: Annotated[
        str | None,
        typer.Option(
            "--evaluation-context",
            help="Identifies the Evaluation and optional test case associated with ingested telemetry. (JSON string)",
        ),
    ] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id", help="Groups related chat-completions calls without forcing them into the same trace."
        ),
    ] = None,
    trace_id: Annotated[
        str | None,
        typer.Option(
            "--trace-id",
            help="Opt into joining an existing trace built via OTel or ATIF. This is not a grouping mechanism for chat-completions calls; use session_id to group related calls.",
        ),
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
    """Ingest Chat Completion

    [bold red]Required fields:[/] request, response

    [green]Examples:[/]
    nemo intake ingest chat-completions create --input-file config.json
    nemo intake ingest chat-completions create --input-data '{"request": {}, "response": {}}'
    echo '{"json": "data"}' | nemo intake ingest chat-completions create --input-file -
    nemo intake ingest chat-completions create --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if request is not None:
        input_payload["request"] = read_payload("request", request)
    if response is not None:
        input_payload["response"] = read_payload("response", response)
    if cost_details is not None:
        input_payload["cost_details"] = read_payload("cost_details", cost_details)
    if cost_input_usd is not None:
        input_payload["cost_input_usd"] = cost_input_usd
    if cost_output_usd is not None:
        input_payload["cost_output_usd"] = cost_output_usd
    if cost_usd is not None:
        input_payload["cost_usd"] = cost_usd
    if evaluation_context is not None:
        input_payload["evaluation_context"] = read_payload("evaluation_context", evaluation_context)
    if provider is not None:
        input_payload["provider"] = provider
    if session_id is not None:
        input_payload["session_id"] = session_id
    if trace_id is not None:
        input_payload["trace_id"] = trace_id
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["request", "response"],
        "intake ingest chat-completions create",
        {
            "request": "Flexible captured chat-completions request. (JSON string) (required)",
            "response": "Flexible captured chat-completions response. (JSON string) (required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["intake", "ingest", "chat_completions"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.intake.ingest.chat_completions.create(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
