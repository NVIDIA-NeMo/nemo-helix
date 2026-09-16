# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from typing import Annotated, Literal

import typer

from nemo_helix_ext.cli.core.api import build_kwargs, merge_filter_dict
from nemo_helix_ext.cli.core.code_generator import handle_code_generation
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.pagination import PaginationType, fetch_all_pages, warn_if_more_pages
from nemo_helix_ext.cli.core.stdin_utils import read_data_input_with_flags, validate_required_fields
from nemo_helix_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="evaluator_results", help="Manage evaluator_results")


@app.command("create")
@collect_warnings
@handle_errors
def create_evaluator_results(
    ctx: typer.Context,
    name: Annotated[
        str | None, typer.Argument(help="Evaluator / metric identity (e.g. 'faithfulness/v1'). (required)")
    ] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    data_type: Annotated[
        Literal["NUMERIC", "CATEGORICAL", "BOOLEAN", "TEXT"] | None,
        typer.Option(
            "--data-type", help="Discriminator for which of value / string_value carries the payload. (required)"
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Session id the target span belongs to. Denormalized so session-scoped reads stay fast. (required)",
        ),
    ] = None,
    span_id: Annotated[
        str | None,
        typer.Option(
            "--span-id", help="Target span id. Not validated against existing spans (loose target policy). (required)"
        ),
    ] = None,
    comment: Annotated[str | None, typer.Option("--comment", help="Free-text rationale or explanation.")] = None,
    string_value: Annotated[
        str | None, typer.Option("--string-value", help="String value. Required when data_type is CATEGORICAL or TEXT.")
    ] = None,
    value: Annotated[
        float | None,
        typer.Option("--value", help="Numeric value. Required when data_type is NUMERIC or BOOLEAN (0|1)."),
    ] = None,
    exist_ok: Annotated[
        bool | None,
        typer.Option(
            "--exist-ok", help="Do not raise an error if the resource already exists. Returns the existing resource."
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
    """Create Evaluator Result

    [bold red]Required fields:[/] data_type, name, session_id, span_id

    [green]Examples:[/]
    nemo intake evaluator-results create <name> --input-file config.json
    nemo intake evaluator-results create <name> --input-data '{"data_type": "value", "name": "value", "session_id": "value", "span_id": "value"}'
    echo '{"json": "data"}' | nemo intake evaluator-results create <name> --input-file -
    nemo intake evaluator-results create <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if data_type is not None:
        input_payload["data_type"] = data_type
    if name is not None:
        input_payload["name"] = name
    if session_id is not None:
        input_payload["session_id"] = session_id
    if span_id is not None:
        input_payload["span_id"] = span_id
    if comment is not None:
        input_payload["comment"] = comment
    if string_value is not None:
        input_payload["string_value"] = string_value
    if value is not None:
        input_payload["value"] = value
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["data_type", "name", "session_id", "span_id"],
        "intake evaluator-results create",
        {
            "data_type": "Discriminator for which of value / string_value carries the payload. (required)",
            "name": "Evaluator / metric identity (e.g. 'faithfulness/v1'). (required)",
            "session_id": "Session id the target span belongs to. Denormalized so session-scoped reads stay fast. (required)",
            "span_id": "Target span id. Not validated against existing spans (loose target policy). (required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["intake", "evaluator_results"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.intake.evaluator_results.create(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("list")
@collect_warnings
@handle_errors
def list_evaluator_results(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  value: {gte: float, lte: float}\n\nFilter evaluator results by span_id, session_id, name, data_type, created_by, value range, and created_at range.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_created_by: Annotated[
        str | None, typer.Option("--filter.created-by", rich_help_panel="Filter Options")
    ] = None,
    filter_data_type: Annotated[
        str | None, typer.Option("--filter.data-type", rich_help_panel="Filter Options")
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_session_id: Annotated[
        str | None, typer.Option("--filter.session-id", rich_help_panel="Filter Options")
    ] = None,
    filter_span_id: Annotated[str | None, typer.Option("--filter.span-id", rich_help_panel="Filter Options")] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[Literal["created_at", "-created_at", "value", "-value"] | None, typer.Option("--sort")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List Evaluator Results"""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)
    validate_stream_output_format(output_format, stream)

    check_output_columns_with_format(columns, output_format)

    default_columns = [
        Column("name", None),
        Column("workspace", None),
        Column("created_at", None),
    ]
    if columns is None or str(columns).strip() == "default":
        columns = default_columns

    kwargs = build_kwargs(
        workspace=workspace,
        filter=merge_filter_dict(
            filter,
            created_by=filter_created_by,
            data_type=filter_data_type,
            name=filter_name,
            session_id=filter_session_id,
            span_id=filter_span_id,
        ),
        page=page,
        page_size=page_size,
        sort=sort,
    )

    if handle_code_generation(["intake", "evaluator_results"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = ()
    pagination_type = PaginationType.PAGE_NUMBER
    if all_pages:
        items = fetch_all_pages(
            client.intake.evaluator_results.list,
            path_args=path_args,
            body_args=kwargs,
            pagination_type=pagination_type,
        )
    else:
        items = client.intake.evaluator_results.list(*path_args, **kwargs)

    format_output(
        items,
        is_list=True,
        output_format=output_format,
        output_columns=columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, pagination_type)


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_evaluator_results(
    ctx: typer.Context,
    evaluator_result_id: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get Evaluator Result"""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
    )
    if handle_code_generation(["intake", "evaluator_results"], "retrieve", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.intake.evaluator_results.retrieve(evaluator_result_id, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
