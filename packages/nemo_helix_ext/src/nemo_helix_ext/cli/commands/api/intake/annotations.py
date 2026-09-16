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
from nemo_helix_ext.cli.core.stdin_utils import read_data_input_with_flags, read_payload, validate_required_fields
from nemo_helix_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="annotations", help="Manage annotations")


@app.command("create")
@collect_warnings
@handle_errors
def create_annotations(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument()] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    kind: Annotated[
        Literal["feedback", "note", "metadata", "label"] | None, typer.Option("--kind", help="(required)")
    ] = None,
    session_id: Annotated[str | None, typer.Option("--session-id", help="(required)")] = None,
    value: Annotated[str | None, typer.Option("--value")] = None,
    span_id: Annotated[str | None, typer.Option("--span-id")] = None,
    text: Annotated[str | None, typer.Option("--text")] = None,
    metadata: Annotated[str | None, typer.Option("--metadata", help="JSON string")] = None,
    value_type: Annotated[Literal["text", "numeric"] | None, typer.Option("--value-type")] = None,
    exist_ok: Annotated[bool | None, typer.Option("--exist-ok")] = None,
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
    """Create annotations.

    [bold red]Required fields:[/] kind, session_id

    [green]Examples:[/]
    nemo intake annotations create <name> --input-file config.json
    nemo intake annotations create <name> --input-data '{"kind": "value", "session_id": "value"}'
    echo '{"json": "data"}' | nemo intake annotations create <name> --input-file -
    nemo intake annotations create <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if kind is not None:
        input_payload["kind"] = kind
    if session_id is not None:
        input_payload["session_id"] = session_id
    if value is not None:
        input_payload["value"] = value
    if span_id is not None:
        input_payload["span_id"] = span_id
    if text is not None:
        input_payload["text"] = text
    if metadata is not None:
        input_payload["metadata"] = read_payload("metadata", metadata)
    if value_type is not None:
        input_payload["value_type"] = value_type
    if name is not None:
        input_payload["name"] = name
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["kind", "session_id"],
        "intake annotations create",
        {
            "kind": "(required)",
            "session_id": "(required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["intake", "annotations"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.intake.annotations.create(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("delete")
@collect_warnings
@handle_errors
def delete_annotations(
    ctx: typer.Context,
    annotation_id: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete Annotation"""
    state: CLIContext = ctx.obj
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
    )
    client.intake.annotations.delete(annotation_id, **kwargs)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_annotations(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  value_numeric: {gte: float, lte: float}\n\nFilter annotations by span_id, session_id, kind, name, created_by, and created_at range.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_created_by: Annotated[
        str | None, typer.Option("--filter.created-by", rich_help_panel="Filter Options")
    ] = None,
    filter_kind: Annotated[str | None, typer.Option("--filter.kind", rich_help_panel="Filter Options")] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_session_id: Annotated[
        str | None, typer.Option("--filter.session-id", rich_help_panel="Filter Options")
    ] = None,
    filter_span_id: Annotated[str | None, typer.Option("--filter.span-id", rich_help_panel="Filter Options")] = None,
    filter_value_text: Annotated[
        str | None, typer.Option("--filter.value-text", rich_help_panel="Filter Options")
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[Literal["created_at", "-created_at"] | None, typer.Option("--sort")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List Annotations"""
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
            kind=filter_kind,
            name=filter_name,
            session_id=filter_session_id,
            span_id=filter_span_id,
            value_text=filter_value_text,
        ),
        page=page,
        page_size=page_size,
        sort=sort,
    )

    if handle_code_generation(["intake", "annotations"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = ()
    pagination_type = PaginationType.PAGE_NUMBER
    if all_pages:
        items = fetch_all_pages(
            client.intake.annotations.list,
            path_args=path_args,
            body_args=kwargs,
            pagination_type=pagination_type,
        )
    else:
        items = client.intake.annotations.list(*path_args, **kwargs)

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
def retrieve_annotations(
    ctx: typer.Context,
    annotation_id: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get Annotation"""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
    )
    if handle_code_generation(["intake", "annotations"], "retrieve", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.intake.annotations.retrieve(annotation_id, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
