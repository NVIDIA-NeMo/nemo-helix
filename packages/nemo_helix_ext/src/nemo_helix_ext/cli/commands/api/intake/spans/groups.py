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
from nemo_helix_ext.cli.core.types import (
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="groups", help="Manage groups")


@app.command("list")
@collect_warnings
@handle_errors
def list_groups(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    by: Annotated[str, typer.Option("--by", help="Comma-separated span fields to group by, e.g.")] = ...,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  started_at: {gte: str, lte: str}\n\nFilter spans by the same fields as the span list endpoint, then group matching spans by the comma-separated fields in the by query parameter.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_agent_id: Annotated[str | None, typer.Option("--filter.agent-id", rich_help_panel="Filter Options")] = None,
    filter_agent_name: Annotated[
        str | None, typer.Option("--filter.agent-name", rich_help_panel="Filter Options")
    ] = None,
    filter_evaluation_id: Annotated[
        str | None, typer.Option("--filter.evaluation-id", rich_help_panel="Filter Options")
    ] = None,
    filter_evaluation_name: Annotated[
        str | None, typer.Option("--filter.evaluation-name", rich_help_panel="Filter Options")
    ] = None,
    filter_kind: Annotated[str | None, typer.Option("--filter.kind", rich_help_panel="Filter Options")] = None,
    filter_model: Annotated[str | None, typer.Option("--filter.model", rich_help_panel="Filter Options")] = None,
    filter_parent_span_id: Annotated[
        str | None, typer.Option("--filter.parent-span-id", rich_help_panel="Filter Options")
    ] = None,
    filter_project: Annotated[str | None, typer.Option("--filter.project", rich_help_panel="Filter Options")] = None,
    filter_provider: Annotated[str | None, typer.Option("--filter.provider", rich_help_panel="Filter Options")] = None,
    filter_session_id: Annotated[
        str | None, typer.Option("--filter.session-id", rich_help_panel="Filter Options")
    ] = None,
    filter_source: Annotated[str | None, typer.Option("--filter.source", rich_help_panel="Filter Options")] = None,
    filter_status: Annotated[str | None, typer.Option("--filter.status", rich_help_panel="Filter Options")] = None,
    filter_test_case_id: Annotated[
        str | None, typer.Option("--filter.test-case-id", rich_help_panel="Filter Options")
    ] = None,
    filter_test_case_name: Annotated[
        str | None, typer.Option("--filter.test-case-name", rich_help_panel="Filter Options")
    ] = None,
    filter_tool_name: Annotated[
        str | None, typer.Option("--filter.tool-name", rich_help_panel="Filter Options")
    ] = None,
    filter_trace_id: Annotated[str | None, typer.Option("--filter.trace-id", rich_help_panel="Filter Options")] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[
        Literal["span_count", "-span_count", "started_at", "-started_at"] | None,
        typer.Option(
            "--sort",
            help="Sort groups by size or by start time. Use -started_at for the traces or sessions that began most recently, which answers 'what ran lately' in one call instead of paging through spans. A group's time is its earliest matching span, so this orders by when work started and not by when it was last active.",
        ),
    ] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List Span Groups"""
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
        by=by,
        filter=merge_filter_dict(
            filter,
            agent_id=filter_agent_id,
            agent_name=filter_agent_name,
            evaluation_id=filter_evaluation_id,
            evaluation_name=filter_evaluation_name,
            kind=filter_kind,
            model=filter_model,
            parent_span_id=filter_parent_span_id,
            project=filter_project,
            provider=filter_provider,
            session_id=filter_session_id,
            source=filter_source,
            status=filter_status,
            test_case_id=filter_test_case_id,
            test_case_name=filter_test_case_name,
            tool_name=filter_tool_name,
            trace_id=filter_trace_id,
        ),
        page=page,
        page_size=page_size,
        sort=sort,
    )

    if handle_code_generation(["intake", "spans", "groups"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = ()
    pagination_type = PaginationType.PAGE_NUMBER
    if all_pages:
        items = fetch_all_pages(
            client.intake.spans.groups.list,
            path_args=path_args,
            body_args=kwargs,
            pagination_type=pagination_type,
        )
    else:
        items = client.intake.spans.groups.list(*path_args, **kwargs)

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
