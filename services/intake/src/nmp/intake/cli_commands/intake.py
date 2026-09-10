# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo intake`` command group, backed by the typed Intake client."""

from __future__ import annotations

from typing import Annotated, Literal, cast

import typer
from nemo_platform_ext.cli.core.api import build_kwargs, merge_filter_dict
from nemo_platform_ext.cli.core.code_generator import handle_code_generation
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_platform_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_platform_ext.cli.core.pagination import (
    PaginationType,
    collect_offset_pages,
    warn_if_more_pages,
)
from nemo_platform_ext.cli.core.stdin_utils import (
    build_request_body,
    read_data_input_with_flags,
    read_payload,
    validate_required_fields,
)
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)
from nemo_platform_plugin.intake.client import IntakeClient
from nemo_platform_plugin.intake.types import (
    ANNOTATION_INPUT_ADAPTER,
    AtifCreateRequest,
    ChatCompletionsIngestRequest,
    DirectSpansIngestRequest,
    EvaluatorResultCreateRequest,
    ListAnnotationsQueryParams,
    ListEvaluatorResultsQueryParams,
    ListSpanGroupsQueryParams,
    ListSpansQueryParams,
    ListTracesQueryParams,
    RetrieveTraceQueryParams,
    TraceMetricsQueryParams,
)
from nmp.intake.cli_commands.common import list_query_params, without_keys

app = create_typer_app(name="intake", help="Intake operations")
annotations_app = create_typer_app(name="annotations", help="Manage annotations")
evaluator_results_app = create_typer_app(name="evaluator_results", help="Manage evaluator_results")
ingest_app = create_typer_app(name="ingest", help="Ingest operations")
ingest_atif_app = create_typer_app(name="atif", help="Manage atif")
ingest_chat_completions_app = create_typer_app(name="chat_completions", help="Manage chat_completions")
ingest_spans_app = create_typer_app(name="spans", help="Manage spans")
sessions_app = create_typer_app(name="sessions", help="Manage sessions")
spans_app = create_typer_app(name="spans", help="Manage spans")
spans_evaluator_results_app = create_typer_app(name="evaluator_results", help="Manage evaluator_results")
spans_groups_app = create_typer_app(name="groups", help="Manage groups")
traces_app = create_typer_app(name="traces", help="Manage traces")

app.add_typer(annotations_app, name="annotations")
app.add_typer(evaluator_results_app, name="evaluator-results")
app.add_typer(ingest_app, name="ingest")
app.add_typer(sessions_app, name="sessions")
app.add_typer(spans_app, name="spans")
app.add_typer(traces_app, name="traces")
ingest_app.add_typer(ingest_atif_app, name="atif")
ingest_app.add_typer(ingest_chat_completions_app, name="chat-completions")
ingest_app.add_typer(ingest_spans_app, name="spans")
spans_app.add_typer(spans_evaluator_results_app, name="evaluator-results")
spans_app.add_typer(spans_groups_app, name="groups")

_DEFAULT_LIST_COLUMNS = [
    Column("name", None),
    Column("workspace", None),
    Column("created_at", None),
]

_TRACE_FILTER_JSON_ONLY = "JSON-only fields:\n  started_at: {gte: str, lte: str}\n\n"
_FILTER_HELP_PREFIX = (
    "Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. "
    "Both can be combined, with field options taking precedence.\n"
)
_SPAN_FILTER_DESCRIPTION = (
    "Filter spans by session_id, trace_id, parent_span_id, project, evaluation_name, test_case_name, source, "
    "kind, status, model, tool_name, provider, agent_id, agent_name, and started_at. Every field takes one exact "
    "value, except started_at, which takes gte and lte."
)
_SPAN_GROUP_FILTER_DESCRIPTION = (
    "Filter spans by the same fields as the span list endpoint, then group matching spans by the comma-separated "
    "fields in the by query parameter."
)
_TRACE_LIST_FILTER_DESCRIPTION = (
    "Filter root-span-backed traces by id, session_id, root status, root span started_at, evaluation_name, "
    "test_case_name, and agent_name."
)
_TRACE_METRICS_FILTER_DESCRIPTION = (
    "Filter the traces the metrics are computed over. Accepts the same fields as the traces list, so agent_name "
    "scopes the rollup to one agent. Without a started_at lower bound the rollup covers the last 7 days."
)


def _resolve_columns(columns: str | None) -> str | list[Column] | None:
    if columns is None or str(columns).strip() == "default":
        return _DEFAULT_LIST_COLUMNS
    return columns


def _format_entity(state: CLIContext, result: object, resolved_output_format: str) -> None:
    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------


@traces_app.command("get-metrics")
@collect_warnings
@handle_errors
def get_metrics_traces(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    bucket: Annotated[
        Literal["total", "hour", "day", "week", "month"] | None,
        typer.Option("--bucket", help="Time bucket granularity."),
    ] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help=_FILTER_HELP_PREFIX + _TRACE_FILTER_JSON_ONLY + _TRACE_METRICS_FILTER_DESCRIPTION,
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_id: Annotated[str | None, typer.Option("--filter.id", rich_help_panel="Filter Options")] = None,
    filter_agent_name: Annotated[
        str | None, typer.Option("--filter.agent-name", rich_help_panel="Filter Options")
    ] = None,
    filter_evaluation_id: Annotated[
        str | None, typer.Option("--filter.evaluation-id", rich_help_panel="Filter Options")
    ] = None,
    filter_evaluation_name: Annotated[
        str | None, typer.Option("--filter.evaluation-name", rich_help_panel="Filter Options")
    ] = None,
    filter_session_id: Annotated[
        str | None, typer.Option("--filter.session-id", rich_help_panel="Filter Options")
    ] = None,
    filter_status: Annotated[str | None, typer.Option("--filter.status", rich_help_panel="Filter Options")] = None,
    filter_test_case_id: Annotated[
        str | None, typer.Option("--filter.test-case-id", rich_help_panel="Filter Options")
    ] = None,
    filter_test_case_name: Annotated[
        str | None, typer.Option("--filter.test-case-name", rich_help_panel="Filter Options")
    ] = None,
    timezone: Annotated[
        str | None,
        typer.Option("--timezone", help="IANA timezone the buckets are aligned to, e.g. America/Los_Angeles."),
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get Trace Metrics"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    query_params = cast(
        TraceMetricsQueryParams | None,
        list_query_params(
            bucket=bucket,
            filter=merge_filter_dict(
                filter,
                id=filter_id,
                agent_name=filter_agent_name,
                evaluation_id=filter_evaluation_id,
                evaluation_name=filter_evaluation_name,
                session_id=filter_session_id,
                status=filter_status,
                test_case_id=filter_test_case_id,
                test_case_name=filter_test_case_name,
            ),
            timezone=timezone,
        ),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)
    if handle_code_generation(IntakeClient, "get_trace_metrics", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).get_trace_metrics(workspace=workspace, query_params=query_params)
    _format_entity(state, result, resolved_output_format)


@traces_app.command("list")
@collect_warnings
@handle_errors
def list_traces(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help=_FILTER_HELP_PREFIX + _TRACE_FILTER_JSON_ONLY + _TRACE_LIST_FILTER_DESCRIPTION,
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_id: Annotated[str | None, typer.Option("--filter.id", rich_help_panel="Filter Options")] = None,
    filter_agent_name: Annotated[
        str | None, typer.Option("--filter.agent-name", rich_help_panel="Filter Options")
    ] = None,
    filter_evaluation_id: Annotated[
        str | None, typer.Option("--filter.evaluation-id", rich_help_panel="Filter Options")
    ] = None,
    filter_evaluation_name: Annotated[
        str | None, typer.Option("--filter.evaluation-name", rich_help_panel="Filter Options")
    ] = None,
    filter_session_id: Annotated[
        str | None, typer.Option("--filter.session-id", rich_help_panel="Filter Options")
    ] = None,
    filter_status: Annotated[str | None, typer.Option("--filter.status", rich_help_panel="Filter Options")] = None,
    filter_test_case_id: Annotated[
        str | None, typer.Option("--filter.test-case-id", rich_help_panel="Filter Options")
    ] = None,
    filter_test_case_name: Annotated[
        str | None, typer.Option("--filter.test-case-name", rich_help_panel="Filter Options")
    ] = None,
    mode: Annotated[
        Literal["summary", "preview", "detailed"] | None,
        typer.Option(
            "--mode",
            help="Response mode. summary returns root-span fields without payloads or rollups; preview adds token, cost, and span-count rollups plus 300-character input/output previews; detailed returns rollups and full payloads.",
        ),
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[Literal["started_at", "-started_at"] | None, typer.Option("--sort")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List Traces"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)
    output_columns = _resolve_columns(columns)

    query_params = cast(
        ListTracesQueryParams | None,
        list_query_params(
            filter=merge_filter_dict(
                filter,
                id=filter_id,
                agent_name=filter_agent_name,
                evaluation_id=filter_evaluation_id,
                evaluation_name=filter_evaluation_name,
                session_id=filter_session_id,
                status=filter_status,
                test_case_id=filter_test_case_id,
                test_case_name=filter_test_case_name,
            ),
            mode=mode,
            page=page,
            page_size=page_size,
            sort=sort,
        ),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)

    if handle_code_generation(IntakeClient, "list_traces", kwargs, resolved_output_format, state, result="list"):
        return

    response = state.typed_client(IntakeClient).list_traces(workspace=workspace, query_params=query_params)
    pagination_type = PaginationType.PAGE_NUMBER
    items = collect_offset_pages(response, all_pages=all_pages)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, pagination_type)


@traces_app.command("get")
@collect_warnings
@handle_errors
def retrieve_traces(
    ctx: typer.Context,
    id: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    mode: Annotated[
        Literal["summary", "preview", "detailed"] | None, typer.Option("--mode", help="Response mode.")
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get Trace"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    query_params = cast(RetrieveTraceQueryParams | None, list_query_params(mode=mode))
    kwargs = build_kwargs(id=id, workspace=workspace, query_params=query_params)
    if handle_code_generation(IntakeClient, "get_trace", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).get_trace(id=id, workspace=workspace, query_params=query_params)
    _format_entity(state, result, resolved_output_format)


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------


@spans_app.command("list")
@collect_warnings
@handle_errors
def list_spans(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help=_FILTER_HELP_PREFIX + _TRACE_FILTER_JSON_ONLY + _SPAN_FILTER_DESCRIPTION,
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
    mode: Annotated[
        Literal["summary", "preview", "detailed"] | None,
        typer.Option(
            "--mode",
            help="Response mode. summary omits payloads and raw attributes; preview includes input and output truncated to 300 characters; detailed returns full payloads and raw attributes.",
        ),
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[Literal["started_at", "-started_at"] | None, typer.Option("--sort")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List Spans"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)
    output_columns = _resolve_columns(columns)

    query_params = cast(
        ListSpansQueryParams | None,
        list_query_params(
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
            mode=mode,
            page=page,
            page_size=page_size,
            sort=sort,
        ),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)

    if handle_code_generation(IntakeClient, "list_spans", kwargs, resolved_output_format, state, result="list"):
        return

    response = state.typed_client(IntakeClient).list_spans(workspace=workspace, query_params=query_params)
    pagination_type = PaginationType.PAGE_NUMBER
    items = collect_offset_pages(response, all_pages=all_pages)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, pagination_type)


@spans_app.command("get")
@collect_warnings
@handle_errors
def retrieve_spans(
    ctx: typer.Context,
    span_id: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get Span"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(span_id=span_id, workspace=workspace)
    if handle_code_generation(IntakeClient, "get_span", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).get_span(span_id=span_id, workspace=workspace)
    _format_entity(state, result, resolved_output_format)


@spans_groups_app.command("list")
@collect_warnings
@handle_errors
def list_groups(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    by: Annotated[str, typer.Option("--by", help="Comma-separated span fields to group by, e.g.")] = ...,  # ty: ignore[invalid-parameter-default]
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help=_FILTER_HELP_PREFIX + _TRACE_FILTER_JSON_ONLY + _SPAN_GROUP_FILTER_DESCRIPTION,
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
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)
    output_columns = _resolve_columns(columns)

    query_params = cast(
        ListSpanGroupsQueryParams,
        list_query_params(
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
        ),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)

    if handle_code_generation(IntakeClient, "list_span_groups", kwargs, resolved_output_format, state, result="list"):
        return

    response = state.typed_client(IntakeClient).list_span_groups(workspace=workspace, query_params=query_params)
    items = collect_offset_pages(response, all_pages=all_pages)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, PaginationType.PAGE_NUMBER)


@spans_evaluator_results_app.command("list")
@collect_warnings
@handle_errors
def list_span_evaluator_results(
    ctx: typer.Context,
    span_id: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """List Evaluator Results For Span"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)
    output_columns = _resolve_columns(columns)

    kwargs = build_kwargs(span_id=span_id, workspace=workspace)

    if handle_code_generation(IntakeClient, "list_evaluator_results_for_span", kwargs, resolved_output_format, state):
        return

    items = state.typed_client(IntakeClient).list_evaluator_results_for_span(span_id=span_id, workspace=workspace)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------


@annotations_app.command("create")
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

    # Annotations are always freshly created server-side (no name-based identity), so
    # --exist-ok has nothing to resolve against and is accepted without effect.
    body = ANNOTATION_INPUT_ADAPTER.validate_python(without_keys(input_payload, {"workspace", "exist_ok"}))
    kwargs = build_kwargs(workspace=input_payload.get("workspace"), body=body)
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(IntakeClient, "create_annotation", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).create_annotation(**kwargs)
    _format_entity(state, result, resolved_output_format)


@annotations_app.command("delete")
@collect_warnings
@handle_errors
def delete_annotations(
    ctx: typer.Context,
    annotation_id: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete Annotation"""
    state: CLIContext = ctx.obj
    state.typed_client(IntakeClient).delete_annotation(annotation_id=annotation_id, workspace=workspace)

    typer.echo("✓ Deleted successfully")


@annotations_app.command("list")
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
            help=_FILTER_HELP_PREFIX
            + "JSON-only fields:\n  created_at: {gte: str, lte: str}\n  value_numeric: {gte: float, lte: float}\n\n"
            + "Filter annotations by span_id, session_id, kind, name, created_by, and created_at range.",
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
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)
    output_columns = _resolve_columns(columns)

    query_params = cast(
        ListAnnotationsQueryParams | None,
        list_query_params(
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
        ),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)

    if handle_code_generation(IntakeClient, "list_annotations", kwargs, resolved_output_format, state, result="list"):
        return

    response = state.typed_client(IntakeClient).list_annotations(workspace=workspace, query_params=query_params)
    pagination_type = PaginationType.PAGE_NUMBER
    items = collect_offset_pages(response, all_pages=all_pages)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, pagination_type)


@annotations_app.command("get")
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
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(annotation_id=annotation_id, workspace=workspace)
    if handle_code_generation(IntakeClient, "get_annotation", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).get_annotation(annotation_id=annotation_id, workspace=workspace)
    _format_entity(state, result, resolved_output_format)


# ---------------------------------------------------------------------------
# evaluator-results
# ---------------------------------------------------------------------------


@evaluator_results_app.command("create")
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

    # Evaluator results upsert on (session, span, name) server-side and never 409, so
    # --exist-ok has nothing to resolve against and is accepted without effect.
    body = build_request_body(
        EvaluatorResultCreateRequest,
        input_payload,
        exclude={"workspace", "exist_ok"},
        command_name="intake evaluator-results create",
    )
    kwargs = build_kwargs(workspace=input_payload.get("workspace"), body=body)
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(IntakeClient, "create_evaluator_result", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).create_evaluator_result(**kwargs)
    _format_entity(state, result, resolved_output_format)


@evaluator_results_app.command("list")
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
            help=_FILTER_HELP_PREFIX
            + "JSON-only fields:\n  created_at: {gte: str, lte: str}\n  value: {gte: float, lte: float}\n\n"
            + "Filter evaluator results by span_id, session_id, name, data_type, created_by, value range, and created_at range.",
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
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)
    output_columns = _resolve_columns(columns)

    query_params = cast(
        ListEvaluatorResultsQueryParams | None,
        list_query_params(
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
        ),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)

    if handle_code_generation(
        IntakeClient, "list_evaluator_results", kwargs, resolved_output_format, state, result="list"
    ):
        return

    response = state.typed_client(IntakeClient).list_evaluator_results(workspace=workspace, query_params=query_params)
    pagination_type = PaginationType.PAGE_NUMBER
    items = collect_offset_pages(response, all_pages=all_pages)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, pagination_type)


@evaluator_results_app.command("get")
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
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(evaluator_result_id=evaluator_result_id, workspace=workspace)
    if handle_code_generation(IntakeClient, "get_evaluator_result", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).get_evaluator_result(
        evaluator_result_id=evaluator_result_id, workspace=workspace
    )
    _format_entity(state, result, resolved_output_format)


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------


@sessions_app.command("get")
@collect_warnings
@handle_errors
def retrieve_sessions(
    ctx: typer.Context,
    id: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get Session"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(id=id, workspace=workspace)
    if handle_code_generation(IntakeClient, "get_session", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).get_session(id=id, workspace=workspace)
    _format_entity(state, result, resolved_output_format)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@ingest_atif_app.command("create")
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

    body = build_request_body(
        AtifCreateRequest, input_payload, exclude={"workspace"}, command_name="intake ingest atif create"
    )
    kwargs = build_kwargs(workspace=input_payload.get("workspace"), body=body)
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(IntakeClient, "create_atif", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).create_atif(**kwargs)
    _format_entity(state, result, resolved_output_format)


@ingest_chat_completions_app.command("create")
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

    body = build_request_body(
        ChatCompletionsIngestRequest,
        input_payload,
        exclude={"workspace"},
        command_name="intake ingest chat-completions create",
    )
    kwargs = build_kwargs(workspace=input_payload.get("workspace"), body=body)
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(IntakeClient, "create_chat_completion", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).create_chat_completion(**kwargs)
    _format_entity(state, result, resolved_output_format)


@ingest_spans_app.command("create")
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

    body = build_request_body(
        DirectSpansIngestRequest, input_payload, exclude={"workspace"}, command_name="intake ingest spans create"
    )
    kwargs = build_kwargs(workspace=input_payload.get("workspace"), body=body)
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(IntakeClient, "create_spans", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IntakeClient).create_spans(**kwargs)
    _format_entity(state, result, resolved_output_format)
