# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intake trace querying for standalone Eval Author.

Two raw queries and one composite.

``query_spans`` and ``query_traces`` pass a filter straight to Intake and return
plain rows. The agent writes its own filters, so it can ask for error spans of
one tool, or the traces of one evaluation case, without a new method here.
Server-side filtering is exact, which client-side counting over a capped page
never is.

``find_agent_traces`` exists because no single endpoint answers "the recent
traces of this agent". Only spans carry ``agent_name``, so ``traces.list``
cannot be scoped to an agent. The composite groups spans by trace id to get the
recent ones, then reads the summary of exactly those ids. It is built on the two
raw queries, so it is also a worked example of composing them.

Reading and diagnosis are delegated, not rebuilt: ``read_trace`` calls
``TraceExplorer`` and ``analyze_trace`` calls ``TraceAnalyzer``.

Failure-mode discovery stays with the Analyst. Nothing here clusters traces.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, cast

from nemo_experimentalist_plugin.entities import ResourceRef, Task, TrialResult
from nemo_experimentalist_plugin.experimentalist.components.trace_analyzer import Diagnostic, TraceAnalyzer
from nemo_experimentalist_plugin.experimentalist.components.trace_explorer import TraceExplorer
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import (
    ListSpanGroupsQueryParams,
    ListSpansQueryParams,
    ListTracesQueryParams,
    SpanFilterParam,
    TraceFilterParam,
    TraceMode,
)

DEFAULT_ROW_LIMIT = 100
MAX_ROW_LIMIT = 1000
DEFAULT_TRACE_LIMIT = 50
MAX_TRACE_LIMIT = 200
_MAX_PAGE_SIZE = 100
# Trace ids travel in the query string, so one wide ``$in`` can overrun the URL limit
# of the server. At 50 ids the query stays near 2 KB.
_TRACE_ID_CHUNK = 50


class TraceQueryError(RuntimeError):
    """An Intake read failed. The message names the next thing to try."""


def _explain(exc: Exception, *, doing: str, workspace: str) -> TraceQueryError:
    """Turn an SDK failure into an error that names a corrective action."""
    if isinstance(exc, NemoHTTPError):
        if exc.status_code in (401, 403):
            hint = (
                "Credentials were rejected. Run `nemo auth login`, then confirm that NMP_BASE_URL "
                f"points at the cluster that owns workspace '{workspace}'."
            )
        elif exc.status_code == 404:
            hint = f"Intake returned 404. Confirm that workspace '{workspace}' exists and that the trace id is correct."
        elif exc.status_code == 400:
            hint = (
                f"Intake rejected the query: {exc}. Check the filter fields and operators against the "
                "vocabulary in the docstring of the query you called."
            )
        else:
            hint = f"Intake returned HTTP {exc.status_code}: {exc}"
        return TraceQueryError(f"{doing} failed. {hint}")
    if isinstance(exc, NemoTransportError):
        return TraceQueryError(
            f"{doing} failed: the platform is unreachable. Check NMP_BASE_URL and that the services run."
        )
    if isinstance(exc, ValueError):
        # TraceExplorer raises ValueError both for an unreadable ref and for a trace with
        # no spans. Its message is specific; only the workspace is missing from it.
        return TraceQueryError(
            f"{doing} failed in workspace '{workspace}': {exc}. Confirm the trace id, and that "
            "this workspace is the one the trace was ingested into."
        )
    return TraceQueryError(f"{doing} failed: {type(exc).__name__}: {exc}")


def _page_size(limit: int) -> int:
    return max(1, min(limit, _MAX_PAGE_SIZE))


async def _drain(paginator: Any, *, limit: int) -> tuple[list[dict[str, Any]], bool]:
    """Pull up to ``limit`` rows as plain dicts. The flag reports that more matched."""
    rows: list[dict[str, Any]] = []
    truncated = False
    async for item in paginator:
        if len(rows) >= limit:
            truncated = True
            break
        rows.append(item.model_dump(mode="json", exclude_none=True))
    return rows, truncated


def _with_trace_ref(row: dict[str, Any]) -> dict[str, Any]:
    """Attach the ref that opens the whole trace a row came from.

    A span is one step of a trajectory. Carrying the ref on every row keeps the way
    back to its context one call away instead of a string the caller has to build.
    """
    trace_id = row.get("trace_id") or row.get("group", {}).get("trace_id")
    if trace_id:
        row["trace_ref"] = f"intake://{trace_id}"
    return row


async def query_spans(
    client: AsyncNemoClient,
    *,
    workspace: str,
    filter: dict[str, Any] | None = None,
    group_by: str | None = None,
    sort: str | None = None,
    mode: str = "summary",
    limit: int = DEFAULT_ROW_LIMIT,
) -> dict[str, Any]:
    """Query the spans of Intake, flat or rolled up into groups.

    Use this to choose which traces to open, not to read what happened. A span is one
    step of a trajectory, and read on its own it omits what led to it and what followed.
    A slow call may be slow because of the retry before it; an error may be the handled
    kind that the next span recovers from. Every row carries ``trace_ref``, so open the
    trace with ``read_trace`` before you treat a span as evidence, quote it, or build a
    test case from it.

    Args:
        client: Platform client.
        workspace: Workspace to search.
        filter: Intake span filter, sent to the server. Combine fields to narrow
            server-side. Every field takes one exact value, except ``started_at``:

            - identity: ``trace_id``, ``session_id``, ``parent_span_id`` (the direct
              children of one span)
            - agent: ``agent_name``, ``agent_id``
            - call: ``kind`` (``LLM``, ``TOOL``, ``AGENT``, ``CHAIN``, ``EVALUATOR``),
              ``status`` (``success``, ``error``, ``cancelled``, ``unknown``),
              ``tool_name``, ``model``, ``provider``
            - context: ``project``, ``source``, ``evaluation_id``, ``test_case_id``
            - time: ``started_at`` as ``{"gte": ..., "lte": ...}``, either bound or both

            No other field is filterable, and ``$in`` is not accepted here: only
            ``id`` on ``query_traces`` takes a list.
        group_by: When set, roll the matching spans up server-side into one row per
            group. Only ``"trace_id"`` and ``"session_id"`` are groupable. Use this to
            get the distinct traces of any filter, then pass those ids to
            ``query_traces``.
        sort: ``"-started_at"`` (default, newest first) or ``"started_at"``. Grouped
            results also take ``"-span_count"`` and ``"span_count"``, for the busiest
            traces rather than the newest. A group's time is its earliest matching
            span, so ``"-started_at"`` returns the traces whose matching work began
            most recently.
        mode: ``"summary"`` omits payloads, ``"preview"`` truncates them, ``"detailed"``
            returns them in full. Stay on ``"summary"``, which is all that choosing a
            trace needs. Payloads pulled from many traces at once are the fragments this
            docstring warns about; ``read_trace`` gives you the same text in context.
        limit: Maximum rows to return. Clamped to ``MAX_ROW_LIMIT``.

    Returns:
        Flat: ``{"spans": [...], "count": int, "truncated": bool}``. Grouped:
        ``{"groups": [...], "grouped_by": str, "count": int, "truncated": bool}``,
        where a group is ``{"group": {...}, "span_count": int, "started_at": str}``.
        A group's counts and time cover only the spans that matched, so use
        ``query_traces`` for whole-trace totals. Rows that belong to a
        trace also carry ``trace_ref`` for ``read_trace``. ``truncated`` means that more
        rows matched than ``limit``, so narrow the filter or raise ``limit``.

    Raises:
        TraceQueryError: The Intake read failed.
    """
    limit = max(1, min(limit, MAX_ROW_LIMIT))
    intake = AsyncIntakeClient.from_client(client)
    try:
        if group_by is not None:
            group_params: ListSpanGroupsQueryParams = {
                "by": group_by,
                "page_size": _page_size(limit),
                "sort": sort or "-started_at",
            }
            if filter is not None:
                group_params["filter"] = cast(SpanFilterParam, filter)
            response = await intake.list_span_groups(workspace=workspace, query_params=group_params)
            rows, truncated = await _drain(response.items(), limit=limit)
            groups = [_with_trace_ref(row) for row in rows]
            return {"groups": groups, "grouped_by": group_by, "count": len(groups), "truncated": truncated}
        span_params: ListSpansQueryParams = {
            "page_size": _page_size(limit),
            "sort": sort or "-started_at",
            "mode": cast(TraceMode, mode),
        }
        if filter is not None:
            span_params["filter"] = cast(SpanFilterParam, filter)
        response = await intake.list_spans(workspace=workspace, query_params=span_params)
        rows, truncated = await _drain(response.items(), limit=limit)
        spans = [_with_trace_ref(row) for row in rows]
        return {"spans": spans, "count": len(spans), "truncated": truncated}
    except Exception as exc:
        raise _explain(exc, doing="Querying spans", workspace=workspace) from exc


async def query_traces(
    client: AsyncNemoClient,
    *,
    workspace: str,
    filter: dict[str, Any] | None = None,
    sort: str | None = None,
    mode: str = "preview",
    limit: int = DEFAULT_ROW_LIMIT,
) -> dict[str, Any]:
    """Query whole traces, with the rollups that the server computes.

    Prefer this over counting spans yourself. A trace row carries the exact
    ``span_count`` and ``error_count`` of the whole trace, which a capped span query
    cannot give you.

    Args:
        client: Platform client.
        workspace: Workspace to search.
        filter: Intake trace filter, sent to the server. Six fields are filterable:
            ``id``, ``session_id``, ``status``, ``evaluation_id``, ``test_case_id``,
            and ``started_at`` as ``{"gte": ..., "lte": ...}``.

            ``id`` is the only field in Intake that takes a list, as
            ``{"id": {"$in": [...]}}``. That makes it the way to resolve many trace
            ids at once. Keep a batch near 50 ids, because the filter travels in the
            query string. Every other field takes one exact value.

            There is no ``agent_name`` field here, because only spans carry it. To
            scope by agent, use ``find_agent_traces``, or group ``query_spans`` by
            ``trace_id`` and pass the ids here.
        sort: ``"-started_at"`` (default, newest first) or ``"started_at"``.
        mode: ``"summary"`` omits both payloads and rollups, ``"preview"`` (default)
            adds the rollups and 300-character payload previews, ``"detailed"`` returns
            full payloads. The rollups need ``"preview"`` or ``"detailed"``.
        limit: Maximum rows to return. Clamped to ``MAX_ROW_LIMIT``.

    Returns:
        ``{"traces": [...], "count": int, "truncated": bool}``.

    Raises:
        TraceQueryError: The Intake read failed.
    """
    limit = max(1, min(limit, MAX_ROW_LIMIT))
    query_params: ListTracesQueryParams = {
        "mode": cast(TraceMode, mode),
        "sort": sort or "-started_at",
        "page_size": _page_size(limit),
    }
    if filter is not None:
        query_params["filter"] = cast(TraceFilterParam, filter)
    try:
        response = await AsyncIntakeClient.from_client(client).list_traces(
            workspace=workspace, query_params=query_params
        )
        rows, truncated = await _drain(response.items(), limit=limit)
    except Exception as exc:
        raise _explain(exc, doing="Querying traces", workspace=workspace) from exc
    return {"traces": rows, "count": len(rows), "truncated": truncated}


async def find_agent_traces(
    client: AsyncNemoClient,
    *,
    agent: str,
    workspace: str,
    since: datetime | None = None,
    limit: int = DEFAULT_TRACE_LIMIT,
) -> dict[str, Any]:
    """Find the most recent traces of one agent, newest first.

    Two calls, because no single endpoint answers this. Grouping spans by trace id
    gives the traces an agent touched, since only spans carry ``agent_name``, and
    sorting those groups by time gives the recent ones. Reading the summary of
    exactly those ids then gives whole-trace counts, which a group cannot report
    because a group counts only the spans that matched the filter.

    Args:
        client: Platform client.
        agent: Value that the agent reports to Intake as ``agent_name``.
        workspace: Workspace to search.
        since: Optional lower bound on span start time.
        limit: Maximum traces to return. Clamped to ``MAX_TRACE_LIMIT``.

    Returns:
        ``{"traces": [...], "count": int, "truncated": bool}``, and a ``note`` when
        nothing matched. Each trace carries ``trace_ref`` for ``read_trace``,
        ``trace_id`` for a follow-up ``query_spans``, and the server summary:
        ``started_at``, ``status``, ``span_count``, ``error_count``, ``duration_ms``,
        and ``name``.

    Raises:
        TraceQueryError: The Intake read failed.
    """
    limit = max(1, min(limit, MAX_TRACE_LIMIT))
    span_filter: dict[str, Any] = {"agent_name": agent}
    if since is not None:
        span_filter["started_at"] = {"gte": since.isoformat()}

    grouped = await query_spans(
        client,
        workspace=workspace,
        filter=span_filter,
        group_by="trace_id",
        sort="-started_at",
        limit=limit,
    )

    truncated = grouped["truncated"]
    ordered = [row["group"]["trace_id"] for row in grouped["groups"]]

    summaries: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ordered), _TRACE_ID_CHUNK):
        chunk = ordered[start : start + _TRACE_ID_CHUNK]
        page = await query_traces(
            client,
            workspace=workspace,
            filter={"id": {"$in": chunk}},
            sort="-started_at",
            # preview is the cheapest mode that carries the rollups.
            mode="preview",
            limit=len(chunk),
        )
        for row in page["traces"]:
            summaries[row["id"]] = row

    traces = [_trace_entry(trace_id, summaries.get(trace_id)) for trace_id in ordered]
    # The server ordered the groups by the agent's earliest span, but each row reports the
    # whole trace's start. Ordering by the time the row actually shows keeps the list
    # readable, rather than descending by a number the caller cannot see.
    traces.sort(key=lambda entry: entry["started_at"] or "", reverse=True)

    result: dict[str, Any] = {"traces": traces, "count": len(traces), "truncated": truncated}
    if not traces:
        window = f" since {since.isoformat()}" if since is not None else ""
        result["note"] = (
            f"No spans matched agent_name='{agent}' in workspace '{workspace}'{window}. "
            "An empty result is not an error. Check the agent name against the value that the "
            "agent reports to Intake, because a wrong name is the usual cause."
        )
    return result


def _trace_entry(trace_id: str, summary: dict[str, Any] | None) -> dict[str, Any]:
    """Build one result row. A trace whose root span was never ingested has no summary.

    Its unknown fields stay ``None`` rather than zero, so an unknown never reads as a
    healthy trace.
    """
    summary = summary or {}
    return {
        "trace_ref": f"intake://{trace_id}",
        "trace_id": trace_id,
        "started_at": summary.get("started_at"),
        "status": summary.get("status", "unknown"),
        "span_count": summary.get("span_count"),
        "error_count": summary.get("error_count"),
        "duration_ms": summary.get("duration_ms"),
        "name": summary.get("name"),
    }


async def read_trace(client: AsyncNemoClient, ref: str, *, workspace: str) -> TraceExplorer:
    """Read one production trace in full.

    Args:
        client: Platform client.
        ref: A bare trace id, ``intake://<id>``, or ``intake://traces/<id>``. The first
            two are what ``find_agent_traces`` and ``Insight.trace_refs`` produce.
        workspace: Workspace that holds the trace.

    Returns:
        A loaded ``TraceExplorer``.

    Raises:
        TraceQueryError: The trace is missing or unreadable.
    """
    uri = ref if ref.startswith("intake://") else f"intake://{ref}"
    try:
        return await TraceExplorer.from_ref(ResourceRef(uri=uri), client, workspace)
    except Exception as exc:
        raise _explain(exc, doing=f"Reading trace '{ref}'", workspace=workspace) from exc


async def analyze_trace(
    client: AsyncNemoClient,
    ref: str,
    *,
    workspace: str,
    experiment_dir: Path,
    agent_path: Path,
) -> Diagnostic:
    """Diagnose one production trace with the Experimentalist trace analyzer.

    ``TraceAnalyzer.run`` expects an evaluator trial. A production trace has none, so
    this function synthesizes the ``Task`` and ``TrialResult`` around the trace ref.
    The task declares no dependencies, so ``Task.start_deps()`` starts nothing.

    Args:
        client: Platform client.
        ref: A bare trace id, ``intake://<id>``, or ``intake://traces/<id>``.
        workspace: Workspace that holds the trace.
        experiment_dir: Directory for analyzer artifacts.
        agent_path: Local path to the source of the agent under test.

    Returns:
        The diagnosis: outcome, summary, failure point, and root cause.
    """
    trace_id = ref.removeprefix("intake://").removeprefix("traces/")
    task = Task(id=f"trace-{trace_id}", description="Production trace with no evaluator task behind it.")
    trial = TrialResult(
        id=f"trace-{trace_id}",
        task_id=task.id,
        status="completed",
        trace=ResourceRef(uri=f"intake://{trace_id}", description="Production trace read from Intake."),
        metadata={"source": "intake"},
    )

    async def load_trace(reference: ResourceRef) -> TraceExplorer:
        """Resolve a trace ref with this call's client, matching the Experimentalist's
        `ctx.load_trace`: the analyzer takes a loader rather than a platform client, so
        its signature names no platform type."""
        return await TraceExplorer.from_ref(reference, client, workspace)

    return await TraceAnalyzer(experiment_dir=experiment_dir).run(
        trial=trial,
        task=task,
        agent_path=agent_path,
        insight=None,
        load_trace=load_trace,
    )
