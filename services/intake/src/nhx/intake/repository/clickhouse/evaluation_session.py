# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse implementation of per-session Evaluation reads.

Returns one row per ingested session (test case execution), using ``trace_index``
for root/session membership and per-session aggregates from all spans (tokens +
cost), plus per-evaluator session-mean scores from ``evaluator_results``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.clickhouse.trace import current_spans_sql
from nhx.intake.repository.evaluation_session import (
    EvaluationSessionPage,
    EvaluationSessionRepository,
    EvaluationSessionRow,
    MetricSortTooLargeError,
)
from nhx.intake.spans.domain import IntakeResponseMode, SpanStatus
from nhx.intake.spans.span_attribute_catalog import COST_SCALE, SpanAttributeField, spec_for_field
from nhx.intake.spans.storage import (
    float_or_none,
    int_or_none,
    normalize_span_status,
    str_or_none,
    text_query_parameters,
    text_select_for_mode,
)

# Sort fields that require a pre-pagination spans join to compute. These values live in the
# `spans` table (not `trace_index`), so they don't exist until after the session_metrics join.
# Sorting by them globally requires computing them for ALL sessions before LIMIT/OFFSET runs —
# see `_metric_sort_page_refs_sql` and the conditional in `list_sessions`.
_PRE_METRICS_SORT_FIELDS = frozenset({"cost_total_usd", "tokens"})

# Maps each API sort field to its SQL expression in the page query. Simple fields come
# from scoped_sessions; pre-metrics fields reference the pre_page_metrics CTE.
_SORT_EXPR_PAGE: dict[str, str] = {
    "started_at": "start_time",
    "ended_at": "end_time",
    "latency_ms": "latency_ms",
    "status": "root_span_status",
    "test_case_name": "test_case_name",
    "cost_total_usd": "pm.cost_total_usd",
    "tokens": "pm.total_tokens",
}

# Sessions above this threshold will not be sorted by cost or tokens. Pre-metrics sort joins
# spans for EVERY scoped session before paginating — an unbounded aggregation on the request
# path. Mirror the evaluations list cap (_MAX_GROUP_EVALUATIONS) to prevent runaway queries.
# Raise this if legitimate evaluations routinely exceed it; the right long-term fix is to
# denormalise cost/tokens into trace_index so no pre-pagination join is needed.
_MAX_METRIC_SORT_SESSIONS = 10_000


@dataclass(frozen=True)
class _SessionPageRef:
    """One session's identity and complete ``trace_index`` sorting key."""

    session_id: str
    trace_id: str
    root_span_id: str
    start_time_us: int

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> _SessionPageRef:
        return cls(
            session_id=str(row["session_id"]),
            trace_id=str(row["trace_id"]),
            root_span_id=str(row["root_span_id"]),
            start_time_us=int(row["start_time_us"]),
        )

    @property
    def storage_key(self) -> tuple[int, str, str]:
        return self.start_time_us, self.trace_id, self.root_span_id


class ClickHouseEvaluationSessionRepository(EvaluationSessionRepository):
    def __init__(self, executor: ClickHouseExecutor) -> None:
        self._executor = executor

    async def list_sessions(
        self,
        *,
        workspace: str,
        evaluation_name: str,
        status: SpanStatus | None = None,
        test_case_name: str | None = None,
        page: int,
        page_size: int,
        mode: IntakeResponseMode,
        sort_keys: list[tuple[str, bool]] | None = None,
    ) -> EvaluationSessionPage:
        trace_index_table = self._executor.table(ClickHouseTable.TRACE_INDEX)
        spans_table = self._executor.table(ClickHouseTable.SPANS)
        evaluator_results_table = self._executor.table(ClickHouseTable.EVALUATOR_RESULTS)

        scoped_filter_sql, scoped_filter_parameters = _scoped_filter(test_case_name=test_case_name, status=status)

        base_parameters: dict[str, Any] = {
            "workspace": workspace,
            "evaluation_name": evaluation_name,
            "input_tokens_key": spec_for_field(SpanAttributeField.INPUT_TOKENS).bag_key,
            "output_tokens_key": spec_for_field(SpanAttributeField.OUTPUT_TOKENS).bag_key,
            "cached_tokens_key": spec_for_field(SpanAttributeField.CACHED_TOKENS).bag_key,
            "cost_key": spec_for_field(SpanAttributeField.COST_TOTAL_USD).bag_key,
        }

        count_sql = _count_sql(
            trace_index_table=trace_index_table,
            scoped_filter_sql=scoped_filter_sql,
        )
        count_value = await self._executor.fetch_scalar(
            ClickHouseQuery(
                name="evaluation_sessions.count",
                statement=count_sql,
                parameters={**base_parameters, **scoped_filter_parameters},
            )
        )
        total = int(count_value) if count_value is not None else 0
        if total == 0:
            return EvaluationSessionPage(rows=[], total=0)

        needs_pre_metrics = sort_keys is not None and any(f in _PRE_METRICS_SORT_FIELDS for f, _ in sort_keys)
        if needs_pre_metrics and total > _MAX_METRIC_SORT_SESSIONS:
            raise MetricSortTooLargeError(total, _MAX_METRIC_SORT_SESSIONS)

        offset = (page - 1) * page_size

        if needs_pre_metrics:
            assert sort_keys is not None
            page_refs_sql = _metric_sort_page_refs_sql(
                trace_index_table=trace_index_table,
                spans_table=spans_table,
                scoped_filter_sql=scoped_filter_sql,
                sort_keys=sort_keys,
            )
            page_query_name = "evaluation_sessions.metric_sort.page"
        else:
            page_refs_sql = _page_refs_sql(
                trace_index_table=trace_index_table,
                scoped_filter_sql=scoped_filter_sql,
                sort_keys=sort_keys or [],
            )
            page_query_name = "evaluation_sessions.page"

        page_rows = await self._executor.fetch_all(
            ClickHouseQuery(
                name=page_query_name,
                statement=page_refs_sql,
                parameters={
                    **base_parameters,
                    **scoped_filter_parameters,
                    "limit": page_size,
                    "offset": offset,
                },
            )
        )
        page_refs = [_SessionPageRef.from_row(record) for record in page_rows]
        if not page_refs:
            return EvaluationSessionPage(rows=[], total=total)

        hydrate_rows = await self._executor.fetch_all(
            ClickHouseQuery(
                name="evaluation_sessions.hydrate",
                statement=_hydrate_by_refs_sql(
                    trace_index_table=trace_index_table,
                    spans_table=spans_table,
                    evaluator_results_table=evaluator_results_table,
                    mode=mode,
                ),
                parameters={
                    **base_parameters,
                    **text_query_parameters(mode),
                    **_page_ref_parameters(page_refs),
                },
            )
        )
        rows_by_id = {record["session_id"]: _row(record) for record in hydrate_rows}
        # ClickHouse does not preserve the page query's order through the hydration joins.
        rows = [rows_by_id[ref.session_id] for ref in page_refs if ref.session_id in rows_by_id]

        return EvaluationSessionPage(rows=rows, total=total)


def _scoped_filter(*, test_case_name: str | None, status: SpanStatus | None) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    parameters: dict[str, Any] = {}
    if test_case_name is not None:
        parameters["test_case_name"] = test_case_name
        clauses.append("test_case_name = %(test_case_name)s")
    if status is not None:
        parameters["status"] = status.value
        clauses.append("root_status = %(status)s")
    return "".join(f"\n            AND {clause}" for clause in clauses), parameters


def _scoped_sessions_sql(
    trace_index_table: str,
    *,
    scoped_filter_sql: str,
    mode: IntakeResponseMode,
) -> str:
    select_columns = [
        "workspace",
        "evaluation_name",
        "session_id",
        "test_case_name",
        "trace_id",
        "root_span_id",
        "root_started_at AS start_time",
        "root_ended_at AS end_time",
        "latency_ms",
        "root_status AS root_span_status",
    ]
    select_columns.extend(
        (
            text_select_for_mode("root_input", alias="input", mode=mode),
            text_select_for_mode("root_output", alias="output", mode=mode),
        )
    )
    select_sql = ",\n            ".join(select_columns)
    return f"""
        SELECT
            {select_sql}
        FROM {trace_index_table} FINAL
        WHERE workspace = %(workspace)s
            AND is_deleted = 0
            AND evaluation_name = %(evaluation_name)s
            {scoped_filter_sql}
        ORDER BY root_started_at ASC, root_span_id ASC
        LIMIT 1 BY workspace, session_id, evaluation_name
    """


def _count_sql(
    *,
    trace_index_table: str,
    scoped_filter_sql: str,
) -> str:
    scoped_sessions_sql = _scoped_sessions_sql(
        trace_index_table,
        scoped_filter_sql=scoped_filter_sql,
        mode="summary",
    )
    return f"""
        SELECT count()
        FROM (
            {scoped_sessions_sql}
        ) AS scoped_sessions
    """


def _build_order_by(sort_keys: list[tuple[str, bool]], expr_map: dict[str, str], tiebreaker: str) -> str:
    """Build a comma-separated ORDER BY clause from an ordered list of (field, descending) pairs.

    Each field is looked up in expr_map to get its SQL expression in the current query scope
    (column names differ between page_sessions and the final SELECT — callers pass the right map).
    NULLS LAST on every user key so missing values sort at the end rather than the front.
    Always appends tiebreaker as the final stable sort key so pages are deterministic when the
    user's keys produce ties.
    """
    parts = [f"{expr_map[field]} {'DESC' if desc else 'ASC'} NULLS LAST" for field, desc in sort_keys]
    parts.append(tiebreaker)
    return ", ".join(parts)


def _page_refs_sql(
    *,
    trace_index_table: str,
    scoped_filter_sql: str,
    sort_keys: list[tuple[str, bool]],
) -> str:
    """Select one ordered page without reading root payload columns."""

    scoped_sessions_sql = _scoped_sessions_sql(
        trace_index_table,
        scoped_filter_sql=scoped_filter_sql,
        mode="summary",
    )
    order_by = (
        _build_order_by(sort_keys, _SORT_EXPR_PAGE, "s.root_span_id ASC")
        if sort_keys
        else "s.start_time ASC, s.root_span_id ASC"
    )
    return f"""
        WITH scoped_sessions AS (
            {scoped_sessions_sql}
        )
        SELECT
            s.session_id,
            s.trace_id,
            s.root_span_id,
            toUnixTimestamp64Micro(s.start_time) AS start_time_us
        FROM scoped_sessions AS s
        ORDER BY {order_by}
        LIMIT %(limit)s OFFSET %(offset)s
    """


def _metric_sort_page_refs_sql(
    *,
    trace_index_table: str,
    spans_table: str,
    scoped_filter_sql: str,
    sort_keys: list[tuple[str, bool]],
) -> str:
    """Select an ordered page after computing cost/token metrics for all sessions.

    Aggregates cost/tokens across ALL scoped sessions exactly once, then applies
    ORDER BY + LIMIT/OFFSET. Only storage references are returned; payload hydration
    is a separate query.

    Why separate: ClickHouse inlines regular CTEs rather than materialising them, so a
    single query that references `page_sessions` from multiple CTEs would re-execute
    the expensive all-session span aggregation once per reference. Returning IDs here
    and hydrating in _hydrate_by_refs_sql ensures the aggregation runs exactly once.
    """
    # summary mode: text payloads are not needed for sorting
    scoped_sessions_sql = _scoped_sessions_sql(
        trace_index_table,
        scoped_filter_sql=scoped_filter_sql,
        mode="summary",
    )
    all_scoped_spans = current_spans_sql(
        spans_table,
        extra_where_sql=(
            "(span_versions.workspace, span_versions.session_id) IN (SELECT workspace, session_id FROM scoped_sessions)"
        ),
    )
    order_by = _build_order_by(sort_keys, _SORT_EXPR_PAGE, "s.root_span_id ASC")
    return f"""
        WITH
        scoped_sessions AS (
            {scoped_sessions_sql}
        ),
        pre_page_metrics AS (
            SELECT
                s.workspace AS workspace,
                s.session_id AS session_id,
                {_guarded_sum_sql("cost_key", scale=COST_SCALE)} AS cost_total_usd,
                if(
                    {_guarded_sum_sql("input_tokens_key")} IS NULL AND {_guarded_sum_sql("output_tokens_key")} IS NULL,
                    NULL,
                    coalesce({_guarded_sum_sql("input_tokens_key")}, 0) + coalesce({_guarded_sum_sql("output_tokens_key")}, 0)
                ) AS total_tokens
            FROM scoped_sessions AS s
            LEFT JOIN {all_scoped_spans} AS spans
                ON s.workspace = spans.workspace
                AND s.session_id = spans.session_id
                AND spans.is_deleted = 0
            GROUP BY s.workspace, s.session_id
        )
        SELECT
            s.session_id,
            s.trace_id,
            s.root_span_id,
            toUnixTimestamp64Micro(s.start_time) AS start_time_us
        FROM scoped_sessions AS s
        LEFT JOIN pre_page_metrics AS pm
            ON s.workspace = pm.workspace AND s.session_id = pm.session_id
        ORDER BY {order_by}
        LIMIT %(limit)s OFFSET %(offset)s
    """


def _hydrate_by_refs_sql(
    *,
    trace_index_table: str,
    spans_table: str,
    evaluator_results_table: str,
    mode: IntakeResponseMode,
) -> str:
    """Hydrate payloads, span metrics, and evaluator scores for one selected page."""
    select_columns = [
        "workspace",
        "evaluation_name",
        "session_id",
        "test_case_name",
        "trace_id",
        "root_span_id",
        "root_started_at AS start_time",
        "root_ended_at AS end_time",
        "latency_ms",
        "root_status AS root_span_status",
    ]
    select_columns.extend(
        (
            text_select_for_mode("root_input", alias="input", mode=mode),
            text_select_for_mode("root_output", alias="output", mode=mode),
        )
    )
    select_sql = ",\n            ".join(select_columns)

    page_spans = current_spans_sql(
        spans_table,
        extra_where_sql="span_versions.session_id IN %(page_session_ids)s",
    )

    return f"""
        WITH
        page_sessions AS (
            SELECT
                {select_sql}
            FROM {trace_index_table} FINAL
            WHERE workspace = %(workspace)s
                AND is_deleted = 0
                AND evaluation_name = %(evaluation_name)s
                -- The flat predicates engage the bloom indexes; the tuple preserves
                -- exact trace_index identity. The time range engages the primary key.
                AND session_id IN %(page_session_ids)s
                AND trace_id IN %(page_trace_ids)s
                AND root_started_at >= fromUnixTimestamp64Micro(%(page_started_at_min_us)s)
                AND root_started_at <= fromUnixTimestamp64Micro(%(page_started_at_max_us)s)
                AND (
                    toUnixTimestamp64Micro(root_started_at),
                    trace_id,
                    root_span_id
                ) IN %(page_storage_keys)s
            LIMIT 1 BY workspace, session_id, evaluation_name
        ),
        current_page_spans AS (
            {page_spans}
        ),
        session_metrics AS (
            SELECT
                spans.workspace AS workspace,
                spans.session_id AS session_id,
                {_guarded_sum_sql("input_tokens_key")} AS input_tokens,
                {_guarded_sum_sql("output_tokens_key")} AS output_tokens,
                {_guarded_sum_sql("cached_tokens_key")} AS cached_tokens,
                {_guarded_sum_sql("cost_key", scale=COST_SCALE)} AS cost_total_usd
            FROM current_page_spans AS spans
            WHERE spans.is_deleted = 0
            GROUP BY spans.workspace, spans.session_id
        ),
        session_scores AS (
            SELECT
                workspace,
                session_id,
                mapFromArrays(groupArray(evaluator_name), groupArray(mean_score)) AS evaluator_scores
            FROM (
                SELECT
                    results.workspace AS workspace,
                    results.session_id AS session_id,
                    results.name AS evaluator_name,
                    avg(results.value) AS mean_score
                FROM (
                    SELECT workspace, session_id, name, value
                    FROM {evaluator_results_table} FINAL
                    WHERE workspace = %(workspace)s
                        AND session_id IN %(page_session_ids)s
                        AND data_type IN ('NUMERIC', 'BOOLEAN')
                        AND value IS NOT NULL
                ) AS results
                GROUP BY results.workspace, results.session_id, results.name
            )
            GROUP BY workspace, session_id
        )
        SELECT
            sessions.workspace AS workspace,
            sessions.evaluation_name,
            sessions.session_id AS session_id,
            sessions.test_case_name,
            sessions.trace_id AS trace_id,
            sessions.root_span_id AS root_span_id,
            sessions.start_time AS start_time,
            sessions.end_time AS end_time,
            sessions.latency_ms AS latency_ms,
            sessions.root_span_status AS root_span_status,
            sessions.input AS input,
            sessions.output AS output,
            metrics.input_tokens AS input_tokens,
            metrics.output_tokens AS output_tokens,
            metrics.cached_tokens AS cached_tokens,
            metrics.cost_total_usd AS cost_total_usd,
            scores.evaluator_scores AS evaluator_scores
        FROM page_sessions AS sessions
        LEFT JOIN session_metrics AS metrics
            ON sessions.workspace = metrics.workspace
            AND sessions.session_id = metrics.session_id
        LEFT JOIN session_scores AS scores
            ON sessions.workspace = scores.workspace
            AND sessions.session_id = scores.session_id
    """


def _page_ref_parameters(refs: list[_SessionPageRef]) -> dict[str, object]:
    start_times_us = [ref.start_time_us for ref in refs]
    return {
        "page_session_ids": [ref.session_id for ref in refs],
        "page_trace_ids": [ref.trace_id for ref in refs],
        "page_started_at_min_us": min(start_times_us),
        "page_started_at_max_us": max(start_times_us),
        "page_storage_keys": [ref.storage_key for ref in refs],
    }


def _guarded_sum_sql(parameter_name: str, *, scale: int = 1) -> str:
    key = f"%({parameter_name})s"
    sum_expr = f"sumIf(spans.attributes_number[{key}], has(mapKeys(spans.attributes_number), {key}))"
    if scale != 1:
        sum_expr = f"{sum_expr} / {scale}"
    return f"""
        if(
            countIf(has(mapKeys(spans.attributes_number), {key})) = 0,
            NULL,
            {sum_expr}
        )
    """


def _row(record: dict[str, Any]) -> EvaluationSessionRow:
    return EvaluationSessionRow(
        workspace=record["workspace"],
        evaluation_name=record["evaluation_name"],
        session_id=record["session_id"],
        test_case_name=str_or_none(record["test_case_name"]),
        trace_id=record["trace_id"],
        root_span_id=record["root_span_id"],
        started_at=record["start_time"],
        ended_at=record["end_time"],
        latency_ms=float_or_none(record["latency_ms"]),
        status=normalize_span_status(record["root_span_status"]),
        input=str_or_none(record["input"]),
        output=str_or_none(record["output"]),
        input_tokens=int_or_none(record["input_tokens"]),
        output_tokens=int_or_none(record["output_tokens"]),
        cached_tokens=int_or_none(record["cached_tokens"]),
        cost_total_usd=float_or_none(record["cost_total_usd"]),
        evaluator_scores=_score_map(record.get("evaluator_scores")),
    )


def _score_map(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    return {str(key): float(score) for key, score in dict(value).items()}
