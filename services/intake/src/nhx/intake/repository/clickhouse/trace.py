# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse implementation of Intake trace reads."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any

from nhx.common.api.common import PaginatedResult
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseExternalData, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.trace import TraceRepository
from nhx.intake.spans.domain import (
    CostRollup,
    IntakeTrace,
    LatencyRollup,
    TokenRollup,
    TraceListFilter,
    TraceMetricBucket,
    TraceMetricPoint,
    TraceMode,
)
from nhx.intake.spans.span_attribute_catalog import SpanAttributeField, spec_for_field
from nhx.intake.spans.span_rollups import METRIC_ATTRIBUTE_FIELDS, metric_aggregate_columns
from nhx.intake.spans.storage import (
    float_or_none,
    int_or_none,
    make_pagination,
    normalize_span_status,
    text_query_parameters,
    text_select_for_mode,
)

logger = logging.getLogger(__name__)

TRACE_SORT_COLUMNS = {
    "started_at": "started_at",
}

TRACE_COLUMNS = [
    "id",
    "workspace",
    "session_id",
    "source_format",
    "root_span_id",
    "name",
    "input",
    "output",
    "project",
    "evaluation_name",
    "test_case_name",
    "started_at",
    "ended_at",
    "status",
    *METRIC_ATTRIBUTE_FIELDS.keys(),
    "models",
    "providers",
    "span_count",
    "error_count",
    "ingested_at",
]

_CURRENT_SPAN_IDENTITY_COLUMNS = (
    "workspace",
    "source_format",
    "trace_id",
    "external_span_id",
    "id",
)
_CURRENT_SPAN_VALUE_COLUMNS = (
    "session_id",
    "external_parent_span_id",
    "kind",
    "name",
    "status",
    "start_time",
    "end_time",
    "attributes_string",
    "attributes_number",
    "attributes_bool",
    "event_ts",
    "is_deleted",
)

_ZERO_DATETIME = datetime.fromtimestamp(0, tz=timezone.utc)


@dataclass(frozen=True)
class _TracePageRef:
    source_format: str
    trace_id: str
    session_id: str
    started_at_us: int

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> _TracePageRef:
        return cls(
            source_format=str(row["source_format"]),
            trace_id=str(row["id"]),
            session_id=str(row["session_id"]),
            started_at_us=int(row["started_at_us"]),
        )

    @property
    def trace_key(self) -> tuple[str, str]:
        return self.source_format, self.trace_id


class ClickHouseTraceRepository(TraceRepository):
    def __init__(self, executor: ClickHouseExecutor) -> None:
        self._executor = executor

    async def list_traces(
        self,
        *,
        filters: TraceListFilter,
        page: int,
        page_size: int,
        sort: str,
        mode: TraceMode,
    ) -> PaginatedResult[IntakeTrace]:
        trace_index_table = self._executor.table(ClickHouseTable.TRACE_INDEX)
        spans_table = self._executor.table(ClickHouseTable.SPANS)
        count_trace_index_sql, parameters = _trace_index_sql(
            trace_index_table,
            filters,
            mode="summary",
        )

        total_results = int(
            await self._executor.fetch_scalar(
                ClickHouseQuery(
                    name="traces.list.count",
                    statement=f"""
                    SELECT count()
                    FROM ({count_trace_index_sql}) AS traces
                    """,
                    parameters=parameters,
                )
            )
            or 0
        )

        offset = (page - 1) * page_size
        page_sql = _trace_page_sql(trace_index_sql=count_trace_index_sql, sort=sort)
        page_rows = await self._executor.fetch_all(
            ClickHouseQuery(
                name="traces.list.page",
                statement=page_sql,
                parameters={
                    **parameters,
                    "limit": page_size,
                    "offset": offset,
                },
            )
        )
        rows = page_rows
        if mode != "summary" and page_rows:
            page_refs = [_TracePageRef.from_row(row) for row in page_rows]
            hydration_sql, hydration_parameters = _trace_hydration_sql(
                trace_index_table=trace_index_table,
                spans_table=spans_table,
                mode=mode,
            )
            rows = await self._executor.fetch_all(
                ClickHouseQuery(
                    name="traces.list.hydrate",
                    statement=hydration_sql,
                    parameters={
                        "workspace": filters.workspace,
                        **hydration_parameters,
                        **_trace_page_parameters(page_refs),
                    },
                )
            )
            rows, dropped_refs = _reconcile_hydrated_page(page_refs, rows)
            if dropped_refs:
                logger.warning(
                    "Trace page hydration omitted refs returned by the page query",
                    extra={
                        "workspace": filters.workspace,
                        "page": page,
                        "dropped_trace_refs": [ref.trace_key for ref in dropped_refs],
                    },
                )
                total_results = max(offset + len(rows), total_results - len(dropped_refs))
        traces = [_row_to_trace(row) for row in rows]
        return PaginatedResult(
            data=traces,
            pagination=make_pagination(
                page=page,
                page_size=page_size,
                current_page_size=len(traces),
                total_results=total_results,
            ),
        )

    async def get_trace(self, *, workspace: str, trace_id: str, mode: TraceMode) -> IntakeTrace | None:
        result = await self.list_traces(
            filters=TraceListFilter(workspace=workspace, trace_ids=[trace_id]),
            page=1,
            page_size=1,
            sort="-started_at",
            mode=mode,
        )
        return result.data[0] if result.data else None

    async def trace_metrics(
        self,
        *,
        filters: TraceListFilter,
        bucket: TraceMetricBucket,
        timezone_name: str,
    ) -> list[TraceMetricPoint]:
        trace_index_table = self._executor.table(ClickHouseTable.TRACE_INDEX)
        spans_table = self._executor.table(ClickHouseTable.SPANS)
        roots_sql, parameters = _metric_roots_sql(
            trace_index_table=trace_index_table,
            filters=filters,
            bucket=bucket,
        )
        # Reuses the same per-trace span rollup as trace hydration, rescoped from a
        # page of trace refs to whatever the filter selected.
        rollups_sql, rollup_parameters = _trace_aggregates_sql(
            spans_table,
            extra_where_sql=(
                # session_id engages the spans primary key, trace_id drives the bloom
                # filter, and the tuple keeps a trace_id shared across source formats
                # from pulling in unrelated spans.
                "span_versions.session_id IN (SELECT session_id FROM roots)\n"
                "                AND span_versions.trace_id IN (SELECT trace_id FROM roots)\n"
                "                AND (span_versions.source_format, span_versions.trace_id) "
                "IN (SELECT source_format, trace_id FROM roots)"
            ),
        )
        parameters.update(rollup_parameters)
        parameters["metrics_timezone"] = timezone_name

        statement = f"""
            WITH
            roots AS (
                {roots_sql}
            ),
            rollups AS (
                {rollups_sql}
            )
            SELECT
                {_metric_select_columns()}
            FROM roots
            LEFT JOIN rollups
                ON roots.workspace = rollups.workspace
                AND roots.source_format = rollups.source_format
                AND roots.trace_id = rollups.trace_id
            GROUP BY bucket_start
            ORDER BY bucket_start ASC
        """
        rows = await self._executor.fetch_all(
            ClickHouseQuery(name="traces.metrics", statement=statement, parameters=parameters)
        )
        return [_row_to_metric_point(row, bucket=bucket) for row in rows]

    async def latest_trace_started_at_by_group(
        self,
        *,
        workspace: str,
        trace_refs_by_group: dict[str, list[str]],
    ) -> dict[str, datetime]:
        pairs = [
            (group_id, trace_id)
            for group_id, trace_ids in trace_refs_by_group.items()
            for trace_id in dict.fromkeys(trace_ids)
        ]
        if not pairs:
            return {}

        trace_index_table = self._executor.table(ClickHouseTable.TRACE_INDEX)
        rows = await self._executor.fetch_all(
            ClickHouseQuery(
                name="traces.latest_started_at_by_group",
                statement=f"""
                WITH
                refs AS (
                    SELECT group_id, trace_id
                    FROM trace_refs
                ),
                traces AS (
                    SELECT
                        trace_roots.trace_id AS id,
                        trace_roots.root_started_at AS started_at
                    FROM {trace_index_table} AS trace_roots FINAL
                    WHERE trace_roots.workspace = %(workspace)s
                        AND trace_roots.is_deleted = 0
                        AND trace_roots.trace_id IN (SELECT trace_id FROM refs)
                    ORDER BY trace_roots.root_started_at ASC, trace_roots.root_span_id ASC
                    LIMIT 1 BY trace_roots.workspace, trace_roots.source_format, trace_roots.trace_id
                )
                SELECT refs.group_id, max(traces.started_at) AS started_at
                FROM refs
                INNER JOIN traces ON traces.id = refs.trace_id
                GROUP BY refs.group_id
                """,
                parameters={"workspace": workspace},
                external_data=ClickHouseExternalData(
                    file_name="trace_refs.jsonl",
                    data=b"\n".join(
                        json.dumps({"group_id": group_id, "trace_id": trace_id}).encode()
                        for group_id, trace_id in pairs
                    ),
                    fmt="JSONEachRow",
                    structure="group_id String, trace_id String",
                ),
            )
        )
        return {str(row["group_id"]): row["started_at"] for row in rows}


def _trace_page_sql(*, trace_index_sql: str, sort: str) -> str:
    """Build the first-phase query that selects lightweight trace roots for one page."""

    return f"""
        SELECT
            {_trace_select_columns(include_aggregates=False)},
            toUnixTimestamp64Micro(traces.started_at) AS started_at_us
        FROM ({trace_index_sql}) AS traces
        ORDER BY {_order_by(sort, table_alias="traces")}
        LIMIT %(limit)s OFFSET %(offset)s
    """


def _trace_hydration_sql(*, trace_index_table: str, spans_table: str, mode: TraceMode) -> tuple[str, dict[str, Any]]:
    """Build the second-phase query that hydrates page roots and span aggregates."""

    trace_roots_sql, trace_parameters = _page_trace_roots_sql(trace_index_table=trace_index_table, mode=mode)
    aggregates_sql, aggregate_parameters = _trace_aggregates_sql(spans_table)
    query = f"""
        WITH
        traces AS (
            {trace_roots_sql}
        ),
        rollups AS (
            {aggregates_sql}
        )
        SELECT
            {_trace_select_columns(include_aggregates=True)}
        FROM traces
        LEFT JOIN rollups
            ON traces.workspace = rollups.workspace
            AND traces.source_format = rollups.source_format
            AND traces.id = rollups.trace_id
    """
    return query, {**trace_parameters, **aggregate_parameters}


def _page_trace_roots_sql(*, trace_index_table: str, mode: TraceMode) -> tuple[str, dict[str, Any]]:
    trace_columns, parameters = _trace_index_select_columns(mode=mode)
    query = f"""
        SELECT
            {trace_columns}
        FROM {trace_index_table} AS trace_roots FINAL
        WHERE trace_roots.workspace = %(workspace)s
            AND trace_roots.is_deleted = 0
            -- The time range engages the primary key before FINAL; the flat trace ID
            -- engages its bloom index, while the tuple preserves source-format identity.
            AND trace_roots.root_started_at >= fromUnixTimestamp64Micro(%(page_started_at_min_us)s)
            AND trace_roots.root_started_at <= fromUnixTimestamp64Micro(%(page_started_at_max_us)s)
            AND trace_roots.trace_id IN %(page_trace_ids)s
            AND (
                trace_roots.source_format,
                trace_roots.trace_id
            ) IN %(page_trace_keys)s
        ORDER BY trace_roots.root_started_at ASC, trace_roots.root_span_id ASC
        LIMIT 1 BY trace_roots.workspace, trace_roots.source_format, trace_roots.trace_id
    """
    return query, parameters


def _trace_select_columns(*, include_aggregates: bool) -> str:
    aggregate_columns = [
        f"rollups.{column} AS {column}" if include_aggregates else f"NULL AS {column}"
        for column in (
            *METRIC_ATTRIBUTE_FIELDS.keys(),
            "models",
            "providers",
            "span_count",
            "error_count",
        )
    ]
    columns = [
        "traces.id AS id",
        "traces.workspace AS workspace",
        "traces.session_id AS session_id",
        "traces.source_format AS source_format",
        "traces.root_span_id AS root_span_id",
        "traces.name AS name",
        "traces.input AS input",
        "traces.output AS output",
        "traces.project AS project",
        "traces.evaluation_name AS evaluation_name",
        "traces.test_case_name AS test_case_name",
        "traces.agent_name AS agent_name",
        "traces.agent_version AS agent_version",
        "traces.started_at AS started_at",
        "traces.ended_at AS ended_at",
        "traces.status AS status",
        *aggregate_columns,
        "traces.ingested_at AS ingested_at",
    ]
    return ",\n            ".join(columns)


def _trace_index_sql(
    table: str,
    filters: TraceListFilter,
    *,
    mode: TraceMode,
) -> tuple[str, dict[str, Any]]:
    where_sql, parameters = _trace_index_where(filters, qualifier="trace_roots")
    select_columns, select_parameters = _trace_index_select_columns(mode=mode)
    parameters.update(select_parameters)
    query = f"""
        SELECT
            {select_columns}
        FROM {table} AS trace_roots FINAL
        WHERE {where_sql}
        ORDER BY trace_roots.root_started_at ASC, trace_roots.root_span_id ASC
        LIMIT 1 BY trace_roots.workspace, trace_roots.source_format, trace_roots.trace_id
    """
    return query, parameters


def _trace_index_select_columns(*, mode: TraceMode) -> tuple[str, dict[str, Any]]:
    payload_columns = (
        text_select_for_mode("trace_roots.root_input", alias="input", mode=mode),
        text_select_for_mode("trace_roots.root_output", alias="output", mode=mode),
    )
    columns = (
        "trace_roots.trace_id AS id",
        "trace_roots.workspace AS workspace",
        "trace_roots.session_id AS session_id",
        "trace_roots.source_format AS source_format",
        "nullIf(trace_roots.root_span_id, '') AS root_span_id",
        "nullIf(trace_roots.root_name, '') AS name",
        *payload_columns,
        "nullIf(trace_roots.project, '') AS project",
        "nullIf(trace_roots.evaluation_name, '') AS evaluation_name",
        "nullIf(trace_roots.test_case_name, '') AS test_case_name",
        "nullIf(trace_roots.agent_name, '') AS agent_name",
        "nullIf(trace_roots.agent_version, '') AS agent_version",
        "trace_roots.root_started_at AS started_at",
        "trace_roots.root_ended_at AS ended_at",
        "trace_roots.root_status AS status",
        "trace_roots.event_ts AS ingested_at",
    )
    return ",\n            ".join(columns), text_query_parameters(mode)


# session_id leads the spans sort key after workspace, so restricting it lets the
# primary key prune before the trace-id bloom filter widens the granule set.
_PAGE_TRACE_REFS_WHERE_SQL = (
    "span_versions.session_id IN %(page_session_ids)s\n"
    "                AND span_versions.trace_id IN %(page_trace_ids)s\n"
    "                AND (span_versions.source_format, span_versions.trace_id) "
    "IN %(page_trace_keys)s"
)


def _trace_aggregates_sql(
    table: str,
    *,
    extra_where_sql: str = _PAGE_TRACE_REFS_WHERE_SQL,
    extra_select_sql: str = "",
) -> tuple[str, dict[str, Any]]:
    """Build the per-trace span rollup shared by trace hydration and metric buckets.

    Both callers must read through ``current_spans_sql``: spans is a
    ReplacingMergeTree, so summing raw rows would double-count re-ingested spans.
    """
    source_alias = "trace_spans"
    select_columns, parameters = _trace_aggregate_select_columns(source_alias)
    if extra_select_sql:
        select_columns = f"{select_columns},\n            {extra_select_sql}"
    current_spans = current_spans_sql(table, extra_where_sql=extra_where_sql)
    query = f"""
        SELECT
            {select_columns}
        FROM {current_spans} AS {source_alias}
        WHERE {source_alias}.is_deleted = 0
        GROUP BY {source_alias}.workspace, {source_alias}.source_format, {source_alias}.trace_id
    """
    return query, parameters


def _trace_aggregate_select_columns(source_alias: str) -> tuple[str, dict[str, Any]]:
    metric_columns, parameters = metric_aggregate_columns(source_alias)
    parameters.update(
        model_key=spec_for_field(SpanAttributeField.MODEL).bag_key,
        provider_key=spec_for_field(SpanAttributeField.PROVIDER).bag_key,
    )
    columns = (
        f"{source_alias}.workspace AS workspace",
        f"{source_alias}.source_format AS source_format",
        f"{source_alias}.trace_id AS trace_id",
        metric_columns,
        _unique_string_attribute_aggregate(source_alias, parameter="model_key", alias="models"),
        _unique_string_attribute_aggregate(source_alias, parameter="provider_key", alias="providers"),
        "count() AS span_count",
        f"countIf({source_alias}.status = 'error') AS error_count",
    )
    return ",\n            ".join(columns), parameters


def _unique_string_attribute_aggregate(source_alias: str, *, parameter: str, alias: str) -> str:
    attribute = f"{source_alias}.attributes_string[%({parameter})s]"
    return f"""arraySort(groupUniqArrayIf(
                {attribute},
                has(mapKeys({source_alias}.attributes_string), %({parameter})s)
                    AND {attribute} != ''
            )) AS {alias}"""


def _trace_page_parameters(refs: Sequence[_TracePageRef]) -> dict[str, object]:
    started_at_us = [ref.started_at_us for ref in refs]
    return {
        "page_trace_ids": [ref.trace_id for ref in refs],
        "page_trace_keys": [ref.trace_key for ref in refs],
        "page_session_ids": sorted({ref.session_id for ref in refs}),
        "page_started_at_min_us": min(started_at_us),
        "page_started_at_max_us": max(started_at_us),
    }


def _reconcile_hydrated_page(
    page_refs: Sequence[_TracePageRef],
    hydrated_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[_TracePageRef]]:
    rows_by_key = {(str(row["source_format"]), str(row["id"])): row for row in hydrated_rows}
    rows = [rows_by_key[ref.trace_key] for ref in page_refs if ref.trace_key in rows_by_key]
    dropped_refs = [ref for ref in page_refs if ref.trace_key not in rows_by_key]
    return rows, dropped_refs


# Maps API/filter field names to their physical trace_index columns.
_TRACE_INDEX_FILTER_COLUMNS = {
    "evaluation_name": "evaluation_name",
    "test_case_name": "test_case_name",
    "agent_name": "agent_name",
}


def _trace_index_where(filters: TraceListFilter, *, qualifier: str) -> tuple[str, dict[str, Any]]:
    def column(name: str) -> str:
        return f"{qualifier}.{name}"

    clauses = [f"{column('workspace')} = %(workspace)s", f"{column('is_deleted')} = 0"]
    parameters: dict[str, Any] = {"workspace": filters.workspace}

    if filters.trace_ids is not None:
        clauses.append(f"{column('trace_id')} IN %(trace_ids)s")
        parameters["trace_ids"] = filters.trace_ids
    if filters.session_id is not None:
        clauses.append(f"{column('session_id')} = %(session_id)s")
        parameters["session_id"] = filters.session_id
    if filters.source_format is not None:
        clauses.append(f"{column('source_format')} = %(source_format)s")
        parameters["source_format"] = filters.source_format
    if filters.status is not None:
        clauses.append(f"{column('root_status')} = %(status)s")
        parameters["status"] = filters.status.value
    if filters.started_at_gte is not None:
        clauses.append(f"{column('root_started_at')} >= %(started_at_gte)s")
        parameters["started_at_gte"] = filters.started_at_gte
    if filters.started_at_lte is not None:
        clauses.append(f"{column('root_started_at')} <= %(started_at_lte)s")
        parameters["started_at_lte"] = filters.started_at_lte

    for field, filter_column in _TRACE_INDEX_FILTER_COLUMNS.items():
        value = getattr(filters, field)
        if value is None:
            continue
        parameter_name = f"filter_{field}"
        clauses.append(f"{column(filter_column)} = %({parameter_name})s")
        parameters[parameter_name] = value

    return " AND ".join(clauses), parameters


def current_spans_sql(
    table: str,
    *,
    extra_where_sql: str | None = None,
) -> str:
    """Read current span versions without materialising wide aggregate states.

    ``spans`` is a ReplacingMergeTree, so ``FINAL`` applies its latest-write-wins
    semantics. Callers should scope by ``session_id`` when possible because it
    follows ``workspace`` in the table's sorting key.
    """
    source_alias = "span_versions"
    columns = [
        f"{source_alias}.{column} AS {column}"
        for column in (*_CURRENT_SPAN_IDENTITY_COLUMNS, *_CURRENT_SPAN_VALUE_COLUMNS)
    ]
    columns_sql = ",\n                ".join(columns)
    where_sql = f"{source_alias}.workspace = %(workspace)s"
    if extra_where_sql is not None:
        where_sql = f"{where_sql}\n                AND {extra_where_sql}"
    return f"""
        (
            SELECT
                {columns_sql}
            FROM {table} AS {source_alias} FINAL
            WHERE {where_sql}
        )
    """


def _order_by(sort: str, *, table_alias: str | None = None) -> str:
    direction = "DESC" if sort.startswith("-") else "ASC"
    field = sort.removeprefix("-")
    column = TRACE_SORT_COLUMNS.get(field)
    if column is None:
        raise ValueError(f"Unsupported trace sort field: {field}")
    if table_alias is not None:
        column = f"{table_alias}.{column}"
        id_column = f"{table_alias}.id"
    else:
        id_column = "id"
    return f"{column} {direction}, {id_column} ASC"


def _row_to_trace(row: dict[str, Any]) -> IntakeTrace:
    ended_at = _none_if_zero_datetime(row.get("ended_at"))
    return IntakeTrace(
        id=row["id"],
        root_span_id=row.get("root_span_id") or None,
        workspace=row["workspace"],
        session_id=row["session_id"],
        source_format=row["source_format"],
        name=row.get("name") or None,
        input=row.get("input") or None,
        output=row.get("output") or None,
        project=row.get("project") or None,
        evaluation_name=row.get("evaluation_name") or None,
        test_case_name=row.get("test_case_name") or None,
        agent_name=row.get("agent_name") or None,
        agent_version=row.get("agent_version") or None,
        started_at=row["started_at"],
        ended_at=ended_at,
        duration_ms=_duration_ms(row["started_at"], ended_at),
        ingested_at=row["ingested_at"],
        status=normalize_span_status(row.get("status")),
        input_tokens=int_or_none(row.get("input_tokens")),
        output_tokens=int_or_none(row.get("output_tokens")),
        cached_tokens=int_or_none(row.get("cached_tokens")),
        total_tokens=int_or_none(row.get("total_tokens")),
        cost_usd=float_or_none(row.get("cost_usd")),
        cost_input_usd=float_or_none(row.get("cost_input_usd")),
        cost_output_usd=float_or_none(row.get("cost_output_usd")),
        models=_string_list_or_none(row.get("models")),
        providers=_string_list_or_none(row.get("providers")),
        span_count=int_or_none(row.get("span_count")),
        error_count=int_or_none(row.get("error_count")),
    )


# ClickHouse resolves these against the caller's timezone so buckets line up with the
# user's calendar rather than the server's. Week starts Monday (mode 1).
# toStartOfWeek/Month return Date, toStartOfDay returns DateTime. Cast so every bucket
# yields the same type; otherwise some buckets deserialize to naive datetimes and others
# to timezone-aware ones, and chart clients see inconsistent offsets.
_METRIC_BUCKET_EXPRESSIONS = {
    "hour": "toStartOfHour(trace_roots.root_started_at, %(metrics_timezone)s)",
    "day": "toStartOfDay(trace_roots.root_started_at, %(metrics_timezone)s)",
    "week": "toDateTime(toStartOfWeek(trace_roots.root_started_at, 1, %(metrics_timezone)s), %(metrics_timezone)s)",
    "month": "toDateTime(toStartOfMonth(trace_roots.root_started_at, %(metrics_timezone)s), %(metrics_timezone)s)",
}


def _metric_roots_sql(
    *,
    trace_index_table: str,
    filters: TraceListFilter,
    bucket: TraceMetricBucket,
) -> tuple[str, dict[str, Any]]:
    """Build the deduplicated root-span CTE the metric buckets group over."""

    where_sql, parameters = _trace_index_where(filters, qualifier="trace_roots")
    query = f"""
        SELECT
            trace_roots.workspace AS workspace,
            trace_roots.source_format AS source_format,
            trace_roots.trace_id AS trace_id,
            trace_roots.session_id AS session_id,
            {_metric_bucket_expression(bucket)} AS bucket_start,
            trace_roots.root_status AS root_status,
            trace_roots.latency_ms AS latency_ms
        FROM {trace_index_table} AS trace_roots FINAL
        WHERE {where_sql}
        ORDER BY trace_roots.root_started_at ASC, trace_roots.root_span_id ASC
        LIMIT 1 BY trace_roots.workspace, trace_roots.source_format, trace_roots.trace_id
    """
    return query, parameters


_TOKEN_ROLLUP_FIELDS = ("input_tokens", "output_tokens", "cached_tokens", "total_tokens")
_SUMMED_ROLLUP_QUANTILES = (0.9, 0.99)
_LATENCY_QUANTILES = (0.5, 0.9, 0.95, 0.99)


def _quantiles_expression(expression: str, quantiles: Sequence[float]) -> str:
    # One combined aggregate rather than a quantile state per percentile.
    return f"quantiles({', '.join(str(quantile) for quantile in quantiles)})({expression})"


def _metric_select_columns() -> str:
    columns = [
        "roots.bucket_start AS bucket_start",
        "count() AS run_count",
        "countIf(roots.root_status = 'error') AS failed_run_count",
    ]
    for field in (*_TOKEN_ROLLUP_FIELDS, "cost_usd"):
        source = f"rollups.{field}"
        columns.extend(
            (
                f"sum({source}) AS {field}_sum",
                f"avg({source}) AS {field}_mean",
                f"{_quantiles_expression(source, _SUMMED_ROLLUP_QUANTILES)} AS {field}_quantiles",
            )
        )
    columns.extend(
        (
            # Percentiles cannot yield a mean, and the design needs one; it is also what
            # makes an aggregate latency-per-token ratio derivable client-side.
            "avg(roots.latency_ms) AS latency_ms_mean",
            f"{_quantiles_expression('roots.latency_ms', _LATENCY_QUANTILES)} AS latency_ms_quantiles",
        )
    )
    return ",\n                ".join(columns)


def _metric_bucket_expression(bucket: TraceMetricBucket) -> str:
    if bucket == "total":
        # A constant collapses the filtered range into one row; the value is discarded.
        return "toDateTime(0)"
    try:
        return _METRIC_BUCKET_EXPRESSIONS[bucket]
    except KeyError:
        raise ValueError(f"Unsupported trace metric bucket: {bucket}") from None


def _row_to_metric_point(row: dict[str, Any], *, bucket: TraceMetricBucket) -> TraceMetricPoint:
    return TraceMetricPoint(
        bucket_start=None if bucket == "total" else row["bucket_start"],
        run_count=int(row["run_count"]),
        # Counts failed *runs* (root status), unlike the span-level error_count on the
        # trace rollups, which counts failed spans within one trace.
        failed_run_count=int(row["failed_run_count"]),
        **{field: _token_rollup(row, field) for field in _TOKEN_ROLLUP_FIELDS},
        cost_usd=CostRollup(
            sum=float_or_none(row.get("cost_usd_sum")),
            mean=_finite_or_none(row.get("cost_usd_mean")),
            **_named_quantiles(row, "cost_usd", _SUMMED_ROLLUP_QUANTILES),
        ),
        latency_ms=LatencyRollup(
            mean=_finite_or_none(row.get("latency_ms_mean")),
            **_named_quantiles(row, "latency_ms", _LATENCY_QUANTILES),
        ),
    )


def _token_rollup(row: dict[str, Any], field: str) -> TokenRollup:
    return TokenRollup(
        sum=int_or_none(row.get(f"{field}_sum")),
        mean=_finite_or_none(row.get(f"{field}_mean")),
        **_named_quantiles(row, field, _SUMMED_ROLLUP_QUANTILES),
    )


def _named_quantiles(row: dict[str, Any], field: str, quantiles: Sequence[float]) -> dict[str, float | None]:
    values: Sequence[Any] = row.get(f"{field}_quantiles") or ()
    return {
        f"p{round(quantile * 100)}": _finite_or_none(values[index]) if index < len(values) else None
        for index, quantile in enumerate(quantiles)
    }


def _finite_or_none(value: Any) -> float | None:
    """Drop the NaN ClickHouse yields for an aggregate over an empty bucket."""

    number = float_or_none(value)
    return None if number is None or not isfinite(number) else number


def _duration_ms(started_at: datetime, ended_at: datetime | None) -> float | None:
    if ended_at is None:
        return None
    return (ended_at - started_at).total_seconds() * 1000


def _none_if_zero_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if value == _ZERO_DATETIME or value.timestamp() == 0:
        return None
    return value


def _string_list_or_none(value: Any) -> list[str] | None:
    if value is None:
        return None
    values = [str(item) for item in value if str(item)]
    return values
