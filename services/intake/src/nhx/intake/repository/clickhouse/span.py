# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse implementation of Intake span storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from nhx.common.api.common import PaginatedResult
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseInsert, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.span import SpanRepository
from nhx.intake.spans.domain import NEMO_STEP_ID_ATTRIBUTE, IntakeResponseMode, IntakeSpan, SpanGroup, SpanListFilter
from nhx.intake.spans.span_attribute_catalog import where_clause
from nhx.intake.spans.storage import (
    dict_to_row,
    make_pagination,
    normalize_span_kind,
    normalize_span_status,
    text_query_parameters,
    text_select_for_mode,
)

SPAN_COLUMNS = [
    "workspace",
    "session_id",
    "trace_id",
    "id",
    "source_format",
    "external_span_id",
    "external_parent_span_id",
    "kind",
    "name",
    "status",
    "start_time",
    "end_time",
    "attributes_string",
    "attributes_number",
    "attributes_bool",
    "input",
    "output",
    "event_ts",
    "is_deleted",
]
SPAN_INSERT_COLUMNS = [column for column in SPAN_COLUMNS if column != "id"]

SPAN_SORT_COLUMNS = {
    "started_at": "start_time",
}

SPAN_GROUP_COLUMN_FIELDS = {
    "trace_id": "trace_id",
    "session_id": "session_id",
}

# Selected by the grouped query itself, so the outer ORDER BY can name them directly.
SPAN_GROUP_SORT_FIELDS = frozenset({"span_count", "started_at"})

_ZERO_DATETIME = datetime.fromtimestamp(0, tz=timezone.utc)


@dataclass(frozen=True)
class _GroupExpression:
    select_sql: str
    group_sql: str
    required_sql: str


@dataclass(frozen=True)
class _SpanStorageRef:
    """The complete sorting key needed to read one current span efficiently."""

    session_id: str
    start_time_us: int
    id: int

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> _SpanStorageRef:
        return cls(
            session_id=str(row["session_id"]),
            start_time_us=int(row["start_time_us"]),
            id=int(row["id"]),
        )


class ClickHouseSpanRepository(SpanRepository):
    def __init__(self, executor: ClickHouseExecutor) -> None:
        self._executor = executor

    async def save_spans(self, spans: list[IntakeSpan]) -> None:
        rows = [dict_to_row(_span_to_row(span), SPAN_INSERT_COLUMNS) for span in spans]
        await self._executor.insert(
            ClickHouseInsert(
                name="spans.save",
                table=ClickHouseTable.SPANS,
                rows=rows,
                column_names=SPAN_INSERT_COLUMNS,
            )
        )

    async def list_spans(
        self,
        *,
        filters: SpanListFilter,
        page: int,
        page_size: int,
        sort: str,
        mode: IntakeResponseMode,
    ) -> PaginatedResult[IntakeSpan]:
        where_sql, parameters = _span_where(filters)
        table = self._executor.table(ClickHouseTable.SPANS)
        total_results = int(
            await self._executor.fetch_scalar(
                ClickHouseQuery(
                    name="spans.list.count",
                    statement=f"SELECT count() FROM {table} FINAL WHERE {where_sql}",
                    parameters=parameters,
                )
            )
            or 0
        )
        offset = (page - 1) * page_size
        columns_sql = _span_select_columns(mode=mode)
        rows_parameters: dict[str, Any] = {
            **parameters,
            **text_query_parameters(mode),
            "limit": page_size,
            "offset": offset,
        }
        rows = await self._executor.fetch_all(
            ClickHouseQuery(
                name="spans.list.rows",
                statement=f"""
                SELECT {columns_sql}
                FROM {table} FINAL
                WHERE {where_sql}
                ORDER BY {_order_by(sort)}
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                parameters=rows_parameters,
            )
        )
        spans = _rows_to_spans(rows)
        return PaginatedResult(
            data=spans,
            pagination=make_pagination(
                page=page, page_size=page_size, current_page_size=len(spans), total_results=total_results
            ),
        )

    async def list_span_groups(
        self,
        *,
        filters: SpanListFilter,
        group_by: list[str],
        page: int,
        page_size: int,
        sort: str,
    ) -> PaginatedResult[SpanGroup]:
        group_expressions = _group_expressions(group_by)
        where_sql, parameters = _span_where(filters)

        required_sql = " AND ".join(f"({expression.required_sql})" for expression in group_expressions)
        if required_sql:
            where_sql = f"{where_sql} AND {required_sql}"

        table = self._executor.table(ClickHouseTable.SPANS)
        select_sql = ", ".join(expression.select_sql for expression in group_expressions)
        group_sql = ", ".join(expression.group_sql for expression in group_expressions)
        grouped_sql = f"""
            SELECT {select_sql}, count() AS span_count, min(start_time) AS started_at
            FROM {table} FINAL
            WHERE {where_sql}
            GROUP BY {group_sql}
        """

        total_results = int(
            await self._executor.fetch_scalar(
                ClickHouseQuery(
                    name="spans.list_groups.count",
                    statement=f"SELECT count() FROM ({grouped_sql}) AS span_groups",
                    parameters=parameters,
                )
            )
            or 0
        )
        offset = (page - 1) * page_size
        rows = await self._executor.fetch_all(
            ClickHouseQuery(
                name="spans.list_groups.rows",
                statement=f"""
                SELECT *
                FROM ({grouped_sql}) AS span_groups
                ORDER BY {_group_order_by(sort, group_by)}
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                parameters={**parameters, "limit": page_size, "offset": offset},
            )
        )
        groups = [_row_to_group(row, group_by=group_by) for row in rows]
        return PaginatedResult(
            data=groups,
            pagination=make_pagination(
                page=page, page_size=page_size, current_page_size=len(groups), total_results=total_results
            ),
        )

    async def get_span(self, *, workspace: str, span_id: str) -> IntakeSpan | None:
        table = self._executor.table(ClickHouseTable.SPANS)
        ref_rows = await self._executor.fetch_all(
            _span_storage_ref_query(table).bind(workspace=workspace, span_id=span_id)
        )
        if not ref_rows:
            return None

        ref = _SpanStorageRef.from_row(ref_rows[0])
        rows = await self._executor.fetch_all(
            _span_hydration_query(table).bind(
                workspace=workspace,
                session_id=ref.session_id,
                start_time_us=ref.start_time_us,
                id=ref.id,
            )
        )
        if not rows:
            return None
        return _row_to_span(rows[0])


def _span_storage_ref_query(table: str) -> ClickHouseQuery:
    """Resolve an external span ID to its current sorting key using only narrow columns.

    ``external_span_id`` has a bloom index but is not part of the sorting key. Applying
    ``FINAL`` directly makes ClickHouse expand the bloom-filter ranges across overlapping
    parts before reading every payload column. Resolving the current key with ``argMax``
    avoids that expansion; the full row can then be read by its exact sorting key.
    """

    return ClickHouseQuery(
        name="spans.get.resolve",
        statement=f"""
        SELECT
            id,
            argMax(session_id, (event_ts, is_deleted)) AS session_id,
            toUnixTimestamp64Micro(argMax(start_time, (event_ts, is_deleted))) AS start_time_us,
            argMax(is_deleted, (event_ts, is_deleted)) AS current_is_deleted
        FROM {table}
        PREWHERE
            workspace = %(workspace)s
            AND external_span_id = %(span_id)s
        GROUP BY workspace, source_format, trace_id, external_span_id, id
        HAVING current_is_deleted = 0
        LIMIT 1
        """,
    )


def _span_hydration_query(table: str) -> ClickHouseQuery:
    columns_sql = ", ".join(SPAN_COLUMNS)
    return ClickHouseQuery(
        name="spans.get.hydrate",
        statement=f"""
        SELECT {columns_sql}
        FROM {table} FINAL
        PREWHERE
            workspace = %(workspace)s
            AND session_id = %(session_id)s
            AND start_time = fromUnixTimestamp64Micro(%(start_time_us)s)
            AND id = %(id)s
        WHERE is_deleted = 0
        LIMIT 1
        """,
    )


def _span_select_columns(*, mode: IntakeResponseMode) -> str:
    columns = []
    for column in SPAN_COLUMNS:
        if column not in {"input", "output"}:
            columns.append(column)
        else:
            columns.append(text_select_for_mode(column, alias=column, mode=mode))
    return ", ".join(columns)


def _span_where(filters: SpanListFilter) -> tuple[str, dict[str, Any]]:
    clauses = ["workspace = %(workspace)s", "is_deleted = 0"]
    parameters: dict[str, Any] = {"workspace": filters.workspace}
    if filters.session_id is not None:
        clauses.append("session_id = %(session_id)s")
        parameters["session_id"] = filters.session_id
    if filters.trace_id is not None:
        clauses.append("trace_id = %(trace_id)s")
        parameters["trace_id"] = filters.trace_id
    if filters.external_parent_span_id is not None:
        clauses.append("external_parent_span_id = %(external_parent_span_id)s")
        parameters["external_parent_span_id"] = filters.external_parent_span_id
    if filters.source_format is not None:
        clauses.append("source_format = %(source_format)s")
        parameters["source_format"] = filters.source_format
    if filters.kind is not None:
        clauses.append("kind = %(kind)s")
        parameters["kind"] = filters.kind.value
    if filters.status is not None:
        clauses.append("status = %(status)s")
        parameters["status"] = filters.status.value
    if filters.started_at_gte is not None:
        clauses.append("start_time >= %(started_at_gte)s")
        parameters["started_at_gte"] = filters.started_at_gte
    if filters.started_at_lte is not None:
        clauses.append("start_time <= %(started_at_lte)s")
        parameters["started_at_lte"] = filters.started_at_lte
    for index, attribute_filter in enumerate(filters.attribute_filters):
        clause, clause_parameters = where_clause(
            attribute_filter.field,
            attribute_filter.operator,
            attribute_filter.value,
            param_prefix=f"attr_{index}",
        )
        clauses.append(f"({clause})")
        parameters.update(clause_parameters)
    return " AND ".join(clauses), parameters


def _group_expressions(group_by: list[str]) -> list[_GroupExpression]:
    if not group_by:
        raise ValueError("At least one span group-by field is required.")
    return [_group_expression(field) for field in group_by]


def _group_expression(field: str) -> _GroupExpression:
    if field in SPAN_GROUP_COLUMN_FIELDS:
        column = SPAN_GROUP_COLUMN_FIELDS[field]
        return _GroupExpression(
            select_sql=f"{column} AS {field}",
            group_sql=column,
            required_sql=f"{column} != ''",
        )
    raise ValueError(f"Unsupported span group-by field: {field}")


def _order_by(sort: str) -> str:
    direction = "DESC" if sort.startswith("-") else "ASC"
    field = sort.removeprefix("-")
    column = SPAN_SORT_COLUMNS.get(field)
    if column is None:
        raise ValueError(f"Unsupported span sort field: {field}")
    nemo_step_id = f"nullIf(attributes_number['{NEMO_STEP_ID_ATTRIBUTE}'], 0)"
    return f"{column} {direction}, {nemo_step_id} {direction} NULLS LAST, id ASC"


def _group_order_by(sort: str, group_by: list[str]) -> str:
    direction = "DESC" if sort.startswith("-") else "ASC"
    field = sort.removeprefix("-")
    if field not in SPAN_GROUP_SORT_FIELDS:
        raise ValueError(f"Unsupported span group sort field: {field}")
    tie_break = ", ".join(f"{key} ASC" for key in group_by)
    if tie_break:
        return f"{field} {direction}, {tie_break}"
    return f"{field} {direction}"


def _span_to_row(span: IntakeSpan) -> dict[str, Any]:
    return {
        "workspace": span.workspace,
        "session_id": span.session_id,
        "trace_id": span.trace_id,
        "source_format": span.source_format,
        "external_span_id": span.external_span_id,
        "external_parent_span_id": span.external_parent_span_id,
        "kind": span.kind.value,
        "name": span.name,
        "status": span.status.value,
        "start_time": span.start_time,
        "end_time": span.end_time or _ZERO_DATETIME,
        "attributes_string": span.attributes_string,
        "attributes_number": {key: float(value) for key, value in span.attributes_number.items()},
        "attributes_bool": span.attributes_bool,
        "input": span.input,
        "output": span.output,
        "event_ts": span.event_ts,
        "is_deleted": span.is_deleted,
    }


def _rows_to_spans(rows: list[dict[str, Any]]) -> list[IntakeSpan]:
    # Parent linkage is reconstructed only within the returned result set. Partial
    # views still expose external_parent_span_id for clients that need full trace assembly.
    id_by_external = {
        (row["workspace"], row["source_format"], row["trace_id"], row["external_span_id"]): int(row["id"])
        for row in rows
    }
    return [_row_to_span(row, id_by_external=id_by_external) for row in rows]


def _row_to_group(row: dict[str, Any], *, group_by: list[str]) -> SpanGroup:
    return SpanGroup(
        group={field: str(row[field]) for field in group_by},
        span_count=int(row["span_count"]),
        started_at=row["started_at"],
    )


def _row_to_span(
    row: dict[str, Any],
    *,
    id_by_external: dict[tuple[str, str, str, str], int] | None = None,
) -> IntakeSpan:
    parent_id = None
    external_parent_span_id = row.get("external_parent_span_id") or ""
    if external_parent_span_id and id_by_external is not None:
        parent_id = id_by_external.get(
            (row["workspace"], row["source_format"], row["trace_id"], external_parent_span_id)
        )
    return IntakeSpan(
        workspace=row["workspace"],
        session_id=row["session_id"],
        trace_id=row["trace_id"],
        id=int(row["id"]),
        source_format=row["source_format"],
        external_span_id=row["external_span_id"],
        external_parent_span_id=external_parent_span_id,
        parent_id=parent_id,
        kind=normalize_span_kind(row.get("kind")),
        name=row.get("name") or "",
        status=normalize_span_status(row.get("status")),
        start_time=row["start_time"],
        end_time=_none_if_zero_datetime(row.get("end_time")),
        attributes_string=dict(row.get("attributes_string") or {}),
        attributes_number={key: float(value) for key, value in dict(row.get("attributes_number") or {}).items()},
        attributes_bool={key: bool(value) for key, value in dict(row.get("attributes_bool") or {}).items()},
        input=row.get("input") or "",
        output=row.get("output") or "",
        event_ts=row["event_ts"],
        is_deleted=int(row.get("is_deleted") or 0),
    )


def _none_if_zero_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if value == _ZERO_DATETIME or (value.tzinfo is None and value == _ZERO_DATETIME.replace(tzinfo=None)):
        return None
    return value
