# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse implementation of Intake sessions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.span import SPAN_INSERT_COLUMNS
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.session import SessionRepository, TraceIndexCollisionError
from nhx.intake.spans.domain import IntakeSession
from nhx.intake.spans.span_attribute_catalog import SpanAttributeField, bag_keys, spec_for_field
from nhx.intake.spans.span_rollups import summed_metric_columns, usage_sql
from nhx.intake.spans.storage import float_or_none, int_or_none, normalize_span_status


class ClickHouseSessionRepository(SessionRepository):
    def __init__(self, executor: ClickHouseExecutor) -> None:
        self._executor = executor

    async def rename_evaluation_references(self, *, workspace: str, original_name: str, deleted_name: str) -> None:
        if original_name == deleted_name:
            return
        keys = bag_keys(spec_for_field(SpanAttributeField.EVALUATION_NAME))
        key_parameters = {f"evaluation_key_{index}": key for index, key in enumerate(keys)}
        key_list = ", ".join(f"%({key})s" for key in key_parameters)
        values = ", ".join(f"attributes_string[%({key})s]" for key in key_parameters)
        # Mutate every retained version, including tombstones. Inserting live
        # replacements could revive a deleted span or replace a colliding root.
        # Mutations do not propagate through the insert-time materialized view,
        # so update its trace index explicitly and wait for both mutations.
        await self._executor.execute(
            ClickHouseQuery(
                name="sessions.rename_evaluation_spans",
                statement=f"""
                    ALTER TABLE {self._executor.table(ClickHouseTable.SPANS)}
                    UPDATE attributes_string = mapApply(
                        (key, value) -> (key, if(
                            has([{key_list}], key) AND value = %(original_name)s,
                            %(deleted_name)s, value
                        )), attributes_string
                    )
                    WHERE workspace = %(workspace)s AND has([{values}], %(original_name)s)
                    SETTINGS mutations_sync = 2
                """,
                parameters={
                    "workspace": workspace,
                    "original_name": original_name,
                    "deleted_name": deleted_name,
                    **key_parameters,
                },
            )
        )
        await self._executor.execute(
            ClickHouseQuery(
                name="sessions.rename_evaluation_traces",
                statement=f"""
                    ALTER TABLE {self._executor.table(ClickHouseTable.TRACE_INDEX)}
                    UPDATE evaluation_name = %(deleted_name)s
                    WHERE workspace = %(workspace)s AND evaluation_name = %(original_name)s
                    SETTINGS mutations_sync = 2
                """,
                parameters={"workspace": workspace, "original_name": original_name, "deleted_name": deleted_name},
            )
        )

    async def list_evaluation_session_ids(
        self, *, workspace: str, evaluation_name: str, after: str | None, limit: int
    ) -> list[str]:
        keys = bag_keys(spec_for_field(SpanAttributeField.EVALUATION_NAME))
        values = ", ".join(f"attributes_string[%(evaluation_key_{index})s]" for index in range(len(keys)))
        rows = await self._executor.fetch_all(
            ClickHouseQuery(
                name="sessions.for_evaluation",
                statement=f"""
                    SELECT DISTINCT session_id
                    FROM {self._executor.table(ClickHouseTable.SPANS)} FINAL
                    PREWHERE workspace = %(workspace)s
                    WHERE is_deleted = 0
                        AND (%(first_page)s OR session_id > %(after)s)
                        AND has([{values}], %(evaluation_name)s)
                    ORDER BY session_id
                    LIMIT %(limit)s
                """,
                parameters={
                    "workspace": workspace,
                    "evaluation_name": evaluation_name,
                    "first_page": after is None,
                    "after": after or "",
                    "limit": limit,
                    **{f"evaluation_key_{index}": key for index, key in enumerate(keys)},
                },
            )
        )
        return [str(row["session_id"]) for row in rows]

    async def get_session(self, *, workspace: str, session_id: str) -> IntakeSession | None:
        query = _session_detail_query(self._executor.table(ClickHouseTable.SPANS)).bind(
            workspace=workspace,
            session_id=session_id,
        )
        rows = await self._executor.fetch_all(query)
        return _row_to_session(rows[0]) if rows else None

    async def get_evaluation_names(self, *, workspace: str, session_id: str) -> list[str]:
        # Inspect all current spans, including child-only and historical attribute
        # spellings, so deleting one evaluation never bypasses another association.
        keys = bag_keys(spec_for_field(SpanAttributeField.EVALUATION_NAME))
        values = ", ".join(f"attributes_string[%(evaluation_key_{index})s]" for index in range(len(keys)))
        rows = await self._executor.fetch_all(
            ClickHouseQuery(
                name="sessions.evaluation_names",
                statement=f"""
                    SELECT DISTINCT arrayJoin([{values}]) AS evaluation_name
                    FROM {self._executor.table(ClickHouseTable.SPANS)} FINAL
                    PREWHERE workspace = %(workspace)s AND session_id = %(session_id)s
                    WHERE is_deleted = 0 AND notEmpty(evaluation_name)
                """,
                parameters={
                    "workspace": workspace,
                    "session_id": session_id,
                    **{f"evaluation_key_{index}": key for index, key in enumerate(keys)},
                },
            )
        )
        return [str(row["evaluation_name"]) for row in rows]

    async def delete_session(self, *, workspace: str, session_id: str) -> None:
        collision_query = _session_trace_index_collision_query(
            self._executor.table(ClickHouseTable.SPANS),
            self._executor.table(ClickHouseTable.TRACE_INDEX),
        ).bind(workspace=workspace, session_id=session_id)
        if await self._executor.fetch_scalar(collision_query):
            raise TraceIndexCollisionError(workspace, session_id)
        # Keep the full replacement key and root metadata so the trace-index MV
        # receives matching tombstones. Copy inside ClickHouse to bound API memory.
        table = self._executor.table(ClickHouseTable.SPANS)
        columns = ", ".join(SPAN_INSERT_COLUMNS)
        selection = ", ".join(
            "greatest(now64(6), event_ts + INTERVAL 1 MICROSECOND)"
            if column == "event_ts"
            else "toUInt8(1)"
            if column == "is_deleted"
            else column
            for column in SPAN_INSERT_COLUMNS
        )
        await self._executor.execute(
            ClickHouseQuery(
                name="sessions.delete",
                statement=f"""
                    INSERT INTO {table} ({columns})
                    SELECT {selection}
                    FROM {table} FINAL
                    PREWHERE workspace = %(workspace)s AND session_id = %(session_id)s
                    WHERE is_deleted = 0
                """,
                parameters={"workspace": workspace, "session_id": session_id},
            )
        )


def _session_trace_index_collision_query(spans_table: str, trace_index_table: str) -> ClickHouseQuery:
    """Check owners of the exact index keys that root-span tombstones would replace.

    The index key omits session and source. Current spans retain those identities
    even after index merges, while raw index history can retain an owner whose
    root span no longer exists. Keep either kind of evidence and reject ambiguity.
    """
    return ClickHouseQuery(
        name="sessions.trace_index_collision",
        statement=f"""
            WITH target_roots AS (
                SELECT start_time, trace_id, external_span_id
                FROM {spans_table} FINAL
                PREWHERE workspace = %(workspace)s AND session_id = %(session_id)s
                WHERE is_deleted = 0 AND external_parent_span_id = ''
            ), owners AS (
                SELECT
                    start_time AS root_started_at,
                    trace_id,
                    external_span_id AS root_span_id,
                    session_id,
                    source_format
                FROM {spans_table} FINAL
                PREWHERE workspace = %(workspace)s
                    AND (start_time, trace_id, external_span_id) IN (SELECT * FROM target_roots)
                WHERE is_deleted = 0 AND external_parent_span_id = ''

                UNION ALL

                SELECT root_started_at, trace_id, root_span_id, session_id, source_format
                FROM {trace_index_table}
                PREWHERE workspace = %(workspace)s
                    AND (root_started_at, trace_id, root_span_id) IN (SELECT * FROM target_roots)
            )
            SELECT 1
            FROM owners
            GROUP BY root_started_at, trace_id, root_span_id
            HAVING uniqExact(session_id, source_format) > 1
            LIMIT 1
        """,
    )


def _session_detail_query(table: str) -> ClickHouseQuery:
    """Build a primary-key-pruned aggregate over the current rows of one session.

    Usage is totaled per trace first, then added across the session, so a trace that reports model calls
    does not also add a root-only trace's fallback, or the other way around.
    """

    usage = usage_sql(scaled=False)
    statement = f"""
        SELECT
            %(session_id)s AS id,
            any(workspace) AS workspace,
            min(started_at) AS started_at,
            if(sum(open_spans) > 0, NULL, max(ended_at)) AS ended_at,
            multiIf(
                sum(error_spans) > 0, 'error',
                sum(cancelled_spans) > 0, 'cancelled',
                sum(unknown_spans) > 0, 'unknown',
                'success'
            ) AS status,
            {summed_metric_columns("trace_usage")},
            uniqExact(source_format, trace_id) AS trace_count,
            sum(trace_usage.span_count) AS span_count
        FROM (
            SELECT
                source_format,
                trace_id,
                any(workspace) AS workspace,
                min(start_time) AS started_at,
                max(end_time) AS ended_at,
                countIf(end_time = toDateTime64(0, 6)) AS open_spans,
                countIf(status = 'error') AS error_spans,
                countIf(status = 'cancelled') AS cancelled_spans,
                countIf(status = 'unknown') AS unknown_spans,
                count() AS span_count,
                {usage.trace_columns}
            FROM (
                SELECT
                    session_spans.workspace AS workspace,
                    session_spans.source_format AS source_format,
                    session_spans.trace_id AS trace_id,
                    session_spans.start_time AS start_time,
                    session_spans.end_time AS end_time,
                    session_spans.status AS status,
                    {usage.span_columns}
                FROM {table} AS session_spans FINAL
                PREWHERE
                    session_spans.workspace = %(workspace)s
                    AND session_spans.session_id = %(session_id)s
                WHERE session_spans.is_deleted = 0
            ) AS span_rows
            GROUP BY source_format, trace_id
        ) AS trace_usage
        HAVING span_count > 0
    """
    return ClickHouseQuery(
        name="sessions.get",
        statement=statement,
        parameters=usage.parameters,
    )


def _row_to_session(row: dict[str, Any]) -> IntakeSession:
    ended_at = row.get("ended_at")
    return IntakeSession(
        id=row["id"],
        workspace=row["workspace"],
        started_at=row["started_at"],
        ended_at=ended_at,
        duration_ms=_duration_ms(row["started_at"], ended_at),
        status=normalize_span_status(row.get("status")),
        input_tokens=int_or_none(row.get("input_tokens")),
        output_tokens=int_or_none(row.get("output_tokens")),
        cached_tokens=int_or_none(row.get("cached_tokens")),
        total_tokens=int_or_none(row.get("total_tokens")),
        cost_usd=float_or_none(row.get("cost_usd")),
        cost_input_usd=float_or_none(row.get("cost_input_usd")),
        cost_output_usd=float_or_none(row.get("cost_output_usd")),
        trace_count=int(row["trace_count"]),
        span_count=int(row["span_count"]),
    )


def _duration_ms(started_at: datetime, ended_at: datetime | None) -> float | None:
    if ended_at is None:
        return None
    return (ended_at - started_at).total_seconds() * 1000
