# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse implementation of Intake session reads."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.session import SessionRepository
from nhx.intake.spans.domain import IntakeSession
from nhx.intake.spans.span_rollups import summed_metric_columns, usage_sql
from nhx.intake.spans.storage import float_or_none, int_or_none, normalize_span_status


class ClickHouseSessionRepository(SessionRepository):
    def __init__(self, executor: ClickHouseExecutor) -> None:
        self._executor = executor

    async def get_session(self, *, workspace: str, session_id: str) -> IntakeSession | None:
        query = _session_detail_query(self._executor.table(ClickHouseTable.SPANS)).bind(
            workspace=workspace,
            session_id=session_id,
        )
        rows = await self._executor.fetch_all(query)
        return _row_to_session(rows[0]) if rows else None


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
