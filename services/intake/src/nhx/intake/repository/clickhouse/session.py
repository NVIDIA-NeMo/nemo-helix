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
from nhx.intake.spans.span_rollups import metric_aggregate_columns
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
    """Build a primary-key-pruned aggregate over the current rows of one session."""

    source_alias = "session_spans"
    metric_columns, parameters = metric_aggregate_columns(source_alias)
    statement = f"""
        SELECT
            %(session_id)s AS id,
            any({source_alias}.workspace) AS workspace,
            min({source_alias}.start_time) AS started_at,
            if(
                countIf({source_alias}.end_time = toDateTime64(0, 6)) > 0,
                NULL,
                max({source_alias}.end_time)
            ) AS ended_at,
            multiIf(
                countIf({source_alias}.status = 'error') > 0, 'error',
                countIf({source_alias}.status = 'cancelled') > 0, 'cancelled',
                countIf({source_alias}.status = 'unknown') > 0, 'unknown',
                'success'
            ) AS status,
            {metric_columns},
            uniqExact({source_alias}.source_format, {source_alias}.trace_id) AS trace_count,
            count() AS span_count
        FROM {table} AS {source_alias} FINAL
        PREWHERE
            {source_alias}.workspace = %(workspace)s
            AND {source_alias}.session_id = %(session_id)s
        WHERE {source_alias}.is_deleted = 0
        HAVING span_count > 0
    """
    return ClickHouseQuery(
        name="sessions.get",
        statement=statement,
        parameters=parameters,
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
