# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session repository tests."""

from datetime import datetime, timedelta, timezone

import pytest
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.session import ClickHouseSessionRepository, _session_detail_query
from nhx.intake.repository.clickhouse.tables import ClickHouseTable


class _Executor(ClickHouseExecutor):
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows or []
        self.queries: list[ClickHouseQuery] = []

    def table(self, table: ClickHouseTable) -> str:
        assert table is ClickHouseTable.SPANS
        return "spans"

    async def fetch_all(self, query: ClickHouseQuery) -> list[dict[str, object]]:
        self.queries.append(query)
        return self.rows


def _repository(executor: _Executor) -> ClickHouseSessionRepository:
    return ClickHouseSessionRepository(executor)


def test_session_query_is_primary_key_pruned_and_payload_free() -> None:
    query = _session_detail_query("spans")

    assert query.name == "sessions.get"
    assert "FROM spans AS session_spans FINAL" in query.statement
    assert "PREWHERE" in query.statement
    assert "session_spans.workspace = %(workspace)s" in query.statement
    assert "session_spans.session_id = %(session_id)s" in query.statement
    assert "trace_index" not in query.statement
    assert "JOIN" not in query.statement
    assert "session_spans.input" not in query.statement
    assert "session_spans.output" not in query.statement
    assert "uniqExact(session_spans.source_format, session_spans.trace_id) AS trace_count" in query.statement
    assert "count() AS span_count" in query.statement
    assert query.parameters["input_tokens_key"] == "llm.token_count.prompt"
    assert query.parameters["cost_usd_key"] == "cost.total"


@pytest.mark.asyncio
async def test_get_session_maps_aggregate_row() -> None:
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ended_at = started_at + timedelta(seconds=8)
    values: dict[str, object | None] = {
        "id": "session-a",
        "workspace": "workspace-a",
        "started_at": started_at,
        "ended_at": ended_at,
        "status": "error",
        "input_tokens": 420,
        "output_tokens": 310,
        "cached_tokens": 128,
        "total_tokens": 858,
        "cost_usd": 0.0061,
        "cost_input_usd": 0.0024,
        "cost_output_usd": 0.0037,
        "trace_count": 2,
        "span_count": 5,
    }
    executor = _Executor([values])

    session = await _repository(executor).get_session(workspace="workspace-a", session_id="session-a")

    assert session is not None
    assert session.id == "session-a"
    assert session.workspace == "workspace-a"
    assert session.started_at == started_at
    assert session.ended_at == ended_at
    assert session.duration_ms == 8000
    assert session.status.value == "error"
    assert session.trace_count == 2
    assert session.span_count == 5
    assert session.total_tokens == 858
    assert session.cost_usd == 0.0061
    assert executor.queries[0].parameters["workspace"] == "workspace-a"
    assert executor.queries[0].parameters["session_id"] == "session-a"


@pytest.mark.asyncio
async def test_get_session_returns_none_when_no_current_spans_exist() -> None:
    workspace = "workspace' OR 1 = 1 --"
    session_id = "session'); DROP TABLE spans; --"
    executor = _Executor()

    session = await _repository(executor).get_session(workspace=workspace, session_id=session_id)

    assert session is None
    assert workspace not in executor.queries[0].statement
    assert session_id not in executor.queries[0].statement
    assert executor.queries[0].parameters["workspace"] == workspace
    assert executor.queries[0].parameters["session_id"] == session_id
