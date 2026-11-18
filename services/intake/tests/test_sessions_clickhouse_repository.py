# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session repository tests."""

from datetime import datetime, timedelta, timezone

import pytest
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.session import (
    ClickHouseSessionRepository,
    _session_detail_query,
    _session_trace_index_collision_query,
)
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.session import TraceIndexCollisionError


class _Executor(ClickHouseExecutor):
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows or []
        self.queries: list[ClickHouseQuery] = []

    def table(self, table: ClickHouseTable) -> str:
        assert table in (ClickHouseTable.SPANS, ClickHouseTable.TRACE_INDEX)
        return table.value

    async def fetch_all(self, query: ClickHouseQuery) -> list[dict[str, object]]:
        self.queries.append(query)
        return self.rows

    async def execute(self, query: ClickHouseQuery) -> None:
        self.queries.append(query)


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
    assert "uniqExact(source_format, trace_id) AS trace_count" in query.statement
    assert "count() AS span_count" in query.statement
    assert "sum(trace_usage.span_count) AS span_count" in query.statement
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


@pytest.mark.asyncio
async def test_delete_session_writes_scoped_tombstones_without_loading_payloads() -> None:
    executor = _Executor()
    workspace = "workspace' OR 1 = 1 --"
    session_id = "session'); DROP TABLE spans; --"

    await _repository(executor).delete_session(workspace=workspace, session_id=session_id)

    collision_query, query = executor.queries
    assert collision_query.name == "sessions.trace_index_collision"
    assert collision_query.parameters == {"workspace": workspace, "session_id": session_id}
    assert query.name == "sessions.delete"
    assert "INSERT INTO spans" in query.statement
    assert "FROM spans FINAL" in query.statement
    assert "PREWHERE workspace = %(workspace)s AND session_id = %(session_id)s" in query.statement
    assert "WHERE is_deleted = 0" in query.statement
    assert "greatest(now64(6), event_ts + INTERVAL 1 MICROSECOND)" in query.statement
    assert "toUInt8(1)" in query.statement
    assert workspace not in query.statement
    assert session_id not in query.statement
    assert query.parameters == {"workspace": workspace, "session_id": session_id}


@pytest.mark.asyncio
async def test_delete_session_rejects_collision_before_any_write() -> None:
    executor = _Executor([{"1": 1}])

    with pytest.raises(TraceIndexCollisionError, match="conflicting trace-index ownership"):
        await _repository(executor).delete_session(workspace="workspace-a", session_id="session-a")

    assert [query.name for query in executor.queries] == ["sessions.trace_index_collision"]


@pytest.mark.asyncio
async def test_collision_lookup_failure_prevents_deletion() -> None:
    class _FailingExecutor(_Executor):
        async def fetch_all(self, query: ClickHouseQuery) -> list[dict[str, object]]:
            self.queries.append(query)
            raise RuntimeError("telemetry store unavailable")

    executor = _FailingExecutor()
    with pytest.raises(RuntimeError, match="telemetry store unavailable"):
        await _repository(executor).delete_session(workspace="workspace-a", session_id="session-a")

    assert [query.name for query in executor.queries] == ["sessions.trace_index_collision"]


def test_collision_query_checks_exact_keys_in_current_spans_and_raw_index_history() -> None:
    query = _session_trace_index_collision_query("spans", "trace_index")

    assert "FROM spans FINAL" in query.statement
    assert "session_id = %(session_id)s" in query.statement
    assert "external_parent_span_id = ''" in query.statement
    assert "(start_time, trace_id, external_span_id) IN" in query.statement
    assert "(root_started_at, trace_id, root_span_id) IN" in query.statement
    assert "FROM trace_index\n" in query.statement
    assert "FROM trace_index FINAL" not in query.statement
    assert "HAVING uniqExact(session_id, source_format) > 1" in query.statement
    assert query.statement.count("PREWHERE workspace = %(workspace)s") == 3
    assert "input" not in query.statement
    assert "output" not in query.statement


@pytest.mark.asyncio
@pytest.mark.parametrize("linked", [True, False])
async def test_evaluation_names_checks_current_scoped_spans_and_legacy_keys(linked: bool) -> None:
    executor = _Executor([{"evaluation_name": "evaluation-a"}, {"evaluation_name": "evaluation-b"}] if linked else [])
    workspace = "workspace' OR 1 = 1 --"
    session_id = "session'); DROP TABLE spans; --"

    names = await _repository(executor).get_evaluation_names(workspace=workspace, session_id=session_id)
    assert names == (["evaluation-a", "evaluation-b"] if linked else [])

    (query,) = executor.queries
    assert query.name == "sessions.evaluation_names"
    assert "FROM spans FINAL" in query.statement
    assert "PREWHERE workspace = %(workspace)s AND session_id = %(session_id)s" in query.statement
    assert "WHERE is_deleted = 0" in query.statement
    assert "SELECT DISTINCT arrayJoin" in query.statement
    assert "notEmpty(evaluation_name)" in query.statement
    assert "external_parent_span_id" not in query.statement
    assert "nemo.evaluation.name" in query.parameters.values()
    assert "nemo.experiment.id" in query.parameters.values()
    assert workspace not in query.statement
    assert session_id not in query.statement
    assert query.parameters["workspace"] == workspace
    assert query.parameters["session_id"] == session_id
