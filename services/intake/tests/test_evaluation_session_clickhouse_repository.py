# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation session ClickHouse repository tests."""

from datetime import datetime, timezone
from typing import Any

import pytest
from nhx.intake.repository.clickhouse.evaluation_session import (
    _MAX_METRIC_SORT_SESSIONS,
    _SORT_EXPR_PAGE,
    ClickHouseEvaluationSessionRepository,
    _build_order_by,
)
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.evaluation_session import MetricSortTooLargeError
from nhx.intake.spans.domain import SpanStatus


class _Executor(ClickHouseExecutor):
    def __init__(self, query_results: list[list[dict[str, Any]]]) -> None:
        self.queries: list[ClickHouseQuery] = []
        self.query_results = query_results
        self.tables: list[ClickHouseTable] = []

    def table(self, table: ClickHouseTable) -> str:
        self.tables.append(table)
        return table.value

    async def fetch_all(self, query: ClickHouseQuery) -> list[dict[str, Any]]:
        self.queries.append(query)
        return self.query_results.pop(0)


def _repository(executor: _Executor) -> ClickHouseEvaluationSessionRepository:
    return ClickHouseEvaluationSessionRepository(executor)


def _session_record(session_id: str) -> dict[str, Any]:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {
        "workspace": "default",
        "evaluation_name": "evaluation-a",
        "session_id": session_id,
        "test_case_name": "case-a",
        "trace_id": f"trace-{session_id}",
        "root_span_id": f"root-{session_id}",
        "start_time": now,
        "end_time": now,
        "latency_ms": 125.0,
        "root_span_status": "success",
        "input": "input",
        "output": "output",
        "input_tokens": 10,
        "output_tokens": 5,
        "cached_tokens": 2,
        "cost_total_usd": 0.01,
        "evaluator_scores": {"reward": 0.9},
    }


def _page_ref(session_id: str, *, start_time_us: int = 1767225600000000) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "trace_id": f"trace-{session_id}",
        "root_span_id": f"root-{session_id}",
        "start_time_us": start_time_us,
    }


@pytest.mark.asyncio
async def test_list_sessions_maps_rows_and_binds_all_request_values() -> None:
    workspace = "workspace' OR 1 = 1 --"
    evaluation_name = "evaluation'); DROP TABLE trace_index; --"
    test_case_id = "case' UNION ALL SELECT secret --"
    executor = _Executor([[{"count()": 1}], [_page_ref("session-a")], [_session_record("session-a")]])

    page = await _repository(executor).list_sessions(
        workspace=workspace,
        evaluation_name=evaluation_name,
        status=SpanStatus.ERROR,
        test_case_name=test_case_id,
        page=2,
        page_size=5,
        mode="preview",
        sort_keys=[("latency_ms", True)],
    )

    assert page.total == 1
    assert [row.session_id for row in page.rows] == ["session-a"]
    assert page.rows[0].evaluator_scores == {"reward": 0.9}
    assert executor.tables == [
        ClickHouseTable.TRACE_INDEX,
        ClickHouseTable.SPANS,
        ClickHouseTable.EVALUATOR_RESULTS,
    ]
    assert [query.name for query in executor.queries] == [
        "evaluation_sessions.count",
        "evaluation_sessions.page",
        "evaluation_sessions.hydrate",
    ]
    for query in executor.queries:
        assert workspace not in query.statement
        assert evaluation_name not in query.statement
        assert test_case_id not in query.statement
        assert query.parameters["workspace"] == workspace
        assert query.parameters["evaluation_name"] == evaluation_name
        assert "evaluation_name = %(evaluation_name)s" in query.statement
        assert "evaluation_id = %(evaluation_name)s" not in query.statement
    for query in executor.queries[:2]:
        assert query.parameters["test_case_name"] == test_case_id
        assert query.parameters["status"] == "error"
        assert "test_case_name = %(test_case_name)s" in query.statement
    assert executor.queries[1].parameters["limit"] == 5
    assert executor.queries[1].parameters["offset"] == 5
    assert "'' AS input" in executor.queries[1].statement
    assert "'' AS output" in executor.queries[1].statement
    assert "substringUTF8(root_input, 1, %(payload_char_limit)s) AS input" in executor.queries[2].statement
    assert "FROM spans AS span_versions FINAL" in executor.queries[2].statement
    assert "argMax(span_versions." not in executor.queries[2].statement
    assert executor.queries[2].parameters["page_session_ids"] == ["session-a"]
    assert executor.queries[2].parameters["page_trace_ids"] == ["trace-session-a"]
    assert executor.queries[2].parameters["page_storage_keys"] == [
        (1767225600000000, "trace-session-a", "root-session-a")
    ]


@pytest.mark.asyncio
async def test_metric_sort_uses_bounded_page_then_restores_hydration_order() -> None:
    executor = _Executor(
        [
            [{"count()": 2}],
            [_page_ref("session-b"), _page_ref("session-a")],
            [_session_record("session-a"), _session_record("session-b")],
        ]
    )

    page = await _repository(executor).list_sessions(
        workspace="default",
        evaluation_name="evaluation-a",
        page=1,
        page_size=2,
        mode="summary",
        sort_keys=[("cost_total_usd", True)],
    )

    assert page.total == 2
    assert [row.session_id for row in page.rows] == ["session-b", "session-a"]
    assert [query.name for query in executor.queries] == [
        "evaluation_sessions.count",
        "evaluation_sessions.metric_sort.page",
        "evaluation_sessions.hydrate",
    ]
    assert "FROM spans AS span_versions FINAL" in executor.queries[1].statement
    assert "argMax(span_versions." not in executor.queries[1].statement
    assert "FROM spans AS span_versions FINAL" in executor.queries[2].statement
    assert executor.queries[2].parameters["page_session_ids"] == ["session-b", "session-a"]


@pytest.mark.asyncio
async def test_empty_result_stops_after_count() -> None:
    executor = _Executor([[{"count()": 0}]])

    page = await _repository(executor).list_sessions(
        workspace="default",
        evaluation_name="evaluation-a",
        page=1,
        page_size=100,
        mode="detailed",
    )

    assert page.total == 0
    assert page.rows == []
    assert [query.name for query in executor.queries] == ["evaluation_sessions.count"]


@pytest.mark.asyncio
async def test_metric_sort_rejects_unbounded_session_set_after_count() -> None:
    executor = _Executor([[{"count()": _MAX_METRIC_SORT_SESSIONS + 1}]])

    with pytest.raises(MetricSortTooLargeError) as exc_info:
        await _repository(executor).list_sessions(
            workspace="default",
            evaluation_name="evaluation-a",
            page=1,
            page_size=100,
            mode="summary",
            sort_keys=[("tokens", False)],
        )

    assert exc_info.value.total == _MAX_METRIC_SORT_SESSIONS + 1
    assert exc_info.value.limit == _MAX_METRIC_SORT_SESSIONS
    assert [query.name for query in executor.queries] == ["evaluation_sessions.count"]


def test_build_order_by_uses_registered_expressions_in_key_order() -> None:
    assert _SORT_EXPR_PAGE["test_case_name"] == "test_case_name"
    assert (
        _build_order_by(
            [("cost_total_usd", True), ("latency_ms", False)],
            _SORT_EXPR_PAGE,
            "s.root_span_id ASC",
        )
        == "pm.cost_total_usd DESC NULLS LAST, latency_ms ASC NULLS LAST, s.root_span_id ASC"
    )
    assert (
        _build_order_by(
            [("tokens", False)],
            _SORT_EXPR_PAGE,
            "s.root_span_id ASC",
        )
        == "pm.total_tokens ASC NULLS LAST, s.root_span_id ASC"
    )
