# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation comparison ClickHouse repository tests."""

from typing import Any

import pytest
from nhx.intake.experiments.compare import parse_compare_metric
from nhx.intake.repository.clickhouse.evaluation_compare import (
    ClickHouseEvaluationCompareRepository,
)
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable


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


@pytest.mark.asyncio
async def test_get_metric_sessions_reads_latency_without_a_limit() -> None:
    executor = _Executor(
        [
            [
                {"evaluation_name": "base", "test_case_name": "tc1", "value": 100.0},
                {"evaluation_name": "base", "test_case_name": "tc1", "value": None},
                {"evaluation_name": "cand", "test_case_name": "tc1", "value": 80.0},
            ]
        ]
    )

    await ClickHouseEvaluationCompareRepository(executor).get_metric_sessions(
        workspace="default",
        evaluation_names=["base", "cand"],
        metric=parse_compare_metric("latency_ms"),
    )

    (query,) = executor.queries
    assert query.name == "evaluation_compare.latency_sessions"
    assert "LIMIT 1 BY workspace, session_id, evaluation_name" in query.statement
    assert "OFFSET" not in query.statement
    assert "LIMIT %(limit)s" not in query.statement
    # Both evaluations are scoped in one query, parameterized against injection.
    assert "evaluation_name IN (%(evaluation_name_0)s, %(evaluation_name_1)s)" in query.statement
    assert "base" not in query.statement
    assert "cand" not in query.statement
    assert query.parameters["evaluation_name_0"] == "base"
    assert query.parameters["evaluation_name_1"] == "cand"


@pytest.mark.asyncio
async def test_get_metric_sessions_reads_one_evaluator_across_both_evaluations() -> None:
    executor = _Executor(
        [
            [
                {"evaluation_name": "base", "test_case_name": "tc1", "value": 0.5, "failed": 0},
                {"evaluation_name": "cand", "test_case_name": "tc1", "value": float("nan"), "failed": 1},
            ]
        ]
    )

    sessions = await ClickHouseEvaluationCompareRepository(executor).get_metric_sessions(
        workspace="default",
        evaluation_names=["base", "cand"],
        metric=parse_compare_metric("evaluators.accuracy"),
    )

    (query,) = executor.queries
    assert query.name == "evaluation_compare.evaluator_sessions"
    assert executor.tables == [ClickHouseTable.TRACE_INDEX, ClickHouseTable.EVALUATOR_RESULTS]
    assert query.parameters["evaluator_name"] == "accuracy"
    assert "name = %(evaluator_name)s" in query.statement
    assert sessions[0].value == 0.5
    assert sessions[0].failed is False
    assert sessions[1].value is None
    assert sessions[1].failed is True


@pytest.mark.asyncio
async def test_get_metric_sessions_reads_tokens_and_cost_from_span_usage() -> None:
    for metric_id in ("cost_usd", "tokens"):
        executor = _Executor(
            [
                [
                    {"evaluation_name": "base", "test_case_name": "tc1", "value": 12.0},
                    {"evaluation_name": "cand", "test_case_name": "tc1", "value": None},
                ]
            ]
        )

        sessions = await ClickHouseEvaluationCompareRepository(executor).get_metric_sessions(
            workspace="default",
            evaluation_names=["base", "cand"],
            metric=parse_compare_metric(metric_id),
        )

        (query,) = executor.queries
        assert query.name == "evaluation_compare.usage_sessions"
        assert executor.tables == [ClickHouseTable.TRACE_INDEX, ClickHouseTable.SPANS]
        # The usage pipeline reads span attribute bags by parameterized key; those keys
        # must be bound or the query fails before it reaches ClickHouse.
        assert query.parameters["input_tokens_key"]
        assert query.parameters["output_tokens_key"]
        assert query.parameters["cost_usd_key"]
        # Sessions without usage keep a NULL value row rather than dropping out, so the
        # pairing rules can omit them from the mean.
        assert sessions[-1].value is None
        assert "LEFT JOIN session_usage" in query.statement


@pytest.mark.asyncio
async def test_get_metric_sessions_without_evaluations_skips_the_store() -> None:
    executor = _Executor([])

    sessions = await ClickHouseEvaluationCompareRepository(executor).get_metric_sessions(
        workspace="default",
        evaluation_names=[],
        metric=parse_compare_metric("tokens"),
    )

    assert sessions == []
    assert executor.queries == []
