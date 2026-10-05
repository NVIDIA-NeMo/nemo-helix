# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation rollup repository tests."""

import pytest
from nhx.intake.repository.clickhouse.evaluation_rollup import ClickHouseEvaluationRollupRepository
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable


class _Executor(ClickHouseExecutor):
    def __init__(self, query_results: list[list[dict[str, object]]]) -> None:
        self.queries: list[ClickHouseQuery] = []
        self.query_results = query_results

    def table(self, table: ClickHouseTable) -> str:
        return table.value

    async def fetch_all(self, query: ClickHouseQuery) -> list[dict[str, object]]:
        self.queries.append(query)
        return self.query_results.pop(0)


def _repository(executor: _Executor) -> ClickHouseEvaluationRollupRepository:
    return ClickHouseEvaluationRollupRepository(executor)


@pytest.mark.asyncio
async def test_evaluation_rollups_anchor_on_root_session_membership():
    executor = _Executor(
        [
            [{"evaluation_name": "exp-a", "run_count": 3, "test_case_count": 2}],
            [
                {
                    "evaluation_name": "exp-a",
                    "evaluator_name": "reward",
                    "sum": 3.0,
                    "mean": 0.75,
                    "median": 0.8,
                    "p90": 1.0,
                    "p95": 1.0,
                    "p99": 1.0,
                    "count": 4,
                }
            ],
            [
                {
                    "evaluation_name": "exp-a",
                    "model_names": ["model-b", "model-a"],
                    "agent_names": ["agent-a"],
                    "agent_versions": ["1.0.0", "1.0.1"],
                    "cost_sum": 0.65,
                    "cost_mean": 0.1625,
                    "cost_median": 0.2,
                    "cost_p90": 0.3,
                    "cost_p95": 0.3,
                    "cost_p99": 0.3,
                    "cost_count": 4,
                    "latency_sum": 7000.0,
                    "latency_mean": 1750.0,
                    "latency_median": 2000.0,
                    "latency_p90": 3000.0,
                    "latency_p95": 3000.0,
                    "latency_p99": 3000.0,
                    "latency_count": 4,
                    "tokens_sum": 6000.0,
                    "tokens_mean": 1500.0,
                    "tokens_median": 1500.0,
                    "tokens_p90": 2500.0,
                    "tokens_p95": 2500.0,
                    "tokens_p99": 2500.0,
                    "tokens_count": 4,
                }
            ],
        ]
    )
    repository = _repository(executor)

    rollups = await repository.get_rollups(workspace="default", evaluation_names=["exp-a"])

    rollup = rollups["exp-a"]
    assert rollup.run_count == 3
    assert rollup.test_case_count == 2
    assert rollup.evaluator_names == ["reward"]
    assert rollup.model_names == ["model-a", "model-b"]
    assert rollup.agent_names == ["agent-a"]
    assert rollup.agent_versions == ["1.0.0", "1.0.1"]
    assert rollup.evaluator_scores["reward"].sum == 3.0
    assert rollup.evaluator_scores["reward"].mean == 0.75
    assert rollup.evaluator_scores["reward"].median == 0.8
    assert rollup.evaluator_scores["reward"].p90 == 1.0
    assert rollup.evaluator_scores["reward"].p95 == 1.0
    assert rollup.evaluator_scores["reward"].p99 == 1.0
    assert rollup.evaluator_scores["reward"].count == 4
    assert rollup.cost_usd is not None
    assert rollup.cost_usd.sum == 0.65
    assert rollup.cost_usd.mean == 0.1625
    assert rollup.cost_usd.median == 0.2
    assert rollup.cost_usd.p90 == 0.3
    assert rollup.cost_usd.p95 == 0.3
    assert rollup.cost_usd.p99 == 0.3
    assert rollup.cost_usd.count == 4
    assert rollup.latency_ms is not None
    assert rollup.latency_ms.sum == 7000
    assert rollup.latency_ms.mean == 1750
    assert rollup.latency_ms.median == 2000
    assert rollup.latency_ms.p90 == 3000
    assert rollup.latency_ms.p95 == 3000
    assert rollup.latency_ms.p99 == 3000
    assert rollup.latency_ms.count == 4
    assert rollup.tokens is not None
    assert rollup.tokens.sum == 6000
    assert rollup.tokens.mean == 1500
    assert rollup.tokens.median == 1500
    assert rollup.tokens.p90 == 2500
    assert rollup.tokens.p95 == 2500
    assert rollup.tokens.p99 == 2500
    assert rollup.tokens.count == 4

    assert [query.name for query in executor.queries] == [
        "evaluation_rollups.run_counts",
        "evaluation_rollups.scores",
        "evaluation_rollups.metrics",
    ]
    statements = [query.statement for query in executor.queries]
    assert "FROM trace_index FINAL" in statements[0]
    assert "evaluation_name" in statements[0]
    assert "test_case_name" in statements[0]
    assert "count() AS run_count" in statements[0]
    assert "uniqExactIf(test_case_name, test_case_name != '')" in statements[0]
    assert "AS test_case_count" in statements[0]
    assert "evaluation_name IN (%(evaluation_name_0)s)" in statements[0]
    assert "LIMIT 1 BY workspace, session_id, evaluation_name" in statements[0]
    assert "ORDER BY root_started_at ASC, root_span_id ASC" in statements[0]
    assert "FROM evaluator_results FINAL" in statements[1]
    assert "quantileExact(0.5)(value) AS median" in statements[1]
    assert "quantileExact(0.99)(value) AS p99" in statements[1]
    assert "AND (workspace, session_id) IN (" in statements[1]
    assert "sessions.session_id = results.session_id" in statements[1]
    # Scores are reduced to one value per (session, evaluator), then averaged per test case before the
    # distribution rollup, so the mean is test-case-weighted and count tracks test cases.
    assert (
        "GROUP BY sessions.evaluation_name, sessions.session_id, sessions.test_case_name, results.name" in statements[1]
    )
    assert "test_case_scores AS" in statements[1]
    assert "WHERE sessions.test_case_name != ''" in statements[1]
    assert "test_case_metrics AS" in statements[2]
    assert "current_session_spans AS" in statements[2]
    assert "(workspace, session_id) IN (SELECT DISTINCT workspace, session_id FROM scoped_sessions)" in statements[2]
    assert "LEFT JOIN current_session_spans AS spans" in statements[2]
    assert "sessions.session_id = spans.dedup_session_id" in statements[2]
    assert "arraySort(arrayDistinct(arrayFlatten(groupArray(model_names)))) AS model_names" in statements[2]
    assert "quantileExactIf(0.5)" in statements[2]
    assert "cost_median" in statements[2]
    assert "quantileExactIf(0.99)" in statements[2]
    assert "latency_p99" in statements[2]
    assert "tokens_p99" in statements[2]
    assert "sessions.trace_id = spans.trace_id" not in statements[2]
    assert executor.queries[0].parameters["evaluation_name_0"] == "exp-a"
    assert executor.queries[2].parameters["model_key"] == "gen_ai.request.model"
    assert "input_tokens_key" in executor.queries[2].parameters
    assert "output_tokens_key" in executor.queries[2].parameters


def test_score_rollup_cte_builders_compose_the_pipeline():
    from nhx.intake.repository.clickhouse.evaluation_rollup import (
        _evaluators_cte,
        _session_scores_cte,
        _test_case_scores_cte,
        _test_case_sessions_cte,
    )

    # Stage 1: one value per (session, evaluator), averaged and grouped per session.
    session_scores = _session_scores_cte("evaluator_results")
    assert "avg(results.value) AS value" in session_scores
    assert "FROM evaluator_results FINAL" in session_scores
    assert (
        "GROUP BY sessions.evaluation_name, sessions.session_id, sessions.test_case_name, results.name"
        in session_scores
    )

    # Fixed denominator: distinct sessions per test case.
    assert "count(DISTINCT session_id) AS session_count" in _test_case_sessions_cte()

    # Evaluator axis of the grid: distinct evaluators, read from evaluator_results (not session_scores).
    evaluators = _evaluators_cte("evaluator_results")
    assert "SELECT DISTINCT" in evaluators
    assert "FROM evaluator_results FINAL" in evaluators
    assert "session_scores" not in evaluators

    # Stage 2: zero-filled per-(test case, evaluator) score using the fixed denominator.
    test_case_scores = _test_case_scores_cte()
    assert "coalesce(sum(scores.value), 0) / test_cases.session_count AS value" in test_case_scores
    assert "LEFT JOIN session_scores AS scores" in test_case_scores
    assert "INNER JOIN evaluators ON evaluators.evaluation_name = test_cases.evaluation_name" in test_case_scores
