# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API tests for the paired baseline-vs-candidate evaluation comparison."""

from __future__ import annotations

from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.intake.api.v2.experiments.dependencies import get_evaluation_compare_repository
from nhx.intake.experiments.compare import EvaluationMetricSession
from nhx.intake.repository.evaluation_compare import EvaluationCompareRepository

EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"
COMPARE = f"{EVALUATIONS}/compare"

_SESSIONS: list[EvaluationMetricSession] = [
    # baseline: tc1 averages (1.0 + failed 0) / 2 = 0.5; tc2 = 0.5
    EvaluationMetricSession("base-eval", "tc1", 1.0),
    EvaluationMetricSession("base-eval", "tc1", None, failed=True),
    EvaluationMetricSession("base-eval", "tc2", 0.5),
    EvaluationMetricSession("base-eval", "only-base", 1.0),
    EvaluationMetricSession("base-eval", None, 1.0),  # no test case: dropped
    # candidate: tc1 = 0.75; tc2 averages (None omitted, 1.0) = 1.0
    EvaluationMetricSession("cand-eval", "tc1", 0.75),
    EvaluationMetricSession("cand-eval", "tc2", None),
    EvaluationMetricSession("cand-eval", "tc2", 1.0),
    EvaluationMetricSession("cand-eval", "only-cand", 1.0),
]


class _ScriptedCompareRepository(EvaluationCompareRepository):
    def __init__(self, sessions: list[EvaluationMetricSession]) -> None:
        self._sessions = sessions
        self.calls: list[tuple[str, list[str], str]] = []

    async def get_metric_sessions(
        self,
        *,
        workspace: str,
        evaluation_names: list[str],
        metric: object,
    ) -> list[EvaluationMetricSession]:
        self.calls.append((workspace, list(evaluation_names), str(getattr(metric, "id", metric))))
        return self._sessions


def _create_group_and_evaluations(client: TestClient) -> None:
    group = client.post("/apis/intake/v2/workspaces/default/experiments", json={"name": "grp"})
    assert group.status_code == 201, group.text
    for name in ("base-eval", "cand-eval"):
        created = client.post(
            EVALUATIONS,
            json={"name": name, "experiment_group_id": group.json()["id"], "dataset_name": "ds"},
        )
        assert created.status_code == 201, created.text


@pytest.fixture
def scripted_repository(client: TestClient) -> _ScriptedCompareRepository:
    _create_group_and_evaluations(client)
    repository = _ScriptedCompareRepository(_SESSIONS)
    app = cast(FastAPI, client.app)
    app.dependency_overrides[get_evaluation_compare_repository] = lambda: repository
    return repository


def test_compare_pairs_test_cases_and_reports_the_interval(
    client: TestClient, scripted_repository: _ScriptedCompareRepository
) -> None:
    response = client.get(
        COMPARE,
        params={"baseline": "base-eval", "candidate": "cand-eval", "metric": "evaluators.accuracy"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    # tc1: 0.5 -> 0.75 (better); tc2: 0.5 -> 1.0 (better). The single-sided test
    # cases and the test-case-less session are not paired.
    assert body["n_pairs"] == 2
    assert (body["better"], body["same"], body["worse"]) == (2, 0, 0)
    assert body["mean_paired_difference"] == pytest.approx(0.375)
    assert body["interval_low"] == pytest.approx(0.25)
    assert body["interval_high"] == pytest.approx(0.5)
    assert body["p_candidate_better"] == pytest.approx(1.0)
    assert body["higher_is_better"] is True
    assert body["metric"] == "evaluators.accuracy"
    assert body["workspace"] == "default"
    assert body["interval_caveat"] is not None
    assert scripted_repository.calls == [("default", ["base-eval", "cand-eval"], "evaluators.accuracy")]


def test_compare_route_is_not_captured_as_an_evaluation_name(
    client: TestClient, scripted_repository: _ScriptedCompareRepository
) -> None:
    # Registered before /evaluations/{name}, so the literal path wins over the name route.
    response = client.get(COMPARE, params={"baseline": "base-eval", "candidate": "cand-eval"})

    # Missing metric is a 400 from the compare handler, not a 404 "Evaluation
    # 'default/compare' not found".
    assert response.status_code == 422, response.text
    assert "metric" in response.text


def test_compare_reports_lower_is_better_for_rollup_metrics(
    client: TestClient, scripted_repository: _ScriptedCompareRepository
) -> None:
    response = client.get(
        COMPARE,
        params={"baseline": "base-eval", "candidate": "cand-eval", "metric": "cost_usd"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["higher_is_better"] is False
    assert body["metric"] == "cost_usd"


def test_compare_returns_404_for_an_unknown_evaluation(client: TestClient) -> None:
    _create_group_and_evaluations(client)
    response = client.get(
        COMPARE,
        params={"baseline": "missing-eval", "candidate": "cand-eval", "metric": "evaluators.accuracy"},
    )

    assert response.status_code == 404, response.text
    assert "missing-eval" in response.json()["detail"]


def test_compare_rejects_an_unsupported_metric(
    client: TestClient, scripted_repository: _ScriptedCompareRepository
) -> None:
    response = client.get(
        COMPARE,
        params={"baseline": "base-eval", "candidate": "cand-eval", "metric": "duration_ms"},
    )

    assert response.status_code == 400, response.text
    assert scripted_repository.calls == []


def test_compare_returns_503_when_telemetry_is_unavailable(client: TestClient) -> None:
    _create_group_and_evaluations(client)
    app = cast(FastAPI, client.app)
    app.dependency_overrides[get_evaluation_compare_repository] = lambda: None
    response = client.get(
        COMPARE,
        params={"baseline": "base-eval", "candidate": "cand-eval", "metric": "evaluators.accuracy"},
    )

    assert response.status_code == 503, response.text


def test_compare_without_shared_test_cases_reports_an_empty_interval(
    client: TestClient, scripted_repository: _ScriptedCompareRepository
) -> None:
    disjoint = [
        EvaluationMetricSession("base-eval", "tc1", 1.0),
        EvaluationMetricSession("cand-eval", "tc2", 1.0),
    ]
    scripted_repository._sessions = disjoint

    response = client.get(
        COMPARE,
        params={"baseline": "base-eval", "candidate": "cand-eval", "metric": "evaluators.accuracy"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_pairs"] == 0
    assert body["mean_paired_difference"] is None
    assert body["interval_low"] is None
    assert body["interval_high"] is None
    assert body["p_candidate_better"] is None
    assert body["interval_caveat"] is not None
