# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation compare integration tests, executing the real ClickHouse queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

ATIF_INGEST = "/apis/intake/v2/workspaces/default/ingest/atif"
EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"
EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"
COMPARE = f"{EVALUATIONS}/compare"

BASELINE = "compare-base"
CANDIDATE = "compare-cand"

# (test_case, score, cost_usd, latency_ms, prompt_tokens, completion_tokens)
BASE_ROWS = [
    ("case-1", 0.4, 0.10, 1000, 100, 10),
    ("case-2", 0.6, 0.20, 2000, 100, 10),
    ("base-only", 0.5, 0.15, 1500, 100, 10),
]
CANDIDATE_ROWS = [
    ("case-1", 0.8, 0.05, 800, 50, 10),
    ("case-2", 0.6, 0.20, 2000, 100, 10),
    ("cand-only", 0.9, 0.01, 500, 100, 10),
]


def _ensure_group(client: TestClient, name: str = "compare-test-group") -> str:
    response = client.post(EXPERIMENTS, json={"name": name})
    if response.status_code == 409:
        response = client.get(f"{EXPERIMENTS}/{name}")
    response.raise_for_status()
    return response.json()["id"]


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _atif_body(
    *,
    started_at: datetime,
    evaluation_id: str,
    run_id: str,
    test_case_id: str,
    score: float,
    cost_usd: float,
    latency_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    offset_seconds: int,
) -> dict[str, Any]:
    session_started_at = started_at + timedelta(seconds=offset_seconds)
    finished_at = session_started_at + timedelta(milliseconds=latency_ms)
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": f"{evaluation_id}-{run_id}-{test_case_id}",
        "evaluation_context": {
            "evaluation_name": evaluation_id,
            "test_case_name": test_case_id,
        },
        "extra": {
            "task_id": test_case_id,
            "task_name": test_case_id,
            "verifier": {
                "started_at": _iso(session_started_at),
                "finished_at": _iso(finished_at),
            },
            "verifier_result": {"rewards": {"reward": score}},
        },
        "agent": {
            "name": "compare-agent",
            "version": "1.0.0",
            "model_name": "provider/sample-model",
        },
        "steps": [
            {
                "step_id": 1,
                "timestamp": _iso(session_started_at),
                "source": "agent",
                "model_name": "provider/sample-model",
                "message": f"solved {test_case_id}",
                "metrics": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost_usd,
                },
            }
        ],
    }


def _ingest(client: TestClient, evaluation_name: str, rows: list[tuple]) -> None:
    group_id = _ensure_group(client)
    created = client.post(
        EVALUATIONS,
        json={
            "name": evaluation_name,
            "experiment_group_id": group_id,
            "dataset_name": "compare-dataset",
            "dataset_version": "v1",
        },
    )
    assert created.status_code == 201, created.text

    started_at = datetime.now(timezone.utc).replace(microsecond=0)
    for index, (test_case, score, cost, latency, prompt, completion) in enumerate(rows):
        response = client.post(
            ATIF_INGEST,
            json=_atif_body(
                started_at=started_at,
                evaluation_id=evaluation_name,
                run_id=f"run-{index}",
                test_case_id=test_case,
                score=score,
                cost_usd=cost,
                latency_ms=latency,
                prompt_tokens=prompt,
                completion_tokens=completion,
                offset_seconds=index * 10,
            ),
        )
        assert response.status_code == 201, response.text


@pytest.fixture
def compared_evaluations(client: TestClient) -> None:
    _ingest(client, BASELINE, BASE_ROWS)
    _ingest(client, CANDIDATE, CANDIDATE_ROWS)


@pytest.mark.parametrize(
    ("metric", "higher_is_better", "better", "same", "worse", "mean_difference"),
    [
        ("evaluators.reward", True, 1, 1, 0, pytest.approx(0.2)),
        ("cost_usd", False, 1, 1, 0, pytest.approx(-0.025)),
        ("latency_ms", False, 1, 1, 0, pytest.approx(-100.0)),
        ("tokens", False, 1, 1, 0, pytest.approx(-25.0)),
    ],
)
def test_compare_executes_each_metric_against_clickhouse(
    client: TestClient,
    compared_evaluations: None,
    metric: str,
    higher_is_better: bool,
    better: int,
    same: int,
    worse: int,
    mean_difference: Any,
) -> None:
    response = client.get(
        COMPARE,
        params={"baseline": BASELINE, "candidate": CANDIDATE, "metric": metric},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["higher_is_better"] is higher_is_better
    assert body["n_pairs"] == 2  # base-only and cand-only are not paired
    assert (body["better"], body["same"], body["worse"]) == (better, same, worse)
    assert body["mean_paired_difference"] == mean_difference
    assert body["interval_low"] is not None
    assert body["interval_high"] is not None
    assert body["interval_low"] <= body["mean_paired_difference"] <= body["interval_high"]
    assert 0.0 <= body["p_candidate_better"] <= 1.0
    assert body["interval_caveat"] is not None  # 2 pairs is a small sample


def test_compare_interval_is_deterministic_across_calls(client: TestClient, compared_evaluations: None) -> None:
    params = {"baseline": BASELINE, "candidate": CANDIDATE, "metric": "evaluators.reward"}
    first = client.get(COMPARE, params=params).json()
    second = client.get(COMPARE, params=params).json()

    assert first == second
    assert (first["better"], first["same"], first["worse"]) == (1, 1, 0)
    assert first["interval_low"] == pytest.approx(0.0)
    assert first["interval_high"] == pytest.approx(0.4)
    assert first["p_candidate_better"] == pytest.approx(0.75, abs=0.02)


def test_compare_without_shared_test_cases_reports_an_empty_interval(client: TestClient) -> None:
    _ingest(client, "compare-disjoint-a", [("case-a", 0.4, 0.10, 1000, 100, 10)])
    _ingest(client, "compare-disjoint-b", [("case-b", 0.8, 0.20, 2000, 100, 10)])

    response = client.get(
        COMPARE,
        params={
            "baseline": "compare-disjoint-a",
            "candidate": "compare-disjoint-b",
            "metric": "evaluators.reward",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_pairs"] == 0
    assert body["interval_low"] is None
    assert body["p_candidate_better"] is None


def test_compare_rejects_unknowns_through_the_real_stack(client: TestClient, compared_evaluations: None) -> None:
    missing = client.get(
        COMPARE,
        params={"baseline": BASELINE, "candidate": "no-such-evaluation", "metric": "evaluators.reward"},
    )
    assert missing.status_code == 404, missing.text

    unsupported = client.get(
        COMPARE,
        params={"baseline": BASELINE, "candidate": CANDIDATE, "metric": "duration_ms"},
    )
    assert unsupported.status_code == 400, unsupported.text
