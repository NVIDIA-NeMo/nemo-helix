# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the per-session evaluation endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

ATIF_INGEST = "/apis/intake/v2/workspaces/default/ingest/atif"
EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"
EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"


def _ensure_group(client: TestClient, name: str = "sessions-test-group") -> str:
    """Create or fetch an ExperimentGroup; returns its id."""
    response = client.post(EXPERIMENTS, json={"name": name})
    if response.status_code == 409:
        response = client.get(f"{EXPERIMENTS}/{name}")
    response.raise_for_status()
    return response.json()["id"]


def _session_started_at() -> datetime:
    """A session start well before the evaluation entities created above.

    Runs are often published after they happened, so rollups and session lists bound each
    evaluation's lifetime by ingest time; a historical span start must still be counted.
    """
    return (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)


def test_list_evaluation_sessions_returns_joined_session_rows(client: TestClient) -> None:
    evaluation_name = "sessions-exp"
    group_id = _ensure_group(client)
    created = client.post(
        EVALUATIONS,
        json={
            "name": evaluation_name,
            "experiment_group_id": group_id,
            "dataset_name": "sessions-dataset",
            "dataset_version": "v1",
        },
    )
    assert created.status_code == 201, created.text

    started_at = _session_started_at()
    seeds = [
        ("case-a", 1.0, 0.05, 1000, 100, 10, 0),
        ("case-b", 0.5, 0.10, 2000, 200, 20, 5),
        ("case-c", 0.0, 0.20, 3000, 300, 30, 10),
    ]
    for index, (
        test_case_id,
        score,
        cost_usd,
        latency_ms,
        prompt_tokens,
        completion_tokens,
        offset_seconds,
    ) in enumerate(seeds):
        response = client.post(
            ATIF_INGEST,
            json=_atif_body(
                started_at=started_at,
                evaluation_name=evaluation_name,
                test_case_id=test_case_id,
                score=score,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                offset_seconds=offset_seconds,
                run_id=f"run-{index}",
            ),
        )
        assert response.status_code == 201, response.text

    listed = client.get(f"{EVALUATIONS}/{evaluation_name}/sessions")
    assert listed.status_code == 200, listed.text
    body = listed.json()
    assert body["pagination"]["total_results"] == 3
    assert len(body["data"]) == 3

    rows_by_case = {row["test_case_name"]: row for row in body["data"]}
    assert set(rows_by_case) == {"case-a", "case-b", "case-c"}
    assert all(row["test_case_id"] == row["test_case_name"] for row in body["data"])

    case_a = rows_by_case["case-a"]
    assert case_a["evaluation_name"] == evaluation_name
    assert case_a["session_id"]
    assert case_a["trace_id"]
    assert case_a["root_span_id"]
    assert case_a["latency_ms"] == pytest.approx(1000.0)
    assert case_a["input_tokens"] == 100
    assert case_a["output_tokens"] == 10
    assert case_a["cost_total_usd"] == pytest.approx(0.05)
    assert case_a["evaluator_scores"] == {"reward": pytest.approx(1.0)}
    assert case_a["status"] in {"success", "unknown"}
    assert case_a["output"] == "solved case-a"

    summary = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"mode": "summary", "page_size": 3},
    )
    assert summary.status_code == 200, summary.text
    assert all(row["input"] is None and row["output"] is None for row in summary.json()["data"])

    preview = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"mode": "preview", "page_size": 3},
    )
    assert preview.status_code == 200, preview.text
    preview_by_case = {row["test_case_name"]: row for row in preview.json()["data"]}
    assert preview_by_case["case-a"]["output"] == "solved case-a"

    paged = client.get(f"{EVALUATIONS}/{evaluation_name}/sessions", params={"page": 2, "page_size": 1})
    assert paged.status_code == 200, paged.text
    paged_body = paged.json()
    assert paged_body["pagination"]["total_results"] == 3
    assert len(paged_body["data"]) == 1
    assert paged_body["data"][0]["test_case_name"] == "case-b"
    assert paged_body["data"][0]["evaluator_scores"] == {"reward": pytest.approx(0.5)}

    latency_sorted = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"sort": "-latency_ms", "page_size": 3},
    )
    assert latency_sorted.status_code == 200, latency_sorted.text
    assert [row["test_case_name"] for row in latency_sorted.json()["data"]] == ["case-c", "case-b", "case-a"]

    cost_sorted = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"sort": "-cost_total_usd", "page_size": 3},
    )
    assert cost_sorted.status_code == 200, cost_sorted.text
    assert [row["test_case_name"] for row in cost_sorted.json()["data"]] == ["case-c", "case-b", "case-a"]

    tokens_sorted = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"sort": "tokens", "page_size": 3},
    )
    assert tokens_sorted.status_code == 200, tokens_sorted.text
    assert [row["test_case_name"] for row in tokens_sorted.json()["data"]] == ["case-a", "case-b", "case-c"]

    test_case_sorted = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"sort": "-test_case_name", "page_size": 3},
    )
    assert test_case_sorted.status_code == 200, test_case_sorted.text
    assert [row["test_case_name"] for row in test_case_sorted.json()["data"]] == ["case-c", "case-b", "case-a"]

    deprecated_sort = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"sort": "-test_case_id", "page_size": 3},
    )
    assert deprecated_sort.status_code == 200, deprecated_sort.text
    assert [row["test_case_name"] for row in deprecated_sort.json()["data"]] == ["case-c", "case-b", "case-a"]


def test_list_evaluation_sessions_filter_by_test_case(client: TestClient) -> None:
    evaluation_name = "sessions-filter-exp"
    group_id = _ensure_group(client)
    created = client.post(
        EVALUATIONS,
        json={
            "name": evaluation_name,
            "experiment_group_id": group_id,
            "dataset_name": "sessions-dataset",
            "dataset_version": "v1",
        },
    )
    assert created.status_code == 201, created.text

    started_at = _session_started_at()
    adversarial_test_case_id = "alpha') OR 1 = 1 --"
    for index, test_case_id in enumerate([adversarial_test_case_id, "beta"]):
        response = client.post(
            ATIF_INGEST,
            json=_atif_body(
                started_at=started_at,
                evaluation_name=evaluation_name,
                test_case_id=test_case_id,
                score=1.0,
                cost_usd=0.01,
                latency_ms=100,
                prompt_tokens=50,
                completion_tokens=5,
                offset_seconds=index * 5,
                run_id=f"run-{index}",
            ),
        )
        assert response.status_code == 201, response.text

    filtered = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"filter[test_case_name]": adversarial_test_case_id},
    )
    assert filtered.status_code == 200, filtered.text
    body = filtered.json()
    assert body["pagination"]["total_results"] == 1
    assert len(body["data"]) == 1
    assert body["data"][0]["test_case_name"] == adversarial_test_case_id

    deprecated_filter = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"filter[test_case_id]": adversarial_test_case_id},
    )
    assert deprecated_filter.status_code == 200, deprecated_filter.text
    assert deprecated_filter.json()["data"][0]["test_case_name"] == adversarial_test_case_id

    conflicting_filters = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={
            "filter[test_case_name]": adversarial_test_case_id,
            "filter[test_case_id]": "beta",
        },
    )
    assert conflicting_filters.status_code == 400, conflicting_filters.text
    assert conflicting_filters.json()["detail"] == "Conflicting evaluation session filters for test_case_name"


def test_list_evaluation_sessions_filter_by_status(client: TestClient) -> None:
    # ATIF ingest doesn't expose explicit per-span status, so all seeded sessions land with the
    # default root-span status. This test verifies the filter is wired through (no SQL break and
    # mismatched filters return zero) rather than per-status seeding.
    evaluation_name = "sessions-status-exp"
    group_id = _ensure_group(client)
    created = client.post(
        EVALUATIONS,
        json={
            "name": evaluation_name,
            "experiment_group_id": group_id,
            "dataset_name": "sessions-dataset",
            "dataset_version": "v1",
        },
    )
    assert created.status_code == 201, created.text

    response = client.post(
        ATIF_INGEST,
        json=_atif_body(
            started_at=_session_started_at(),
            evaluation_name=evaluation_name,
            test_case_id="case-1",
            score=1.0,
            cost_usd=0.01,
            latency_ms=100,
            prompt_tokens=50,
            completion_tokens=5,
            offset_seconds=0,
            run_id="run-1",
        ),
    )
    assert response.status_code == 201, response.text

    listed = client.get(f"{EVALUATIONS}/{evaluation_name}/sessions")
    assert listed.status_code == 200, listed.text
    seeded_status = listed.json()["data"][0]["status"]

    matching = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"filter[status]": seeded_status},
    )
    assert matching.status_code == 200, matching.text
    assert matching.json()["pagination"]["total_results"] == 1

    other_status = "error" if seeded_status != "error" else "cancelled"
    mismatched = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"filter[status]": other_status},
    )
    assert mismatched.status_code == 200, mismatched.text
    assert mismatched.json()["pagination"]["total_results"] == 0


def test_list_evaluation_sessions_returns_404_for_unknown_evaluation(client: TestClient) -> None:
    response = client.get(f"{EVALUATIONS}/does-not-exist/sessions")
    assert response.status_code == 404, response.text


def test_list_evaluation_sessions_rejects_unknown_query_param(client: TestClient) -> None:
    response = client.get(
        f"{EVALUATIONS}/does-not-exist/sessions",
        params={"test_caseid": "case-1"},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Unsupported query parameter(s): test_caseid"


def _atif_body(
    *,
    started_at: datetime,
    evaluation_name: str,
    test_case_id: str,
    score: float,
    cost_usd: float,
    latency_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    offset_seconds: int,
    run_id: str,
) -> dict[str, Any]:
    session_started_at = started_at + timedelta(seconds=offset_seconds)
    finished_at = session_started_at + timedelta(milliseconds=latency_ms)
    session_id = f"{evaluation_name}-{run_id}-{test_case_id}"
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "evaluation_context": {
            "evaluation_name": evaluation_name,
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
            "name": "sample-agent",
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


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
