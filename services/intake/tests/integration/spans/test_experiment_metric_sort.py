# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The evaluations list sorts by a ClickHouse rollup metric (Option A app-merge), end to end."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi.testclient import TestClient

ATIF_INGEST = "/apis/intake/v2/workspaces/default/ingest/atif"
EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"
EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"


def _ensure_group(client: TestClient, name: str) -> str:
    response = client.post(EXPERIMENTS, json={"name": name})
    if response.status_code == 409:
        response = client.get(f"{EXPERIMENTS}/{name}")
    response.raise_for_status()
    return response.json()["id"]


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _atif_body(*, started_at: datetime, evaluation_id: str, cost_usd: float, offset_seconds: int) -> dict[str, Any]:
    session_started_at = started_at + timedelta(seconds=offset_seconds)
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": f"{evaluation_id}-session",
        "evaluation_context": {"evaluation_name": evaluation_id, "test_case_name": "case-1"},
        "extra": {"task_name": "case-1", "verifier_result": {"rewards": {"reward": 1.0}}},
        "agent": {"name": "sample-agent", "version": "1.0.0", "model_name": "provider/sample-model"},
        "steps": [
            {
                "step_id": 1,
                "timestamp": _iso(session_started_at),
                "source": "agent",
                "model_name": "provider/sample-model",
                "message": "done",
                "metrics": {"prompt_tokens": 100, "completion_tokens": 10, "cost_usd": cost_usd},
            }
        ],
    }


def _create_evaluation(client: TestClient, group_id: str, name: str) -> None:
    response = client.post(
        EVALUATIONS,
        json={"name": name, "experiment_group_id": group_id, "dataset_name": "ds"},
    )
    assert response.status_code == 201, response.text


def test_list_sorts_by_cost_metric_missing_last(client: TestClient) -> None:
    # Unique per run so reruns/shared integration state can't collide on group or evaluation names.
    suffix = uuid.uuid4().hex
    group_id = _ensure_group(client, name=f"metric-sort-group-{suffix}")
    started_at = datetime.now(timezone.utc).replace(microsecond=0)
    cheap, pricey, mid = f"exp-cheap-{suffix}", f"exp-pricey-{suffix}", f"exp-mid-{suffix}"
    for index, (name, cost) in enumerate([(cheap, 0.10), (pricey, 0.90), (mid, 0.50)]):
        _create_evaluation(client, group_id, name)
        response = client.post(
            ATIF_INGEST,
            json=_atif_body(started_at=started_at, evaluation_id=name, cost_usd=cost, offset_seconds=index * 10),
        )
        assert response.status_code == 201, response.text
    # No ingest -> no cost rollup -> must sort last regardless of direction.
    norun = f"exp-norun-{suffix}"
    _create_evaluation(client, group_id, norun)

    # Filter by this group so the assertion only inspects evaluations this test created.
    listed = client.get(
        EVALUATIONS,
        params={"filter[experiment_group_id]": group_id, "sort": "-cost_usd.mean", "page_size": 50},
    )
    assert listed.status_code == 200, listed.text
    names = [row["name"] for row in listed.json()["data"]]
    assert names == [pricey, mid, cheap, norun]


def test_list_filters_by_cost_metric(client: TestClient) -> None:
    # Same shape as the sort test, but filtering. Combine an entity filter (group) with two metric
    # filters on different fields: cost_usd.mean <= 0.50 excludes pricey; run_count >= 1 excludes the
    # never-ingested evaluation (whose cost rollup is also missing). Sort by cost so the order is
    # deterministic by value rather than by creation time.
    suffix = uuid.uuid4().hex
    group_id = _ensure_group(client, name=f"metric-filter-group-{suffix}")
    started_at = datetime.now(timezone.utc).replace(microsecond=0)
    cheap, pricey, mid = f"exp-cheap-{suffix}", f"exp-pricey-{suffix}", f"exp-mid-{suffix}"
    for index, (name, cost) in enumerate([(cheap, 0.10), (pricey, 0.90), (mid, 0.50)]):
        _create_evaluation(client, group_id, name)
        response = client.post(
            ATIF_INGEST,
            json=_atif_body(started_at=started_at, evaluation_id=name, cost_usd=cost, offset_seconds=index * 10),
        )
        assert response.status_code == 201, response.text
    _create_evaluation(client, group_id, f"exp-norun-{suffix}")  # no ingest -> excluded by both predicates

    listed = client.get(
        EVALUATIONS,
        params={
            "filter[experiment_group_id]": group_id,
            "filter[cost_usd.mean][lte]": "0.50",
            "filter[run_count][gte]": "1",
            "sort": "cost_usd.mean",
            "page_size": 50,
        },
    )
    assert listed.status_code == 200, listed.text
    names = [row["name"] for row in listed.json()["data"]]
    assert names == [cheap, mid]


def test_list_rejects_unknown_sort_field(client: TestClient) -> None:
    response = client.get(EVALUATIONS, params={"sort": "bogus.field"})
    assert response.status_code == 400, response.text
