# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint-level guards for the evaluations list sort (rollups unavailable / bad field).

The shared ``client`` fixture disables the Evaluation rollup repository,
which is exactly the "ClickHouse disabled / unavailable" condition. A metric-backed sort cannot be
computed without rollups, so it must fail loudly rather than silently degrade to name order.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from nhx.intake.readiness import CLICKHOUSE_UNAVAILABLE_MESSAGE

EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"
EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"


def _make_evaluation(client: TestClient, name: str = "exp-1", group: str = "grp-1") -> None:
    group_resp = client.post(EXPERIMENTS, json={"name": group})
    assert group_resp.status_code == 201, group_resp.text
    exp_resp = client.post(
        EVALUATIONS,
        json={"name": name, "experiment_group_id": group_resp.json()["id"], "dataset_name": "ds"},
    )
    assert exp_resp.status_code == 201, exp_resp.text


def test_metric_sort_returns_503_when_rollups_unavailable(client: TestClient) -> None:
    _make_evaluation(client)
    response = client.get(EVALUATIONS, params={"sort": "-cost_usd.mean"})
    assert response.status_code == 503, response.text


def test_run_count_sort_returns_503_when_rollups_unavailable(client: TestClient) -> None:
    _make_evaluation(client)
    response = client.get(EVALUATIONS, params={"sort": "run_count"})
    assert response.status_code == 503, response.text


def test_session_reads_return_actionable_503_when_clickhouse_is_unavailable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_evaluation(client)
    service = client.app.state.intake_service
    monkeypatch.setattr(
        service.clickhouse_client,
        "query",
        AsyncMock(side_effect=ConnectionError("ClickHouse unavailable")),
    )

    response = client.get(f"{EVALUATIONS}/exp-1/sessions")

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == CLICKHOUSE_UNAVAILABLE_MESSAGE


def test_entity_sort_still_succeeds_without_rollups(client: TestClient) -> None:
    _make_evaluation(client)
    for sort in ("name", "-created_at", "pinned_at"):
        response = client.get(EVALUATIONS, params={"sort": sort})
        assert response.status_code == 200, response.text


def test_unknown_sort_field_returns_400(client: TestClient) -> None:
    response = client.get(EVALUATIONS, params={"sort": "bogus.field"})
    assert response.status_code == 400, response.text


def test_multi_field_metric_sort_returns_503_when_rollups_unavailable(client: TestClient) -> None:
    # Entity field first, metric second: the metric key still needs rollups, so the whole sort 503s.
    _make_evaluation(client)
    response = client.get(EVALUATIONS, params={"sort": "name,-cost_usd.mean"})
    assert response.status_code == 503, response.text


def test_multi_field_entity_sort_succeeds_without_rollups(client: TestClient) -> None:
    _make_evaluation(client)
    response = client.get(EVALUATIONS, params={"sort": "name,-created_at"})
    assert response.status_code == 200, response.text


def test_unknown_field_in_multi_field_sort_returns_400(client: TestClient) -> None:
    response = client.get(EVALUATIONS, params={"sort": "name,bogus.field"})
    assert response.status_code == 400, response.text


def test_empty_sort_returns_400(client: TestClient) -> None:
    for value in ("", ","):
        response = client.get(EVALUATIONS, params={"sort": value})
        assert response.status_code == 400, response.text


def test_too_many_evaluations_to_sort_returns_413(client: TestClient, monkeypatch) -> None:
    # The whole filtered set is sorted in memory; over the cap we refuse rather than return a
    # silently truncated result. 413 (distinct from the 400 bad-sort-field case) so a caller can tell
    # the two apart. Shrink the cap so the test stays fast.
    from nhx.intake.api.v2.experiments import endpoints

    monkeypatch.setattr(endpoints, "_MAX_GROUP_EVALUATIONS", 2)
    group_resp = client.post(EXPERIMENTS, json={"name": "big-grp"})
    group_id = group_resp.json()["id"]
    for index in range(3):
        resp = client.post(
            EVALUATIONS,
            json={"name": f"exp-{index}", "experiment_group_id": group_id, "dataset_name": "ds"},
        )
        assert resp.status_code == 201, resp.text

    response = client.get(EVALUATIONS, params={"sort": "name"})
    assert response.status_code == 413, response.text
    assert "exceeding the maximum" in response.json()["detail"]
