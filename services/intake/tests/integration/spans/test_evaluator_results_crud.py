# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD tests for evaluator_results."""

from __future__ import annotations

from fastapi.testclient import TestClient

EVAL_BASE = "/apis/intake/v2/workspaces/default/evaluator-results"


def _make_numeric_body(**overrides) -> dict:
    body = {
        "span_id": "span-target-1",
        "session_id": "session-1",
        "name": "faithfulness",
        "value": 0.85,
        "data_type": "NUMERIC",
        "comment": "looks good",
    }
    body.update(overrides)
    return body


def test_create_and_get_evaluator_result(client: TestClient):
    response = client.post(EVAL_BASE, json=_make_numeric_body())
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["span_id"] == "span-target-1"
    assert payload["name"] == "faithfulness"
    assert payload["data_type"] == "NUMERIC"
    assert payload["value"] == 0.85
    assert payload["comment"] == "looks good"
    assert payload["evaluator_result_id"].startswith("eval-")
    assert payload["created_at"] == payload["ingested_at"]

    fetched = client.get(f"{EVAL_BASE}/{payload['evaluator_result_id']}")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["evaluator_result_id"] == payload["evaluator_result_id"]


def test_list_evaluator_results_returns_paginated_results(client: TestClient):
    for index in range(3):
        response = client.post(
            EVAL_BASE,
            json=_make_numeric_body(name=f"metric-{index}", value=float(index) / 10),
        )
        assert response.status_code == 201, response.text

    listed = client.get(EVAL_BASE, params={"page_size": 50})
    assert listed.status_code == 200, listed.text
    page = listed.json()
    assert page["pagination"]["total_results"] == 3
    names = {row["name"] for row in page["data"]}
    assert names == {"metric-0", "metric-1", "metric-2"}


def test_create_evaluator_result_rejects_value_when_data_type_is_text(client: TestClient):
    response = client.post(
        EVAL_BASE,
        json=_make_numeric_body(data_type="TEXT", value=None, string_value=None),
    )
    assert response.status_code == 422, response.text
    assert "string_value" in response.text


def test_create_evaluator_result_accepts_categorical_string_value(client: TestClient):
    response = client.post(
        EVAL_BASE,
        json={
            "span_id": "span-target-2",
            "session_id": "session-1",
            "name": "hallucination_kind",
            "string_value": "fabricated",
            "data_type": "CATEGORICAL",
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["data_type"] == "CATEGORICAL"
    assert payload["string_value"] == "fabricated"
    assert "value" not in payload


def test_create_evaluator_result_rejects_non_binary_boolean_value(client: TestClient):
    response = client.post(
        EVAL_BASE,
        json=_make_numeric_body(data_type="BOOLEAN", value=0.5),
    )
    assert response.status_code == 422, response.text
    assert "BOOLEAN" in response.text


def test_get_evaluator_result_returns_404_when_missing(client: TestClient):
    response = client.get(f"{EVAL_BASE}/eval-does-not-exist")
    assert response.status_code == 404, response.text


def test_create_evaluator_result_is_idempotent_for_identical_writes(client: TestClient):
    body = _make_numeric_body()

    first = client.post(EVAL_BASE, json=body)
    assert first.status_code == 201, first.text
    second = client.post(EVAL_BASE, json=body)
    assert second.status_code == 201, second.text

    # An identical re-POST hashes to the same deterministic id...
    assert first.json()["evaluator_result_id"] == second.json()["evaluator_result_id"]

    # ...and dedupes on read rather than creating a permanent duplicate row.
    listed = client.get(EVAL_BASE, params={"page_size": 50})
    assert listed.status_code == 200, listed.text
    assert listed.json()["pagination"]["total_results"] == 1


def test_re_post_same_target_with_different_value_upserts_latest(client: TestClient):
    first = client.post(EVAL_BASE, json=_make_numeric_body(value=0.85))
    assert first.status_code == 201, first.text
    second = client.post(EVAL_BASE, json=_make_numeric_body(value=0.42))
    assert second.status_code == 201, second.text

    # Same target (workspace, session, span, name) -> same id; a re-score upserts, not appends.
    assert first.json()["evaluator_result_id"] == second.json()["evaluator_result_id"]

    listed = client.get(EVAL_BASE, params={"page_size": 50})
    assert listed.status_code == 200, listed.text
    page = listed.json()
    assert page["pagination"]["total_results"] == 1
    # Latest write wins.
    assert page["data"][0]["value"] == 0.42


def test_create_failed_evaluator_result_carries_no_value(client: TestClient):
    """An evaluator that ran but could not score records a FAILED row with the reason, not a value."""
    body = _make_numeric_body(name="judge", status="FAILED", comment="judge endpoint unreachable")
    del body["value"]
    response = client.post(EVAL_BASE, json=body)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["status"] == "FAILED"
    assert "value" not in payload and "string_value" not in payload
    assert payload["comment"] == "judge endpoint unreachable"

    fetched = client.get(f"{EVAL_BASE}/{payload['evaluator_result_id']}")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["status"] == "FAILED"


def test_scored_results_default_to_scored_status(client: TestClient):
    response = client.post(EVAL_BASE, json=_make_numeric_body())
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "SCORED"


def test_create_failed_evaluator_result_rejects_a_value(client: TestClient):
    response = client.post(EVAL_BASE, json=_make_numeric_body(status="FAILED"))
    assert response.status_code == 422, response.text
    assert "FAILED" in response.text


def test_filter_evaluator_results_by_status(client: TestClient):
    assert client.post(EVAL_BASE, json=_make_numeric_body(name="scored")).status_code == 201
    failed = _make_numeric_body(name="failed", status="FAILED")
    del failed["value"]
    assert client.post(EVAL_BASE, json=failed).status_code == 201

    listed = client.get(EVAL_BASE, params={"filter[status]": "FAILED", "page_size": 50})
    assert listed.status_code == 200, listed.text
    assert [row["name"] for row in listed.json()["data"]] == ["failed"]
