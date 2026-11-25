# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session deletion against real ClickHouse spans and the trace-index materialized view."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

BASE = "/apis/intake/v2/workspaces/default"


def test_session_delete_updates_metrics_and_preserves_other_sessions(client: TestClient) -> None:
    started_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    for session_id, traces, tokens in (("delete-me", 2, 100), ("keep-me", 1, 30)):
        for index in range(traces):
            ingested = client.post(
                f"{BASE}/ingest/spans",
                json={
                    "source": "metrics-test",
                    "spans": [
                        {
                            "session_id": session_id,
                            "trace_id": f"{session_id}-{index}",
                            "span_id": "root",
                            "name": "root",
                            "started_at": started_at.isoformat(),
                            "ended_at": (started_at + timedelta(seconds=1)).isoformat(),
                            "attributes": {"llm.token_count.prompt": tokens, "llm.cost.total": tokens / 1000},
                        }
                    ],
                },
            )
            assert ingested.status_code == 201, ingested.text

    def metrics(bucket: str, session_id: str | None = None) -> list[dict]:
        params = {"bucket": bucket, "timezone": "UTC"}
        if session_id:
            params["filter[session_id]"] = session_id
        response = client.get(f"{BASE}/traces/metrics", params=params)
        assert response.status_code == 200, response.text
        return response.json()["data"]

    before = metrics("total")[0]
    assert before["run_count"] == 3
    assert before["input_tokens"]["sum"] == 230
    assert before["cost_usd"]["sum"] == pytest.approx(0.23)
    deleted = client.delete(f"{BASE}/sessions/delete-me")
    assert deleted.status_code == 204, deleted.text

    for bucket in ("total", "hour", "day"):
        remaining = metrics(bucket)
        assert sum(point["run_count"] for point in remaining) == 1
        assert sum(point["input_tokens"]["sum"] or 0 for point in remaining) == 30
        assert sum(point["cost_usd"]["sum"] or 0 for point in remaining) == pytest.approx(0.03)
        assert sum(point["run_count"] for point in metrics(bucket, "delete-me")) == 0
    kept = client.get(f"{BASE}/sessions/keep-me")
    assert kept.status_code == 200, kept.text
    assert kept.json()["trace_count"] == 1
    assert kept.json()["input_tokens"] == 30


def test_delete_session_removes_telemetry_and_preserves_independent_records(client: TestClient, make_otlp_request):
    base_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    trace_id = "00000000000000000000000000001001"
    # Reuse trace/span IDs with distinct root times to exercise session scoping.
    for offset, session_id in enumerate(("delete-me", "keep-me")):
        body = make_otlp_request(
            [
                {
                    "span_id": f"{index + 1:016x}",
                    "parent_span_id": "0000000000000001" if index else None,
                    "start_time_unix_nano": base_ns + offset * 10_000_000_000 + index * 100_000_000,
                    "end_time_unix_nano": base_ns + offset * 10_000_000_000 + 1_000_000_000,
                    "attributes": {"gen_ai.conversation.id": session_id},
                }
                for index in range(2)
            ],
            trace_id=trace_id,
        )
        response = client.post(
            f"{BASE}/ingest/otlp/v1/traces", content=body, headers={"Content-Type": "application/x-protobuf"}
        )
        assert response.status_code == 200, response.text

    annotation = client.post(
        f"{BASE}/annotations", json={"session_id": "delete-me", "kind": "note", "text": "retain this"}
    )
    assert annotation.status_code == 201, annotation.text
    result = client.post(
        f"{BASE}/evaluator-results",
        json={
            "span_id": "0000000000000001",
            "session_id": "delete-me",
            "name": "quality",
            "value": 0.8,
            "data_type": "NUMERIC",
        },
    )
    assert result.status_code == 201, result.text

    before = client.get(f"{BASE}/sessions/delete-me")
    assert before.status_code == 200, before.text
    assert before.json()["span_count"] == 2
    response = client.delete(f"{BASE}/sessions/delete-me")
    assert response.status_code == 204, response.text
    assert response.content == b""
    assert client.get(f"{BASE}/sessions/delete-me").status_code == 404
    assert client.delete(f"{BASE}/sessions/delete-me").status_code == 404
    assert client.delete(f"{BASE}/sessions/missing").status_code == 404

    for resource in ("spans", "traces"):
        deleted = client.get(f"{BASE}/{resource}", params={"filter[session_id]": "delete-me"})
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["data"] == []
        kept = client.get(f"{BASE}/{resource}", params={"filter[session_id]": "keep-me"})
        assert kept.status_code == 200, kept.text
        assert len(kept.json()["data"]) == (2 if resource == "spans" else 1)

    kept_session = client.get(f"{BASE}/sessions/keep-me")
    assert kept_session.status_code == 200, kept_session.text
    assert kept_session.json()["span_count"] == 2
    assert client.get(f"{BASE}/annotations/{annotation.json()['annotation_id']}").status_code == 200
    assert client.get(f"{BASE}/evaluator-results/{result.json()['evaluator_result_id']}").status_code == 200


@pytest.mark.parametrize("evaluation_deleted", [False, True])
@pytest.mark.parametrize("child_only", [False, True])
def test_delete_session_requires_evaluation_deletion(
    client: TestClient, make_otlp_request, evaluation_deleted: bool, child_only: bool
) -> None:
    group = client.post(f"{BASE}/experiments", json={"name": "protected-session-group"})
    assert group.status_code == 201, group.text
    evaluation_name = "protected-session-evaluation"
    evaluation = client.post(
        f"{BASE}/evaluations",
        json={
            "name": evaluation_name,
            "experiment_group_id": group.json()["id"],
            "dataset_name": "test",
            "status": "completed",
        },
    )
    assert evaluation.status_code == 201, evaluation.text
    spans = [
        {
            "span_id": f"{index + 1:016x}",
            "parent_span_id": "0000000000000001" if index else None,
            "attributes": {
                "gen_ai.conversation.id": "protected-session",
                **({"nemo.evaluation.name": evaluation_name} if not child_only or index else {}),
            },
        }
        for index in range(2)
    ]
    for trace_id in ("1" * 32, "2" * 32):
        ingested = client.post(
            f"{BASE}/ingest/otlp/v1/traces",
            content=make_otlp_request(spans, trace_id=trace_id),
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert ingested.status_code == 200, ingested.text
        assert ingested.json() == {"errors": []}
    if evaluation_deleted:
        assert client.delete(f"{BASE}/evaluations/{evaluation_name}").status_code == 204
        # Evaluation deletion cascades to the session's traces and spans.
        before = client.get(f"{BASE}/sessions/protected-session")
        assert before.status_code == 404, before.text

    response = client.delete(f"{BASE}/sessions/protected-session")

    session = client.get(f"{BASE}/sessions/protected-session")
    if evaluation_deleted:
        assert response.status_code == 404, response.text
        assert session.status_code == 404, session.text
    else:
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == (
            "Session default/protected-session is associated with an undeleted or unresolved evaluation "
            "and cannot be deleted"
        )
        assert session.status_code == 200, session.text
        assert session.json()["span_count"] == 4
    traces = client.get(f"{BASE}/traces", params={"filter[session_id]": "protected-session"})
    assert traces.status_code == 200, traces.text
    assert len(traces.json()["data"]) == (0 if evaluation_deleted else 2)
    stored_spans = client.get(f"{BASE}/spans", params={"filter[session_id]": "protected-session"})
    assert stored_spans.status_code == 200, stored_spans.text
    assert len(stored_spans.json()["data"]) == (0 if evaluation_deleted else 4)
