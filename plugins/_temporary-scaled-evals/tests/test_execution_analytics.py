# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api import execution_analytics
from scaled_evals.api.db import get_db
from scaled_evals.api.execution_analytics import ANALYTICS_SOURCE, emit_execution_span, failure_category


class _FakeIntake:
    def __init__(self) -> None:
        self.spans: list = []
        self.queries: list[dict] = []
        self.fail = False

    def create_spans(self, *, workspace: str, body) -> None:
        if self.fail:
            raise ConnectionError("intake down")
        assert body.source == ANALYTICS_SOURCE
        self.spans.extend(body.spans)

    def list_spans(self, *, workspace: str, query_params: dict):
        self.queries.append(query_params)
        return SimpleNamespace(
            items=lambda: [SimpleNamespace(raw_attributes=json.dumps(span.attributes)) for span in self.spans]
        )


def _row(evaluation_id: str, status: str, finished_at: datetime, **extra) -> dict:
    return {
        "id": evaluation_id,
        "name": evaluation_id,
        "task_id": "task_1",
        "owner_id": "usr_1",
        "runtime": "sandbox_k8s",
        "status": status,
        "created_at": finished_at - timedelta(minutes=10),
        "finished_at": finished_at,
        "n_trials": 4,
        "parallelism": 2,
        "current_execution": 1,
        **extra,
    }


def test_terminal_spans_round_trip_into_admin_views(monkeypatch: pytest.MonkeyPatch) -> None:
    intake = _FakeIntake()
    monkeypatch.setattr(execution_analytics, "_intake", lambda: intake)
    db = MagicMock()
    db.users.live_usage.return_value = {
        "queued_runs": 1,
        "active_runs": 0,
        "total_parallelism": 3,
        "total_tasks": 7,
        "actors": [{"owner_id": "usr_1", "queued_runs": 1, "active_runs": 0, "total_parallelism": 3}],
    }
    v1.dependency_overrides[get_db] = lambda: db
    recent = datetime.now(UTC) - timedelta(hours=1)
    failed = _row(
        "ev_504",
        "failed",
        recent,
        owner_id="usr_2",
        last_failure_code="APIStatusError",
        status_detail="inference request returned 504 with api_key=sk-do-not-echo",
    )
    usage = {"cpu_samples": 4.0, "cpu_sum": 2.0, "cpu_request_cores": 1.0, "peak_cpu_cores": 1.5}
    try:
        emit_execution_span(_row("ev_ok", "succeeded", recent), usage, workspace="default")
        emit_execution_span(failed, {}, workspace="default")
        emit_execution_span(failed, {}, workspace="default")  # an evidence retry re-emits; views dedupe
        emit_execution_span(_row("ev_old", "failed", recent - timedelta(days=30)), {}, workspace="default")
        intake.fail = True
        emit_execution_span(_row("ev_lost", "succeeded", recent), {}, workspace="default")  # logged, not raised

        usage_body = client.get("/v1/admin/usage").json()
        failures = client.get("/v1/admin/failures", params={"examples": 1}).json()
        compute = client.get("/v1/admin/compute").json()
    finally:
        v1.dependency_overrides.pop(get_db, None)

    assert intake.queries[0]["filter"]["source"] == ANALYTICS_SOURCE
    assert (usage_body["total_runs"], usage_body["succeeded_runs"], usage_body["failed_runs"]) == (3, 1, 1)
    assert (usage_body["queued_runs"], usage_body["total_tasks"], usage_body["total_trials"]) == (1, 7, 8)
    assert usage_body["total_parallelism"] == 7
    assert usage_body["avg_runtime_seconds"] == 600.0
    assert [(a["owner_id"], a["total_runs"], a["queued_runs"]) for a in usage_body["actors"]] == [
        ("usr_1", 2, 1),
        ("usr_2", 1, 0),
    ]

    assert failures["total_failures"] == 1
    category = failures["categories"][0]
    assert (category["key"], category["label"], category["count"]) == ("inference_http_504", "Inference HTTP 504", 1)
    assert category["examples"][0]["owner_label"] == "usr_2"
    assert "sk-do-not-echo" not in category["examples"][0]["detail"]
    assert sum(point["total"] for point in failures["timeline"]) == 1
    assert len(failures["timeline"]) == 7

    assert (compute["evaluations"], compute["sampled_evaluations"], compute["samples"]) == (2, 1, 4)
    assert compute["avg_cpu_cores"] == 0.5
    assert compute["avg_cpu_request_utilization_percent"] == 50.0
    assert compute["peak_cpu_cores"] == 1.5
    assert compute["runtimes"][0]["runtime"] == "sandbox_k8s"

    assert failure_category(code="RateLimitError", category=None, detail=None, exception_counts=None) == (
        "inference_rate_limit"
    )
    assert failure_category(code="x", category="infrastructure", detail=None, exception_counts=None) == (
        "runtime_infrastructure"
    )
    assert failure_category(code=None, category=None, detail=None, exception_counts={"AgentTimeoutError": 1}) == (
        "evaluation_timeout"
    )
    assert failure_category(code="weird", category=None, detail=None, exception_counts=None) == "other"
