# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for provider-neutral JSON span ingest."""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

INGEST_URL = "/apis/intake/v2/workspaces/default/ingest/spans"
SPANS_URL = "/apis/intake/v2/workspaces/default/spans"


def _timestamp(offset_seconds: int = 0) -> str:
    value = datetime.now(timezone.utc) - timedelta(hours=1) + timedelta(seconds=offset_seconds)
    return value.isoformat()


def _span(**overrides: Any) -> dict[str, Any]:
    span = {
        "span_id": "root-span",
        "trace_id": "provider-trace",
        "name": "agent-run",
        "kind": "AGENT",
        "status": "success",
        "started_at": _timestamp(),
        "ended_at": _timestamp(2),
        "input": {"question": "héllo"},
        "output": {"answer": "world"},
        "attributes": {
            "llm.model_name": "provider-model",
            "llm.provider": "provider-system",
            "custom.number": 7,
            "provider.raw": {
                "nested": [1, True, None, {"unicode": "雪"}],
                "empty": "",
            },
        },
    }
    span.update(overrides)
    return span


def test_direct_span_ingest_round_trips_batch_and_raw_json(client: TestClient):
    body = {
        "source": "provider-store",
        "spans": [
            _span(),
            _span(
                span_id="child/span:any-string",
                parent_span_id="root-span",
                session_id="provider-session",
                name="model-call",
                kind="LLM",
                started_at=_timestamp(1),
                ended_at=None,
                input="plain input",
                output='{"already":"serialized"}',
                attributes={
                    "llm.model_name": "child-model",
                    "llm.token_count.prompt": 12,
                    "llm.token_count.completion": 3,
                    "provider.raw": {"events": [{"name": "token", "value": None}]},
                },
            ),
        ],
    }

    response = client.post(INGEST_URL, json=body)
    assert response.status_code == 201, response.text
    assert response.content == b""

    root_response = client.get(SPANS_URL, params={"filter[session_id]": "provider-trace", "page_size": 10})
    assert root_response.status_code == 200, root_response.text
    root = root_response.json()["data"][0]
    assert root["span_id"] == "root-span"
    assert root["source"] == "provider-store"
    assert root["model"] == "provider-model"
    assert root["provider"] == "provider-system"
    assert json.loads(root["input"]) == {"question": "héllo"}
    assert json.loads(root["output"]) == {"answer": "world"}
    assert json.loads(root["raw_attributes"]) == {
        "custom.number": 7,
        "provider.raw": {
            "nested": [1, True, None, {"unicode": "雪"}],
            "empty": "",
        },
    }

    child_response = client.get(SPANS_URL, params={"filter[session_id]": "provider-session", "page_size": 10})
    assert child_response.status_code == 200, child_response.text
    child = child_response.json()["data"][0]
    assert child["span_id"] == "child/span:any-string"
    assert child["parent_span_id"] == "root-span"
    assert child["input_tokens"] == 12
    assert child["output_tokens"] == 3
    assert child["input"] == "plain input"
    assert child["output"] == '{"already":"serialized"}'
    assert "ended_at" not in child
    assert json.loads(child["raw_attributes"])["provider.raw"]["events"][0]["value"] is None


def test_direct_span_ingest_orders_equal_timestamps_by_step_id(client: TestClient):
    started_at = _timestamp()
    spans = [
        _span(span_id=span_id, started_at=started_at, attributes={"nemo.step_id": step_id})
        for span_id, step_id in {"step-3": 3, "step-1": 1, "too-large": 2**1024}.items()
    ]

    response = client.post(INGEST_URL, json={"source": "provider-store", "spans": spans})
    assert response.status_code == 201, response.text

    listed = client.get(SPANS_URL, params={"filter[session_id]": "provider-trace", "sort": "started_at"})
    assert listed.status_code == 200, listed.text
    assert [span["span_id"] for span in listed.json()["data"]] == ["step-1", "step-3", "too-large"]
    assert listed.json()["data"][0]["raw_attributes"] == '{"nemo.step_id":1}'


def test_direct_span_ingest_replay_upserts(client: TestClient):
    body = {"source": "replay-source", "spans": [_span(trace_id="replay-trace", output={"version": 1})]}
    first = client.post(INGEST_URL, json=body)
    assert first.status_code == 201, first.text

    body["spans"][0]["output"] = {"version": 2}
    second = client.post(INGEST_URL, json=body)
    assert second.status_code == 201, second.text

    listed = client.get(
        SPANS_URL,
        params={"filter[trace_id]": "replay-trace", "filter[source]": "replay-source", "page_size": 10},
    )
    assert listed.status_code == 200, listed.text
    spans = listed.json()["data"]
    assert len(spans) == 1
    assert json.loads(spans[0]["output"]) == {"version": 2}


@pytest.mark.parametrize(
    "spans",
    [
        [_span(parent_span_id="root-span")],
        [_span(started_at=_timestamp(2), ended_at=_timestamp(1))],
        [_span(), _span()],
        [_span(unsupported="value")],
        [_span(started_at="2026-01-01T00:00:00")],
        [_span(attributes={"llm.token_count.prompt": {"bad": "value"}})],
    ],
)
def test_direct_span_ingest_rejects_invalid_batches_without_partial_writes(
    client: TestClient,
    spans: list[dict[str, Any]],
):
    response = client.post(INGEST_URL, json={"source": "invalid-source", "spans": spans})
    assert response.status_code == 422, response.text

    listed = client.get(SPANS_URL, params={"filter[source]": "invalid-source", "page_size": 10})
    assert listed.status_code == 200, listed.text
    assert listed.json()["data"] == []


def test_direct_span_ingest_rejects_data_outside_retention_without_partial_writes(client: TestClient):
    expired = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
    response = client.post(
        INGEST_URL,
        json={
            "source": "expired-import",
            "spans": [
                _span(span_id="fresh", trace_id="fresh-trace"),
                _span(span_id="expired", trace_id="expired-trace", started_at=expired, ended_at=expired),
            ],
        },
    )

    assert response.status_code == 422, response.text
    assert "outside Intake's 90-day ClickHouse retention window" in response.json()["detail"]
    assert "Increase the spans and trace_index table TTL" in response.json()["detail"]

    listed = client.get(SPANS_URL, params={"filter[source]": "expired-import", "page_size": 10})
    assert listed.status_code == 200, listed.text
    assert listed.json()["data"] == []


def test_direct_span_ingest_matches_clickhouse_date_retention_boundary(client: TestClient) -> None:
    retention_date = datetime.now(timezone.utc).date() - timedelta(days=90)
    started_at = datetime.combine(retention_date, time(23, 59), tzinfo=timezone.utc).isoformat()

    response = client.post(
        INGEST_URL,
        json={
            "source": "retention-boundary",
            "spans": [_span(started_at=started_at, ended_at=started_at)],
        },
    )

    assert response.status_code == 422, response.text


def test_direct_span_ingest_accepts_parent_outside_batch(client: TestClient):
    response = client.post(
        INGEST_URL,
        json={
            "source": "partial-export",
            "spans": [_span(span_id="only-child", parent_span_id="not-in-batch", trace_id="partial-trace")],
        },
    )
    assert response.status_code == 201, response.text

    listed = client.get(SPANS_URL, params={"filter[trace_id]": "partial-trace", "page_size": 10})
    assert listed.status_code == 200, listed.text
    assert listed.json()["data"][0]["parent_span_id"] == "not-in-batch"


def test_spans_are_listed_by_the_address_they_came_from_within_a_time_range(client: TestClient):
    """The gateway's call spans are found by caller address and call time, the join a run's sandbox records need."""
    body = {
        "source": "inference-gateway",
        "spans": [
            _span(
                span_id="early",
                trace_id="call-early",
                kind="LLM",
                started_at=_timestamp(0),
                ended_at=_timestamp(1),
                attributes={"client.address": "10.244.0.7"},
            ),
            _span(
                span_id="late",
                trace_id="call-late",
                kind="LLM",
                started_at=_timestamp(100),
                ended_at=_timestamp(101),
                attributes={"client.address": "10.244.0.7"},
            ),
            _span(
                span_id="other",
                trace_id="call-other",
                kind="LLM",
                started_at=_timestamp(0),
                ended_at=_timestamp(1),
                attributes={"client.address": "10.244.0.8"},
            ),
        ],
    }
    assert client.post(INGEST_URL, json=body).status_code == 201

    listed = client.get(
        SPANS_URL,
        params={
            "filter[client_address]": "10.244.0.7",
            "filter[started_at][$gte]": _timestamp(-10),
            "filter[started_at][$lte]": _timestamp(50),
            "page_size": 10,
        },
    )

    assert listed.status_code == 200, listed.text
    spans = listed.json()["data"]
    assert [span["trace_id"] for span in spans] == ["call-early"]
    assert spans[0]["client_address"] == "10.244.0.7"
