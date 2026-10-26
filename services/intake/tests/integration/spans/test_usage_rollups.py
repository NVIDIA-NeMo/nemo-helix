# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Usage rollups count each model call once when parent spans repeat it."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

OTLP_INGEST = "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces"
TRACES = "/apis/intake/v2/workspaces/default/traces"
TRACE_METRICS = "/apis/intake/v2/workspaces/default/traces/metrics"
SESSIONS = "/apis/intake/v2/workspaces/default/sessions"
EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"
EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"


def test_minimal_repro_counts_shared_input_tokens_once(client: TestClient, make_otlp_request) -> None:
    """The issue's two-span repro: root and child both report 120."""
    _ingest(
        client,
        make_otlp_request,
        session_id="s1",
        trace_id=_trace_id(1),
        spans=[
            _span(name="root", span_id=1, session_id="s1", input_tokens=120),
            _span(name="chat", span_id=2, parent_span_id=1, session_id="s1", kind="LLM", input_tokens=120),
        ],
    )

    session = _session(client, "s1")
    trace = _trace(client, "s1")
    assert session["input_tokens"] == 120
    assert trace["input_tokens"] == 120


def test_three_level_trial_counts_calls_and_keeps_root_cost(client: TestClient, make_otlp_request) -> None:
    """Root and invoke_agent repeat the chats, and only the root reports cost."""
    session_id = "trial-session"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(2),
        spans=[
            _span(name="trial", span_id=1, session_id=session_id, input_tokens=300, output_tokens=30, cost_usd=0.012),
            _span(
                name="invoke_agent",
                span_id=2,
                parent_span_id=1,
                session_id=session_id,
                kind="AGENT",
                input_tokens=300,
                output_tokens=30,
            ),
            _span(
                name="chat-1",
                span_id=3,
                parent_span_id=2,
                session_id=session_id,
                kind="LLM",
                input_tokens=100,
                output_tokens=10,
            ),
            _span(
                name="chat-2",
                span_id=4,
                parent_span_id=2,
                session_id=session_id,
                kind="LLM",
                input_tokens=200,
                output_tokens=20,
            ),
        ],
    )

    for usage in (_session(client, session_id), _trace(client, session_id)):
        assert usage["input_tokens"] == 300
        assert usage["output_tokens"] == 30
        assert usage["cost_usd"] == pytest.approx(0.012)
    metrics = _trace_metrics(client, session_id)
    assert metrics["input_tokens"]["sum"] == 300
    assert metrics["output_tokens"]["sum"] == 30
    assert metrics["cost_usd"]["sum"] == pytest.approx(0.012)


def test_invoke_agent_over_chats_counts_the_calls_once(client: TestClient, make_otlp_request) -> None:
    session_id = "agent-over-chats"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(3),
        spans=[
            _span(name="invoke_agent", span_id=1, session_id=session_id, kind="AGENT", input_tokens=300),
            _span(name="chat-1", span_id=2, parent_span_id=1, session_id=session_id, kind="LLM", input_tokens=100),
            _span(name="chat-2", span_id=3, parent_span_id=1, session_id=session_id, kind="LLM", input_tokens=200),
        ],
    )

    assert _trace(client, session_id)["input_tokens"] == 300
    assert _session(client, session_id)["input_tokens"] == 300


def test_root_model_span_does_not_repeat_the_calls_beneath_it(client: TestClient, make_otlp_request) -> None:
    """A root chat, including one typed only by gen_ai.operation.name, is a publisher total."""
    session_id = "root-model-span"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(16),
        spans=[
            _span(name="chat", span_id=1, session_id=session_id, kind="LLM", input_tokens=120),
            _span(
                name="chat",
                span_id=2,
                parent_span_id=1,
                session_id=session_id,
                operation="chat",
                input_tokens=120,
            ),
        ],
    )

    assert _trace(client, session_id)["input_tokens"] == 120
    assert _session(client, session_id)["input_tokens"] == 120


def test_lone_root_chat_operation_still_counts(client: TestClient, make_otlp_request) -> None:
    session_id = "lone-root-chat"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(17),
        spans=[_span(name="chat", span_id=1, session_id=session_id, operation="chat", input_tokens=80)],
    )

    assert _trace(client, session_id)["input_tokens"] == 80
    assert _session(client, session_id)["input_tokens"] == 80


def test_root_totals_over_calls_count_the_calls_once(client: TestClient, make_otlp_request) -> None:
    session_id = "root-over-calls"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(4),
        spans=[
            _span(name="trial", span_id=1, session_id=session_id, input_tokens=300),
            _span(name="chat-1", span_id=2, parent_span_id=1, session_id=session_id, kind="LLM", input_tokens=100),
            _span(name="chat-2", span_id=3, parent_span_id=1, session_id=session_id, kind="LLM", input_tokens=200),
        ],
    )

    assert _trace(client, session_id)["input_tokens"] == 300


def test_per_field_fallback_keeps_root_cost_and_cached_tokens(client: TestClient, make_otlp_request) -> None:
    """Calls report tokens. The root's tokens are ignored; its cost and cached tokens still count."""
    session_id = "per-field"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(5),
        spans=[
            _span(
                name="trial",
                span_id=1,
                session_id=session_id,
                input_tokens=300,
                output_tokens=30,
                cached_tokens=40,
                cost_usd=0.012,
            ),
            _span(
                name="chat-1",
                span_id=2,
                parent_span_id=1,
                session_id=session_id,
                kind="LLM",
                input_tokens=100,
                output_tokens=10,
            ),
            _span(
                name="chat-2",
                span_id=3,
                parent_span_id=1,
                session_id=session_id,
                kind="LLM",
                input_tokens=200,
                output_tokens=20,
            ),
        ],
    )

    trace = _trace(client, session_id)
    assert trace["input_tokens"] == 300
    assert trace["output_tokens"] == 30
    assert trace["cached_tokens"] == 40
    assert trace["cost_usd"] == pytest.approx(0.012)


def test_root_only_usage_counts_once(client: TestClient, make_otlp_request) -> None:
    session_id = "root-only"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(6),
        spans=[
            _span(name="trial", span_id=1, session_id=session_id, input_tokens=80, output_tokens=8, cost_usd=0.004),
            _span(name="chat", span_id=2, parent_span_id=1, session_id=session_id, kind="LLM"),
        ],
    )

    trace = _trace(client, session_id)
    assert trace["input_tokens"] == 80
    assert trace["output_tokens"] == 8
    assert trace["cost_usd"] == pytest.approx(0.004)


def test_agent_only_usage_counts_once(client: TestClient, make_otlp_request) -> None:
    session_id = "agent-only"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(7),
        spans=[
            _span(name="trial", span_id=1, session_id=session_id),
            _span(
                name="invoke_agent", span_id=2, parent_span_id=1, session_id=session_id, kind="AGENT", input_tokens=55
            ),
        ],
    )

    assert _trace(client, session_id)["input_tokens"] == 55
    assert _session(client, session_id)["input_tokens"] == 55


def test_unreported_field_stays_absent(client: TestClient, make_otlp_request) -> None:
    session_id = "missing-output"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(8),
        spans=[
            _span(name="chat", span_id=1, session_id=session_id, kind="LLM", input_tokens=12),
        ],
    )

    trace = _trace(client, session_id)
    session = _session(client, session_id)
    assert trace["input_tokens"] == 12
    assert trace.get("output_tokens") is None
    assert session.get("output_tokens") is None
    assert session.get("cost_usd") is None


def test_explicit_zero_cost_on_calls_blocks_the_root_total(client: TestClient, make_otlp_request) -> None:
    session_id = "zero-cost"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(9),
        spans=[
            _span(name="trial", span_id=1, session_id=session_id, cost_usd=0.012),
            _span(
                name="chat", span_id=2, parent_span_id=1, session_id=session_id, kind="LLM", input_tokens=10, cost_usd=0
            ),
        ],
    )

    trace = _trace(client, session_id)
    assert trace["input_tokens"] == 10
    assert trace["cost_usd"] == pytest.approx(0)


def test_mixed_session_sums_each_trace_on_its_own_tier(client: TestClient, make_otlp_request) -> None:
    """One trace has calls, the other only a root. The session adds those totals."""
    session_id = "mixed-session"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(10),
        spans=[
            _span(name="trial", span_id=1, session_id=session_id, input_tokens=300),
            _span(name="chat-1", span_id=2, parent_span_id=1, session_id=session_id, kind="LLM", input_tokens=100),
            _span(name="chat-2", span_id=3, parent_span_id=1, session_id=session_id, kind="LLM", input_tokens=200),
        ],
    )
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(11),
        spans=[
            _span(name="root-only", span_id=1, session_id=session_id, input_tokens=60),
        ],
    )

    assert _session(client, session_id)["input_tokens"] == 360


def test_republish_counts_the_latest_span_version_once(client: TestClient, make_otlp_request) -> None:
    session_id = "republish"
    trace_id = _trace_id(12)
    # Retention is 90 days, so a fixed historical start time would be dropped before the read.
    started_at = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    first = [
        _span(name="trial", span_id=1, session_id=session_id, input_tokens=100, start_time_unix_nano=started_at),
        _span(
            name="chat",
            span_id=2,
            parent_span_id=1,
            session_id=session_id,
            kind="LLM",
            input_tokens=100,
            start_time_unix_nano=started_at + 1,
        ),
    ]
    second = [
        _span(name="trial", span_id=1, session_id=session_id, input_tokens=150, start_time_unix_nano=started_at),
        _span(
            name="chat",
            span_id=2,
            parent_span_id=1,
            session_id=session_id,
            kind="LLM",
            input_tokens=150,
            start_time_unix_nano=started_at + 1,
        ),
    ]
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=trace_id, spans=first)
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=trace_id, spans=second)

    assert _trace(client, session_id)["input_tokens"] == 150
    assert _session(client, session_id)["input_tokens"] == 150


def test_evaluation_views_count_the_three_level_shape_once(client: TestClient, make_otlp_request) -> None:
    evaluation_name = "usage-rollup-eval"
    _create_evaluation(client, evaluation_name)
    session_id = "eval-trial"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(13),
        spans=[
            _span(
                name="trial",
                span_id=1,
                session_id=session_id,
                input_tokens=300,
                output_tokens=30,
                cost_usd=0.012,
                evaluation_name=evaluation_name,
                test_case_name="case-trial",
            ),
            _span(
                name="invoke_agent",
                span_id=2,
                parent_span_id=1,
                session_id=session_id,
                kind="AGENT",
                input_tokens=300,
                output_tokens=30,
            ),
            _span(
                name="chat-1",
                span_id=3,
                parent_span_id=2,
                session_id=session_id,
                kind="LLM",
                input_tokens=100,
                output_tokens=10,
            ),
            _span(
                name="chat-2",
                span_id=4,
                parent_span_id=2,
                session_id=session_id,
                kind="LLM",
                input_tokens=200,
                output_tokens=20,
            ),
        ],
    )

    listed = client.get(f"{EVALUATIONS}/{evaluation_name}/sessions")
    assert listed.status_code == 200, listed.text
    row = listed.json()["data"][0]
    assert row["input_tokens"] == 300
    assert row["output_tokens"] == 30
    assert row["cost_total_usd"] == pytest.approx(0.012)

    evaluation = client.get(f"{EVALUATIONS}/{evaluation_name}")
    assert evaluation.status_code == 200, evaluation.text
    body = evaluation.json()
    assert body["tokens"]["sum"] == pytest.approx(330)
    assert body["cost_usd"]["sum"] == pytest.approx(0.012)


def _trial_with_gateway_calls(session_id: str, *, evaluation_name: str | None = None) -> list[dict[str, Any]]:
    """An agent reports its two chats, and the gateway reports what the provider billed for each.

    The agent's own counts are off by a few tokens, as agents' often are; the gateway's are authoritative.
    """
    return [
        _span(name="trial", span_id=1, session_id=session_id, evaluation_name=evaluation_name, test_case_name="case"),
        _span(name="claude_code.interaction", span_id=2, parent_span_id=1, session_id=session_id, kind="AGENT"),
        _span(
            name="llm_request-1",
            span_id=3,
            parent_span_id=2,
            session_id=session_id,
            kind="LLM",
            input_tokens=98,
            output_tokens=9,
        ),
        _span(
            name="llm_request-2",
            span_id=4,
            parent_span_id=2,
            session_id=session_id,
            kind="LLM",
            input_tokens=197,
            output_tokens=21,
        ),
        _span(
            name="chat.completions",
            span_id=5,
            parent_span_id=1,
            session_id=session_id,
            operation="chat",
            input_tokens=100,
            output_tokens=10,
            cached_tokens=60,
            gateway_call=True,
        ),
        _span(
            name="chat.completions",
            span_id=6,
            parent_span_id=1,
            session_id=session_id,
            operation="chat",
            input_tokens=200,
            output_tokens=20,
            gateway_call=True,
        ),
    ]


def test_gateway_call_spans_count_a_calls_usage_once(client: TestClient, make_otlp_request) -> None:
    session_id = "gateway-trial"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(20),
        spans=_trial_with_gateway_calls(session_id),
    )

    trace = _trace(client, session_id)
    session = _session(client, session_id)
    for totals in (trace, session):
        assert totals["input_tokens"] == 300
        assert totals["output_tokens"] == 30
        assert totals["cached_tokens"] == 60


def test_the_agents_counts_still_count_when_the_gateway_reports_no_usage(client: TestClient, make_otlp_request) -> None:
    """A gateway span without a usage field (an aborted call) doesn't hide the agent's own count of it."""
    session_id = "gateway-no-usage"
    spans = [span for span in _trial_with_gateway_calls(session_id) if span["name"] != "chat.completions"]
    spans.append(_span(name="chat.completions", span_id=5, parent_span_id=1, session_id=session_id, gateway_call=True))
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(21), spans=spans)

    trace = _trace(client, session_id)
    assert trace["input_tokens"] == 295


def test_evaluation_views_count_gateway_calls_once(client: TestClient, make_otlp_request) -> None:
    evaluation_name = "gateway-usage-eval"
    _create_evaluation(client, evaluation_name)
    session_id = "gateway-eval-trial"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(22),
        spans=_trial_with_gateway_calls(session_id, evaluation_name=evaluation_name),
    )

    listed = client.get(f"{EVALUATIONS}/{evaluation_name}/sessions")
    assert listed.status_code == 200, listed.text
    row = listed.json()["data"][0]
    assert (row["input_tokens"], row["output_tokens"]) == (300, 30)
    evaluation = client.get(f"{EVALUATIONS}/{evaluation_name}")
    assert evaluation.status_code == 200, evaluation.text
    assert evaluation.json()["tokens"]["sum"] == pytest.approx(330)


def test_evaluation_session_token_sort_uses_deduped_totals(client: TestClient, make_otlp_request) -> None:
    """A parent total must not push a small call ahead of a larger one."""
    evaluation_name = "usage-rollup-sort"
    _create_evaluation(client, evaluation_name)
    _ingest(
        client,
        make_otlp_request,
        session_id="sort-small",
        trace_id=_trace_id(14),
        spans=[
            _span(
                name="trial",
                span_id=1,
                session_id="sort-small",
                input_tokens=1000,
                evaluation_name=evaluation_name,
                test_case_name="case-small",
            ),
            _span(
                name="chat",
                span_id=2,
                parent_span_id=1,
                session_id="sort-small",
                kind="LLM",
                input_tokens=10,
            ),
        ],
    )
    _ingest(
        client,
        make_otlp_request,
        session_id="sort-large",
        trace_id=_trace_id(15),
        spans=[
            _span(
                name="chat",
                span_id=1,
                session_id="sort-large",
                kind="LLM",
                input_tokens=100,
                evaluation_name=evaluation_name,
                test_case_name="case-large",
            ),
        ],
    )

    listed = client.get(
        f"{EVALUATIONS}/{evaluation_name}/sessions",
        params={"sort": "tokens", "page_size": 10},
    )
    assert listed.status_code == 200, listed.text
    assert [row["test_case_name"] for row in listed.json()["data"]] == ["case-small", "case-large"]


def _create_evaluation(client: TestClient, name: str) -> None:
    group = client.post(EXPERIMENTS, json={"name": f"{name}-group"})
    if group.status_code == 409:
        group = client.get(f"{EXPERIMENTS}/{name}-group")
    group.raise_for_status()
    created = client.post(
        EVALUATIONS,
        json={
            "name": name,
            "experiment_group_id": group.json()["id"],
            "dataset_name": f"{name}-dataset",
            "dataset_version": "v1",
        },
    )
    assert created.status_code == 201, created.text


def _ingest(
    client: TestClient,
    make_otlp_request,
    *,
    session_id: str,
    trace_id: str,
    spans: list[dict[str, Any]],
) -> None:
    response = client.post(
        OTLP_INGEST,
        content=make_otlp_request(spans, trace_id=trace_id),
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert response.status_code == 200, response.text
    assert all(span["attributes"]["gen_ai.conversation.id"] == session_id for span in spans)


def _trace(client: TestClient, session_id: str) -> dict[str, Any]:
    response = client.get(TRACES, params={"filter[session_id]": session_id, "page_size": 20})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert len(data) >= 1
    return data[0]


def _session(client: TestClient, session_id: str) -> dict[str, Any]:
    response = client.get(f"{SESSIONS}/{session_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _trace_metrics(client: TestClient, session_id: str) -> dict[str, Any]:
    response = client.get(
        TRACE_METRICS,
        params={"bucket": "total", "filter[session_id]": session_id},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert len(data) == 1
    return data[0]


def _span(
    *,
    name: str,
    span_id: int,
    session_id: str,
    parent_span_id: int | None = None,
    kind: str | None = None,
    operation: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_tokens: int | None = None,
    cost_usd: float | None = None,
    evaluation_name: str | None = None,
    test_case_name: str | None = None,
    start_time_unix_nano: int | None = None,
    gateway_call: bool = False,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {"gen_ai.conversation.id": session_id}
    if gateway_call:
        attributes["nemo.inference_gateway.call"] = True
    if kind is not None:
        attributes["openinference.span.kind"] = kind
    if operation is not None:
        attributes["gen_ai.operation.name"] = operation
    if input_tokens is not None:
        attributes["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attributes["gen_ai.usage.output_tokens"] = output_tokens
    if cached_tokens is not None:
        attributes["gen_ai.usage.cached_tokens"] = cached_tokens
    if cost_usd is not None:
        attributes["llm.cost.total"] = cost_usd
    if evaluation_name is not None:
        attributes["nemo.evaluation.name"] = evaluation_name
    if test_case_name is not None:
        attributes["nemo.test_case.name"] = test_case_name
    span: dict[str, Any] = {
        "name": name,
        "span_id": f"{span_id:016x}",
        "attributes": attributes,
    }
    if parent_span_id is not None:
        span["parent_span_id"] = f"{parent_span_id:016x}"
    if start_time_unix_nano is not None:
        span["start_time_unix_nano"] = start_time_unix_nano
        span["end_time_unix_nano"] = start_time_unix_nano + 1_000_000
    return span


def _trace_id(number: int) -> str:
    return f"{number:032x}"
