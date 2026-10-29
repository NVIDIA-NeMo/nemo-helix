# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Usage rollups count each model call once when parent spans repeat it."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nemo_helix_plugin.intake.span_attributes import GATEWAY_CALL_ATTRIBUTE, USAGE_SUPERSEDED_ATTRIBUTE

OTLP_INGEST = "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces"
TRACES = "/apis/intake/v2/workspaces/default/traces"
TRACE_METRICS = "/apis/intake/v2/workspaces/default/traces/metrics"
SESSIONS = "/apis/intake/v2/workspaces/default/sessions"
EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"
EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"
ATIF_INGEST = "/apis/intake/v2/workspaces/default/ingest/atif"
DIRECT_INGEST = "/apis/intake/v2/workspaces/default/ingest/spans"
SPANS = "/apis/intake/v2/workspaces/default/spans"


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


def _trial_with_linked_gateway_calls(session_id: str, *, evaluation_name: str | None = None) -> list[dict[str, Any]]:
    """An agent that sends a ``traceparent``: the gateway's span for each call sits beneath the agent's own.

    The agent's own counts are off by a few tokens, as agents' often are; the gateway's are what the provider billed.
    """
    return [
        _span(name="trial", span_id=1, session_id=session_id, evaluation_name=evaluation_name, test_case_name="case"),
        _span(name="codex-turn", span_id=2, parent_span_id=1, session_id=session_id, kind="AGENT"),
        _span(
            name="openai.responses",
            span_id=3,
            parent_span_id=2,
            session_id=session_id,
            kind="LLM",
            input_tokens=98,
            output_tokens=9,
        ),
        _span(
            name="openai.responses",
            span_id=4,
            parent_span_id=2,
            session_id=session_id,
            kind="LLM",
            input_tokens=197,
            output_tokens=21,
        ),
        _span(
            name="responses",
            span_id=5,
            parent_span_id=3,
            session_id=session_id,
            input_tokens=100,
            output_tokens=10,
            cached_tokens=60,
            gateway_call=True,
        ),
        _span(
            name="responses",
            span_id=6,
            parent_span_id=4,
            session_id=session_id,
            input_tokens=200,
            output_tokens=20,
            gateway_call=True,
        ),
    ]


def test_gateway_call_spans_in_the_agents_trace_count_each_call_once(client: TestClient, make_otlp_request) -> None:
    session_id = "gateway-linked"
    _ingest(
        client,
        make_otlp_request,
        session_id=session_id,
        trace_id=_trace_id(20),
        spans=_trial_with_linked_gateway_calls(session_id),
    )

    for totals in (_trace(client, session_id), _session(client, session_id)):
        assert (totals["input_tokens"], totals["output_tokens"], totals["cached_tokens"]) == (300, 30, 60)


def test_the_agents_counts_still_count_when_the_gateway_reports_no_usage(client: TestClient, make_otlp_request) -> None:
    """A gateway span without a usage field (an aborted call) doesn't hide the agent's own count of it."""
    session_id = "gateway-no-usage"
    spans = [span for span in _trial_with_linked_gateway_calls(session_id) if span["name"] != "responses"]
    spans.append(_span(name="responses", span_id=5, parent_span_id=3, session_id=session_id, gateway_call=True))
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(21), spans=spans)

    assert _trace(client, session_id)["input_tokens"] == 295


def test_a_gateway_call_without_usage_leaves_the_agents_counts_for_the_whole_trace(
    client: TestClient, make_otlp_request
) -> None:
    """Counting only the gateway's reported call would drop the other; counting both would double the first."""
    session_id = "gateway-partial-linked"
    spans = _trial_with_linked_gateway_calls(session_id)
    del spans[-1]["attributes"]["gen_ai.usage.input_tokens"], spans[-1]["attributes"]["gen_ai.usage.output_tokens"]
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(23), spans=spans)

    trace = _trace(client, session_id)
    assert (trace["input_tokens"], trace["output_tokens"]) == (295, 30)


def test_a_marked_trajectory_still_counts_when_the_gateway_missed_a_calls_usage(
    client: TestClient, make_otlp_request
) -> None:
    session_id = "gateway-partial-marked"
    trial, agent = _gateway_and_agent_traces(session_id, marked=True)
    del trial[-1]["attributes"]["gen_ai.usage.input_tokens"], trial[-1]["attributes"]["gen_ai.usage.output_tokens"]
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(24), spans=trial)
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(25), spans=agent)

    session = _session(client, session_id)
    assert (session["input_tokens"], session["output_tokens"]) == (295, 30)


def test_a_gateway_span_reporting_zero_tokens_still_wins(client: TestClient, make_otlp_request) -> None:
    """A stored 0 is a report: the provider billed nothing, so the agent's estimate doesn't stand in for it."""
    session_id = "gateway-zero"
    spans = [span for span in _trial_with_linked_gateway_calls(session_id) if span["name"] != "responses"]
    spans.append(
        _span(name="responses", span_id=5, parent_span_id=3, session_id=session_id, input_tokens=0, gateway_call=True)
    )
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(22), spans=spans)

    assert _trace(client, session_id)["input_tokens"] == 0


def _gateway_and_agent_traces(
    session_id: str,
    *,
    marked: bool,
    evaluation_name: str | None = None,
    agent_cost_usd: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A trial whose agent sent no ``traceparent``: two traces of one session.

    The trial trace holds the gateway's spans for the agent's two calls. The agent's published trajectory is a trace
    of its own that reports the same two calls again, marked as superseded when the publisher knows the gateway
    recorded them.
    """
    trial = [
        _span(name="trial", span_id=1, session_id=session_id, evaluation_name=evaluation_name, test_case_name="case"),
        _span(
            name="responses",
            span_id=2,
            parent_span_id=1,
            session_id=session_id,
            input_tokens=100,
            output_tokens=10,
            gateway_call=True,
        ),
        _span(
            name="responses",
            span_id=3,
            parent_span_id=1,
            session_id=session_id,
            input_tokens=200,
            output_tokens=20,
            gateway_call=True,
        ),
    ]
    agent = [
        _span(
            name="invoke_agent",
            span_id=11,
            session_id=session_id,
            kind="AGENT",
            evaluation_name=evaluation_name,
            test_case_name="case",
            cost_usd=agent_cost_usd,
            usage_superseded=marked,
        ),
        _span(
            name="chat",
            span_id=12,
            parent_span_id=11,
            session_id=session_id,
            kind="LLM",
            input_tokens=98,
            output_tokens=9,
        ),
        _span(
            name="chat",
            span_id=13,
            parent_span_id=11,
            session_id=session_id,
            kind="LLM",
            input_tokens=197,
            output_tokens=21,
        ),
    ]
    return trial, agent


def _ingest_both(client: TestClient, make_otlp_request, session_id: str, number: int, **kwargs: Any) -> None:
    trial, agent = _gateway_and_agent_traces(session_id, **kwargs)
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(number), spans=trial)
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(number + 1), spans=agent)


def test_a_marked_trajectory_doesnt_add_the_calls_the_gateway_recorded(client: TestClient, make_otlp_request) -> None:
    session_id = "gateway-marked"
    _ingest_both(client, make_otlp_request, session_id, 30, marked=True)

    session = _session(client, session_id)
    assert (session["input_tokens"], session["output_tokens"], session["trace_count"]) == (300, 30, 2)
    metrics = _trace_metrics(client, session_id)
    assert (metrics["run_count"], metrics["input_tokens"]["sum"], metrics["output_tokens"]["sum"]) == (2, 300, 30)


def test_an_unmarked_trajectory_still_adds_its_usage(client: TestClient, make_otlp_request) -> None:
    """Without the mark Intake can't tell a repeat from other calls, so a session adds up every trace."""
    session_id = "gateway-unmarked"
    _ingest_both(client, make_otlp_request, session_id, 32, marked=False)

    assert _session(client, session_id)["input_tokens"] == 595
    assert _trace_metrics(client, session_id)["input_tokens"]["sum"] == 595


def test_a_marked_trajectory_keeps_its_own_totals_for_inspection(client: TestClient, make_otlp_request) -> None:
    session_id = "gateway-marked-trace"
    _ingest_both(client, make_otlp_request, session_id, 34, marked=True)

    traces = client.get(TRACES, params={"filter[session_id]": session_id})
    assert traces.status_code == 200, traces.text
    assert sorted(trace["input_tokens"] for trace in traces.json()["data"]) == [295, 300]


def test_a_metric_the_gateway_doesnt_report_still_counts_from_a_marked_trajectory(
    client: TestClient, make_otlp_request
) -> None:
    session_id = "gateway-no-cost"
    _ingest_both(client, make_otlp_request, session_id, 36, marked=True, agent_cost_usd=0.02)

    session = _session(client, session_id)
    assert session["input_tokens"] == 300
    assert session["cost_usd"] == pytest.approx(0.02)
    metrics = _trace_metrics(client, session_id)
    assert metrics["input_tokens"]["sum"] == 300
    assert metrics["cost_usd"]["sum"] == pytest.approx(0.02)


def test_evaluation_views_count_a_marked_sessions_calls_once(client: TestClient, make_otlp_request) -> None:
    evaluation_name = "gateway-marked-eval"
    _create_evaluation(client, evaluation_name)
    _ingest_both(
        client, make_otlp_request, "gateway-marked-eval-trial", 38, marked=True, evaluation_name=evaluation_name
    )

    listed = client.get(f"{EVALUATIONS}/{evaluation_name}/sessions")
    assert listed.status_code == 200, listed.text
    [row] = listed.json()["data"]
    assert (row["input_tokens"], row["output_tokens"]) == (300, 30)
    by_tokens = client.get(f"{EVALUATIONS}/{evaluation_name}/sessions", params={"sort": "-tokens"})
    assert by_tokens.status_code == 200, by_tokens.text
    assert by_tokens.json()["data"][0]["input_tokens"] == 300
    evaluation = client.get(f"{EVALUATIONS}/{evaluation_name}")
    assert evaluation.status_code == 200, evaluation.text
    assert evaluation.json()["tokens"]["sum"] == pytest.approx(330)


def test_a_marked_trace_with_its_own_gateway_calls_still_counts_them(client: TestClient, make_otlp_request) -> None:
    """A publisher marks every gateway-recorded trial; when the agent sent a traceparent the calls are in its trace."""
    session_id = "gateway-linked-marked"
    spans = _trial_with_linked_gateway_calls(session_id)
    spans[0]["attributes"][USAGE_SUPERSEDED_ATTRIBUTE] = True
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(42), spans=spans)

    assert _session(client, session_id)["input_tokens"] == 300


def test_evaluation_cost_counts_the_gateways_cost_over_a_marked_trajectorys(
    client: TestClient, make_otlp_request
) -> None:
    evaluation_name = "gateway-marked-cost-eval"
    _create_evaluation(client, evaluation_name)
    session_id = "gateway-marked-cost-trial"
    trial, agent = _gateway_and_agent_traces(
        session_id, marked=True, evaluation_name=evaluation_name, agent_cost_usd=0.05
    )
    for span in trial[1:]:
        span["attributes"]["llm.cost.total"] = 0.01
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(44), spans=trial)
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(45), spans=agent)

    evaluation = client.get(f"{EVALUATIONS}/{evaluation_name}")
    assert evaluation.status_code == 200, evaluation.text
    assert evaluation.json()["cost_usd"]["sum"] == pytest.approx(0.02)


def test_a_marked_atif_trajectory_doesnt_add_the_calls_the_gateway_recorded(
    client: TestClient, make_otlp_request
) -> None:
    session_id = "gateway-marked-atif"
    trial, _ = _gateway_and_agent_traces(session_id, marked=True)
    _ingest(client, make_otlp_request, session_id=session_id, trace_id=_trace_id(40), spans=trial)
    response = client.post(
        ATIF_INGEST,
        json={
            "schema_version": "ATIF-v1.7",
            "session_id": session_id,
            "agent": {"name": "codex", "version": "1"},
            "extra": {USAGE_SUPERSEDED_ATTRIBUTE: True},
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "hi",
                    "metrics": {"prompt_tokens": 98, "completion_tokens": 9},
                },
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "done",
                    "metrics": {"prompt_tokens": 197, "completion_tokens": 21},
                },
            ],
        },
    )
    assert response.status_code == 201, response.text

    session = _session(client, session_id)
    assert (session["input_tokens"], session["trace_count"]) == (300, 2)


def test_direct_span_ingest_stores_both_attributes_where_the_usage_rule_reads_them(client: TestClient) -> None:
    session_id = "gateway-direct"
    started = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

    def span(trace_id: str, span_id: str, **fields: Any) -> dict[str, Any]:
        return {"trace_id": trace_id, "span_id": span_id, "session_id": session_id, "started_at": started, **fields}

    response = client.post(
        DIRECT_INGEST,
        json={
            "source": "provider-store",
            "spans": [
                span("trial-trace", "trial", name="trial", kind="AGENT"),
                span(
                    "trial-trace",
                    "call",
                    parent_span_id="trial",
                    name="responses",
                    kind="LLM",
                    attributes={GATEWAY_CALL_ATTRIBUTE: True, "gen_ai.usage.input_tokens": 7},
                ),
                span("agent-trace", "agent", name="agent", kind="AGENT", attributes={USAGE_SUPERSEDED_ATTRIBUTE: True}),
                span(
                    "agent-trace",
                    "chat",
                    parent_span_id="agent",
                    name="chat",
                    kind="LLM",
                    attributes={"gen_ai.usage.input_tokens": 5},
                ),
            ],
        },
    )
    assert response.status_code == 201, response.text

    assert _session(client, session_id)["input_tokens"] == 7
    spans = client.get(SPANS, params={"filter[session_id]": session_id, "page_size": 10})
    assert spans.status_code == 200, spans.text
    raw = " ".join(row.get("raw_attributes") or "" for row in spans.json()["data"])
    assert GATEWAY_CALL_ATTRIBUTE not in raw and USAGE_SUPERSEDED_ATTRIBUTE not in raw


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
    usage_superseded: bool = False,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {"gen_ai.conversation.id": session_id}
    if gateway_call:
        attributes[GATEWAY_CALL_ATTRIBUTE] = True
    if usage_superseded:
        attributes[USAGE_SUPERSEDED_ATTRIBUTE] = True
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
