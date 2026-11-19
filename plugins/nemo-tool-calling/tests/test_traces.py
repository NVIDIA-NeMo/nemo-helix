# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from nemo_helix_plugin.intake.types import Span
from nemo_tool_calling_plugin.traces import (
    NoTracedSessionsError,
    TracedSession,
    TraceWindow,
    call_from_span,
    first_user_message,
    group_sessions,
    load_traced_baseline,
    replay_prompts,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "intake-spans.json").read_text())
TOOL = "generate_data_designer_job_request"
SESSION = FIXTURES["session_tool_spans"][0]["session_id"]
WINDOW = TraceWindow(since=datetime(2026, 11, 18, tzinfo=UTC), until=datetime(2026, 11, 20, tzinfo=UTC))


def _span(raw: dict[str, Any], **overrides: Any) -> Span:
    return Span.model_validate({**raw, **overrides})


def _session_spans() -> list[Span]:
    return [_span(raw) for raw in FIXTURES["session_tool_spans"]]


def test_execute_tool_span_parses_arguments_and_rejection() -> None:
    call = call_from_span(_session_spans()[0])

    assert call is not None
    assert call.tool == TOOL
    assert isinstance(call.arguments, dict) and "job_request" in call.arguments
    assert call.result_text.startswith("TOOL CALL REJECTED: 422 Unprocessable Entity")
    assert call.signals == ["tool_error"]


def test_accepted_execute_tool_span_has_no_signals() -> None:
    call = call_from_span(_session_spans()[-1])

    assert call is not None
    assert json.loads(call.result_text)["status"] == "accepted"
    assert call.signals == []


def test_error_status_span_uses_its_error_message() -> None:
    call = call_from_span(_span(FIXTURES["error_status_tool_span"]))

    assert call is not None
    assert call.is_error
    assert "422 Unprocessable Entity" in call.result_text
    assert call.reasons[0].startswith("columns[")


def test_relay_duplicate_span_is_skipped() -> None:
    relay = _span(FIXTURES["relay_tool_span"])

    assert relay.name == TOOL
    assert call_from_span(relay) is None
    assert group_sessions([relay]) == []


def test_group_sessions_orders_calls_and_newest_session_first() -> None:
    spans = list(reversed(_session_spans()))
    older = _span(FIXTURES["error_status_tool_span"], session_id="session-older", started_at="2026-11-01T00:00:00")

    sessions = group_sessions([*spans, older, _span(FIXTURES["relay_tool_span"])])

    assert [session.session_id for session in sessions] == [SESSION, "session-older"]
    assert [call.rejected for call in sessions[0].calls] == [True, True, True, False]


def test_first_user_message_reads_genai_parts_from_the_chat_span() -> None:
    llm = FIXTURES["session_first_llm_span"]

    assert llm["name"].startswith("chat ")
    assert first_user_message(llm["input"]) == (
        "100 product reviews for electronics with a star rating, a sentiment label and the review text"
    )


@pytest.mark.parametrize(
    "llm_input",
    [None, "", "not json", "[]", json.dumps([{"role": "system", "parts": [{"type": "text", "content": "x"}]}])],
)
def test_first_user_message_without_a_user_turn(llm_input: str | None) -> None:
    assert first_user_message(llm_input) is None


def test_first_user_message_reads_openai_messages() -> None:
    messages = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "make orders"}]}

    assert first_user_message(json.dumps(messages)) == "make orders"


def _session(session_id: str, prompt: str | None) -> TracedSession:
    return TracedSession(session_id=session_id, started_at=datetime(2026, 11, 19, tzinfo=UTC), prompt=prompt)


def test_replay_prompts_dedupes_ranks_and_caps() -> None:
    sessions = [
        _session("s1", "Orders dataset with a total"),
        _session("s2", "100 product reviews for electronics."),
        _session("s3", "100 product reviews for  electronics"),
        _session("s4", "Support tickets"),
        _session("s5", None),
    ]

    prompts, ids = replay_prompts(sessions, limit=2)

    assert [prompt.id for prompt in prompts] == ["100-product-reviews-for-electronics", "orders-dataset-with-a-total"]
    assert prompts[0].prompt == "100 product reviews for electronics."
    assert ids["s3"] == ids["s2"] == "100-product-reviews-for-electronics"
    assert ids["s4"] == "support-tickets"
    assert ids["s5"] == "s5"


class FakeSpanSource:
    def __init__(self, spans: list[Span], llm_inputs: dict[str, str]) -> None:
        self._spans = spans
        self._llm_inputs = llm_inputs
        self.queries: list[dict[str, Any]] = []

    def tool_spans(self, *, agent_name: str, tool_names: Sequence[str], window: TraceWindow) -> list[Span]:
        self.queries.append({"agent_name": agent_name, "tool_names": list(tool_names), "window": window})
        return self._spans

    def first_llm_input(self, session_id: str) -> str | None:
        return self._llm_inputs.get(session_id)


def test_load_traced_baseline_scores_sessions_and_collects_prompts() -> None:
    source = FakeSpanSource(_session_spans(), {SESSION: FIXTURES["session_first_llm_span"]["input"]})

    traced = load_traced_baseline(
        source,
        agent_name="data-designer-toolcall",
        tool_names=[TOOL],
        window=WINDOW,
        max_sessions=50,
        max_replay_prompts=5,
    )

    assert source.queries == [{"agent_name": "data-designer-toolcall", "tool_names": [TOOL], "window": WINDOW}]
    assert [prompt.id for prompt in traced.prompts] == ["100-product-reviews-for-electronics-with"]
    [run] = traced.runs
    assert run.prompt_id == "100-product-reviews-for-electronics-with"
    assert len(run.calls) == 4
    assert run.retry_loop(call_budget=3)


def test_load_traced_baseline_keeps_the_newest_sessions() -> None:
    spans = [
        _span(FIXTURES["error_status_tool_span"], session_id=f"s{day}", started_at=f"2026-11-{day:02d}T00:00:00")
        for day in (10, 12, 11)
    ]

    traced = load_traced_baseline(
        FakeSpanSource(spans, {}),
        agent_name="a",
        tool_names=[],
        window=WINDOW,
        max_sessions=2,
        max_replay_prompts=5,
    )

    assert [session.session_id for session in traced.sessions] == ["s12", "s11"]
    assert traced.prompts == []


def test_no_traced_sessions_tells_the_user_what_to_do() -> None:
    with pytest.raises(NoTracedSessionsError, match="Chat with the agent first.*pass a prompt dataset"):
        load_traced_baseline(
            FakeSpanSource([_span(FIXTURES["relay_tool_span"])], {}),
            agent_name="data-designer-toolcall",
            tool_names=[TOOL],
            window=WINDOW,
            max_sessions=50,
            max_replay_prompts=5,
        )


def test_trace_window_defaults_to_hours_before_until() -> None:
    until = datetime(2026, 11, 19, 12, tzinfo=UTC)

    window = TraceWindow.resolve(since=None, until=until, hours=6)

    assert window.since == until - timedelta(hours=6)
    assert window.to_json() == {"since": "2026-11-19T06:00:00Z", "until": "2026-11-19T12:00:00Z"}
