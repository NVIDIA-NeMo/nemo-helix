# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Gym rollout -> OTLP/JSON projection.

Imports nothing from the SDK but the subpackage under test, which is what keeps it liftable.
"""

from __future__ import annotations

from typing import Any

import pytest
from nhx_evals_sdk.ng_trajectory_otlp import rollout_to_resource_spans
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status


def _spans(record: dict[str, Any], *, rollout_id: str = "trial-1", **kwargs: Any) -> list[Span]:
    resource_spans = rollout_to_resource_spans(record, rollout_id=rollout_id, **kwargs)
    return list(resource_spans[0].scope_spans[0].spans)


def _attributes(span: Span) -> dict[str, Any]:
    """Attribute values unwrapped from their ``AnyValue`` oneof, so a test reads what was written."""
    values: dict[str, Any] = {}
    for attribute in span.attributes:
        # None only for an AnyValue with nothing set, which nothing here writes.
        field = attribute.value.WhichOneof("value")
        assert field is not None, f"{attribute.key} carries no value"
        values[attribute.key] = getattr(attribute.value, field)
    return values


def _kind(span: Span) -> str:
    return _attributes(span)["openinference.span.kind"]


def _message(text: str) -> dict[str, Any]:
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def test_a_rollout_becomes_an_agent_root_with_a_model_call_beneath_it() -> None:
    spans = _spans({"response": {"output_text": "hi"}}, task_id="task-1")

    assert [_kind(span) for span in spans] == ["AGENT", "LLM"]
    assert spans[0].name == "task-1"
    assert spans[0].parent_span_id == b""
    assert spans[1].parent_span_id == spans[0].span_id
    assert {span.trace_id for span in spans} == {spans[0].trace_id}


def test_the_root_carries_the_answer_a_content_metric_scores() -> None:
    spans = _spans({"response": {"output": [_message("The answer is 4.")]}})

    assert _attributes(spans[0])["output.value"] == "The answer is 4."


def test_an_output_of_only_tool_calls_is_not_an_answer() -> None:
    # A tool result is not the agent's answer; claiming otherwise would score the tool's output.
    record = {
        "response": {
            "output": [
                {"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "42"},
            ]
        }
    }

    assert "output.value" not in _attributes(_spans(record)[0])


def test_a_function_call_becomes_a_tool_span_paired_with_its_result() -> None:
    record = {
        "response": {
            "output": [
                {"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"q":"otlp"}'},
                {"type": "function_call_output", "call_id": "c1", "output": "found"},
                _message("done"),
            ]
        }
    }

    tool = next(span for span in _spans(record) if _kind(span) == "TOOL")

    assert tool.name == "search"
    assert _attributes(tool) == {
        "openinference.span.kind": "TOOL",
        "tool.name": "search",
        "tool_call.id": "c1",
        "input.value": '{"q":"otlp"}',
        "output.value": "found",
    }


def test_a_call_without_an_id_does_not_inherit_an_unkeyed_output() -> None:
    # Results are joined by call id. An output missing one used to be filed under ``None``, which a
    # call also missing one would then read back — reporting another tool's output as its own.
    record = {
        "response": {
            "output": [
                {"type": "function_call", "name": "search", "arguments": "{}"},
                {"type": "function_call_output", "output": "someone else's result"},
            ]
        }
    }

    tool = next(span for span in _spans(record) if _kind(span) == "TOOL")

    assert "output.value" not in _attributes(tool)
    assert "tool_call.id" not in _attributes(tool)


def test_a_non_string_call_id_is_not_used_as_a_join_key() -> None:
    # The id attribute is already guarded on ``str``; the join has to agree, or a call keeps an
    # id too malformed to report while still using it to claim a result.
    record = {
        "response": {
            "output": [
                {"type": "function_call", "call_id": 7, "name": "search", "arguments": "{}"},
                {"type": "function_call_output", "call_id": 7, "output": "found"},
            ]
        }
    }

    tool = next(span for span in _spans(record) if _kind(span) == "TOOL")

    assert "output.value" not in _attributes(tool)


def test_a_function_call_without_a_name_is_dropped_rather_than_named_for_us() -> None:
    record = {"response": {"output": [{"type": "function_call", "call_id": "c1", "arguments": "{}"}]}}

    assert [_kind(span) for span in _spans(record)] == ["AGENT", "LLM"]


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"input_tokens": 12, "output_tokens": 3}, (12, 3)),
        ({"prompt_tokens": 12, "completion_tokens": 3}, (12, 3)),
        ({"input_tokens": 0, "output_tokens": 0}, (0, 0)),
    ],
)
def test_token_usage_is_read_in_either_vocabulary(usage: dict[str, Any], expected: tuple[int, int]) -> None:
    # Gym's model servers speak the Responses API's names or Chat Completions', by adapter.
    llm = _spans({"response": {"usage": usage}})[1]
    attributes = _attributes(llm)

    assert (int(attributes["gen_ai.usage.input_tokens"]), int(attributes["gen_ai.usage.output_tokens"])) == expected


def test_absent_usage_is_left_unreported_rather_than_called_zero() -> None:
    attributes = _attributes(_spans({"response": {}})[1])

    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.output_tokens" not in attributes


@pytest.mark.parametrize(
    ("value", "expected_field"),
    [(0.5, "double_value"), (1, "int_value"), (True, "bool_value")],
)
def test_an_attribute_lands_in_the_any_value_field_matching_its_type(value: Any, expected_field: str) -> None:
    # bool is an int subclass, so an ordering slip would write True into int_value.
    spans = _spans({"reward": value, "response": {}})
    reward = next((attribute for attribute in spans[0].attributes if attribute.key == "nemo.gym.reward"), None)

    if isinstance(value, bool):
        # A reward is a measurement, not a flag: a bool is refused rather than coerced to 1.
        assert reward is None
    else:
        assert reward is not None
        assert reward.value.WhichOneof("value") == expected_field


def test_ids_are_stable_across_conversions_and_distinct_across_rollouts() -> None:
    # Re-publishing a trial must replace its spans, which needs the ids to be a function of the
    # rollout rather than of when it was converted -- and two rollouts must never share them, or
    # a consumer keyed on span identity stores only the last.
    first = _spans({"response": {}}, rollout_id="task-1:0", task_id="task-1")
    again = _spans({"response": {}}, rollout_id="task-1:0", task_id="task-1")
    retry = _spans({"response": {}}, rollout_id="task-1:1", task_id="task-1")

    assert [span.span_id for span in first] == [span.span_id for span in again]
    assert first[0].trace_id != retry[0].trace_id
    assert {span.span_id for span in first}.isdisjoint({span.span_id for span in retry})


def test_an_answer_recorded_as_a_bare_string_is_still_the_answer() -> None:
    # Not every Gym adapter writes the Responses API payload; some record the answer directly.
    spans = _spans({"response": "Lyon"})

    assert _attributes(spans[0])["output.value"] == "Lyon"


def _capture(**overrides: Any) -> dict[str, Any]:
    call = {
        "model_call_id": "c0",
        "model_ref": {"type": "responses_api_models", "name": "policy_model"},
        "status_code": 200,
        "started_at": 1788534870.5,
        "completed_at": 1788534870.75,
        "latency_ttft_ms": 12.5,
        "response": {"usage": {"input_tokens": 5, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 3}}},
    }
    return {**call, **overrides}


def test_each_captured_model_call_becomes_its_own_timed_span() -> None:
    # The rollout record reports one usage block summed over the turn, so per-call tokens and
    # timing exist in the capture or nowhere.
    spans = _spans({"response": {}}, model_calls=[_capture(), _capture(model_call_id="c1")])

    llm = [span for span in spans if _kind(span) == "LLM"]
    assert len(llm) == 2
    assert llm[0].start_time_unix_nano == 1788534870_500_000_000
    assert llm[0].end_time_unix_nano == 1788534870_750_000_000
    assert llm[0].span_id != llm[1].span_id


def test_a_captured_call_reports_its_own_tokens_including_the_cached_ones() -> None:
    attributes = _attributes(
        [span for span in _spans({"response": {}}, model_calls=[_capture()]) if _kind(span) == "LLM"][0]
    )

    assert attributes["gen_ai.usage.input_tokens"] == 5
    assert attributes["gen_ai.usage.output_tokens"] == 2
    assert attributes["gen_ai.usage.cache_read.input_tokens"] == 3
    assert attributes["nemo.gym.latency_ttft_ms"] == pytest.approx(12.5)


def test_a_failed_model_call_marks_its_span_an_error() -> None:
    span = [
        s
        for s in _spans({"response": {}}, model_calls=[_capture(status_code=503, error_category="upstream")])
        if _kind(s) == "LLM"
    ][0]

    assert span.status.code == Status.STATUS_CODE_ERROR
    assert span.status.message == "upstream"


def test_without_a_capture_the_rollout_still_yields_one_untimed_model_call() -> None:
    # Observability is what produces captures; a run without it must still yield a usable trace.
    llm = [
        span
        for span in _spans({"response": {"usage": {"input_tokens": 9, "output_tokens": 1}}})
        if _kind(span) == "LLM"
    ]

    assert len(llm) == 1
    assert llm[0].start_time_unix_nano == 0
    assert _attributes(llm[0])["gen_ai.usage.input_tokens"] == 9


@pytest.mark.parametrize("timestamp", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_capture_timestamp_costs_the_timestamp_not_the_trace(timestamp: float) -> None:
    # json.loads accepts Infinity and NaN, and int(inf) raises OverflowError -- which the caller's
    # (TypeError, ValueError) guard does not catch, so it would abort collecting the whole rollout.
    spans = _spans({"response": {}}, model_calls=[_capture(started_at=timestamp, completed_at=timestamp)])

    llm = [span for span in spans if _kind(span) == "LLM"]
    assert len(llm) == 1
    assert llm[0].start_time_unix_nano == 0


def test_a_message_whose_content_is_plain_text_is_still_the_answer() -> None:
    # The Responses API allows content to be a bare string rather than a list of parts.
    spans = _spans({"response": {"output": [{"type": "message", "content": "Four."}]}})

    assert _attributes(spans[0])["output.value"] == "Four."


def test_an_answer_is_taken_from_the_last_message_that_has_one() -> None:
    # A trailing item with unusable content must not mask an earlier real answer.
    record = {"response": {"output": [_message("Four."), {"type": "message", "content": {"unusable": True}}]}}

    assert _attributes(_spans(record)[0])["output.value"] == "Four."


def test_a_prompt_falls_back_to_instructions_when_input_is_empty() -> None:
    # Gym rollouts that carry only a system prompt leave `input` empty rather than absent.
    record = {"responses_create_params": {"input": [], "instructions": "Answer briefly."}, "response": {}}

    assert _attributes(_spans(record)[0])["input.value"] == "Answer briefly."


def test_a_capture_reporting_no_usable_token_counts_omits_them() -> None:
    # An absent count and a zero count are different claims; only the zero should reach the span.
    spans = _spans({"response": {}}, model_calls=[_capture(response={"usage": {"something_else": 3}})])

    attributes = _attributes([span for span in spans if _kind(span) == "LLM"][0])
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.output_tokens" not in attributes


def test_the_answer_is_the_agent_s_last_message_not_its_first() -> None:
    # A multi-turn rollout ends with the answer; taking the first message would score the agent's
    # opening move, which is often a plan or a restatement of the question.
    record = {"response": {"output": [_message("Let me check."), _message("Four.")]}}

    assert _attributes(_spans(record)[0])["output.value"] == "Four."
