# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
from nemo_tool_calling_plugin.scoring import (
    RunRecord,
    ToolCall,
    annotate_schema_errors,
    calls_from_messages,
    calls_from_relay_records,
    looks_like_error,
    loose_schema_findings,
    metrics,
    rejected_examples,
    rejection_reasons,
)

TOOL = "generate_data_designer_job_request"
REJECTION = (
    "TOOL CALL REJECTED: 422 Unprocessable Entity:\n"
    "- columns[0] (review): unknown field 'sampler_type '\n"
    "- model_configs[0]: provider 'openai' is not registered in workspace 'default'"
)
ACCEPTED = '{"status": "accepted", "name": "reviews", "num_records": 10, "columns": ["review"], "errors": []}'


def _start(uuid: str, name: str, args: Any, call_id: str | None = None) -> dict[str, Any]:
    record = {"kind": "scope", "scope_category": "start", "category": "tool", "name": name, "uuid": uuid, "data": args}
    if call_id:
        record["category_profile"] = {"tool_call_id": call_id}
    return record


def _end(uuid: str, name: str, content: Any, *, is_error: bool = False, call_id: str | None = None) -> dict[str, Any]:
    record = {
        "kind": "scope",
        "scope_category": "end",
        "category": "tool",
        "name": name,
        "uuid": uuid,
        "data": {"schema": "relay.tool_result", "content": content, "is_error": is_error},
    }
    if call_id:
        record["category_profile"] = {"tool_call_id": call_id}
    return record


def _call(result: str = ACCEPTED, *, tool: str = TOOL, args: Any = None, is_error: bool = False) -> ToolCall:
    return ToolCall(tool=tool, arguments=args or {"job_request": {}}, result_text=result, is_error=is_error)


def test_relay_records_pair_start_and_end_into_calls() -> None:
    records = [
        {"kind": "scope", "scope_category": "start", "category": "agent", "name": "agent", "uuid": "a"},
        _start("t1", TOOL, {"job_request": {"name": "x"}}, call_id="c1"),
        _end("t1", TOOL, [{"type": "text", "text": REJECTION}], call_id="c1"),
        _start("t2", "write_todos", {"todos": []}),
        _end("t2", "write_todos", "ok"),
        _start("t3", TOOL, {"job_request": {"name": "y"}}),
        _end("t3", TOOL, ACCEPTED),
    ]

    calls = calls_from_relay_records(records)

    assert [(call.tool, call.rejected) for call in calls] == [(TOOL, True), ("write_todos", False), (TOOL, False)]
    assert calls[0].arguments == {"job_request": {"name": "x"}}
    assert calls[0].reasons[0] == "columns[0] (review): unknown field 'sampler_type '"


def test_relay_mcp_error_result_is_a_tool_error() -> None:
    calls = calls_from_relay_records([_start("t", TOOL, {}), _end("t", TOOL, "boom", is_error=True)])

    assert calls[0].signals == ["tool_error"]


def test_messages_fallback_pairs_tool_calls_with_tool_messages() -> None:
    messages = [
        {"role": "human", "content": "make data"},
        {"role": "ai", "content": "", "tool_calls": [{"name": TOOL, "args": {"a": 1}, "id": "c1"}]},
        {"role": "tool", "content": REJECTION},
        {"role": "ai", "content": "", "tool_calls": [{"name": TOOL, "args": {"a": 2}, "id": "c2"}]},
        {"role": "tool", "content": ACCEPTED},
        {"role": "ai", "content": "Done."},
    ]

    calls = calls_from_messages(messages)

    assert [(call.arguments, call.rejected) for call in calls] == [({"a": 1}, True), ({"a": 2}, False)]


def test_schema_invalid_signal_uses_the_advertised_schema() -> None:
    schema = {
        "type": "object",
        "required": ["job_request"],
        "properties": {"job_request": {"type": "object", "additionalProperties": False, "properties": {}}},
    }
    run = RunRecord("p", 1, [_call(args={"job_request": {"model_configs": []}})])

    annotate_schema_errors([run], {TOOL: schema})

    call = run.calls[0]
    assert call.signals == ["schema_invalid"]
    assert call.rejected
    assert call.reasons == call.schema_errors
    assert call.schema_errors[0].startswith("job_request: Additional properties are not allowed")


def test_tool_rejection_text_wins_over_schema_errors_for_reasons() -> None:
    call = _call(REJECTION)
    call.schema_errors = ["job_request: something"]

    assert call.signals == ["tool_error", "schema_invalid"]
    assert call.reasons[0].startswith("columns[0]")


@pytest.mark.parametrize(
    "text",
    [
        "Error: connection refused",
        "Traceback (most recent call last):\n  File ...",
        "Invalid argument: num_records",
        "upstream returned HTTP 503",
        "404 Not Found",
        '{"error": "bad request"}',
        "ValueError: something. Exception: wrapped",
    ],
)
def test_error_text_heuristic_matches_error_shapes(text: str) -> None:
    assert looks_like_error(text)
    assert _call(text).signals == ["error_text"]


@pytest.mark.parametrize(
    "text",
    [
        ACCEPTED,
        '{"status": "ok", "error": null, "exception": ""}',
        "Created 3 rows; 0 invalid_rows found.",
        "The error budget is fine and the job is queued.",
        "Saved 404 records.",
        "",
    ],
)
def test_error_text_heuristic_ignores_successful_results(text: str) -> None:
    assert not looks_like_error(text)
    assert _call(text).signals == []


def test_error_text_is_skipped_when_the_tool_already_reported_an_error() -> None:
    assert _call("TOOL CALL REJECTED: Error: invalid value").signals == ["tool_error"]


def test_retry_loop_counts_a_retry_after_a_bad_call() -> None:
    run = RunRecord("p", 1, [_call(REJECTION), _call()])

    assert run.retry_loop(call_budget=3)


def test_retry_loop_counts_runs_over_the_call_budget() -> None:
    run = RunRecord("p", 1, [_call(), _call(tool="other"), _call(), _call(tool="other")])

    assert run.retry_loop(call_budget=3)
    assert not run.retry_loop(call_budget=4)


def test_no_retry_loop_for_a_clean_run() -> None:
    assert not RunRecord("p", 1, [_call()]).retry_loop(call_budget=3)


def test_metrics_match_the_contract() -> None:
    runs = [
        RunRecord("a", 1, [_call(REJECTION), _call(REJECTION), _call()]),
        RunRecord("a", 2, [_call()]),
        RunRecord("b", 1, [_call(REJECTION)], error="deepagents_recursion_limit_reached: limit"),
        RunRecord("b", 2, [_call(), _call("Error: tool crashed")]),
    ]

    assert metrics(runs, call_budget=3) == {
        "runs": 4,
        "runs_with_errors": 1,
        "tool_calls": 7,
        "rejected_calls": 4,
        "tool_error_rate": 0.571,
        "signals": {"tool_error": 3, "schema_invalid": 0, "error_text": 1, "retry_loop_runs": 1},
        "first_call_accepted_rate": 0.5,
        "task_success_rate": 0.5,
        "avg_calls_per_run": 1.75,
    }


def test_rejection_reasons_and_examples() -> None:
    runs = [RunRecord("a", 1, [_call(REJECTION, args={"job_request": {"x": "y" * 1000}})])]

    assert rejection_reasons(runs) == {"corrupted or unknown key": 1, "model/provider not on platform": 1}
    [example] = rejected_examples(runs)
    assert example["prompt_id"] == "a"
    assert example["signals"] == ["tool_error"]
    assert example["arguments_excerpt"].endswith("…")
    assert "unknown field" in example["rejection"]
    assert example["schema_errors"] == []


def test_loose_schema_finding_when_the_schema_passes_calls_the_tool_rejects() -> None:
    runs = [RunRecord("a", 1, [_call(REJECTION), _call(REJECTION), _call()])]

    [finding] = loose_schema_findings(runs)

    assert finding["title"] == "Schema too loose to catch its own errors"
    assert finding["severity"] == "high"
    assert finding["source"] == "schema-lint"
    assert "2 of 2" in finding["detail"]
