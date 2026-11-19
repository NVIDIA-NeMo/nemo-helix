# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline config models, static checks and runner helpers."""

from __future__ import annotations

import re
from typing import Any

import pytest
from nemo_helix_plugin.agents.pipelines import (
    PIPELINE_MAX_STEPS,
    PipelineConfig,
    PipelineInline,
    PipelineStepConfig,
    collect_static_issues,
    ensure_runnable,
    parse_step_response,
    render_step,
    resolve_run_inputs,
    should_run,
    step_input_template,
)
from nemo_helix_plugin.agents.types import CreatePipelineConfigRequest, UpdatePipelineConfigRequest
from pydantic import ValidationError


def _cfg(*steps: dict[str, Any], inputs: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "name": "p1",
        "inputs": {"input": {}} if inputs is None else inputs,
        "steps": list(steps),
        **extra,
    }


def _errors(data: dict[str, Any]) -> list[str]:
    body = {key: value for key, value in data.items() if key != "name"}
    return [f"{i.location}: {i.message}" for i in collect_static_issues(PipelineInline.model_validate(body))]


# ---------------------------------------------------------------------------
# Valid configs and defaults
# ---------------------------------------------------------------------------


def test_minimal_chain_is_valid_and_uses_defaults() -> None:
    pipeline = PipelineConfig.model_validate(_cfg({"name": "a", "agent": "x"}, {"name": "b", "agent": "ws/y"}))
    assert step_input_template(pipeline, 0) == "{{ inputs.input }}"
    assert step_input_template(pipeline, 1) == "{{ previous.response }}"


def test_json_references_are_valid_with_response_format_json() -> None:
    data = _cfg(
        {"name": "triage", "agent": "x", "response_format": "json"},
        {
            "name": "reply",
            "agent": "y",
            "input": "{{ previous.response.verdict }} {{ input }}",
            "context": {
                "indicators": "{{ steps.triage.response.indicators }}",
                "n": [{"o": "{{ steps.triage.output.a[0] }}"}],
            },
            "retries": 2,
        },
        {
            "name": "last",
            "agent": "z",
            "input": "{{ inputs.input }}",
            "when": {"path": "steps.triage.response.verdict", "in": ["x"]},
        },
    )
    assert _errors(data) == []


def test_default_input_is_the_single_text_input() -> None:
    inputs = {"task": {}, "review": {"type": "boolean", "default": True}}
    pipeline = PipelineConfig.model_validate(_cfg({"name": "a", "agent": "x"}, inputs=inputs))
    assert step_input_template(pipeline, 0) == "{{ inputs.task }}"


def test_input_is_short_for_inputs_input() -> None:
    assert _errors(_cfg({"name": "a", "agent": "x", "input": "{{ input }}"})) == []
    errors = _errors(_cfg({"name": "a", "agent": "x", "input": "{{ input }}"}, inputs={"email": {}}))
    assert any("Unknown input 'input'" in error for error in errors), errors


# ---------------------------------------------------------------------------
# Static errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (_cfg({"name": "a", "agent": "x"}, {"name": "a", "agent": "y"}), "Duplicate step name 'a'"),
        (
            _cfg({"name": "a", "agent": "x", "input": "{{ steps.b.response }}"}, {"name": "b", "agent": "y"}),
            "runs later",
        ),
        (_cfg({"name": "a", "agent": "x", "input": "{{ steps.a.response }}"}), "cannot reference itself"),
        (_cfg({"name": "a", "agent": "x", "input": "{{ steps.zz.response }}"}), "Unknown step 'zz'"),
        (_cfg({"name": "a", "agent": "x", "input": "{{ previous.response }}"}), "not available in the first step"),
        (
            _cfg({"name": "a", "agent": "x"}, {"name": "b", "agent": "y", "input": "{{ previous.output }}"}),
            "Only previous.response",
        ),
        (
            _cfg({"name": "a", "agent": "x"}, {"name": "b", "agent": "y", "input": "{{ previous.response.f }}"}),
            "response_format: json",
        ),
        (
            _cfg({"name": "a", "agent": "x"}, {"name": "b", "agent": "y", "input": "{{ steps.a.response.f }}"}),
            "response_format: json",
        ),
        (
            _cfg({"name": "a", "agent": "x"}, {"name": "b", "agent": "y", "input": "{{ steps.a.status }}"}),
            "steps.a.response or steps.a.output",
        ),
        (_cfg({"name": "a", "agent": "x", "input": "{{ input.field }}"}), "is a string, so it has no fields"),
        (_cfg({"name": "a", "agent": "x", "input": "{{ inputs.zz }}"}), "Unknown input 'zz'"),
        (_cfg({"name": "a", "agent": "x", "input": "{{ env.HOME }}"}), "Unknown name 'env'"),
        (_cfg({"name": "a", "agent": "x", "input": "{{ input | upper }}"}), "Filters are not supported"),
        (_cfg({"name": "a", "agent": "x", "context": {"k": "{% if x %}"}}), "steps[a].context.k"),
        (_cfg({"name": "a", "agent": "x", "when": {"path": "steps.a.response", "equals": 1}}), "steps[a].when.path"),
        (_cfg({"name": "a", "agent": "x"}, version=3), "Unsupported version 3"),
        (_cfg({"name": "a", "agent": "x", "input": "hi"}, inputs={}), "Declare at least one input"),
        (_cfg({"name": "a", "agent": "x"}, inputs={"a": {}, "b": {}}), "no single text input"),
        (_cfg({"name": "a", "agent": "x"}, inputs={"n": {"type": "number"}}), "no single text input"),
        (_cfg({"name": "a", "agent": "x", "input": "x"}, inputs={"1bad": {}}), "Input names use"),
        (_cfg({"name": "a", "agent": "x"}, input={}), "Unknown key(s): input"),
    ],
)
def test_static_errors(data: dict[str, Any], expected: str) -> None:
    errors = _errors(data)
    assert any(expected in error for error in errors), errors
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(data)


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (
            {"name": "a", "agent": "x", "model": "m"},
            "Unknown key(s): model. Agent settings go in the agent's agent.yaml",
        ),
        ({"name": "a", "agent": "x", "image": "i", "timeout_seconds": 5}, "Unknown key(s): image, timeout_seconds."),
        ({"name": "a", "agent": {"config": {}}}, "register it from its agent.yaml first"),
        ({"name": "A", "agent": "x"}, "Step names use lowercase letters"),
        ({"name": "a.b", "agent": "x"}, "Step names use lowercase letters"),
        ({"name": "a", "agent": "bad ref!"}, "Use 'name' or 'workspace/name'"),
        ({"name": "a", "agent": "x", "retries": 9}, "retries is between 0 and 5"),
        ({"name": "a", "agent": "x", "when": {"path": "input", "equals": 1, "in": [1]}}, "exactly one"),
        ({"name": "a", "agent": "x", "when": {"path": "input", "equals": 1, "bogus": 2}}, "Unknown key(s): bogus"),
    ],
)
def test_step_shape_errors(step: dict[str, Any], expected: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(expected)):
        PipelineConfig.model_validate(_cfg(step))


def test_pipeline_name_follows_platform_naming() -> None:
    with pytest.raises(ValidationError, match="name"):
        PipelineConfig.model_validate({**_cfg({"name": "a", "agent": "x"}), "name": "Bad--Name"})


def test_step_count_is_limited() -> None:
    steps = [{"name": f"s{i}", "agent": "x"} for i in range(PIPELINE_MAX_STEPS + 1)]
    with pytest.raises(ValidationError, match="1 to 20 steps"):
        PipelineConfig.model_validate(_cfg(*steps))


def test_input_default_rules() -> None:
    with pytest.raises(ValidationError, match="both required and have a default"):
        PipelineConfig.model_validate(
            _cfg({"name": "a", "agent": "x"}, inputs={"input": {"required": True, "default": "x"}})
        )
    with pytest.raises(ValidationError, match="type number"):
        PipelineConfig.model_validate(
            _cfg(
                {"name": "a", "agent": "x", "input": "{{ inputs.n }}"},
                inputs={"n": {"type": "number", "default": "x"}},
            )
        )
    pipeline = PipelineConfig.model_validate(_cfg({"name": "a", "agent": "x"}, inputs={"input": {"default": "x"}}))
    assert pipeline.inputs["input"].is_required is False


def test_no_inputs_reports_only_the_missing_inputs() -> None:
    assert [e for e in _errors(_cfg({"name": "a", "agent": "x"}, inputs={})) if "input" in e.lower()] == [
        "inputs: Declare at least one input, e.g. inputs: {input: {}}."
    ]


# ---------------------------------------------------------------------------
# Conditional steps
# ---------------------------------------------------------------------------


def _conditional(**third: Any) -> dict[str, Any]:
    return _cfg(
        {"name": "a", "agent": "x"},
        {"name": "b", "agent": "y", "input": "{{ input }}", "when": {"path": "input", "equals": "go"}},
        {"name": "c", "agent": "z", **third},
    )


@pytest.mark.parametrize(
    "third",
    [
        {"input": "{{ steps.b.response }}"},
        {"input": "{{ previous.response }}"},
        {},  # the implicit default input is previous.response
        {"input": "{{ input }}", "when": {"path": "steps.b.response", "equals": 1}},
    ],
)
def test_steps_that_may_be_skipped_cannot_be_referenced(third: dict[str, Any]) -> None:
    errors = _errors(_conditional(**third))
    assert any("Step 'b' has a 'when' condition and may be skipped" in error for error in errors), errors


def test_steps_after_a_conditional_step_can_use_earlier_steps() -> None:
    assert _errors(_conditional(input="{{ steps.a.response }} {{ input }}")) == []


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("inputs.input", None),
        ("previous.response", None),
        ("steps.zz.response", "Unknown step 'zz'"),
        ("env.HOME", "Unknown name 'env'"),
    ],
)
def test_when_paths_are_checked(path: str, expected: str | None) -> None:
    data = _cfg(
        {"name": "a", "agent": "x"},
        {"name": "b", "agent": "y", "input": "{{ input }}", "when": {"path": path, "equals": 1}},
    )
    errors = _errors(data)
    if expected is None:
        assert errors == []
    else:
        assert any(expected in error and "when.path" in error for error in errors), errors


@pytest.mark.parametrize(
    "when",
    [{"path": "input", "equals": None}, {"path": "input", "not_equals": "x"}, {"path": "input", "in": ["a", 1]}],
)
def test_conditions_round_trip_through_a_full_dump(when: dict[str, Any]) -> None:
    pipeline = PipelineInline.model_validate({"version": 1, "steps": [{"name": "a", "agent": "x", "when": when}]})
    dumped = pipeline.model_dump(mode="json")  # what responses and the entity store write: no exclude_unset
    assert dumped["steps"][0]["when"] == when
    assert PipelineInline.model_validate(dumped) == pipeline


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        ({"path": "input", "in": "abc"}, "Input should be a valid list"),
        ({"path": "input"}, "Set exactly one of"),
        ({"equals": 1}, "path"),
    ],
)
def test_malformed_conditions_give_one_clear_error(when: dict[str, Any], expected: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        PipelineConfig.model_validate(_cfg({"name": "a", "agent": "x", "when": when}))
    assert excinfo.value.error_count() == 1
    assert expected in str(excinfo.value)


# ---------------------------------------------------------------------------
# Stored shape and request types
# ---------------------------------------------------------------------------


def test_stored_shape_does_not_run_rules() -> None:
    stored = {"version": 1, "future": True, "steps": [{"name": "a", "agent": "x", "retries": 9, "new_key": 1}]}
    pipeline = PipelineInline.model_validate(stored)
    assert pipeline.steps[0].retries == 9
    assert pipeline.model_dump()["steps"][0]["new_key"] == 1
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({**stored, "name": "p1"})


def test_request_types_round_trip_the_in_alias() -> None:
    request = CreatePipelineConfigRequest.model_validate(
        _cfg({"name": "a", "agent": "x", "when": {"path": "input", "in": ["a"]}})
    )
    assert '"in":["a"]' in request.model_dump_json(exclude_unset=True)
    assert CreatePipelineConfigRequest.model_validate_json(request.model_dump_json()) == request


def test_update_request_accepts_an_optional_name() -> None:
    body = {k: v for k, v in _cfg({"name": "a", "agent": "x"}).items() if k != "name"}
    assert UpdatePipelineConfigRequest.model_validate(body).name is None
    assert UpdatePipelineConfigRequest.model_validate({**body, "name": "p1"}).name == "p1"


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _inline(*steps: dict[str, Any], inputs: dict[str, Any] | None = None) -> PipelineInline:
    return PipelineInline.model_validate({"version": 1, "inputs": inputs or {"input": {}}, "steps": list(steps)})


def test_resolve_run_inputs() -> None:
    single = _inline({"name": "a", "agent": "x"})
    assert resolve_run_inputs(single, "text") == {"inputs": {"input": "text"}, "input": "text"}

    typed = _inline(
        {"name": "a", "agent": "x", "input": "{{ inputs.t }} {{ inputs.n }}"},
        inputs={"t": {}, "n": {"type": "number", "default": 3}},
    )
    assert resolve_run_inputs(typed, {"t": "x"}) == {"inputs": {"t": "x", "n": 3}}
    assert resolve_run_inputs(typed, "x") == {"inputs": {"t": "x", "n": 3}}  # the single text input
    with pytest.raises(ValueError, match="Missing required input 't'"):
        resolve_run_inputs(typed, {})
    with pytest.raises(ValueError, match="type number"):
        resolve_run_inputs(typed, {"t": "x", "n": "3"})
    with pytest.raises(ValueError, match="Unknown inputs: z"):
        resolve_run_inputs(typed, {"t": "x", "z": 1})


def test_ensure_runnable_rejects_stale_or_unsupported_configs() -> None:
    with pytest.raises(ValueError, match=r"cannot run: steps\[0\]\.retries"):
        resolve_run_inputs(_inline({"name": "a", "agent": "x", "retries": 9}), "text")
    future = PipelineInline.model_validate(
        {"version": 3, "inputs": {"input": {}}, "steps": [{"name": "a", "agent": "x"}]}
    )
    with pytest.raises(ValueError, match="Unsupported version 3"):
        ensure_runnable(future)


def test_render_step_and_conditions() -> None:
    pipeline = _inline(
        {"name": "triage", "agent": "x", "response_format": "json"},
        {
            "name": "reply",
            "agent": "y",
            "input": "Verdict: {{ previous.response.verdict }} / {{ input }}",
            "context": {"indicators": "{{ steps.triage.response.indicators }}"},
        },
        {
            "name": "notify",
            "agent": "z",
            "input": "{{ input }}",
            "when": {"path": "steps.triage.response.verdict", "not_equals": "clean"},
        },
    )
    response = parse_step_response(pipeline.steps[0], '{"verdict": "phishing", "indicators": ["a"]}')
    context = {
        **resolve_run_inputs(pipeline, "the email"),
        "steps": {"triage": {"response": response, "output": {}}},
        "previous": {"response": response},
    }
    assert render_step(pipeline, 0, context) == ("the email", {})
    assert render_step(pipeline, 1, context) == ("Verdict: phishing / the email", {"indicators": ["a"]})
    assert should_run(pipeline.steps[2], context) is True
    context["steps"]["triage"]["response"] = {"verdict": "clean"}
    assert should_run(pipeline.steps[2], context) is False


def test_parse_step_response() -> None:
    json_step = PipelineStepConfig(name="a", agent="x", response_format="json")
    text_step = PipelineStepConfig(name="b", agent="x")
    assert parse_step_response(json_step, '{"a": 1}') == {"a": 1}
    assert parse_step_response(json_step, {"already": "object"}) == {"already": "object"}
    assert parse_step_response(text_step, '{"a": 1}') == '{"a": 1}'
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_step_response(json_step, "not json")
