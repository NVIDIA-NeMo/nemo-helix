# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import logging
from unittest.mock import AsyncMock

import pytest
from nemo_evaluator_sdk.enums import ModelFormat
from nemo_evaluator_sdk.session import begin_evaluation_session
from nemo_evaluator_sdk.structured_output import (
    InferenceStructuredOutput,
    Model,
    StructuredOutputMode,
    _default_probe_schema,
    _looks_like_unsupported_guided_json_error,
    default_structured_output_mode,
    detect_structured_output_mode,
)
from pydantic import ValidationError

_PROBE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def _test_model() -> Model:
    return Model(url="https://example.com/v1/chat/completions", name="test/model")


def test_inference_structured_output_validation_error():
    with pytest.raises(ValueError, match="structured_output cannot be empty"):
        InferenceStructuredOutput(StructuredOutputMode.NVEXT_GUIDED_JSON, {})

    with pytest.raises(ValueError, match="Unsupported structured output mode"):
        InferenceStructuredOutput("unsupported-format", {"schema": {}})  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValidationError, match="schema\n +Input should be a valid dictionary"):
        structured_output = {"schema": "string"}
        InferenceStructuredOutput(StructuredOutputMode.NVEXT_GUIDED_JSON, structured_output)


def test_inference_structured_output():
    structured_output = {
        "schema": {"type": "object", "properties": {"quality": {"type": "number"}}},
        "strict": True,  # ignored by NIM
    }

    request = {
        "model": "meta/llama-3.2-1b-instruct",
        "messages": [],
    }

    hook = InferenceStructuredOutput(StructuredOutputMode.NVEXT_GUIDED_JSON, structured_output)
    modified_request = hook.preprocess(request)

    assert "extra_body" in modified_request
    assert "nvext" in modified_request["extra_body"]
    assert "guided_json" in modified_request["extra_body"]["nvext"]
    assert modified_request["extra_body"]["nvext"]["guided_json"] == structured_output["schema"]


def test_inference_structured_output_preserves_existing_extra_body():
    structured_output = {
        "schema": {"type": "object", "properties": {"quality": {"type": "number"}}},
        "strict": True,
    }
    request = {
        "model": "meta/llama-3.2-1b-instruct",
        "messages": [],
        "extra_body": {
            "nvext": {
                "max_thinking_tokens": 256,
            }
        },
    }

    hook = InferenceStructuredOutput(StructuredOutputMode.NVEXT_GUIDED_JSON, structured_output)
    modified_request = hook.preprocess(request)

    assert modified_request["extra_body"]["nvext"] == {
        "max_thinking_tokens": 256,
        "guided_json": structured_output["schema"],
    }


def test_inference_structured_output_openai():
    """Test that OpenAI format produces correct response_format structure."""
    structured_output = {
        "schema": {"type": "object", "properties": {"quality": {"type": "number"}}},
        "strict": True,
    }
    request = {
        "model": "openai/model",
        "messages": [{"role": "user", "content": "test"}],
    }

    hook = InferenceStructuredOutput(StructuredOutputMode.OPENAI_RESPONSE_FORMAT, structured_output)
    modified_request = hook.preprocess(request)

    assert "response_format" in modified_request
    response_format = modified_request["response_format"]
    assert response_format["type"] == "json_schema"
    assert "json_schema" in response_format
    assert response_format["json_schema"]["name"] == "structured_output"
    assert response_format["json_schema"]["schema"] == structured_output["schema"]
    assert response_format["json_schema"]["strict"] is True


def test_inference_structured_output_openai_default_strict():
    """Test that OpenAI format uses the strict value from input (defaults to False)."""
    structured_output = {
        "schema": {"type": "object", "properties": {"quality": {"type": "number"}}},
    }

    hook = InferenceStructuredOutput(StructuredOutputMode.OPENAI_RESPONSE_FORMAT, structured_output)

    # StructuredOutput.strict defaults to False when not specified
    response_format = hook.inference_param.get("response_format")
    assert isinstance(response_format, dict)
    json_schema = response_format.get("json_schema")
    assert isinstance(json_schema, dict)
    assert json_schema["strict"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("format", [ModelFormat.OPEN_AI, ModelFormat.NVIDIA_NIM, ModelFormat.LLAMA_STACK, None])
async def test_detect_structured_output_mode_probes_regardless_of_format(format):
    """Support is a property of the endpoint, so the label must not short-circuit detection.

    Previously OPEN_AI returned without probing and every other non-NIM format returned UNSUPPORTED
    without probing, which meant the label decided enforcement rather than the endpoint.
    """
    requests: list[dict] = []

    async def inference_fn(model, request, max_retries, **kwargs):
        requests.append(request)
        return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    mode = await detect_structured_output_mode(
        format=format,
        model=_test_model(),
        inference_fn=inference_fn,
        api_key=None,
        probe_schema=_PROBE_SCHEMA,
    )

    assert mode == StructuredOutputMode.OPENAI_RESPONSE_FORMAT
    assert len(requests) == 1


def test_default_structured_output_mode_ignores_format():
    for format in (ModelFormat.OPEN_AI, ModelFormat.NVIDIA_NIM, ModelFormat.LLAMA_STACK, None):
        assert default_structured_output_mode(format) == StructuredOutputMode.OPENAI_RESPONSE_FORMAT


@pytest.mark.asyncio
async def test_detect_structured_output_mode_prefers_response_format_for_nim():
    """NIM-labelled endpoints serving an OpenAI-compatible route must resolve to response_format.

    Hosted endpoints such as integrate.api.nvidia.com reject nvext.guided_json and silently ignore
    root guided_json, so probing the guided_json placements first yields UNSUPPORTED even though
    response_format is enforced there.
    """
    requests: list[dict] = []

    async def inference_fn(model, request, max_retries, **kwargs):
        requests.append(request)
        return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    mode = await detect_structured_output_mode(
        format=ModelFormat.NVIDIA_NIM,
        model=_test_model(),
        inference_fn=inference_fn,
        api_key="secret",
        probe_schema=_PROBE_SCHEMA,
    )

    assert mode == StructuredOutputMode.OPENAI_RESPONSE_FORMAT
    assert len(requests) == 1
    assert "extra_body" not in requests[0]
    # The probe must send the same payload the hook sends in production, or detection can select a
    # mode that then fails during evaluation.
    assert requests[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "structured_output", "schema": _PROBE_SCHEMA, "strict": False},
    }


@pytest.mark.asyncio
async def test_detect_structured_output_mode_falls_back_to_root_guided_json():
    requests: list[dict] = []

    async def inference_fn(model, request, max_retries, **kwargs):
        requests.append(request)
        if "response_format" in request:
            # Endpoint ignores response_format and answers in prose.
            return {"choices": [{"message": {"content": "Sure, the value is true."}}]}
        return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    mode = await detect_structured_output_mode(
        format=ModelFormat.NVIDIA_NIM,
        model=_test_model(),
        inference_fn=inference_fn,
        api_key=None,
        probe_schema=_PROBE_SCHEMA,
    )

    assert mode == StructuredOutputMode.ROOT_GUIDED_JSON
    assert len(requests) == 2
    assert requests[1]["extra_body"] == {"guided_json": _PROBE_SCHEMA}


@pytest.mark.asyncio
async def test_detect_structured_output_mode_falls_back_to_nvext_when_root_invalid():
    requests: list[dict] = []

    async def inference_fn(model, request, max_retries, **kwargs):
        requests.append(request)
        if "response_format" in request:
            return {"choices": [{"message": {"content": "Sure, the value is true."}}]}
        if request.get("extra_body", {}).get("guided_json") is not None:
            # Invalid for schema ("ok" must be boolean)
            return {"choices": [{"message": {"content": '{"ok":"not-bool"}'}}]}
        return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    mode = await detect_structured_output_mode(
        format=ModelFormat.NVIDIA_NIM,
        model=_test_model(),
        inference_fn=inference_fn,
        api_key=None,
        probe_schema=_PROBE_SCHEMA,
    )

    assert mode == StructuredOutputMode.NVEXT_GUIDED_JSON
    assert len(requests) == 3
    assert "response_format" in requests[0]
    assert "guided_json" in requests[1]["extra_body"]
    assert requests[2]["extra_body"] == {"nvext": {"guided_json": _PROBE_SCHEMA}}


@pytest.mark.asyncio
async def test_detect_structured_output_mode_returns_unsupported_on_probe_exceptions():
    async def inference_fn(model, request, max_retries, **kwargs):
        raise RuntimeError("Error code: 500 - {'detail': 'internal server error'}")

    mode = await detect_structured_output_mode(
        format=ModelFormat.NVIDIA_NIM,
        model=_test_model(),
        inference_fn=inference_fn,
        api_key=None,
        probe_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )

    assert mode == StructuredOutputMode.UNSUPPORTED


@pytest.mark.asyncio
async def test_detect_structured_output_mode_unsupported_signature_then_unsupported():
    async def inference_fn(model, request, max_retries, **kwargs):
        raise RuntimeError("extra_forbidden: extra inputs are not permitted for extra_body.guided_json")

    mode = await detect_structured_output_mode(
        format=ModelFormat.NVIDIA_NIM,
        model=_test_model(),
        inference_fn=inference_fn,
        api_key=None,
        probe_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )

    assert mode == StructuredOutputMode.UNSUPPORTED


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("extra_forbidden for extra_body.guided_json", True),
        ("extra inputs are not permitted: nvext.guided_json", True),
        ("unexpected keyword argument 'guided_json'", True),
        ("unexpected keyword argument 'nvext'", True),
        ("extra inputs are not permitted", False),
    ],
)
def test_looks_like_unsupported_guided_json_error(message: str, expected: bool):
    assert _looks_like_unsupported_guided_json_error(message) is expected


@pytest.mark.asyncio
async def test_detect_structured_output_mode_does_not_abort_on_malformed_probe_schema():
    """A bad probe schema must degrade to the prompt fallback, not abort evaluation startup."""
    inference_fn = AsyncMock()

    mode = await detect_structured_output_mode(
        model=_test_model(),
        inference_fn=inference_fn,
        api_key=None,
        probe_schema={"type": "not-a-real-json-schema-type"},
    )

    assert mode == StructuredOutputMode.UNSUPPORTED
    inference_fn.assert_not_called()


@pytest.mark.asyncio
async def test_probe_schema_default_is_not_shared_between_calls():
    """The default probe schema must not be reachable-and-mutable from a request payload.

    Pydantic copies only the top level of a dict field, so a module-level constant handed straight
    to InferenceStructuredOutput stays nested-shared with the outgoing request. One mutation of that
    payload would corrupt the schema every later probe validates against, and a corrupted schema
    silently mis-selects the structured output mode.
    """
    captured: list[dict] = []

    async def inference_fn(model, request, max_retries, **kwargs):
        # Snapshot before mutating: `request` is the live object, so reading it back afterwards
        # would only show this function's own edit.
        captured.append(copy.deepcopy(request))
        # Simulate any downstream code mutating the request payload it was handed.
        schema = request["response_format"]["json_schema"]["schema"]
        schema["properties"]["__nhx_probe_score"]["type"] = "POISONED"
        return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}

    # use_cache=False: this exercises schema isolation between probes, not the endpoint cache.
    first = await detect_structured_output_mode(
        model=_test_model(), inference_fn=inference_fn, api_key=None, use_cache=False
    )
    second = await detect_structured_output_mode(
        model=_test_model(), inference_fn=inference_fn, api_key=None, use_cache=False
    )

    assert first == StructuredOutputMode.OPENAI_RESPONSE_FORMAT
    assert second == StructuredOutputMode.OPENAI_RESPONSE_FORMAT
    assert _default_probe_schema()["properties"]["__nhx_probe_score"]["type"] == "integer"
    assert captured[1]["response_format"]["json_schema"]["schema"]["properties"]["__nhx_probe_score"] == {
        "type": "integer"
    }


@pytest.mark.asyncio
async def test_detection_is_cached_per_endpoint_within_a_session():
    calls: list[dict] = []

    async def inference_fn(model, request, max_retries, **kwargs):
        calls.append(request)
        return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}

    async with begin_evaluation_session():
        first = await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)
        second = await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)

    assert first == second == StructuredOutputMode.OPENAI_RESPONSE_FORMAT
    assert len(calls) == 1, "second call should have been served from the run cache"


@pytest.mark.asyncio
async def test_detection_is_not_cached_outside_a_session():
    """Without a run boundary there is nowhere safe to cache, so every call probes."""
    calls: list[dict] = []

    async def inference_fn(model, request, max_retries, **kwargs):
        calls.append(request)
        return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}

    await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)
    await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_unsupported_is_cached_within_a_run_but_re_probed_by_the_next():
    """Run scoping is what makes caching a negative safe.

    Within a run an unsupported endpoint must not be re-probed per row. Across runs the negative
    must not persist, so a transient failure cannot silently disable enforcement thereafter.
    """
    attempts = {"n": 0}

    async def flaky_inference_fn(model, request, max_retries, **kwargs):
        attempts["n"] += 1
        if attempts["n"] <= 3:  # first run: every candidate fails transiently
            raise RuntimeError("Error code: 429 - rate limited")
        return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}

    async with begin_evaluation_session():
        first = await detect_structured_output_mode(model=_test_model(), inference_fn=flaky_inference_fn, api_key=None)
        repeat = await detect_structured_output_mode(model=_test_model(), inference_fn=flaky_inference_fn, api_key=None)

    assert first == repeat == StructuredOutputMode.UNSUPPORTED
    assert attempts["n"] == 3, "the negative must be cached within the run, not re-probed per caller"

    async with begin_evaluation_session():
        later = await detect_structured_output_mode(model=_test_model(), inference_fn=flaky_inference_fn, api_key=None)

    assert later == StructuredOutputMode.OPENAI_RESPONSE_FORMAT, "a new run must re-probe"


def test_unresolved_hook_warns_once_when_used_without_probing(caplog):
    """A generation path that never probes must be visible, not silent."""
    hook = InferenceStructuredOutput(
        StructuredOutputMode.OPENAI_RESPONSE_FORMAT, {"schema": _PROBE_SCHEMA}, resolved=False
    )

    with caplog.at_level(logging.WARNING):
        hook.preprocess({"messages": []})
        hook.preprocess({"messages": []})

    warnings = [r for r in caplog.records if "never resolved against the endpoint" in r.message]
    assert len(warnings) == 1, "warn once per hook, not per row"


def test_resolved_hook_does_not_warn(caplog):
    hook = InferenceStructuredOutput(
        StructuredOutputMode.OPENAI_RESPONSE_FORMAT, {"schema": _PROBE_SCHEMA}, resolved=False
    )
    hook.set_mode(StructuredOutputMode.NVEXT_GUIDED_JSON)

    with caplog.at_level(logging.WARNING):
        hook.preprocess({"messages": []})

    assert not [r for r in caplog.records if "never resolved against the endpoint" in r.message]


def test_explicitly_constructed_hook_is_resolved_by_default(caplog):
    """A caller passing a deliberate mode is not guessing, so it must not be warned at."""
    hook = InferenceStructuredOutput(StructuredOutputMode.NVEXT_GUIDED_JSON, {"schema": _PROBE_SCHEMA})

    with caplog.at_level(logging.WARNING):
        hook.preprocess({"messages": []})

    assert not [r for r in caplog.records if "never resolved against the endpoint" in r.message]


@pytest.mark.asyncio
async def test_concurrent_detections_in_one_session_probe_the_endpoint_once():
    """Per-row callers must not stampede the endpoint.

    Also pins that the session ContextVar is visible inside tasks created by asyncio.gather: if it
    were not, each task would get no cache and probe independently.
    """
    calls: list[dict] = []

    async def slow_inference_fn(model, request, max_retries, **kwargs):
        calls.append(request)
        await asyncio.sleep(0.01)  # widen the window for a stampede
        return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}

    async with begin_evaluation_session():
        modes = await asyncio.gather(
            *(
                detect_structured_output_mode(model=_test_model(), inference_fn=slow_inference_fn, api_key=None)
                for _ in range(8)
            )
        )

    assert set(modes) == {StructuredOutputMode.OPENAI_RESPONSE_FORMAT}
    assert len(calls) == 1, f"expected a single probe for 8 concurrent callers, got {len(calls)}"


@pytest.mark.asyncio
async def test_nested_sessions_are_re_entrant_and_share_one_cache():
    """Entry points nest (backend opens one, evaluate_metric opens another).

    A nested session must join the outer one; creating a fresh inner cache would re-probe endpoints
    the run had already resolved.
    """
    calls: list[dict] = []

    async def inference_fn(model, request, max_retries, **kwargs):
        calls.append(request)
        return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}

    async with begin_evaluation_session():
        await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)
        async with begin_evaluation_session():
            await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)
        await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)

    assert len(calls) == 1, "the nested session must join the outer cache, not start a new one"


@pytest.mark.asyncio
async def test_separate_runs_do_not_share_a_cache():
    """Re-entrancy must not leak between sibling runs."""
    calls: list[dict] = []

    async def inference_fn(model, request, max_retries, **kwargs):
        calls.append(request)
        return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}

    async with begin_evaluation_session():
        await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)
    async with begin_evaluation_session():
        await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_concurrent_runs_get_isolated_caches():
    """Two concurrent runs must not share detection results.

    A long-lived process (the evaluator service) handles jobs as sibling asyncio tasks. Each task
    gets a copy of the context at creation, so each session must be its own cache -- otherwise one
    job's transient failure would decide the encoding for another job's endpoint.
    """
    calls: list[str] = []

    async def run_one(tag: str) -> StructuredOutputMode:
        async def inference_fn(model, request, max_retries, **kwargs):
            calls.append(tag)
            await asyncio.sleep(0.01)
            return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}

        async with begin_evaluation_session():
            return await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)

    modes = await asyncio.gather(run_one("a"), run_one("b"))

    assert set(modes) == {StructuredOutputMode.OPENAI_RESPONSE_FORMAT}
    assert sorted(calls) == ["a", "b"], f"each concurrent run must probe in its own cache, got {calls}"


@pytest.mark.asyncio
async def test_truncated_probe_is_reported_rather_than_silently_unsupported(caplog):
    """A reasoning model can spend the whole probe budget thinking and return empty content.

    Truncation is indistinguishable from an endpoint ignoring the encoding, so it must at least be
    surfaced instead of quietly resolving to the unenforced prompt fallback.
    """

    async def truncating_inference_fn(model, request, max_retries, **kwargs):
        return {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}

    with caplog.at_level(logging.WARNING):
        mode = await detect_structured_output_mode(
            model=_test_model(), inference_fn=truncating_inference_fn, api_key=None
        )

    assert mode == StructuredOutputMode.UNSUPPORTED
    assert [r for r in caplog.records if "could not be determined" in r.message]


@pytest.mark.asyncio
async def test_probe_budget_accommodates_a_reasoning_model():
    """The probe must not use a budget a reasoning model burns before emitting content."""
    seen: list[int] = []

    async def inference_fn(model, request, max_retries, **kwargs):
        seen.append(request["max_tokens"])
        return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}

    await detect_structured_output_mode(model=_test_model(), inference_fn=inference_fn, api_key=None)

    assert seen[0] >= 2048, f"probe budget {seen[0]} is too small for a reasoning model"
