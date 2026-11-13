# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym's rendered usage survives Intake normalization and storage-bag readback."""

import pytest
from nhx.intake.spans.span_semantic_attributes import SpanSemanticAttributes
from nhx_evals_sdk.ng_trajectory_otlp import rollout_to_resource_spans
from opentelemetry.proto.trace.v1.trace_pb2 import Span


def _attributes(span: Span) -> dict[str, object]:
    attributes: dict[str, object] = {}
    for entry in span.attributes:
        field = entry.value.WhichOneof("value")
        assert field is not None
        attributes[entry.key] = getattr(entry.value, field)
    return attributes


@pytest.mark.parametrize("count", [None, 0, 7])
def test_gym_usage_details_round_trip_through_intake(count: int | None) -> None:
    usage = {
        "input_tokens": 31,
        "output_tokens": 11,
        "input_tokens_details": {"cached_tokens": count, "cache_write_tokens": count},
        "output_tokens_details": {"reasoning_tokens": count},
    }
    (resource,) = rollout_to_resource_spans(
        {"response": {}},
        run_id="run-1",
        rollout_id="trial",
        model_calls=[{"response": {"usage": usage}}],
    )
    _, call = resource.scope_spans[0].spans
    attributes = _attributes(call)

    semantic, _ = SpanSemanticAttributes.from_source_attributes(attributes)
    restored = SpanSemanticAttributes.from_bags(semantic.to_bags())
    for usage in (semantic, restored):
        assert usage.input_tokens == 31
        assert usage.output_tokens == 11
        assert usage.total_tokens is None
        assert usage.cached_tokens == count
        assert usage.prompt_cache_write_tokens == count
        assert usage.completion_reasoning_tokens == count


@pytest.mark.parametrize("reported_total", [None, 0, 999])
@pytest.mark.parametrize("normalized", [False, True])
def test_intake_preserves_rendered_reported_totals(reported_total: int | None, normalized: bool) -> None:
    call = {"response": {"usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": reported_total}}}
    if normalized:
        call.update(tokens_in=12, tokens_out=8, tokens_total=reported_total)
    (resource,) = rollout_to_resource_spans({}, run_id="run", rollout_id="trial", model_calls=[call])
    _, span = resource.scope_spans[0].spans
    attributes = _attributes(span)
    semantic, _ = SpanSemanticAttributes.from_source_attributes(attributes)
    restored = SpanSemanticAttributes.from_bags(semantic.to_bags())
    assert semantic.total_tokens == restored.total_tokens == reported_total


def test_normalized_capture_counters_round_trip_independently() -> None:
    call = {
        "tokens_in": 31,
        "tokens_out": 11,
        "tokens_total": 42,
        "cached_tokens": 7,
        "cache_creation_tokens": 3,
        "tokens_reasoning": 5,
        "response": {"usage": {"input_tokens": 999, "output_tokens": 999}},
    }
    (resource,) = rollout_to_resource_spans(
        {"ng_model_call_capture": {"calls": [call]}}, run_id="run", rollout_id="trial"
    )
    _, span = resource.scope_spans[0].spans
    attributes = _attributes(span)
    semantic, _ = SpanSemanticAttributes.from_source_attributes(attributes)
    restored = SpanSemanticAttributes.from_bags(semantic.to_bags())
    for result in (semantic, restored):
        assert result.input_tokens == 31
        assert result.output_tokens == 11
        assert result.total_tokens == 42
        assert result.cached_tokens == 7
        assert result.prompt_cache_write_tokens == 3
        assert result.completion_reasoning_tokens == 5
