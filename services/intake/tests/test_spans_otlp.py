# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP ingest helper tests."""

import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, cast

import pytest
from fastapi import HTTPException, Request
from nhx.intake.spans.ingest.otlp import _read_limited_body, _span_to_domain

DEFAULT_TRACE_ID = bytes.fromhex("0" * 31 + "1")
DEFAULT_SPAN_ID = bytes.fromhex("0000000000000001")


class StreamingRequest:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def stream(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_read_limited_body_accepts_stream_under_limit():
    body = await _read_limited_body(cast(Request, StreamingRequest([b"abc", b"def"])), max_bytes=6)

    assert body == b"abcdef"


@pytest.mark.asyncio
async def test_read_limited_body_rejects_stream_over_limit():
    with pytest.raises(HTTPException) as exc_info:
        await _read_limited_body(cast(Request, StreamingRequest([b"abc", b"def"])), max_bytes=5)

    assert exc_info.value.status_code == 413


@pytest.mark.parametrize(
    ("trace_id", "span_id", "parent_span_id", "match"),
    [
        (bytes(16), bytes.fromhex("0000000000000001"), b"", "trace_id is required"),
        (bytes.fromhex("0" * 31 + "1"), bytes(8), b"", "span_id is required"),
        (bytes.fromhex("0" * 31 + "1"), bytes.fromhex("0000000000000001"), bytes(8), "parent_span_id"),
    ],
)
def test_span_to_domain_rejects_empty_or_zero_otlp_ids(
    trace_id: bytes, span_id: bytes, parent_span_id: bytes, match: str
):
    span = _make_span(trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id)

    with pytest.raises(ValueError, match=match):
        _span_to_domain(
            workspace="default",
            span=span,
            resource_attributes={},
            scope_data={},
            ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_span_to_domain_filters_resource_raw_attributes():
    span = _make_span()
    raw_resource_attributes = {
        "service.name": "intake-span-test",
        "session.id": "resource-session",
        "gen_ai.project": "project-a",
        "user.id": "user-a",
    }

    domain_span = _span_to_domain(
        workspace="default",
        span=span,
        resource_attributes=raw_resource_attributes,
        scope_data={},
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert domain_span.attributes_string["service.name"] == "intake-span-test"
    assert domain_span.attributes_string["project.name"] == "project-a"
    assert domain_span.attributes_string["user.id"] == "user-a"


def test_span_to_domain_skips_empty_scope_data():
    span = _make_span()

    domain_span = _span_to_domain(
        workspace="default",
        span=span,
        resource_attributes={},
        scope_data={"name": None, "version": None},
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert "otel.scope" not in domain_span.attributes_string


def test_span_to_domain_does_not_duplicate_model_aliases():
    span = _make_span()
    request_model = span.attributes.add()
    request_model.key = "gen_ai.request.model"
    request_model.value.string_value = "request-model"
    response_model = span.attributes.add()
    response_model.key = "gen_ai.response.model"
    response_model.value.string_value = "response-model"

    domain_span = _span_to_domain(
        workspace="default",
        span=span,
        resource_attributes={},
        scope_data={},
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert domain_span.attributes_string["gen_ai.request.model"] == "request-model"
    assert "gen_ai.response.model" not in domain_span.attributes_string


def test_span_to_domain_promotes_pydantic_ai_model_messages():
    input_messages = [{"role": "user", "parts": [{"type": "text", "content": "Analyze traces."}]}]
    output_messages = [{"role": "assistant", "parts": [{"type": "text", "content": "Found a recurring issue."}]}]
    span = _make_span(name="chat aws/anthropic/bedrock-claude-opus-4-8")
    _add_string_attr(span, "gen_ai.input.messages", json.dumps(input_messages))
    _add_string_attr(span, "gen_ai.output.messages", json.dumps(output_messages))
    _add_string_attr(
        span, "gen_ai.system_instructions", json.dumps([{"type": "text", "content": "You are an analyst."}])
    )

    domain_span = _span_to_domain(
        workspace="default",
        span=span,
        resource_attributes={"gen_ai.agent.name": "nemo-optimizer-analyst"},
        scope_data={},
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert json.loads(domain_span.input) == input_messages
    assert json.loads(domain_span.output) == output_messages
    assert domain_span.attributes_string["gen_ai.system_instructions"] == json.dumps(
        [{"type": "text", "content": "You are an analyst."}]
    )


def test_span_to_domain_promotes_pydantic_ai_agent_run_result():
    all_messages = [
        {"role": "user", "parts": [{"type": "text", "content": "Analyze traces."}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "Summary."}]},
    ]
    final_result = {"summary": "Created one insight.", "new_insights": []}
    span = _make_span(name="nemo-optimizer-analyst run")
    _add_string_attr(span, "pydantic_ai.all_messages", json.dumps(all_messages))
    _add_string_attr(span, "final_result", json.dumps(final_result))

    domain_span = _span_to_domain(
        workspace="default",
        span=span,
        resource_attributes={"gen_ai.agent.name": "nemo-optimizer-analyst"},
        scope_data={},
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert domain_span.input == ""
    assert json.loads(domain_span.output) == final_result
    assert domain_span.attributes_string["pydantic_ai.all_messages"] == json.dumps(all_messages)


def test_span_to_domain_promotes_pydantic_ai_tool_arguments_and_response():
    span = _make_span(name="running tool")
    _add_string_attr(span, "gen_ai.tool.name", "run_code")
    _add_string_attr(span, "gen_ai.tool.call.arguments", '{"code":"await fetch_spans()"}')
    _add_string_attr(span, "tool_response", '{"return_value":{"count":3}}')

    domain_span = _span_to_domain(
        workspace="default",
        span=span,
        resource_attributes={"gen_ai.agent.name": "nemo-optimizer-analyst"},
        scope_data={},
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert json.loads(domain_span.input) == {"code": "await fetch_spans()"}
    assert json.loads(domain_span.output) == {"return_value": {"count": 3}}


def test_span_to_domain_carries_evaluation_id_attribute() -> None:
    # OTLP associates a span with an evaluation via the nemo.evaluation.name attribute; the ingest path
    # reads it off the built span to mark that evaluation dirty for facet denormalization.
    span = _make_span()
    _add_string_attr(span, "nemo.evaluation.name", "my-eval")
    domain_span = _span_to_domain(
        workspace="default",
        span=span,
        resource_attributes={},
        scope_data={},
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert domain_span.attributes_string.get("nemo.evaluation.name") == "my-eval"


def test_span_to_domain_normalizes_legacy_evaluation_id_key() -> None:
    # A pre-rename producer still emitting nemo.experiment.id must keep associating: ingest normalizes
    # the legacy source key to the canonical nemo.evaluation.name bag key.
    span = _make_span()
    _add_string_attr(span, "nemo.experiment.id", "legacy-eval")
    domain_span = _span_to_domain(
        workspace="default",
        span=span,
        resource_attributes={},
        scope_data={},
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert domain_span.attributes_string.get("nemo.evaluation.name") == "legacy-eval"
    assert "nemo.experiment.id" not in domain_span.attributes_string


def test_span_to_domain_without_evaluation_attribute_has_no_id() -> None:
    domain_span = _span_to_domain(
        workspace="default",
        span=_make_span(),
        resource_attributes={},
        scope_data={},
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert domain_span.attributes_string.get("nemo.evaluation.name") is None


def _make_span(
    *,
    trace_id: bytes = DEFAULT_TRACE_ID,
    span_id: bytes = DEFAULT_SPAN_ID,
    parent_span_id: bytes = b"",
    name: str = "test-span",
) -> Any:
    from opentelemetry.proto.trace.v1 import trace_pb2

    span = trace_pb2.Span()
    span.trace_id = trace_id
    span.span_id = span_id
    span.parent_span_id = parent_span_id
    span.name = name
    span.start_time_unix_nano = 1_700_000_000_000_000_000
    span.end_time_unix_nano = 1_700_000_000_001_000_000
    return span


def _add_string_attr(span: Any, key: str, value: str) -> None:
    attr = span.attributes.add()
    attr.key = key
    attr.value.string_value = value
