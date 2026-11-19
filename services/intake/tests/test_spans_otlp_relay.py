# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native full-capture mapping; see fixtures/README.md for provenance and reproduction."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from google.protobuf.json_format import ParseDict
from nhx.intake.spans.domain import SpanKind, SpanStatus
from nhx.intake.spans.ingest.otlp import _attributes_to_dict, _span_to_domain
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span

FIXTURE = Path(__file__).parent / "fixtures" / "relay_full_0_9_3.json"
RESPONSE_KEY = "nemo_relay.end.output.__nemo_relay_integrations_langchain_model_response"


def _domain(span: Span, resource: dict[str, Any] | None = None):
    return _span_to_domain(
        workspace="default",
        span=span,
        resource_attributes=resource or {},
        scope_data={},
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _span(attrs: dict[str, Any], name: str = "recorded") -> Span:
    span = Span(trace_id=b"t" * 16, span_id=b"s" * 8, name=name)
    for key, value in attrs.items():
        attr = span.attributes.add(key=key)
        if isinstance(value, bool):
            attr.value.bool_value = value
        elif isinstance(value, int):
            attr.value.int_value = value
        else:
            attr.value.string_value = value if isinstance(value, str) else json.dumps(value)
    return span


def _response(data: dict[str, Any]) -> dict[str, Any]:
    return {RESPONSE_KEY: {"messages": [{"type": "ai", "data": {"content": "answer", **data}}]}}


def test_existing_native_full_capture():
    spans = {}
    for request in json.loads(FIXTURE.read_text()):
        export = ParseDict(request, ExportTraceServiceRequest())
        for resource in export.resource_spans:
            for scope in resource.scope_spans:
                for source in scope.spans:
                    domain = _domain(source, _attributes_to_dict(resource.resource.attributes))
                    spans[source.name] = domain
                    assert domain.trace_id == source.trace_id.hex()
                    assert domain.external_span_id == source.span_id.hex()
                    assert domain.external_parent_span_id == source.parent_span_id.hex()
                    assert domain.start_time == datetime.fromtimestamp(source.start_time_unix_nano / 1e9, timezone.utc)
                    assert domain.end_time == datetime.fromtimestamp(source.end_time_unix_nano / 1e9, timezone.utc)
                    for key, value in _attributes_to_dict(source.attributes).items():
                        if key.startswith("nemo_relay."):
                            assert domain.attributes_string[key] == value
    # Compare one normalized shape, excluding volatile source IDs/timestamps and
    # raw bags whose losslessness is checked independently above.
    semantic_keys = {
        "gen_ai.agent.name",
        "gen_ai.request.model",
        "gen_ai.system",
        "gen_ai.tool.name",
        "exception.message",
        "exception.type",
    }

    def payload(value: str):
        return json.loads(value) if value.startswith(("{", "[")) else value

    actual = {
        name: {
            "kind": span.kind.value,
            "status": span.status.value,
            "input": payload(span.input),
            "output": payload(span.output),
            "identity_and_error": {key: value for key, value in span.attributes_string.items() if key in semantic_keys},
            "usage": span.attributes_number,
        }
        for name, span in spans.items()
    }
    assert actual == {
        "outer-agent": {
            "kind": "AGENT",
            "status": "success",
            "input": {"instruction": "hello"},
            "output": "hello back",
            "identity_and_error": {"gen_ai.agent.name": "outer-agent"},
            "usage": {},
        },
        "nested-agent": {
            "kind": "AGENT",
            "status": "success",
            "input": {"text": "hello"},
            "output": {"answer": "hello back"},
            "identity_and_error": {"gen_ai.agent.name": "nested-agent"},
            "usage": {},
        },
        "recorded-model": {
            "kind": "LLM",
            "status": "success",
            "input": {
                "model": "recorded-model",
                "messages": [
                    {
                        "role": "user",
                        "type": "human",
                        "content": "hello",
                        "id": None,
                        "name": None,
                        "additional_kwargs": {},
                        "response_metadata": {},
                    }
                ],
            },
            "output": {
                "messages": [
                    {
                        "role": "assistant",
                        "type": "ai",
                        "content": "hello back",
                        "id": "msg-1",
                        "name": None,
                        "additional_kwargs": {},
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                        "response_metadata": {"model_name": "recorded-model", "model_provider": "recorded-provider"},
                        "usage_metadata": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                    }
                ]
            },
            "identity_and_error": {"gen_ai.request.model": "recorded-model", "gen_ai.system": "recorded-provider"},
            "usage": {"llm.token_count.prompt": 11, "llm.token_count.completion": 3, "llm.token_count.total": 14},
        },
        "lookup": {
            "kind": "TOOL",
            "status": "success",
            "input": {"query": "hello"},
            "output": {"answer": "world"},
            "identity_and_error": {"gen_ai.tool.name": "lookup"},
            "usage": {},
        },
        "broken": {
            "kind": "TOOL",
            "status": "error",
            "input": "",
            "output": "",
            "identity_and_error": {
                "gen_ai.tool.name": "broken",
                "exception.type": "ValueError",
                "exception.message": "internal error: ValueError: recorded failure",
            },
            "usage": {},
        },
    }


@pytest.mark.parametrize("direction", ["input", "output"])
def test_deeply_nested_native_json_remains_raw_instead_of_failing_ingest(direction):
    value = "[" * 100_000
    key = "nemo_relay.start.input" if direction == "input" else "nemo_relay.end.output"
    domain = _domain(_span({"nemo_relay.scope_type": "agent", key: value}))
    assert getattr(domain, direction) == value
    assert domain.attributes_string[key] == value


@pytest.mark.parametrize(
    "scope,kind",
    [
        ("agent", SpanKind.AGENT),
        ("llm", SpanKind.LLM),
        ("tool", SpanKind.TOOL),
        ("function", SpanKind.CHAIN),
        ("retriever", SpanKind.RETRIEVER),
        ("embedder", SpanKind.EMBEDDING),
        ("reranker", SpanKind.RERANKER),
        ("guardrail", SpanKind.GUARDRAIL),
        ("evaluator", SpanKind.EVALUATOR),
        ("custom", SpanKind.UNKNOWN),
        ("unknown", SpanKind.UNKNOWN),
    ],
)
def test_native_scope_kinds(scope, kind):
    assert _domain(_span({"nemo_relay.scope_type": scope})).kind is kind


def test_standard_semantics_win_over_native_capture():
    attrs = {
        "nemo_relay.scope_type": "llm",
        "nemo_relay.model_name": "native-model",
        "gen_ai.request.model": "standard-model",
        "gen_ai.provider.name": "standard-provider",
        "gen_ai.usage.input_tokens": 0,
        "llm.token_count.completion": 5,
        "input.value": "",
        "output.value": "standard-output",
        "nemo_relay.start.input.content": {"messages": []},
        **_response(
            {
                "usage_metadata": {"input_tokens": 11, "output_tokens": 3},
                "response_metadata": {"model_provider": "native-provider"},
            }
        ),
    }
    domain = _domain(_span(attrs))
    assert domain.input == ""
    assert domain.output == "standard-output"
    assert domain.attributes_string["gen_ai.request.model"] == "standard-model"
    assert domain.attributes_string["gen_ai.system"] == "standard-provider"
    assert domain.attributes_number["llm.token_count.prompt"] == 0
    assert domain.attributes_number["llm.token_count.completion"] == 5
    assert domain.attributes_string["nemo_relay.model_name"] == "native-model"
    for explicit in ({"openinference.span.kind": "TOOL"}, {"gen_ai.operation.name": "invoke_agent"}):
        assert _domain(_span({**attrs, **explicit})).kind is not SpanKind.LLM
    identity = _domain(
        _span(
            {"nemo_relay.scope_type": "agent", "gen_ai.agent.name": "explicit", "gen_ai.agent.version": "v1"},
            name="scope-name",
        )
    )
    assert identity.attributes_string["gen_ai.agent.name"] == "explicit"
    assert identity.attributes_string["agent.version"] == "v1"


def test_native_messages_preserve_content_tool_calls_and_recorded_usage_details():
    content = [{"type": "text", "text": "thinking"}, {"type": "image_url", "image_url": {"url": "data:image/test"}}]
    calls = [{"id": "call-1", "name": "lookup", "args": {"query": "hello"}, "type": "tool_call"}]
    domain = _domain(
        _span(
            {
                "nemo_relay.scope_type": "llm",
                **_response(
                    {
                        "content": content,
                        "tool_calls": calls,
                        "id": "m1",
                        "usage_metadata": {
                            "input_tokens": 0,
                            "output_tokens": 7,
                            "input_token_details": {"cache_read": 2, "cache_creation": 1, "audio": 3},
                            "output_token_details": {"reasoning": 4, "audio": 1},
                        },
                    }
                ),
            }
        )
    )
    message = json.loads(domain.output)["messages"][0]
    assert message["content"] == content
    assert message["tool_calls"] == calls
    assert message["id"] == "m1"
    assert domain.attributes_number["llm.token_count.cached"] == 2
    assert domain.attributes_number["llm.token_count.prompt_details.cache_write"] == 1
    assert domain.attributes_number["llm.token_count.completion_details.reasoning"] == 4
    assert "llm.token_count.total" not in domain.attributes_number
    assert "gen_ai.system" not in domain.attributes_string


@pytest.mark.parametrize("payload", ["not json", "{bad", [], {"messages": "invalid"}, {"messages": []}])
def test_malformed_native_response_stays_raw_without_guessing(payload):
    domain = _domain(
        _span({"nemo_relay.scope_type": "llm", "nemo_relay.model_name": "gpt-not-a-provider", RESPONSE_KEY: payload})
    )
    assert "gen_ai.system" not in domain.attributes_string
    assert not domain.attributes_number
    assert RESPONSE_KEY in domain.attributes_string
    assert domain.output


@pytest.mark.parametrize("standard", [{"gen_ai.usage.input_tokens": 0}, {"llm.cost.total": 0}])
def test_any_standard_usage_suppresses_the_entire_native_usage_group(standard):
    native = _response(
        {
            "usage_metadata": {
                "input_tokens": 9,
                "output_tokens": 5,
                "total_tokens": 14,
                "input_token_details": {"cache_read": 2},
                "output_token_details": {"reasoning": 1},
            }
        }
    )
    domain = _domain(_span({"nemo_relay.scope_type": "llm", **native, **standard}))
    expected = {"llm.token_count.prompt": 0} if "gen_ai.usage.input_tokens" in standard else {"cost.total": 0}
    assert domain.attributes_number == expected
    assert RESPONSE_KEY in domain.attributes_string


@pytest.mark.parametrize(
    "standard", [{"gen_ai.request.model": "standard-model"}, {"gen_ai.provider.name": "standard-provider"}]
)
def test_standard_model_or_provider_suppresses_native_identity_as_a_group(standard):
    domain = _domain(
        _span(
            {
                "nemo_relay.scope_type": "llm",
                "nemo_relay.model_name": "native-model",
                **standard,
                **_response(
                    {"response_metadata": {"model_name": "response-model", "model_provider": "native-provider"}}
                ),
            }
        )
    )
    assert domain.attributes_string.get("gen_ai.request.model") == standard.get("gen_ai.request.model")
    assert domain.attributes_string.get("gen_ai.system") == standard.get("gen_ai.provider.name")
    assert domain.attributes_string["nemo_relay.model_name"] == "native-model"
    assert RESPONSE_KEY in domain.attributes_string


def test_top_level_payload_projection_does_not_split_dotted_user_keys():
    domain = _domain(
        _span(
            {
                "nemo_relay.scope_type": "tool",
                "nemo_relay.tool_call_id": "call-1",
                "nemo_relay.start.input.a.b": {"nested": [1, 2]},
                "nemo_relay.end.output": False,
            }
        )
    )
    assert json.loads(domain.input) == {"a.b": {"nested": [1, 2]}}
    assert domain.attributes_string["nemo_relay.tool_call_id"] == "call-1"
    assert domain.output == "false"


def test_root_span_payload_layer_wins_over_resource_aliases():
    attrs = {
        "nemo_relay.scope_type": "agent",
        "nemo_relay.start.input": "native request",
        "nemo_relay.end.output": "native answer",
    }
    resource = {"input.value": "resource request", "output.value": "resource answer"}
    domain = _domain(_span(attrs), resource)
    assert domain.input == "native request"
    assert domain.output == "native answer"
    domain = _domain(_span({**attrs, "input.value": "span request", "output.value": ""}), resource)
    assert domain.input == "span request"
    assert domain.output == ""


def test_non_relay_root_uses_its_own_gen_ai_payload_before_resource_input_value():
    resource = {"input.value": "resource request", "output.value": "resource answer"}
    attrs = {"gen_ai.input.messages": "recorded request", "gen_ai.output.messages": "recorded answer"}
    domain = _domain(_span(attrs), resource)
    assert (domain.input, domain.output) == ("recorded request", "recorded answer")
    empty = _domain(_span({"input.value": "", "output.value": ""}), resource)
    assert (empty.input, empty.output) == ("", "")
    missing = _domain(_span({}), resource)
    assert (missing.input, missing.output) == ("resource request", "resource answer")


def test_root_resource_payloads_do_not_leak_into_nested_calls():
    resource = {"input.value": "root request", "output.value": "root answer"}
    child = _span(
        {
            "nemo_relay.scope_type": "tool",
            "nemo_relay.start.input.query": "child query",
            "nemo_relay.end.output": "child answer",
        }
    )
    child.parent_span_id = b"p" * 8
    domain = _domain(child, resource)
    assert json.loads(domain.input) == {"query": "child query"}
    assert domain.output == "child answer"
    child = _span({"nemo_relay.scope_type": "tool"})
    child.parent_span_id = b"p" * 8
    domain = _domain(child, resource)
    assert domain.input == domain.output == ""


def test_consumed_native_resource_attributes_remain_inspectable():
    native_error = "nemo_relay.end.metadata.exception.message"
    domain = _domain(_span({"nemo_relay.scope_type": "tool"}), {native_error: "recorded failure"})
    assert domain.attributes_string[native_error] == "recorded failure"
    assert domain.attributes_string["exception.message"] == "recorded failure"


def test_resource_semantics_and_flattened_standard_messages_win():
    attrs = {
        "nemo_relay.scope_type": "llm",
        "nemo_relay.model_name": "native-model",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "standard input",
        "gen_ai.output.messages": [{"role": "assistant", "content": "standard output"}],
        "nemo_relay.start.input.content": {"messages": [{"role": "user", "content": "native"}]},
        **_response({"usage_metadata": {"input_tokens": 9}}),
    }
    domain = _domain(_span(attrs), {"gen_ai.request.model": "resource-model", "gen_ai.usage.input_tokens": 2})
    assert json.loads(domain.input)["messages"][0]["content"] == "standard input"
    assert json.loads(domain.output)[0]["content"] == "standard output"
    assert domain.attributes_string["gen_ai.request.model"] == "resource-model"
    assert domain.attributes_number["llm.token_count.prompt"] == 2


@pytest.mark.parametrize("value", [None, -1, True, 1.5, "7"])
def test_native_usage_invalid_or_missing_is_not_zero(value):
    domain = _domain(
        _span(
            {
                "nemo_relay.scope_type": "llm",
                **_response(
                    {
                        "usage_metadata": {"input_tokens": value},
                        "response_metadata": {"model_provider": {"invalid": "provider"}},
                    }
                ),
            }
        )
    )
    assert not domain.attributes_number
    assert "gen_ai.system" not in domain.attributes_string


@pytest.mark.parametrize("exception", [None, "ExplicitError"])
def test_non_relay_error_type_remains_raw_and_distinct(exception):
    attrs = {"error.type": "producer classification"}
    if exception is not None:
        attrs["exception.type"] = exception
    domain = _domain(_span(attrs))
    assert domain.attributes_string["error.type"] == "producer classification"
    assert domain.attributes_string.get("exception.type") == exception


@pytest.mark.parametrize("shape", ["array", "kvlist"])
def test_malformed_native_scope_type_does_not_reject_the_span(shape):
    span = _span({})
    attr = span.attributes.add(key="nemo_relay.scope_type")
    if shape == "array":
        attr.value.array_value.values.add(string_value="llm")
    else:
        attr.value.kvlist_value.values.add(key="scope").value.string_value = "llm"
    assert _domain(span).kind is SpanKind.UNKNOWN


@pytest.mark.parametrize("scope", ["agent", "tool", "llm"])
@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("text", ["[1, 2]", '{ "a": 1, "a": 2 }', '{ "overflow": 1e400 }'])
def test_exact_native_scalar_text_is_not_rewritten_as_json(scope, direction, text):
    prefix = "nemo_relay.start.input" if direction == "input" else "nemo_relay.end.output"
    domain = _domain(_span({"nemo_relay.scope_type": scope, prefix: text}))
    assert getattr(domain, direction) == text


def test_native_error_aliases_and_standard_error_precedence():
    attrs = {
        "nemo_relay.scope_type": "tool",
        "nemo_relay.end.metadata.otel.status_code": "ERROR",
        "nemo_relay.end.metadata.exception.type": "ValueError",
        "nemo_relay.end.metadata.otel.status_description": "recorded error",
    }
    domain = _domain(_span(attrs))
    assert domain.status is SpanStatus.ERROR
    assert domain.attributes_string["exception.type"] == "ValueError"
    assert domain.attributes_string["exception.message"] == "recorded error"
    domain = _domain(_span({**attrs, "exception.type": "ExplicitError", "exception.message": "explicit"}))
    assert domain.attributes_string["exception.type"] == "ExplicitError"
    assert domain.attributes_string["exception.message"] == "explicit"
    span = _span(attrs)
    span.status.message = "standard OTLP status message"
    span.status.code = 1
    domain = _domain(span)
    assert domain.status is SpanStatus.SUCCESS
    assert domain.attributes_string["exception.message"] == "standard OTLP status message"
