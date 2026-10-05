# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple OTLP trace ingest test."""

import json
from decimal import Decimal

from fastapi.testclient import TestClient
from nhx.intake.repository.clickhouse.tables import ClickHouseTable, qualified_table
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient


def test_otlp_ingest_simple_llm_span(client: TestClient, make_otlp_request):
    input_payload = {"messages": [{"role": "user", "content": "hello"}]}
    output_payload = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    body = make_otlp_request(
        [
            {
                "name": "llm-call",
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "gen_ai.conversation.id": "conv-simple",
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": "gpt-4o-mini",
                    "gen_ai.usage.input_tokens": 12,
                    "gen_ai.usage.output_tokens": 8,
                    "gen_ai.usage.total_tokens": 20,
                    "input.value": json.dumps(input_payload, separators=(",", ":")),
                    "output.value": json.dumps(output_payload, separators=(",", ":")),
                },
            }
        ]
    )

    ingest_response = client.post(
        "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces",
        content=body,
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert ingest_response.status_code == 200, ingest_response.text
    assert ingest_response.json() == {"errors": []}

    spans_response = client.get(
        "/apis/intake/v2/workspaces/default/spans",
        params={"filter[session_id]": "conv-simple", "filter[source]": "otel"},
    )
    assert spans_response.status_code == 200, spans_response.text
    spans = spans_response.json()["data"]
    assert len(spans) == 1
    span = spans[0]
    assert span["session_id"] == "conv-simple"
    assert span["kind"] == "LLM"
    assert span["model"] == "gpt-4o-mini"
    assert span["input_tokens"] == 12
    assert span["output_tokens"] == 8
    assert span["total_tokens"] == 20
    assert json.loads(span["input"]) == input_payload
    assert json.loads(span["output"]) == output_payload
    assert "attributes_string" not in span
    assert "attributes_number" not in span


def test_otlp_ingest_openinference_session_id(client: TestClient, make_otlp_request):
    body = make_otlp_request(
        [
            {
                "name": "ChatOpenAI",
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "session.id": "openinference-session",
                    "user.id": "openinference-user",
                    "tag.tags": ["intake-smoke"],
                    "llm.provider": "openai",
                    "llm.system": "openai",
                    "llm.model_name": "openai/openai/gpt-5.5",
                    "llm.token_count.prompt": 183,
                    "llm.token_count.completion": 4,
                    "llm.token_count.total": 187,
                    "llm.token_count.prompt_details.cache_read": 12,
                    "llm.token_count.completion_details.reasoning": 2,
                    "llm.cost.prompt": 0.0021,
                    "llm.cost.completion": 0.0045,
                    "llm.cost.total": 0.0066,
                    "llm.cost.prompt_details.cache_read": 0.0003,
                    "llm.cost.completion_details.reasoning": 0.0024,
                    "exception.message": "",
                    "input.value": '{"langchain":"full-input"}',
                    "output.value": '{"langchain":"full-output"}',
                    "llm.input_messages.0.message.role": "system",
                    "llm.input_messages.0.message.content": "You are terse.",
                    "llm.input_messages.1.message.role": "user",
                    "llm.input_messages.1.message.content": "What is 6 * 7?",
                    "llm.input_messages.2.message.role": "assistant",
                    "llm.input_messages.2.message.tool_calls.0.tool_call.id": "call-1",
                    "llm.input_messages.2.message.tool_calls.0.tool_call.function.name": "multiply",
                    "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments": '{"a":6,"b":7}',
                    "llm.input_messages.3.message.role": "tool",
                    "llm.input_messages.3.message.content": "42",
                    "llm.input_messages.3.message.name": "multiply",
                    "llm.input_messages.3.message.tool_call_id": "call-1",
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": "42",
                    "llm.invocation_parameters": '{"model":"openai/openai/gpt-5.5","tools":[{"large":"schema"}]}',
                    "llm.tools.0.tool.json_schema": '{"type":"function","function":{"name":"multiply"}}',
                },
            }
        ]
    )

    ingest_response = client.post(
        "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces",
        content=body,
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert ingest_response.status_code == 200, ingest_response.text

    spans_response = client.get(
        "/apis/intake/v2/workspaces/default/spans",
        params={"filter[session_id]": "openinference-session", "filter[source]": "otel"},
    )
    assert spans_response.status_code == 200, spans_response.text
    spans = spans_response.json()["data"]
    assert len(spans) == 1
    span = spans[0]
    assert span["session_id"] == "openinference-session"
    assert span["kind"] == "LLM"
    assert span["provider"] == "openai"
    assert span["model"] == "openai/openai/gpt-5.5"
    assert span["input_tokens"] == 183
    assert span["output_tokens"] == 4
    assert span["cached_tokens"] == 12
    assert span["total_tokens"] == 187
    assert span["usage_details"] == {"completion_details.reasoning": 2}
    assert Decimal(str(span["cost_total_usd"])) == Decimal("0.0066")
    assert Decimal(str(span["cost_input_usd"])) == Decimal("0.0021")
    assert Decimal(str(span["cost_output_usd"])) == Decimal("0.0045")
    assert json.loads(span["input"]) == {
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is 6 * 7?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "multiply", "arguments": '{"a":6,"b":7}'},
                    }
                ],
            },
            {"role": "tool", "content": "42", "name": "multiply", "tool_call_id": "call-1"},
        ]
    }
    assert json.loads(span["output"]) == {"messages": [{"role": "assistant", "content": "42"}]}
    assert "attributes_string" not in span
    assert "error_message" not in span


def test_otlp_reingest_same_batch_deduplicates_before_and_after_merge(
    client: TestClient,
    make_otlp_request,
    clickhouse_client: ClickHouseSpanClient,
    run_async,
):
    body = make_otlp_request(
        [
            {
                "name": "repeatable-llm",
                "span_id": "00000000000000aa",
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "gen_ai.conversation.id": "conv-idempotent",
                    "gen_ai.response.model": "model-idempotent",
                    "gen_ai.usage.input_tokens": 123,
                },
            }
        ]
    )

    for _ in range(2):
        ingest_response = client.post(
            "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces",
            content=body,
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert ingest_response.status_code == 200, ingest_response.text
        assert ingest_response.json() == {"errors": []}

    traces_response = client.get(
        "/apis/intake/v2/workspaces/default/traces",
        params={"filter[session_id]": "conv-idempotent"},
    )
    assert traces_response.status_code == 200, traces_response.text
    trace_payload = traces_response.json()
    assert trace_payload["pagination"]["total_results"] == 1
    assert trace_payload["data"][0]["span_count"] == 1
    assert trace_payload["data"][0]["input_tokens"] == 123

    run_async(
        clickhouse_client.command(
            f"OPTIMIZE TABLE {qualified_table(clickhouse_client.database, ClickHouseTable.SPANS)} FINAL"
        )
    )

    spans_response = client.get(
        "/apis/intake/v2/workspaces/default/spans",
        params={"filter[session_id]": "conv-idempotent", "filter[source]": "otel"},
    )
    assert spans_response.status_code == 200, spans_response.text
    payload = spans_response.json()
    assert payload["pagination"]["total_results"] == 1
    assert len(payload["data"]) == 1
    assert payload["data"][0]["span_id"] == "00000000000000aa"
    assert payload["data"][0]["model"] == "model-idempotent"


def test_otlp_ingest_reports_malformed_span(client: TestClient, make_otlp_request):
    body = make_otlp_request([{"name": "missing-trace-id", "trace_id": ""}])

    ingest_response = client.post(
        "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces",
        content=body,
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert ingest_response.status_code == 200, ingest_response.text
    payload = ingest_response.json()
    assert payload["errors"] == ["span 0000000000000001: trace_id is required"]
