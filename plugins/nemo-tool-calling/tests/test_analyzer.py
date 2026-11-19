# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from nemo_tool_calling_plugin.analyzer import (
    AnalysisContext,
    AnalyzerError,
    GatewayAnalyzer,
    SchemaFix,
    ToolDefinition,
    check_schema_fix,
    extract_json,
    parse_config_fix,
    parse_schema_fix,
    short_reason,
)
from nemo_tool_calling_plugin.scoring import ToolCall

TOOL = ToolDefinition(
    server="data_designer",
    name="make",
    description="Make a thing.",
    input_schema={"type": "object", "properties": {"n": {"type": "number"}}},
)


def test_extract_json_from_fenced_content() -> None:
    message = {"content": 'Here you go:\n```json\n{"config_fix": {"system_prompt": "x"}}\n```\nDone.'}

    assert extract_json(message) == {"config_fix": {"system_prompt": "x"}}


def test_extract_json_falls_back_to_reasoning_content() -> None:
    message = {"content": "", "reasoning_content": 'Thinking... the answer is {"findings": []} ok'}

    assert extract_json(message) == {"findings": []}


def test_extract_json_strips_think_blocks() -> None:
    message = {"content": '<think>maybe {"wrong": 1</think>{"findings": [{"title": "t"}]}'}

    assert extract_json(message) == {"findings": [{"title": "t"}]}


def test_extract_json_without_json_raises() -> None:
    with pytest.raises(AnalyzerError):
        extract_json({"content": "no json here"})


def test_parse_config_fix_appends_a_section() -> None:
    fix = parse_config_fix({"config_fix": {"title": "t", "append_to_system_prompt": "Use model m."}}, "Base.")

    assert fix is not None
    assert fix.system_prompt == "Base.\n\nUse model m."


def test_parse_config_fix_ignores_an_unchanged_prompt() -> None:
    assert parse_config_fix({"config_fix": {"system_prompt": "Base."}}, "Base.") is None


def test_parse_schema_fix_keeps_known_tools_only() -> None:
    payload = {
        "schema_fix": {
            "title": "Tighten",
            "tools": [
                {"name": "make", "inputSchema": {"type": "object", "additionalProperties": False}},
                {"name": "unknown", "inputSchema": {}},
            ],
        }
    }

    fix = parse_schema_fix(payload, [TOOL])

    assert fix is not None
    assert [(tool.server, tool.name, tool.description) for tool in fix.tools] == [
        ("data_designer", "make", "Make a thing.")
    ]


def test_check_schema_fix_drops_invalid_schemas() -> None:
    fix = SchemaFix("t", "r", (ToolDefinition("s", "make", "", {"type": "not-a-type"}),))

    kept, findings = check_schema_fix(fix, {})

    assert kept is None
    assert findings[0]["id"] == "schema-patch-invalid-make"


def test_check_schema_fix_reports_accepted_calls_the_patch_rejects() -> None:
    strict = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
    fix = SchemaFix("t", "r", (ToolDefinition("s", "make", "", strict),))
    accepted = {"make": [ToolCall("make", {"n": 1}, "ok"), ToolCall("make", {"n": 1.5}, "ok")]}

    kept, findings = check_schema_fix(fix, accepted)

    assert kept is not None
    assert findings[0]["title"] == "Schema patch rejects 1 of 2 calls the tool accepted"
    assert findings[0]["evidence"] == ["n: 1.5 is not of type 'integer'"]


def _context() -> AnalysisContext:
    return AnalysisContext(
        agent="default/a",
        system_prompt="Base.",
        tools=[TOOL],
        metrics={},
        rejection_reasons={},
        examples=[],
        lint_findings=[],
    )


def _completion(payload: dict[str, Any], *, field: str = "content") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {field: f"```json\n{json.dumps(payload)}\n```"}}]})


@respx.mock
def test_gateway_analyzer_makes_a_config_call_and_a_schema_call() -> None:
    route = respx.post("http://gw/v1/chat/completions").mock(
        side_effect=[
            _completion({"findings": [{"title": "f", "severity": "high"}], "config_fix": {"system_prompt": "New."}}),
            _completion(
                {"schema_fix": {"tools": [{"name": "make", "inputSchema": {"type": "object"}}]}},
                field="reasoning_content",
            ),
        ]
    )

    analysis = GatewayAnalyzer(base_url="http://gw/v1", model="default/m").analyze(_context())

    assert route.call_count == 2
    assert json.loads(route.calls[0].request.content)["model"] == "default/m"
    assert analysis.findings[0].severity == "high"
    assert analysis.config_fix is not None and analysis.config_fix.system_prompt == "New."
    assert analysis.schema_fix is not None and analysis.schema_fix.tools[0].name == "make"
    assert analysis.errors == []


@respx.mock
def test_gateway_analyzer_records_errors_instead_of_failing() -> None:
    respx.post("http://gw/v1/chat/completions").mock(return_value=httpx.Response(503, text="down"))

    analysis = GatewayAnalyzer(base_url="http://gw/v1", model="m").analyze(_context())

    assert analysis.config_fix is None and analysis.schema_fix is None
    assert [(failure.step, failure.status, failure.reason) for failure in analysis.errors] == [
        ("config-fix", 503, "down"),
        ("schema-fix", 503, "down"),
    ]


@respx.mock
def test_gateway_analyzer_names_a_missing_virtual_model() -> None:
    body = {"detail": "No VirtualModel named 'default/nemotron3-5-super' in workspace 'default'. " + "x" * 2000}
    respx.post("http://gw/v1/chat/completions").mock(return_value=httpx.Response(404, json=body))

    analyzer = GatewayAnalyzer(base_url="http://gw/v1", model="default/nemotron3-5-super")
    [failure, _] = analyzer.analyze(_context()).errors

    assert failure.status == 404
    assert failure.reason == "no VirtualModel"
    assert failure.cause() == "analyzer model `default/nemotron3-5-super` is unavailable (HTTP 404, no VirtualModel)"
    assert failure.remedy().startswith("Register the model in the inference gateway or set `analyzer_model`")
    assert len(failure.message) < 400


@respx.mock
def test_gateway_analyzer_reports_a_transport_error() -> None:
    respx.post("http://gw/v1/chat/completions").mock(side_effect=httpx.ConnectError("refused"))

    [failure, _] = GatewayAnalyzer(base_url="http://gw/v1", model="m").analyze(_context()).errors

    assert failure.status is None
    assert failure.cause() == "analyzer model `m` failed (ConnectError)"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ('{"error": {"message": "model overloaded\\ntry later"}}', "model overloaded"),
        ("plain failure", "plain failure"),
        ("x" * 200, "x" * 79 + "…"),
        ("", ""),
    ],
)
def test_short_reason(body: str, reason: str) -> None:
    assert short_reason(body) == reason
