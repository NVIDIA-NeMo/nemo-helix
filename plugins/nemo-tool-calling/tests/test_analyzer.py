# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
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


def test_parse_schema_fix_still_accepts_full_definitions() -> None:
    payload = {
        "schema_fix": {
            "title": "Tighten",
            "tools": [
                {"name": "make", "inputSchema": {"type": "object", "additionalProperties": False}},
                {"name": "unknown", "inputSchema": {}},
            ],
        }
    }

    fix, skipped = parse_schema_fix(payload, [TOOL])

    assert fix is not None
    assert skipped == []
    assert [(tool.server, tool.name, tool.description) for tool in fix.tools] == [
        ("data_designer", "make", "Make a thing.")
    ]
    assert fix.tools[0].input_schema == {"type": "object", "additionalProperties": False}


NESTED = ToolDefinition(
    server="data_designer",
    name="make",
    description="Make a thing.",
    input_schema={
        "type": "object",
        "properties": {
            "job": {"type": "object", "properties": {"columns": {"type": "array", "items": {"type": "object"}}}}
        },
    },
)


def test_parse_schema_fix_applies_pointer_edits_to_the_published_schema() -> None:
    column = {"type": "object", "required": ["name"], "additionalProperties": False, "properties": {"name": {}}}
    payload = {
        "schema_fix": {
            "title": "Close the columns",
            "edits": [
                {"tool": "make", "path": "/properties/job/properties/columns/items", "value": column},
                {"tool": "make", "path": "/properties/job/additionalProperties", "value": False},
                {"tool": "make", "path": "/required", "value": ["job"]},
            ],
            "descriptions": {"make": "Make a thing from columns."},
        }
    }

    fix, skipped = parse_schema_fix(payload, [NESTED])

    assert skipped == []
    assert fix is not None
    [tool] = fix.tools
    assert tool.description == "Make a thing from columns."
    assert tool.input_schema["required"] == ["job"]
    assert tool.input_schema["properties"]["job"]["additionalProperties"] is False
    assert tool.input_schema["properties"]["job"]["properties"]["columns"]["items"] == column
    assert NESTED.input_schema["properties"]["job"]["properties"]["columns"]["items"] == {"type": "object"}


def test_parse_schema_fix_skips_bad_edits_with_a_finding() -> None:
    payload = {
        "schema_fix": {
            "edits": [
                {"tool": "make", "path": "/properties/nope/properties/x", "value": {}},
                {"tool": "make", "path": "properties/job", "value": {}},
                {"tool": "make", "path": "", "value": ["not", "an", "object"]},
                {"tool": "other", "path": "/x", "value": {}},
                {"tool": "make", "path": "/properties/job/description", "value": "The job."},
            ]
        }
    }

    fix, skipped = parse_schema_fix(payload, [NESTED])

    assert fix is not None
    assert fix.tools[0].input_schema["properties"]["job"]["description"] == "The job."
    assert [finding.detail for finding in skipped] == [
        "Edit names unknown tool 'other'.",
        "Could not apply an edit at /properties/nope/properties/x: '/properties/nope' does not exist.",
        "Could not apply an edit at properties/job: a JSON Pointer must be empty or start with '/'.",
        "Could not apply an edit at <root>: the input schema must stay an object.",
    ]
    assert {finding.severity for finding in skipped} == {"low"}


def test_parse_schema_fix_with_only_bad_edits_proposes_nothing() -> None:
    payload = {"schema_fix": {"edits": [{"tool": "make", "path": "/a/b", "value": {}}]}}

    fix, skipped = parse_schema_fix(payload, [NESTED])

    assert fix is None
    assert len(skipped) == 1


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


def _raw_completion(content: str, *, finish_reason: str = "stop") -> httpx.Response:
    choice = {"message": {"content": content, "reasoning_content": "thinking"}, "finish_reason": finish_reason}
    return httpx.Response(200, json={"choices": [choice], "usage": {"prompt_tokens": 10, "completion_tokens": 20}})


GOOD_CONFIG = json.dumps({"findings": [], "config_fix": {"system_prompt": "New."}})
GOOD_SCHEMA = json.dumps({"schema_fix": {"edits": [{"tool": "make", "path": "/additionalProperties", "value": False}]}})


@respx.mock
def test_replies_are_saved_on_success(tmp_path: Path) -> None:
    respx.post("http://gw/v1/chat/completions").mock(
        side_effect=[_raw_completion(GOOD_CONFIG), _raw_completion(GOOD_SCHEMA)]
    )

    analysis = GatewayAnalyzer(base_url="http://gw/v1", model="m", replies_dir=tmp_path).analyze(_context())

    assert analysis.schema_fix is not None
    assert analysis.schema_fix.tools[0].input_schema["additionalProperties"] is False
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config-fix.json", "schema-fix.json"]
    saved = json.loads((tmp_path / "schema-fix.json").read_text())
    assert saved["step"] == "schema-fix"
    assert saved["model"] == "m"
    assert saved["request"]["messages"] == 2
    assert saved["request"]["roles"] == ["system", "user"]
    assert saved["request"]["prompt_chars"] > 0
    assert saved["content"] == GOOD_SCHEMA
    assert saved["reasoning_content"] == "thinking"
    assert saved["finish_reason"] == "stop"
    assert saved["usage"] == {"prompt_tokens": 10, "completion_tokens": 20}
    assert saved["parse"] == {"ok": True}


@respx.mock
def test_replies_are_saved_on_http_failure(tmp_path: Path) -> None:
    respx.post("http://gw/v1/chat/completions").mock(return_value=httpx.Response(404, text="No VirtualModel named m"))

    GatewayAnalyzer(base_url="http://gw/v1", model="m", replies_dir=tmp_path).analyze(_context())

    saved = json.loads((tmp_path / "config-fix.json").read_text())
    assert saved["status"] == 404
    assert saved["error_body"] == "No VirtualModel named m"
    assert saved["parse"]["ok"] is False
    assert "HTTP 404" in saved["parse"]["error"]


@respx.mock
def test_replies_are_saved_on_transport_failure(tmp_path: Path) -> None:
    respx.post("http://gw/v1/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))

    GatewayAnalyzer(base_url="http://gw/v1", model="m", replies_dir=tmp_path).analyze(_context())

    saved = json.loads((tmp_path / "schema-fix.json").read_text())
    assert saved["parse"] == {"ok": False, "error": "ReadTimeout: slow"}


@respx.mock
def test_an_unparseable_reply_is_retried_once_with_the_parse_error(tmp_path: Path) -> None:
    bad = "Here is the patch: {{ column }} and then {broken"
    route = respx.post("http://gw/v1/chat/completions").mock(
        side_effect=[_raw_completion(GOOD_CONFIG), _raw_completion(bad), _raw_completion(GOOD_SCHEMA)]
    )

    analysis = GatewayAnalyzer(base_url="http://gw/v1", model="m", replies_dir=tmp_path).analyze(_context())

    assert route.call_count == 3
    assert analysis.errors == []
    assert analysis.schema_fix is not None
    retry_request = json.loads(route.calls[2].request.content)["messages"]
    assert [message["role"] for message in retry_request] == ["system", "user", "assistant", "user"]
    assert retry_request[2]["content"] == bad
    assert "could not be parsed" in retry_request[3]["content"]
    assert "'schema_fix'" in retry_request[3]["content"]
    first = json.loads((tmp_path / "schema-fix.json").read_text())
    retry = json.loads((tmp_path / "schema-fix-retry.json").read_text())
    assert first["parse"]["ok"] is False
    assert retry["parse"] == {"ok": True}
    assert retry["request"]["roles"] == ["system", "user", "assistant", "user"]


@respx.mock
def test_a_reply_that_fails_twice_is_an_analyzer_failure(tmp_path: Path) -> None:
    respx.post("http://gw/v1/chat/completions").mock(
        side_effect=[_raw_completion(GOOD_CONFIG), _raw_completion("nope"), _raw_completion("still nope")]
    )

    analysis = GatewayAnalyzer(base_url="http://gw/v1", model="m", replies_dir=tmp_path).analyze(_context())

    [failure] = analysis.errors
    assert (failure.step, failure.reason) == ("schema-fix", "no JSON in the reply")
    assert json.loads((tmp_path / "schema-fix-retry.json").read_text())["parse"]["ok"] is False
