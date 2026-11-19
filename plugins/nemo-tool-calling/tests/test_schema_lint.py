# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_tool_calling_plugin.schema_lint import lint_tool

COLUMNS = "job_request.spec.config.columns[]"


def _rules_at(findings: list, path: str) -> set[str]:
    return {finding.rule for finding in findings if finding.path == path}


def test_pre_fix_schema_flags_the_column_shape(pre_fix_tool: dict[str, Any]) -> None:
    findings = lint_tool(pre_fix_tool)

    assert {"open-object", "many-optional", "discriminator-in-prose"} <= _rules_at(findings, COLUMNS)
    assert all(finding.severity == "high" for finding in findings if finding.path == COLUMNS)
    assert "required-not-first" in _rules_at(findings, "job_request")


def test_post_fix_schema_has_no_high_severity_findings(post_fix_tool: dict[str, Any]) -> None:
    findings = lint_tool(post_fix_tool)

    assert [finding for finding in findings if finding.severity == "high"] == []
    assert not any(finding.rule in ("discriminator-in-prose", "open-object") for finding in findings)


def test_post_fix_schema_is_cleaner_than_pre_fix(pre_fix_tool: dict[str, Any], post_fix_tool: dict[str, Any]) -> None:
    assert len(lint_tool(post_fix_tool)) < len(lint_tool(pre_fix_tool))


def _tool(schema: dict[str, Any]) -> dict[str, Any]:
    return {"name": "t", "description": "", "inputSchema": schema}


def test_flags_untyped_array_items() -> None:
    findings = lint_tool(
        _tool({"type": "object", "properties": {"rows": {"type": "array", "items": {"type": "object"}}}})
    )

    assert "untyped-items" in _rules_at(findings, "rows")


def test_flags_enum_listed_only_in_prose() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"unit": {"type": "string", "description": 'One of "D", "W" or "M".'}},
    }

    assert _rules_at(lint_tool(_tool(schema)), "unit") == {"enum-in-prose"}


def test_flags_const() -> None:
    schema = {"type": "object", "additionalProperties": False, "properties": {"kind": {"type": "string", "const": "a"}}}

    assert _rules_at(lint_tool(_tool(schema)), "kind") == {"const"}


def test_closed_object_with_required_keys_first_is_clean() -> None:
    schema = {
        "type": "object",
        "required": ["a"],
        "additionalProperties": False,
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
    }

    assert lint_tool(_tool(schema)) == []


def test_finding_serializes_to_contract_shape(pre_fix_tool: dict[str, Any]) -> None:
    finding = lint_tool(pre_fix_tool)[0].to_finding()

    assert set(finding) == {"id", "source", "severity", "tool", "title", "detail", "evidence"}
    assert finding["source"] == "schema-lint"
    assert finding["id"].startswith("lint-")
