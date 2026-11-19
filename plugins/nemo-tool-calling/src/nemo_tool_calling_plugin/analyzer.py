# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ask an analyzer LLM for a prompt fix and a tool-schema patch, and check what it returns."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx
import jsonschema
from nemo_tool_calling_plugin.scoring import ToolCall

logger = logging.getLogger(__name__)

ANALYZER_TIMEOUT_SECONDS = 900.0
ANALYZER_MAX_TOKENS = 32768
_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

Severity = Literal["high", "medium", "low"]


class AnalyzerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_mcp(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": self.input_schema}


@dataclass
class AnalysisContext:
    agent: str
    system_prompt: str
    tools: list[ToolDefinition]
    metrics: dict[str, Any]
    rejection_reasons: dict[str, int]
    examples: list[dict[str, Any]]
    lint_findings: list[dict[str, Any]]
    models: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnalyzerFinding:
    title: str
    detail: str
    severity: Severity = "medium"
    tool: str | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigFix:
    title: str
    rationale: str
    system_prompt: str


@dataclass(frozen=True)
class SchemaFix:
    title: str
    rationale: str
    tools: tuple[ToolDefinition, ...]


@dataclass
class Analysis:
    findings: list[AnalyzerFinding] = field(default_factory=list)
    config_fix: ConfigFix | None = None
    schema_fix: SchemaFix | None = None
    errors: list[str] = field(default_factory=list)


class Analyzer(Protocol):
    model: str

    def analyze(self, context: AnalysisContext) -> Analysis: ...


def extract_json(message: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON object in a chat message, looking in ``content`` then ``reasoning_content``."""
    for key in ("content", "reasoning_content", "reasoning"):
        text = message.get(key)
        if not isinstance(text, str) or not text.strip():
            continue
        parsed = _json_object_in(text)
        if parsed is not None:
            return parsed
    raise AnalyzerError("The analyzer reply holds no JSON object.")


def _json_object_in(text: str) -> dict[str, Any] | None:
    stripped = _THINK.sub("", text)
    candidates = [match.group(1) for match in _FENCE.finditer(stripped)][::-1]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _severity(value: Any) -> Severity:
    return value if value in ("high", "medium", "low") else "medium"


def parse_findings(payload: Mapping[str, Any]) -> list[AnalyzerFinding]:
    findings: list[AnalyzerFinding] = []
    for item in payload.get("findings") or []:
        if not isinstance(item, Mapping) or not item.get("title"):
            continue
        findings.append(
            AnalyzerFinding(
                title=str(item["title"]),
                detail=str(item.get("detail") or ""),
                severity=_severity(item.get("severity")),
                tool=str(item["tool"]) if item.get("tool") else None,
                evidence=tuple(str(entry) for entry in item.get("evidence") or []),
            )
        )
    return findings


def parse_config_fix(payload: Mapping[str, Any], current_prompt: str) -> ConfigFix | None:
    raw = payload.get("config_fix")
    if not isinstance(raw, Mapping):
        return None
    full = raw.get("system_prompt")
    appended = raw.get("append_to_system_prompt")
    if isinstance(full, str) and full.strip():
        prompt = full.strip()
    elif isinstance(appended, str) and appended.strip():
        prompt = f"{current_prompt.rstrip()}\n\n{appended.strip()}"
    else:
        return None
    if prompt == current_prompt.strip():
        return None
    return ConfigFix(
        title=str(raw.get("title") or "Revise the system prompt"),
        rationale=str(raw.get("rationale") or ""),
        system_prompt=prompt,
    )


def parse_schema_fix(payload: Mapping[str, Any], current: Sequence[ToolDefinition]) -> SchemaFix | None:
    raw = payload.get("schema_fix")
    if not isinstance(raw, Mapping):
        return None
    by_name = {tool.name: tool for tool in current}
    patched: list[ToolDefinition] = []
    for item in raw.get("tools") or []:
        if not isinstance(item, Mapping) or item.get("name") not in by_name:
            continue
        original = by_name[str(item["name"])]
        schema = item.get("inputSchema", item.get("input_schema"))
        if not isinstance(schema, Mapping):
            continue
        description = item.get("description")
        patched.append(
            ToolDefinition(
                server=original.server,
                name=original.name,
                description=description if isinstance(description, str) and description else original.description,
                input_schema=dict(schema),
            )
        )
    if not patched:
        return None
    return SchemaFix(
        title=str(raw.get("title") or "Tighten the tool schema"),
        rationale=str(raw.get("rationale") or ""),
        tools=tuple(patched),
    )


def check_schema_fix(fix: SchemaFix, accepted: Mapping[str, Sequence[ToolCall]]) -> tuple[SchemaFix | None, list[dict]]:
    """Drop patches that are not valid JSON Schema; report accepted calls a patch would now reject."""
    kept: list[ToolDefinition] = []
    findings: list[dict[str, Any]] = []
    for tool in fix.tools:
        try:
            jsonschema.Draft202012Validator.check_schema(tool.input_schema)
        except jsonschema.SchemaError as error:
            findings.append(
                {
                    "id": f"schema-patch-invalid-{tool.name}",
                    "source": "failure-analysis",
                    "severity": "medium",
                    "tool": tool.name,
                    "title": "Suggested schema patch is not valid JSON Schema",
                    "detail": f"Dropped the patch for {tool.name}: {error.message}",
                    "evidence": [],
                }
            )
            continue
        kept.append(tool)
        validator = jsonschema.Draft202012Validator(tool.input_schema)
        calls = accepted.get(tool.name, [])
        broken = [call for call in calls if not validator.is_valid(call.arguments)]
        if broken:
            first_error = next(iter(validator.iter_errors(broken[0].arguments)))
            findings.append(
                {
                    "id": f"schema-patch-regression-{tool.name}",
                    "source": "failure-analysis",
                    "severity": "high",
                    "tool": tool.name,
                    "title": f"Schema patch rejects {len(broken)} of {len(calls)} calls the tool accepted",
                    "detail": "The patched schema is stricter than the tool for calls it accepted during the "
                    "baseline; review it before sending it to the tool's owner.",
                    "evidence": [f"{'/'.join(map(str, first_error.absolute_path)) or '<root>'}: {first_error.message}"],
                }
            )
    if not kept:
        return None, findings
    return SchemaFix(title=fix.title, rationale=fix.rationale, tools=tuple(kept)), findings


_SYSTEM = (
    "You diagnose why an AI agent's tool calls are rejected and propose minimal, concrete fixes. "
    "Reply with exactly one JSON object and nothing else."
)


def _context_block(context: AnalysisContext) -> str:
    tools = [{"server": tool.server, **tool.to_mcp()} for tool in context.tools]
    return json.dumps(
        {
            "agent": context.agent,
            "baseline_metrics": context.metrics,
            "rejection_reason_counts": context.rejection_reasons,
            "rejected_call_examples": context.examples,
            "schema_lint_findings": [
                {key: finding[key] for key in ("severity", "title", "detail")} for finding in context.lint_findings
            ],
            "platform_models": context.models,
            "platform_providers": context.providers,
            "current_system_prompt": context.system_prompt,
            "current_tools": tools,
        },
        indent=1,
        ensure_ascii=False,
    )


def config_fix_messages(context: AnalysisContext) -> list[dict[str, str]]:
    task = (
        "Task: propose an edit to the agent's system prompt that prevents the rejected calls above. You may not "
        "change the tools. Name real values from platform_models / platform_providers when rejections show the "
        "agent inventing models or providers. Keep every instruction that still applies.\n"
        "Reply with JSON:\n"
        '{"findings": [{"title": str, "detail": str, "severity": "high"|"medium"|"low", "tool": str, '
        '"evidence": [str]}],\n'
        ' "config_fix": {"title": str, "rationale": str, "system_prompt": "<the complete new system prompt>"}}'
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": f"{_context_block(context)}\n\n{task}"}]


def schema_fix_messages(context: AnalysisContext) -> list[dict[str, str]]:
    task = (
        "Task: propose a patched tool definition for the tools whose calls are rejected. The tool's own validation "
        "does not change, so the patch must accept a subset of what the tool accepts while making invalid calls "
        "impossible to write. Models with order-constrained grammars emit required keys first and optional keys in "
        "declared order, so: declare required keys first; set additionalProperties: false; split an object whose "
        "keys depend on a discriminator (such as a *_type field) into anyOf variants that each list their own "
        "required keys and at most two optional keys; use enum (never const) for fixed values; give arrays typed "
        "items; drop fields and variants the tool rejects anyway. Shorten the description so it does not repeat "
        "the schema.\n"
        "Reply with JSON:\n"
        '{"schema_fix": {"title": str, "rationale": str, "tools": [{"name": str, "description": str, '
        '"inputSchema": {...}}]}}'
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": f"{_context_block(context)}\n\n{task}"}]


class GatewayAnalyzer:
    """Analyzer backed by an OpenAI-compatible chat completions endpoint (the platform Inference Gateway)."""

    def __init__(self, *, base_url: str, model: str, timeout_seconds: float = ANALYZER_TIMEOUT_SECONDS) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def _complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": ANALYZER_MAX_TOKENS}
        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(f"{self._base_url}/chat/completions", json=body)
        if response.status_code >= 400:
            raise AnalyzerError(f"Analyzer model {self.model!r} returned HTTP {response.status_code}: {response.text}")
        choices = response.json().get("choices") or []
        if not choices:
            raise AnalyzerError(f"Analyzer model {self.model!r} returned no choices.")
        return extract_json(choices[0].get("message") or {})

    def analyze(self, context: AnalysisContext) -> Analysis:
        analysis = Analysis()
        try:
            payload = self._complete(config_fix_messages(context))
            analysis.findings.extend(parse_findings(payload))
            analysis.config_fix = parse_config_fix(payload, context.system_prompt)
        except (AnalyzerError, httpx.HTTPError) as error:
            logger.warning("Config-fix analysis failed: %s", error)
            analysis.errors.append(f"Config-fix analysis failed: {error}")
        try:
            payload = self._complete(schema_fix_messages(context))
            analysis.schema_fix = parse_schema_fix(payload, context.tools)
        except (AnalyzerError, httpx.HTTPError) as error:
            logger.warning("Schema-fix analysis failed: %s", error)
            analysis.errors.append(f"Schema-fix analysis failed: {error}")
        return analysis
