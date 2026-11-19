# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ask an analyzer LLM for a prompt fix and a tool-schema patch, and check what it returns."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import jsonschema
from nemo_tool_calling_plugin.json_extract import json_object_in, parse_error
from nemo_tool_calling_plugin.schema_edits import SchemaEdit, apply_edits
from nemo_tool_calling_plugin.scoring import ToolCall

logger = logging.getLogger(__name__)

ANALYZER_TIMEOUT_SECONDS = 900.0
ANALYZER_MAX_TOKENS = 32768
CONFIG_FIX_KEYS = ("config_fix", "findings")
SCHEMA_FIX_KEYS = ("schema_fix",)

Severity = Literal["high", "medium", "low"]


AnalysisStep = Literal["config-fix", "schema-fix"]
STEP_LABELS: dict[AnalysisStep, str] = {"config-fix": "prompt fix", "schema-fix": "schema patch"}
MAX_REASON_CHARS = 80
MAX_DETAIL_CHARS = 300
_NO_VIRTUAL_MODEL = re.compile(r"No VirtualModel named", re.IGNORECASE)


class AnalyzerError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, reason: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class AnalyzerFailure:
    step: AnalysisStep
    model: str
    message: str
    status: int | None = None
    reason: str = ""

    def cause(self) -> str:
        detail = ", ".join(part for part in (f"HTTP {self.status}" if self.status else "", self.reason) if part)
        verb = "is unavailable" if self.status == 404 else "failed"
        return f"analyzer model `{self.model}` {verb}" + (f" ({detail})" if detail else "")

    def remedy(self) -> str:
        if self.status == 404:
            return "Register the model in the inference gateway or set `analyzer_model`, then re-run."
        if self.status in (401, 403):
            return "Give the job access to the model or set `analyzer_model`, then re-run."
        return "Check the analyzer model or set `analyzer_model`, then re-run."


def short_reason(body: str) -> str:
    """A few words on why a gateway call failed, from its error body."""
    text = body
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, Mapping):
        detail = parsed.get("detail") or parsed.get("message") or parsed.get("error")
        if isinstance(detail, Mapping):
            detail = detail.get("message") or detail.get("detail")
        if isinstance(detail, str):
            text = detail
    if _NO_VIRTUAL_MODEL.search(text):
        return "no VirtualModel"
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first if len(first) <= MAX_REASON_CHARS else f"{first[: MAX_REASON_CHARS - 1]}…"


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
    errors: list[AnalyzerFailure] = field(default_factory=list)


class Analyzer(Protocol):
    model: str

    def analyze(self, context: AnalysisContext) -> Analysis: ...


def extract_json(message: Mapping[str, Any], *, keys: Sequence[str] = ()) -> dict[str, Any]:
    """The JSON object in a chat message, looking in ``content`` then ``reasoning_content``."""
    texts = [message.get(key) for key in ("content", "reasoning_content", "reasoning")]
    texts = [text for text in texts if isinstance(text, str) and text.strip()]
    for text in texts:
        parsed = json_object_in(text, keys=keys)
        if parsed is not None:
            return parsed
    wanted = f" with a {' or '.join(repr(key) for key in keys)} key" if keys else ""
    detail = parse_error(texts[0]) if texts else "the reply is empty"
    raise AnalyzerError(f"The analyzer reply holds no JSON object{wanted}: {detail}", reason="no JSON in the reply")


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


def _skipped_edit(tool: str, detail: str) -> AnalyzerFinding:
    return AnalyzerFinding(title="Skipped a suggested schema edit", detail=detail, severity="low", tool=tool)


def parse_schema_fix(
    payload: Mapping[str, Any], current: Sequence[ToolDefinition]
) -> tuple[SchemaFix | None, list[AnalyzerFinding]]:
    """Build patched tool definitions from ``edits`` (JSON Pointer into ``inputSchema``) or full ``tools``."""
    raw = payload.get("schema_fix")
    if not isinstance(raw, Mapping):
        return None, []
    published = {tool.name: tool for tool in current}
    starts: dict[str, ToolDefinition] = {}
    for item in raw.get("tools") or []:
        if not isinstance(item, Mapping) or item.get("name") not in published:
            continue
        schema = item.get("inputSchema", item.get("input_schema"))
        if isinstance(schema, Mapping):
            original = published[str(item["name"])]
            description = item.get("description")
            starts[original.name] = ToolDefinition(
                server=original.server,
                name=original.name,
                description=description if isinstance(description, str) and description else original.description,
                input_schema=dict(schema),
            )
    descriptions = {
        str(name): text
        for name, text in (raw.get("descriptions") or {}).items()
        if name in published and isinstance(text, str) and text
    }
    edits: dict[str, list[SchemaEdit]] = {}
    skipped: list[AnalyzerFinding] = []
    for item in raw.get("edits") or []:
        if not isinstance(item, Mapping):
            continue
        tool = str(item.get("tool") or (current[0].name if len(current) == 1 else ""))
        if tool not in published:
            skipped.append(_skipped_edit(tool, f"Edit names unknown tool {tool!r}."))
            continue
        if isinstance(item.get("description"), str) and "path" not in item:
            descriptions[tool] = item["description"]
            continue
        if not isinstance(item.get("path"), str) or "value" not in item:
            skipped.append(_skipped_edit(tool, "Edit has no string 'path' or no 'value'."))
            continue
        edits.setdefault(tool, []).append(SchemaEdit(tool=tool, path=item["path"], value=item["value"]))

    patched: list[ToolDefinition] = []
    for name in [*published]:
        if name not in starts and name not in edits and name not in descriptions:
            continue
        start = starts.get(name, published[name])
        schema, problems = apply_edits(start.input_schema, edits.get(name, []))
        skipped.extend(_skipped_edit(name, f"Could not apply an edit at {problem}.") for problem in problems)
        description = descriptions.get(name, start.description)
        original = published[name]
        if schema == original.input_schema and description == original.description:
            continue
        patched.append(ToolDefinition(server=original.server, name=name, description=description, input_schema=schema))
    if not patched:
        return None, skipped
    fix = SchemaFix(
        title=str(raw.get("title") or "Tighten the tool schema"),
        rationale=str(raw.get("rationale") or ""),
        tools=tuple(patched),
    )
    return fix, skipped


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
        "Do not repeat the whole schema. Return edits: each edit replaces the subschema at a JSON Pointer (RFC 6901) "
        "into the tool's inputSchema, or adds a key when the pointer's parent exists, e.g. "
        '"/properties/job_request/properties/spec/properties/config/properties/columns/items". Prefer a few edits '
        "high in the tree over many small ones. Reply with exactly one JSON object, no prose, no comments:\n"
        '{"schema_fix": {"title": str, "rationale": str, '
        '"edits": [{"tool": str, "path": "<JSON Pointer into inputSchema>", "value": <replacement subschema>}], '
        '"descriptions": {"<tool name>": "<new tool description, optional>"}}}'
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": f"{_context_block(context)}\n\n{task}"}]


RETRY_INSTRUCTION = (
    "Your reply could not be parsed: {error}. Reply again with only the JSON object: no prose, no code fences, "
    "no comments, no trailing commas."
)


def _request_summary(messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "messages": len(messages),
        "roles": [message["role"] for message in messages],
        "prompt_chars": sum(len(message["content"]) for message in messages),
    }


class GatewayAnalyzer:
    """Analyzer backed by an OpenAI-compatible chat completions endpoint (the platform Inference Gateway).

    Every reply, including failed calls and retries, is written to ``replies_dir/<step>.json`` when one is given.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = ANALYZER_TIMEOUT_SECONDS,
        replies_dir: Path | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._replies_dir = replies_dir

    def _save(self, name: str, record: dict[str, Any]) -> None:
        if self._replies_dir is None:
            return
        self._replies_dir.mkdir(parents=True, exist_ok=True)
        (self._replies_dir / f"{name}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _post(self, messages: list[dict[str, str]], record: dict[str, Any]) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": ANALYZER_MAX_TOKENS}
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(f"{self._base_url}/chat/completions", json=body)
        except httpx.HTTPError as error:
            record["parse"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            raise
        record["status"] = response.status_code
        if response.status_code >= 400:
            excerpt = (
                response.text if len(response.text) <= MAX_DETAIL_CHARS else f"{response.text[:MAX_DETAIL_CHARS]}…"
            )
            error = AnalyzerError(
                f"Analyzer model {self.model!r} returned HTTP {response.status_code}: {excerpt}",
                status=response.status_code,
                reason=short_reason(response.text),
            )
            record["error_body"] = response.text
            record["parse"] = {"ok": False, "error": str(error)}
            raise error
        reply = response.json()
        choice = (reply.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        record.update(
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content", message.get("reasoning")),
            finish_reason=choice.get("finish_reason"),
            usage=reply.get("usage"),
        )
        if not reply.get("choices"):
            error = AnalyzerError(
                f"Analyzer model {self.model!r} returned no choices.", reason="no choices in the reply"
            )
            record["parse"] = {"ok": False, "error": str(error)}
            raise error
        return message

    def _call(
        self, step: AnalysisStep, messages: list[dict[str, str]], keys: Sequence[str], *, retry: bool = True
    ) -> dict[str, Any]:
        name = step if retry else f"{step}-retry"
        record: dict[str, Any] = {"step": name, "model": self.model, "request": _request_summary(messages)}
        try:
            message = self._post(messages, record)
            try:
                payload = extract_json(message, keys=keys)
            except AnalyzerError as error:
                record["parse"] = {"ok": False, "error": str(error)}
                if not retry:
                    raise
                logger.warning("%s reply did not parse (%s); asking once more for only the JSON", step, error)
                previous = str(message.get("content") or message.get("reasoning_content") or "")
                follow_up = [
                    *messages,
                    {"role": "assistant", "content": previous},
                    {"role": "user", "content": RETRY_INSTRUCTION.format(error=error)},
                ]
                return self._call(step, follow_up, keys, retry=False)
            record["parse"] = {"ok": True}
            return payload
        finally:
            self._save(name, record)

    def _failure(self, step: AnalysisStep, error: AnalyzerError | httpx.HTTPError) -> AnalyzerFailure:
        logger.warning("%s analysis failed: %s", step, error)
        if isinstance(error, AnalyzerError):
            return AnalyzerFailure(step, self.model, str(error), status=error.status, reason=error.reason)
        return AnalyzerFailure(step, self.model, f"{type(error).__name__}: {error}", reason=type(error).__name__)

    def analyze(self, context: AnalysisContext) -> Analysis:
        analysis = Analysis()
        try:
            payload = self._call("config-fix", config_fix_messages(context), CONFIG_FIX_KEYS)
            analysis.findings.extend(parse_findings(payload))
            analysis.config_fix = parse_config_fix(payload, context.system_prompt)
        except (AnalyzerError, httpx.HTTPError) as error:
            analysis.errors.append(self._failure("config-fix", error))
        try:
            payload = self._call("schema-fix", schema_fix_messages(context), SCHEMA_FIX_KEYS)
            analysis.schema_fix, skipped = parse_schema_fix(payload, context.tools)
            analysis.findings.extend(skipped)
        except (AnalyzerError, httpx.HTTPError) as error:
            analysis.errors.append(self._failure("schema-fix", error))
        return analysis
