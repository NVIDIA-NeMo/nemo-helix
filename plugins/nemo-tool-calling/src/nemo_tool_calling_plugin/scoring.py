# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score the tool calls of agent runs: which were accepted, which rejected, and why."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jsonschema

REJECTION_PREFIX = "TOOL CALL REJECTED"
REJECTION_STATUS = "422 Unprocessable Entity"
MCP_ERROR_REASON = "tool returned an error"
MAX_EXAMPLES = 5
DEFAULT_CALL_BUDGET = 3
ARGUMENT_EXCERPT_CHARS = 400

REASON_CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("corrupted or unknown key", re.compile(r"unknown field")),
    ("missing required field", re.compile(r"requires '|is a required property")),
    ("model/provider not on platform", re.compile(r"not registered|not served")),
    ("undeclared model alias", re.compile(r"no model is configured with alias")),
    ("model_configs as a column", re.compile(r"cannot be built from a description")),
    ("bad column reference", re.compile(r"not an earlier column")),
    ("invalid job name", re.compile(r"entity name pattern")),
    ("misplaced or extra property", re.compile(r"Additional properties")),
    ("wrong value type", re.compile(r"is not of type|is not valid under any of the given schemas")),
    ("value not allowed", re.compile(r"is not one of")),
)


ERROR_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\W*(error|exception|failed)\b", re.IGNORECASE),
    re.compile(r"\b(error|exception)\s*:", re.IGNORECASE),
    re.compile(r"traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"^\W*invalid\b|\binvalid (argument|input|value|request|parameter|field)s?\b", re.IGNORECASE),
    re.compile(r"\b(HTTP|status)(\s*code)?\W{0,3}[45]\d\d\b", re.IGNORECASE),
    re.compile(
        r"\b[45]\d\d (Bad Request|Unauthorized|Forbidden|Not Found|Conflict|Unprocessable|Too Many|Internal Server)"
    ),
)
_JSON_ERROR_KEYS = ("error", "errors", "exception")


def looks_like_error(text: str) -> bool:
    """Heuristic for tools that report failure in prose; a JSON result is judged by its error keys only."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return any(pattern.search(text) for pattern in ERROR_TEXT_PATTERNS)
    if isinstance(parsed, Mapping):
        return any(parsed.get(key) for key in _JSON_ERROR_KEYS)
    return False


@dataclass
class ToolCall:
    tool: str
    arguments: Any
    result_text: str
    is_error: bool = False
    schema_errors: list[str] = field(default_factory=list)

    @property
    def tool_error(self) -> bool:
        return is_rejection(self.result_text, is_error=self.is_error)

    @property
    def signals(self) -> list[str]:
        signals: list[str] = []
        if self.tool_error:
            signals.append("tool_error")
        if self.schema_errors:
            signals.append("schema_invalid")
        if not self.tool_error and looks_like_error(self.result_text):
            signals.append("error_text")
        return signals

    @property
    def rejected(self) -> bool:
        return bool(self.signals)

    @property
    def reasons(self) -> list[str]:
        if not self.rejected:
            return []
        if self.tool_error:
            bullets = [line.strip()[2:] for line in self.result_text.splitlines() if line.strip().startswith("- ")]
            return bullets or [self.result_text.strip() or MCP_ERROR_REASON]
        if self.schema_errors:
            return list(self.schema_errors)
        return [self.result_text.strip().splitlines()[0] if self.result_text.strip() else MCP_ERROR_REASON]


@dataclass
class RunRecord:
    prompt_id: str
    attempt: int
    calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None

    def retry_loop(self, call_budget: int) -> bool:
        if len(self.calls) > call_budget:
            return True
        return any(
            previous.rejected and following.tool == previous.tool
            for previous, following in zip(self.calls, self.calls[1:], strict=False)
        )


def is_rejection(text: str, *, is_error: bool = False) -> bool:
    return is_error or text.lstrip().startswith(REJECTION_PREFIX) or REJECTION_STATUS in text


def schema_errors(schema: Mapping[str, Any], arguments: Any) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in validator.iter_errors(arguments if arguments is not None else {})
    ]


def annotate_schema_errors(runs: Iterable[RunRecord], schemas: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate each call against the input schema its variant advertised for that tool."""
    for run in runs:
        for call in run.calls:
            schema = schemas.get(call.tool)
            call.schema_errors = schema_errors(schema, call.arguments) if schema else []


def categorize(reason: str) -> str:
    for label, pattern in REASON_CATEGORIES:
        if pattern.search(reason):
            return label
    return "other"


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        for key in ("text", "content"):
            if key in content:
                return _text_from_content(content[key])
        return json.dumps(content)
    if isinstance(content, Sequence):
        return "\n".join(_text_from_content(item) for item in content)
    return str(content)


def _scope_key(record: Mapping[str, Any]) -> str | None:
    profile = record.get("category_profile")
    if isinstance(profile, Mapping) and isinstance(profile.get("tool_call_id"), str):
        return f"call:{profile['tool_call_id']}"
    uuid = record.get("uuid")
    return f"uuid:{uuid}" if isinstance(uuid, str) else None


def _start_arguments(data: Any) -> Any:
    if isinstance(data, Mapping):
        for key in ("args", "arguments", "input"):
            if key in data:
                return data[key]
    return data


def calls_from_relay_records(records: Iterable[Mapping[str, Any]]) -> list[ToolCall]:
    """Pair Relay ATOF tool scope start/end records into calls, in completion order."""
    pending: dict[str, tuple[str, Any]] = {}
    unkeyed: list[tuple[str, Any]] = []
    calls: list[ToolCall] = []
    for record in records:
        if record.get("kind") != "scope" or record.get("category") != "tool":
            continue
        name = str(record.get("name") or "tool")
        key = _scope_key(record)
        if record.get("scope_category") == "start":
            entry = (name, _start_arguments(record.get("data")))
            if key is None:
                unkeyed.append(entry)
            else:
                pending[key] = entry
            continue
        if record.get("scope_category") != "end":
            continue
        started = pending.pop(key, None) if key is not None else None
        if started is None:
            index = next((i for i, (pending_name, _) in enumerate(unkeyed) if pending_name == name), None)
            started = unkeyed.pop(index) if index is not None else (name, None)
        data = record.get("data")
        is_error = isinstance(data, Mapping) and data.get("is_error") is True
        text = _text_from_content(data.get("content") if isinstance(data, Mapping) else data)
        calls.append(ToolCall(tool=started[0], arguments=started[1], result_text=text, is_error=is_error))
    return calls


def calls_from_messages(messages: Iterable[Mapping[str, Any]]) -> list[ToolCall]:
    """Pair LangChain-style AI ``tool_calls`` with the tool messages that answer them."""
    requested: list[dict[str, Any]] = []
    calls: list[ToolCall] = []
    for message in messages:
        role = str(message.get("role") or message.get("type") or "")
        if role in ("ai", "assistant"):
            for call in message.get("tool_calls") or []:
                if isinstance(call, Mapping):
                    requested.append(dict(call))
            continue
        if role != "tool":
            continue
        call_id = message.get("tool_call_id")
        index = next((i for i, call in enumerate(requested) if call_id and call.get("id") == call_id), 0)
        call = requested.pop(index) if requested else {}
        calls.append(
            ToolCall(
                tool=str(call.get("name") or message.get("name") or "tool"),
                arguments=call.get("args", call.get("arguments")),
                result_text=_text_from_content(message.get("content")),
                is_error=message.get("status") == "error",
            )
        )
    return calls


def only_tools(calls: Iterable[ToolCall], tool_names: set[str]) -> list[ToolCall]:
    return [call for call in calls if not tool_names or call.tool in tool_names]


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def metrics(runs: Sequence[RunRecord], call_budget: int = DEFAULT_CALL_BUDGET) -> dict[str, Any]:
    tool_calls = sum(len(run.calls) for run in runs)
    signal_counts = Counter(signal for run in runs for call in run.calls for signal in call.signals)
    rejected = sum(call.rejected for run in runs for call in run.calls)
    first_accepted = sum(bool(run.calls) and not run.calls[0].rejected for run in runs)
    succeeded = sum(run.error is None and bool(run.calls) and not run.calls[-1].rejected for run in runs)
    return {
        "runs": len(runs),
        "runs_with_errors": sum(run.error is not None for run in runs),
        "tool_calls": tool_calls,
        "rejected_calls": rejected,
        "tool_error_rate": _rate(rejected, tool_calls),
        "signals": {
            "tool_error": signal_counts["tool_error"],
            "schema_invalid": signal_counts["schema_invalid"],
            "error_text": signal_counts["error_text"],
            "retry_loop_runs": sum(run.retry_loop(call_budget) for run in runs),
        },
        "first_call_accepted_rate": _rate(first_accepted, len(runs)),
        "task_success_rate": _rate(succeeded, len(runs)),
        "avg_calls_per_run": round(tool_calls / len(runs), 3) if runs else 0.0,
    }


def rejection_reasons(runs: Sequence[RunRecord]) -> dict[str, int]:
    counts = Counter(categorize(reason) for run in runs for call in run.calls for reason in call.reasons)
    return dict(counts.most_common())


def argument_excerpt(arguments: Any, limit: int = ARGUMENT_EXCERPT_CHARS) -> str:
    text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return text if len(text) <= limit else f"{text[:limit]}…"


def rejected_examples(runs: Sequence[RunRecord], limit: int = MAX_EXAMPLES) -> list[dict[str, Any]]:
    """One example per rejection category first, so the UI shows the spread before the repeats."""
    rejected = [(run, call) for run in runs for call in run.calls if call.rejected]
    seen: set[str] = set()
    picked: list[tuple[RunRecord, ToolCall]] = []
    for run, call in rejected:
        category = categorize(call.reasons[0])
        if category not in seen:
            seen.add(category)
            picked.append((run, call))
    picked_calls = {id(call) for _, call in picked}
    picked.extend(pair for pair in rejected if id(pair[1]) not in picked_calls)
    return [
        {
            "prompt_id": run.prompt_id,
            "tool": call.tool,
            "signals": call.signals,
            "arguments_excerpt": argument_excerpt(call.arguments),
            "rejection": "; ".join(call.reasons),
            "schema_errors": call.schema_errors,
        }
        for run, call in picked[:limit]
    ]


def variant_summary(runs: Sequence[RunRecord], call_budget: int = DEFAULT_CALL_BUDGET) -> dict[str, Any]:
    by_prompt: dict[str, list[RunRecord]] = {}
    for run in runs:
        by_prompt.setdefault(run.prompt_id, []).append(run)
    return {
        "metrics": metrics(runs, call_budget),
        "rejection_reasons": rejection_reasons(runs),
        "per_prompt": {prompt_id: metrics(prompt_runs, call_budget) for prompt_id, prompt_runs in by_prompt.items()},
        "examples": rejected_examples(runs),
    }


def accepted_calls(runs: Sequence[RunRecord], tool: str) -> list[ToolCall]:
    """Calls the tool itself accepted, whatever the published schema said about them."""
    return [call for run in runs for call in run.calls if call.tool == tool and not call.tool_error]


def loose_schema_findings(runs: Sequence[RunRecord]) -> list[dict[str, Any]]:
    """Flag tools whose published schema accepted calls the tool itself rejected."""
    findings: list[dict[str, Any]] = []
    tools = sorted({call.tool for run in runs for call in run.calls})
    for tool in tools:
        tool_errors = [call for run in runs for call in run.calls if call.tool == tool and call.tool_error]
        passed = [call for call in tool_errors if not call.schema_errors]
        if not passed:
            continue
        reasons = Counter(categorize(reason) for call in passed for reason in call.reasons)
        findings.append(
            {
                "id": f"lint-schema-too-loose-{re.sub(r'[^a-z0-9]+', '-', tool.lower()).strip('-')}",
                "source": "schema-lint",
                "severity": "high" if len(passed) * 2 >= len(tool_errors) else "medium",
                "tool": tool,
                "title": "Schema too loose to catch its own errors",
                "detail": f"{len(passed)} of {len(tool_errors)} calls the tool rejected pass its published input "
                "schema, so neither the model's grammar nor a client-side check can stop them.",
                "evidence": [f"{count} rejected for {label}" for label, count in reasons.most_common(3)],
            }
        )
    return findings
