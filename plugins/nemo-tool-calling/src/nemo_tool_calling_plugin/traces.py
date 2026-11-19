# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score an agent's baseline from the tool calls it already made, as traced in Intake.

Relay records each tool call twice: an OpenTelemetry ``execute_tool <tool>`` span that carries the arguments and
result, and an ATIF span named just the tool. Only the first is read. Calls are grouped by OTel ``session_id``,
one session per agent run. The prompt that started a session is the first ``user`` message in the input of the
session's earliest LLM span (``chat <model>``), which holds the request messages as GenAI ``parts``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.intake.client import IntakeClient
from nemo_helix_plugin.intake.types import Span
from nemo_tool_calling_plugin.pipeline import Prompt
from nemo_tool_calling_plugin.scoring import RunRecord, ToolCall

TOOL_SPAN_PREFIX = "execute_tool"
SPANS_PAGE_SIZE = 500
PROMPT_ID_WORDS = 6


class NoTracedSessionsError(LocalRunError):
    pass


@dataclass(frozen=True)
class TraceWindow:
    since: datetime
    until: datetime

    @classmethod
    def resolve(cls, *, since: datetime | None, until: datetime | None, hours: float) -> TraceWindow:
        end = _utc(until) if until is not None else datetime.now(UTC)
        start = _utc(since) if since is not None else end - timedelta(hours=hours)
        return cls(since=start, until=end)

    def to_json(self) -> dict[str, str]:
        return {"since": _iso(self.since), "until": _iso(self.until)}


@dataclass
class TracedSession:
    session_id: str
    started_at: datetime
    calls: list[ToolCall] = field(default_factory=list)
    prompt: str | None = None


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_tool_result_span(span: Span) -> bool:
    return (span.name or "").startswith(TOOL_SPAN_PREFIX)


def _tool_name(span: Span) -> str:
    return span.tool_name or (span.name or "").removeprefix(TOOL_SPAN_PREFIX).strip() or "tool"


def _arguments(span: Span) -> Any:
    if span.input is None:
        return None
    try:
        return json.loads(span.input)
    except ValueError:
        return span.input


def _result(span: Span) -> tuple[str, bool]:
    errored = str(span.status) == "error"
    if span.output:
        try:
            output = json.loads(span.output)
        except ValueError:
            return span.output, errored
        if isinstance(output, Mapping):
            content = output.get("content")
            first = content[0] if isinstance(content, list) and content else None
            text = first.get("text") if isinstance(first, Mapping) else None
            if isinstance(text, str):
                return text, errored or output.get("is_error") is True
    return span.error_message or span.output or "", errored


def call_from_span(span: Span) -> ToolCall | None:
    if not is_tool_result_span(span):
        return None
    text, is_error = _result(span)
    return ToolCall(tool=_tool_name(span), arguments=_arguments(span), result_text=text, is_error=is_error)


def group_sessions(spans: Iterable[Span]) -> list[TracedSession]:
    """Tool-result spans grouped into sessions, newest session first, calls in call order."""
    by_session: dict[str, list[Span]] = {}
    for span in spans:
        if is_tool_result_span(span):
            by_session.setdefault(span.session_id, []).append(span)
    sessions: list[TracedSession] = []
    for session_id, session_spans in by_session.items():
        ordered = sorted(session_spans, key=lambda span: _utc(span.started_at))
        calls = [call for call in map(call_from_span, ordered) if call is not None]
        sessions.append(TracedSession(session_id=session_id, started_at=_utc(ordered[0].started_at), calls=calls))
    return sorted(sessions, key=lambda session: session.started_at, reverse=True)


def _part_text(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""
    texts = [
        str(part.get("content") or part.get("text") or "")
        for part in parts
        if isinstance(part, Mapping) and part.get("type", "text") == "text"
    ]
    return "\n".join(text for text in texts if text)


def first_user_message(llm_input: str | None) -> str | None:
    """The first user message in an LLM span input: GenAI ``parts`` or OpenAI ``content`` messages."""
    if not llm_input:
        return None
    try:
        messages = json.loads(llm_input)
    except ValueError:
        return None
    if isinstance(messages, Mapping):
        messages = messages.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "user":
            text = _part_text(message.get("parts", message.get("content"))).strip()
            if text:
                return text
    return None


def _normalized(text: str) -> str:
    return " ".join(text.split()).rstrip(".").lower()


def _slug(text: str, taken: set[str]) -> str:
    base = "-".join(re.findall(r"[a-z0-9]+", text.lower())[:PROMPT_ID_WORDS]) or "prompt"
    slug, suffix = base, 2
    while slug in taken:
        slug, suffix = f"{base}-{suffix}", suffix + 1
    taken.add(slug)
    return slug


def replay_prompts(sessions: Sequence[TracedSession], limit: int) -> tuple[list[Prompt], dict[str, str]]:
    """Distinct session prompts, most-traced first, capped at *limit*; plus every session's prompt id."""
    counts: dict[str, int] = {}
    first_text: dict[str, str] = {}
    for session in sessions:
        if session.prompt:
            key = _normalized(session.prompt)
            counts[key] = counts.get(key, 0) + 1
            first_text.setdefault(key, session.prompt)
    taken: set[str] = set()
    ids = {key: _slug(first_text[key], taken) for key in first_text}
    ranked = sorted(first_text, key=lambda key: -counts[key])
    prompts = [Prompt(id=ids[key], prompt=first_text[key]) for key in ranked[:limit]]
    session_ids = {
        session.session_id: ids[_normalized(session.prompt)] if session.prompt else session.session_id
        for session in sessions
    }
    return prompts, session_ids


def baseline_runs(sessions: Sequence[TracedSession], prompt_ids: Mapping[str, str]) -> list[RunRecord]:
    attempts: dict[str, int] = {}
    runs: list[RunRecord] = []
    for session in reversed(sessions):
        prompt_id = prompt_ids.get(session.session_id, session.session_id)
        attempts[prompt_id] = attempts.get(prompt_id, 0) + 1
        runs.append(RunRecord(prompt_id=prompt_id, attempt=attempts[prompt_id], calls=list(session.calls)))
    return runs


class SpanSource(Protocol):
    def tool_spans(self, *, agent_name: str, tool_names: Sequence[str], window: TraceWindow) -> list[Span]: ...

    def first_llm_input(self, session_id: str) -> str | None: ...


class IntakeSpanSource:
    def __init__(self, client: IntakeClient, *, workspace: str) -> None:
        self._client = client
        self._workspace = workspace

    def _list(self, filter: dict[str, Any], *, page_size: int = SPANS_PAGE_SIZE) -> Iterable[Span]:
        return self._client.spans.list(
            workspace=self._workspace, page_size=page_size, sort="started_at", filter=filter
        ).items()

    def tool_spans(self, *, agent_name: str, tool_names: Sequence[str], window: TraceWindow) -> list[Span]:
        base: dict[str, Any] = {
            "agent_name": agent_name,
            "started_at": {"$gte": window.since.isoformat(), "$lte": window.until.isoformat()},
        }
        filters = [{**base, "tool_name": name} for name in tool_names] or [base]
        return [span for query in filters for span in self._list(query) if is_tool_result_span(span)]

    def first_llm_input(self, session_id: str) -> str | None:
        page = self._client.spans.list(
            workspace=self._workspace, page_size=5, sort="started_at", filter={"session_id": session_id, "kind": "LLM"}
        ).page()
        return next((span.input for span in page.items if span.input), None)


@dataclass
class TracedBaseline:
    window: TraceWindow
    sessions: list[TracedSession]
    runs: list[RunRecord]
    prompts: list[Prompt]


def load_traced_baseline(
    source: SpanSource,
    *,
    agent_name: str,
    tool_names: Sequence[str],
    window: TraceWindow,
    max_sessions: int,
    max_replay_prompts: int,
) -> TracedBaseline:
    spans = source.tool_spans(agent_name=agent_name, tool_names=tool_names, window=window)
    sessions = group_sessions(spans)[:max_sessions]
    if not sessions:
        raise NoTracedSessionsError(
            f"No traced tool calls for agent '{agent_name}' between {_iso(window.since)} and {_iso(window.until)}. "
            "Chat with the agent first (with telemetry enabled) or widen trace_window_hours, or pass a prompt "
            "dataset with optimize_config + optimize_config_fileset or prompts."
        )
    for session in sessions:
        session.prompt = first_user_message(source.first_llm_input(session.session_id))
    prompts, prompt_ids = replay_prompts(sessions, max_replay_prompts)
    return TracedBaseline(window=window, sessions=sessions, runs=baseline_runs(sessions, prompt_ids), prompts=prompts)
