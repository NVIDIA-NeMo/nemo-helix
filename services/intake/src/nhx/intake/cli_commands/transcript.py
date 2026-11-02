# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transcript view of one trace for ``nemo intake traces show``.

A transcript is one block per selected span, in ``started_at`` order. By
default the selected spans are model turns and tool calls: LLM and TOOL spans,
plus any span with a tool name, because OTel GenAI ``execute_tool`` spans are
stored as UNKNOWN. A block's input is the span input (a tool's arguments) and
its output is the span output (a tool's result), each cut to a character bound.
Tool calls are not paired with their results across spans; each span stands on
its own. Spans that are not selected are counted by kind.

An LLM span's input usually carries the whole conversation so far, so when it
parses as an OpenAI-style message list the block also carries that list's last
message, which is what is new in the turn.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from nemo_helix_plugin.intake.types import Span, SpanKind, Trace
from pydantic import BaseModel

DEFAULT_TRANSCRIPT_KINDS = ("LLM", "TOOL")
DEFAULT_MAX_CHARS = 500
_ELLIPSIS = "…"


class TranscriptMessage(BaseModel):
    role: str
    content: str
    content_truncated: bool = False
    message_count: int


class TranscriptBlock(BaseModel):
    span_id: str
    parent_span_id: str | None = None
    kind: SpanKind
    name: str | None = None
    model: str | None = None
    tool_name: str | None = None
    status: str
    error_message: str | None = None
    started_at: datetime
    duration_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None
    input: str | None = None
    input_truncated: bool = False
    input_last_message: TranscriptMessage | None = None
    output: str | None = None
    output_truncated: bool = False


class Transcript(BaseModel):
    trace_id: str
    name: str | None = None
    session_id: str
    agent_name: str | None = None
    status: str
    started_at: datetime
    duration_ms: float | None = None
    span_count: int
    kinds: list[str] | None
    hidden_kinds: dict[str, int]
    max_chars: int | None
    blocks: list[TranscriptBlock]


def _duration_ms(started_at: datetime, ended_at: datetime | None) -> float | None:
    if ended_at is None:
        return None
    return (ended_at - started_at).total_seconds() * 1000


def _bounded(text: str | None, max_chars: int | None) -> tuple[str | None, bool]:
    if text is None or max_chars is None or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _message_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
        return "\n".join(texts) if texts else None
    return None


def _last_message(text: str | None, max_chars: int | None) -> TranscriptMessage | None:
    """Return the last message of an OpenAI-style ``{"messages": [...]}`` or bare message list."""
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    messages = payload.get("messages") if isinstance(payload, dict) else payload
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return None
    last = messages[-1]
    content = _message_text(last.get("content"))
    if not isinstance(last.get("role"), str) or content is None:
        return None
    bounded, truncated = _bounded(content, max_chars)
    return TranscriptMessage(
        role=last["role"], content=bounded or "", content_truncated=truncated, message_count=len(messages)
    )


def _is_tool_call(kind: str, tool_name: str | None) -> bool:
    return kind == "TOOL" or (kind != "LLM" and tool_name is not None)


def _selected(span: Span, kinds: Iterable[str] | None) -> bool:
    if kinds is None:
        return True
    return span.kind in kinds or ("TOOL" in kinds and _is_tool_call(span.kind, span.tool_name))


def build_transcript(
    trace: Trace, spans: Iterable[Span], *, kinds: Iterable[str] | None, max_chars: int | None
) -> Transcript:
    """Project a trace's spans, already sorted by ``started_at``, into a transcript.

    ``kinds`` selects which span kinds become blocks; ``None`` selects every span.
    """
    kinds = None if kinds is None else list(kinds)
    spans = list(spans)
    blocks: list[TranscriptBlock] = []
    hidden: Counter[str] = Counter()
    for span in spans:
        if not _selected(span, kinds):
            hidden[span.kind] += 1
            continue
        input_text, input_truncated = _bounded(span.input, max_chars)
        output_text, output_truncated = _bounded(span.output, max_chars)
        blocks.append(
            TranscriptBlock(
                span_id=span.span_id,
                parent_span_id=span.parent_span_id,
                kind=span.kind,
                name=span.name,
                model=span.model,
                tool_name=span.tool_name,
                status=span.status,
                error_message=span.error_message,
                started_at=span.started_at,
                duration_ms=_duration_ms(span.started_at, span.ended_at),
                input_tokens=span.input_tokens,
                output_tokens=span.output_tokens,
                cached_tokens=span.cached_tokens,
                total_tokens=span.total_tokens,
                input=input_text,
                input_truncated=input_truncated,
                input_last_message=_last_message(span.input, max_chars) if span.kind == "LLM" else None,
                output=output_text,
                output_truncated=output_truncated,
            )
        )
    return Transcript(
        trace_id=trace.id,
        name=trace.name,
        session_id=trace.session_id,
        agent_name=trace.agent_name,
        status=trace.status,
        started_at=trace.started_at,
        duration_ms=trace.duration_ms,
        span_count=len(spans),
        kinds=kinds,
        hidden_kinds=dict(sorted(hidden.items(), key=lambda item: (-item[1], item[0]))),
        max_chars=max_chars,
        blocks=blocks,
    )


def _format_duration(duration_ms: float | None) -> str | None:
    if duration_ms is None:
        return None
    if duration_ms < 1000:
        return f"{duration_ms:.0f}ms"
    return f"{duration_ms / 1000:.2f}s"


def _format_tokens(block: TranscriptBlock) -> str | None:
    parts = [
        f"{label} {value}"
        for label, value in (
            ("in", block.input_tokens),
            ("out", block.output_tokens),
            ("cached", block.cached_tokens),
        )
        if value is not None
    ]
    if not parts and block.total_tokens is not None:
        parts.append(f"total {block.total_tokens}")
    return f"tokens {' / '.join(parts)}" if parts else None


def _payload_lines(label: str, text: str | None, truncated: bool) -> list[str]:
    if text is None:
        return []
    body = text + _ELLIPSIS if truncated else text
    first, *rest = body.splitlines() or [""]
    prefix = f"    {label}: "
    return [prefix + first, *(" " * len(prefix) + line for line in rest)]


def render_transcript(transcript: Transcript) -> str:
    """Render a transcript as plain text, one block per selected span."""
    header = " · ".join(
        part
        for part in (
            f"Trace {transcript.trace_id}",
            transcript.name,
            transcript.agent_name and f"agent {transcript.agent_name}",
            transcript.status,
            _format_duration(transcript.duration_ms),
        )
        if part
    )
    shown = "all kinds" if transcript.kinds is None else ", ".join(transcript.kinds)
    lines = [
        header,
        f"session {transcript.session_id} · started {transcript.started_at.isoformat()} · "
        f"{len(transcript.blocks)} of {transcript.span_count} spans shown ({shown})",
    ]
    if transcript.hidden_kinds:
        hidden = ", ".join(f"{count} {kind}" for kind, count in transcript.hidden_kinds.items())
        lines.append(f"hidden: {hidden} (add them with --kinds, or --kinds all)")
    if not transcript.blocks:
        lines.extend(["", "No spans of the selected kinds in this trace."])

    for index, block in enumerate(transcript.blocks, start=1):
        is_tool_call = _is_tool_call(block.kind, block.tool_name)
        title = block.tool_name if is_tool_call and block.tool_name else block.name
        summary = " · ".join(
            part
            for part in (
                f"[{index}] {block.kind}" + (f" {title}" if title else ""),
                block.model,
                block.status if block.status != "success" else None,
                _format_duration(block.duration_ms),
                _format_tokens(block),
            )
            if part
        )
        input_label, output_label = ("args", "result") if is_tool_call else ("input", "output")
        lines.extend(["", summary])
        if message := block.input_last_message:
            noun = "message" if message.message_count == 1 else "messages"
            label = f"{input_label} ({message.message_count} {noun}, last shown) {message.role}"
            lines.extend(_payload_lines(label, message.content, message.content_truncated))
        else:
            lines.extend(_payload_lines(input_label, block.input, block.input_truncated))
        lines.extend(_payload_lines(output_label, block.output, block.output_truncated))
        if block.error_message:
            lines.append(f"    error: {block.error_message}")
    return "\n".join(lines)
