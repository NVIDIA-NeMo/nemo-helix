# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse Claude Code session JSONL transcripts from completed Harbor eval runs.

Adapted from tests/agentic-use/shared/trace_reader.py for batch analysis
of completed runs (post-Harbor), rather than in-container verification.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_agents_plugin.improvement.models import TokenUsage, ToolCallSummary


@dataclass
class ToolCall:
    """A single tool call from the session."""

    name: str
    tool_id: str
    input: dict[str, Any]
    timestamp: str | None = None

    @property
    def command(self) -> str | None:
        """For Bash tool calls, return the command string."""
        if self.name == "Bash":
            return self.input.get("command")
        return None


@dataclass
class ToolResult:
    """A tool result returned to the assistant."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class SessionData:
    """Parsed Claude Code session transcript."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    def get_bash_commands(self) -> list[str]:
        return [tc.command for tc in self.tool_calls if tc.command is not None]

    def get_mcp_calls(self) -> list[ToolCall]:
        return [tc for tc in self.tool_calls if tc.name.startswith("mcp__")]

    def get_tool_calls_by_name(self, name: str) -> list[ToolCall]:
        return [tc for tc in self.tool_calls if tc.name == name]

    def get_error_results(self) -> list[ToolResult]:
        return [tr for tr in self.tool_results if tr.is_error]


def find_session_files(trial_dir: Path) -> list[Path]:
    """Find JSONL session files within a completed trial directory.

    Harbor stores session data at:
      <trial_dir>/agent/sessions/projects/-app/<session-id>.jsonl

    Returns paths sorted by modification time, newest first.
    """
    sessions_dir = trial_dir / "agent" / "sessions" / "projects"
    if not sessions_dir.exists():
        return []

    files = list(sessions_dir.rglob("*.jsonl"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def parse_session_jsonl(jsonl_path: Path) -> SessionData:
    """Parse a single session JSONL file into structured data."""
    messages: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []
    tool_results: list[ToolResult] = []

    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages.append(entry)

            if entry.get("type") == "assistant":
                content = entry.get("message", {}).get("content", [])
                timestamp = entry.get("timestamp")
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_calls.append(
                            ToolCall(
                                name=item.get("name", ""),
                                tool_id=item.get("id", ""),
                                input=item.get("input", {}),
                                timestamp=timestamp,
                            )
                        )

            if entry.get("type") == "user":
                content = entry.get("message", {}).get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            result_content = item.get("content", "")
                            if isinstance(result_content, list):
                                # Content can be a list of blocks
                                result_content = "\n".join(
                                    b.get("text", "") for b in result_content if isinstance(b, dict)
                                )
                            tool_results.append(
                                ToolResult(
                                    tool_use_id=item.get("tool_use_id", ""),
                                    content=str(result_content),
                                    is_error=item.get("is_error", False),
                                )
                            )

    session_id = jsonl_path.stem
    for msg in messages:
        if "sessionId" in msg:
            session_id = msg["sessionId"]
            break

    return SessionData(
        session_id=session_id,
        messages=messages,
        tool_calls=tool_calls,
        tool_results=tool_results,
    )


def analyze_tool_calls(trial_dir: Path) -> ToolCallSummary:
    """Count and categorize tool calls from trajectory.json or JSONL sessions.

    Tries trajectory.json (ATIF format) first, then falls back to session JSONL.
    """
    # Try trajectory.json first
    trajectory_file = trial_dir / "agent" / "trajectory.json"
    if trajectory_file.exists():
        try:
            data = json.loads(trajectory_file.read_text())
            by_name: dict[str, int] = {}
            total = 0
            for step in data.get("steps", []):
                for tc in step.get("tool_calls", []):
                    name = tc.get("name", "unknown")
                    by_name[name] = by_name.get(name, 0) + 1
                    total += 1
            # trajectory.json doesn't track errors, so error_count stays 0
            return ToolCallSummary(total=total, by_name=by_name, error_count=0)
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: parse session JSONL
    session_files = find_session_files(trial_dir)
    if not session_files:
        return ToolCallSummary()

    session = parse_session_jsonl(session_files[0])
    by_name = {}
    for tc in session.tool_calls:
        by_name[tc.name] = by_name.get(tc.name, 0) + 1

    error_count = len(session.get_error_results())

    return ToolCallSummary(
        total=len(session.tool_calls),
        by_name=by_name,
        error_count=error_count,
    )


def _token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def extract_token_usage(trial_dir: Path, *, require_token_fields: bool = False) -> TokenUsage | None:
    """Sum token usage across assistant messages in the session JSONL.

    Returns None when no session file is present. Each assistant message in a
    Claude Code session JSONL carries a ``message.usage`` block with the
    Anthropic API breakdown (input_tokens / output_tokens /
    cache_creation_input_tokens / cache_read_input_tokens). When
    ``require_token_fields`` is true, returns None if an assistant usage block
    omits either the input or output token dimension.
    """
    session_files = find_session_files(trial_dir)
    if not session_files:
        return None

    input_total = 0
    output_total = 0
    cache_total = 0
    found_any = False
    with session_files[0].open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "assistant":
                continue
            usage = entry.get("message", {}).get("usage")
            if not isinstance(usage, dict):
                continue
            input_tokens = _token_count(usage.get("input_tokens"))
            output_tokens = _token_count(usage.get("output_tokens"))
            if require_token_fields and (input_tokens is None or output_tokens is None):
                return None
            cache_creation_tokens = _token_count(usage.get("cache_creation_input_tokens"))
            cache_read_tokens = _token_count(usage.get("cache_read_input_tokens"))
            found_any = True
            input_total += input_tokens or 0
            output_total += output_tokens or 0
            cache_total += cache_creation_tokens or 0
            cache_total += cache_read_tokens or 0

    if not found_any:
        return None
    return TokenUsage(
        input_tokens=input_total,
        output_tokens=output_total,
        cache_tokens=cache_total,
    )


def extract_skill_names(trial_dir: Path) -> list[str]:
    """Extract names of skills that were loaded during the session."""
    session_files = find_session_files(trial_dir)
    if not session_files:
        return []

    session = parse_session_jsonl(session_files[0])
    skills: list[str] = []
    for tc in session.tool_calls:
        if tc.name == "Skill":
            skill_name = tc.input.get("skill", "")
            if skill_name and skill_name not in skills:
                skills.append(skill_name)
    return skills


def extract_error_patterns(trial_dir: Path) -> list[str]:
    """Find tool_result entries with is_error=True and extract error snippets."""
    session_files = find_session_files(trial_dir)
    if not session_files:
        return []

    session = parse_session_jsonl(session_files[0])
    errors = []
    for tr in session.get_error_results():
        # Truncate long error messages
        text = tr.content[:500] if tr.content else ""
        if text:
            errors.append(text)
    return errors


def extract_trace_excerpt(trial_dir: Path, max_turns: int = 20) -> str:
    """Get a truncated text representation of the session for LLM analysis.

    Returns a human-readable summary of the agent's actions: tool calls made,
    commands run, errors encountered. Limited to max_turns most recent exchanges.
    """
    session_files = find_session_files(trial_dir)
    if not session_files:
        return "(no session data)"

    session = parse_session_jsonl(session_files[0])

    # Build a lookup of tool results by tool_use_id
    result_lookup: dict[str, ToolResult] = {tr.tool_use_id: tr for tr in session.tool_results}

    # Take the most recent calls (failure tail is more useful than startup noise)
    recent_calls = session.tool_calls[-max_turns:]
    skipped = len(session.tool_calls) - len(recent_calls)

    lines: list[str] = []
    if skipped > 0:
        lines.append(f"... ({skipped} earlier tool calls omitted)")

    for tc in recent_calls:
        if tc.name == "Bash":
            cmd = tc.command or "(empty)"
            lines.append(f"[Bash] {cmd[:200]}")
        elif tc.name.startswith("mcp__"):
            lines.append(f"[MCP] {tc.name}: {json.dumps(tc.input)[:200]}")
        elif tc.name == "Read":
            lines.append(f"[Read] {tc.input.get('file_path', '?')}")
        elif tc.name == "Write":
            lines.append(f"[Write] {tc.input.get('file_path', '?')}")
        elif tc.name == "Edit":
            lines.append(f"[Edit] {tc.input.get('file_path', '?')}")
        else:
            lines.append(f"[{tc.name}] {json.dumps(tc.input)[:150]}")

        # Show errors inline
        result = result_lookup.get(tc.tool_id)
        if result and result.is_error:
            lines.append(f"  ERROR: {result.content[:200]}")

    return "\n".join(lines)
