# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode the Codex 0.162.1 evidence retained by Harbor."""

import json
import re
import shlex
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from nhx_evals_sdk.agent_eval.trials import TrialMeasurements

CODEX_VERSION = "0.162.1"
AGENT_HOME = PurePosixPath("/home/harbor-agent")
_CATALOG_KEYS = ("skills", "host_skills", "cloud_skills")
_CATALOG_ALIAS = re.compile(r"^- `([rec]\d+)` = `([^`]+)`$", re.MULTILINE)
_CATALOG_ENTRY = re.compile(r"^- ([^:]+):.*\((file|executor package|cloud package|custom resource): (.+)\)$")
_TOOL_CALLS = {"custom_tool_call", "function_call", "local_shell_call", "tool_search_call", "web_search_call"}


@dataclass(frozen=True)
class CodexEvidence:
    skill_listed: bool
    activated: bool | None
    measurements: TrialMeasurements


def rollout_events(agent_dir: Path) -> Iterator[dict[str, Any]]:
    """Read the single root `codex exec` rollout for this trial."""
    roots = []
    for path in (agent_dir / "sessions").rglob("rollout-*.jsonl"):
        if not path.resolve().is_relative_to(agent_dir.resolve()):
            raise ValueError("Codex rollout is outside the downloaded agent logs")
        with path.open(encoding="utf-8") as log:
            first = json.loads(log.readline())
        meta = first.get("payload") if isinstance(first, dict) and first.get("type") == "session_meta" else None
        if isinstance(meta, dict) and meta.get("source") == "exec" and meta.get("parent_thread_id") is None:
            roots.append((path, meta))
    if len(roots) != 1:
        raise ValueError(f"Expected one Codex root exec rollout, found {len(roots)}")
    path, meta = roots[0]
    if meta.get("cli_version") != CODEX_VERSION or not meta.get("id") or meta["id"] != meta.get("session_id"):
        raise ValueError(f"Expected a root Codex {CODEX_VERSION} session")
    with path.open(encoding="utf-8") as log:
        for line in log:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Codex rollout record is not a JSON object")
            yield event


def _expand_alias(locator: str, aliases: dict[str, str]) -> str:
    name, _, relative = locator.partition("/")
    if name in aliases:
        return f"{aliases[name].rstrip('/')}/{relative}"
    return locator


def _locator_path(locator: str) -> str:
    # Executor package and resource handles encode the complete executor path after their root ID.
    return urlsplit(locator).path if locator.startswith("skill://") else locator


def _catalog_listed(state: dict[str, Any], skill_name: str, registered: str) -> bool:
    listed = False
    for name in _CATALOG_KEYS:
        if name not in state:
            continue
        catalog = state[name]
        if not isinstance(catalog, dict) or catalog.get("includeInstructions") is not True:
            raise ValueError("Codex launch skill catalog is hidden or incomplete")
        # Codex omits the optional body when discovery is enabled and this catalog has no entries.
        body = catalog.get("body")
        if body is None:
            continue
        if not isinstance(body, str) or "omitted " in body.lower():
            raise ValueError("Codex launch skill catalog is truncated or malformed")
        header, marker, entries = body.partition("\n### Available skills\n")
        if not marker:
            raise ValueError("Codex launch skill catalog has no skill entries section")
        aliases = dict(_CATALOG_ALIAS.findall(header))
        for line in entries.partition("\n### ")[0].splitlines():
            if not line.strip():
                continue
            entry = _CATALOG_ENTRY.fullmatch(line)
            if entry is None:
                raise ValueError("Codex launch skill catalog has an unrecognized entry")
            entry_name, kind, locator = entry.groups()
            if entry_name != skill_name:
                continue
            resolved = _locator_path(_expand_alias(locator, aliases))
            expected = str(PurePosixPath(registered).parent) if kind == "executor package" else registered
            if kind not in {"file", "executor package"} or resolved != expected:
                raise ValueError(f"Codex launch skill catalog lists {skill_name} at an unexpected path")
            listed = True
    return listed


def _token_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _measurements(usage: dict[str, Any]) -> TrialMeasurements:
    prompt = _token_count(usage.get("input_tokens"))
    cache_read = _token_count(usage.get("cached_input_tokens"))
    if prompt is not None and cache_read is not None and cache_read > prompt:
        cache_read = None
    return TrialMeasurements(
        prompt_tokens=prompt,
        completion_tokens=_token_count(usage.get("output_tokens")),
        cache_read_tokens=cache_read,
        cache_creation_tokens=_token_count(usage.get("cache_write_input_tokens")),
    )


def _json_object(text: object) -> dict[str, Any]:
    if not isinstance(text, str):
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _is_document_read(command: object, registered: str) -> bool:
    """Recognize simple reads. Compound shell commands are inconclusive."""
    if isinstance(command, list) and all(isinstance(arg, str) for arg in command):
        if len(command) == 3 and command[1] in {"-c", "-lc"} and PurePosixPath(command[0]).name in {"bash", "sh"}:
            return _is_document_read(command[-1], registered)
        args = command
    elif isinstance(command, str):
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            args = list(lexer)
        except ValueError:
            return False
    else:
        return False
    if len(args) < 2:
        return False
    path = args[-1]
    for prefix in ("$HOME/", "${HOME}/", "~/"):
        if path.startswith(prefix):
            path = str(AGENT_HOME / path.removeprefix(prefix))
            break
    if path != registered:
        return False
    tool = PurePosixPath(args[0]).name
    options = args[1:-1]
    if tool == "cat":
        return options in ([], ["--"])
    if tool == "head":
        return not options or (len(options) == 2 and options[0] == "-n" and options[1].isdigit())
    return (
        tool == "sed"
        and len(options) == 2
        and options[0] == "-n"
        and re.fullmatch(r"(?:\d+,)?\d+p", options[1]) is not None
    )


def _read_document_output(text: object, document: str) -> bool:
    return isinstance(text, str) and bool(text.strip()) and text.strip() in document


def _proves_tool_read(call: dict[str, Any], output: object, registered: str, document: str) -> bool:
    if isinstance(output, list):
        # Codex serializes function output as a string or a list of typed content parts.
        parts = [part.get("text") for part in output if isinstance(part, dict) and part.get("type") == "input_text"]
        output = parts[0] if len(parts) == 1 else None
    if not isinstance(output, str):
        return False
    if (call.get("namespace"), call.get("name")) == ("skills", "read") or call.get("name") == "skills.read":
        response = _json_object(output)
        resource, contents = response.get("resource"), response.get("contents")
        return (
            isinstance(resource, str)
            and _locator_path(resource) == registered
            and _read_document_output(contents, document)
        )
    if call.get("name") != "exec_command" or not _is_document_read(
        _json_object(call.get("arguments")).get("cmd"), registered
    ):
        return False
    status, delimiter, stdout = output.partition("\nOutput:\n")
    return bool(delimiter) and "Process exited with code 0" in status and _read_document_output(stdout, document)


def observe_codex(events: Iterable[dict[str, Any]], skill_name: str, document: str) -> CodexEvidence:
    """Read launch discovery, proven skill use, and final cumulative token usage."""
    registered = str(AGENT_HOME / ".agents" / "skills" / skill_name / "SKILL.md")
    listed = None
    activated = None
    after_tool = False
    pending: dict[str, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if event.get("type") == "world_state" and listed is None and not after_tool and payload.get("full") is True:
            state = payload.get("state")
            if isinstance(state, dict) and any(key in state for key in _CATALOG_KEYS):
                listed = _catalog_listed(state, skill_name, registered)
        elif event.get("type") == "response_item":
            kind, call_id = payload.get("type"), payload.get("call_id")
            if kind in _TOOL_CALLS:
                after_tool = True
                if kind == "function_call" and isinstance(call_id, str):
                    pending[call_id] = payload
            elif listed and kind == "function_call_output" and isinstance(call_id, str):
                if _proves_tool_read(pending.pop(call_id, {}), payload.get("output"), registered, document):
                    activated = True
        elif event.get("type") == "event_msg" and payload.get("type") == "token_count":
            info = payload.get("info")
            if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                usage = info["total_token_usage"]
        elif event.get("type") == "event_msg" and payload.get("type") == "item_completed":
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "CommandExecution":
                after_tool = True
                if (
                    listed
                    and item.get("status") == "completed"
                    and _token_count(item.get("exit_code")) == 0
                    and _is_document_read(item.get("command"), registered)
                    and _read_document_output(item.get("aggregated_output"), document)
                ):
                    activated = True
    if listed is None:
        raise ValueError("Codex rollout has no explicit launch skill catalog before the first tool call")
    return CodexEvidence(listed, activated, _measurements(usage))
