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
from nhx_evals_sdk.values.skill_evidence import SkillToolCall, SkillToolTrace

CODEX_VERSION = "0.162.1"
PUBLISH = "Taskset:publish"
READBACK = "Taskset:readback"
AGENT_HOME = PurePosixPath("/home/harbor-agent")
_CATALOG_KEYS = ("skills", "host_skills", "cloud_skills")
_CATALOG_ALIAS = re.compile(r"^- `([rec]\d+)` = `([^`]+)`$", re.MULTILINE)
_CATALOG_ENTRY = re.compile(r"^- ([^:]+):.*\((file|executor package|cloud package|custom resource): (.+)\)$")
_TOOL_CALLS = {"custom_tool_call", "function_call", "local_shell_call", "tool_search_call", "web_search_call"}
_EXEC_EXIT = re.compile(
    r"\A(?:Chunk ID: [^\n]*\n)?(?:Wall time: [^\n]*\n)?Process exited with code (-?[0-9]+)(?:\n|\Z)"
)


@dataclass(frozen=True)
class CodexEvidence:
    skill_listed: bool
    activated: bool | None
    measurements: TrialMeasurements
    command_outcomes: tuple[bool | None, ...] = ()


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


def _proves_shell_read(command: object, output: object, registered: str, document: str) -> bool:
    """Recognize simple reads and compare the output prefix for a leading read in a command list."""
    if not isinstance(output, str):
        return False
    compound = False
    if isinstance(command, list) and all(isinstance(arg, str) for arg in command):
        if len(command) == 3 and command[1] in {"-c", "-lc"} and PurePosixPath(command[0]).name in {"bash", "sh"}:
            return _proves_shell_read(command[-1], output, registered, document)
        args = command
    elif isinstance(command, str):
        first, *remaining = re.split(r"\s*(?:&&|;)\s*", command, maxsplit=1)
        compound = bool(remaining)
        try:
            lexer = shlex.shlex(first, posix=True, punctuation_chars=True)
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
    try:
        excerpt = _document_lines(PurePosixPath(args[0]).name, args[1:-1])
    except ValueError:
        return False
    if excerpt is None:
        return False
    if not compound:
        return _read_document_output(output, document)
    expected = "".join(document.splitlines(keepends=True)[excerpt])
    return bool(expected.strip()) and output.startswith(expected)


def _document_lines(tool: str, options: list[str]) -> slice | None:
    if tool == "cat" and options in ([], ["--"]):
        return slice(None)
    if tool == "head" and not options:
        return slice(10)
    if len(options) == 2 and options[0] == "-n":
        if tool == "head" and options[1].isascii() and options[1].isdigit():
            return slice(int(options[1]))
        if tool == "sed" and (match := re.fullmatch(r"(?:([0-9]+),)?([0-9]+)p", options[1])):
            last = int(match[2])
            first = int(match[1]) if match[1] else last
            if first > 0:
                return slice(first - 1, max(first, last))
    return None


def _read_document_output(text: object, document: str) -> bool:
    return isinstance(text, str) and bool(text.strip()) and text.strip() in document


def _function_output_text(output: object) -> str | None:
    if isinstance(output, list):
        # Codex serializes function output as a string or a list of typed content parts.
        parts = [part.get("text") for part in output if isinstance(part, dict) and part.get("type") == "input_text"]
        output = parts[0] if len(parts) == 1 else None
    return output if isinstance(output, str) else None


def _command_succeeded(output: str | None) -> bool | None:
    # Only the tool envelope can establish an exit code. Command stdout is untrusted.
    status = _EXEC_EXIT.match(output or "")
    return status[1] == "0" if status else None


def _proves_tool_read(call: dict[str, Any], output: str | None, registered: str, document: str) -> bool:
    if output is None:
        return False
    if (call.get("namespace"), call.get("name")) == ("skills", "read") or call.get("name") == "skills.read":
        response = _json_object(output)
        resource, contents = response.get("resource"), response.get("contents")
        return (
            isinstance(resource, str)
            and _locator_path(resource) == registered
            and _read_document_output(contents, document)
        )
    if call.get("name") != "exec_command":
        return False
    status, delimiter, stdout = output.partition("\nOutput:\n")
    command = _json_object(call.get("arguments")).get("cmd")
    return (
        bool(delimiter)
        and "Process exited with code 0" in status.splitlines()
        and _proves_shell_read(command, stdout, registered, document)
    )


def observe_codex(events: Iterable[dict[str, Any]], skill_name: str, document: str) -> CodexEvidence:
    registered = str(AGENT_HOME / ".agents" / "skills" / skill_name / "SKILL.md")
    listed = None
    activated = None
    after_tool = False
    pending: dict[str, dict[str, Any]] = {}
    command_outcomes: dict[str, bool | None] = {}
    seen_outputs: set[str] = set()
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
                if kind == "function_call" and isinstance(call_id, str) and call_id:
                    if call_id in command_outcomes:
                        command_outcomes[call_id] = None
                        seen_outputs.add(call_id)
                    elif payload.get("name") == "exec_command" and payload.get("namespace") in (None, "functions"):
                        command_outcomes[call_id] = None
                        if call_id in pending:
                            seen_outputs.add(call_id)
                    pending[call_id] = payload
            elif kind == "function_call_output" and isinstance(call_id, str):
                output = _function_output_text(payload.get("output"))
                if call_id in command_outcomes:
                    command_outcomes[call_id] = _command_succeeded(output) if call_id not in seen_outputs else None
                seen_outputs.add(call_id)
                call = pending.pop(call_id, {})
                if listed and _proves_tool_read(call, output, registered, document):
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
                    and _proves_shell_read(item.get("command"), item.get("aggregated_output"), registered, document)
                ):
                    activated = True
    if listed is None:
        raise ValueError("Codex rollout has no explicit launch skill catalog before the first tool call")
    return CodexEvidence(listed, activated, _measurements(usage), tuple(command_outcomes.values()))


def _skill_location(name: str) -> str:
    return str(AGENT_HOME / ".agents" / "skills" / name / "SKILL.md")


def _confirmed_component(command: str, stdout: str, taskset: str, task: str) -> tuple[str | None, dict | None]:
    """Recognize publishing through the CLI and reading the pinned revision through HTTP."""
    if not taskset or not task:
        return None, None
    try:
        args = shlex.split(command.splitlines()[0].split("|", maxsplit=1)[0]) if command.strip() else []
        body, _ = json.JSONDecoder().raw_decode(stdout[stdout.index("{") :])
    except ValueError:
        return None, None
    if not args or not isinstance(body, dict) or body.get("tasks") != [task]:
        return None, None
    if args[:2] == ["uv", "run"]:
        args = args[2:]
    if args and Path(args[0]).name == "nemo" and args[1:4] == ["evals", "tasksets", "create"]:
        if body.get("taskset_ref") == taskset and "--harbor-dir" in args[:-1]:
            return PUBLISH, {"harbor_dir": args[args.index("--harbor-dir") + 1]}
    name, _, digest = taskset.partition("#")
    workspace, _, taskset_name = name.partition("/")
    path = f"/apis/evals/v2/workspaces/{workspace}/tasksets/{taskset_name}/revisions/{digest}"
    if args and args[0] == "curl" and any(arg == f"http://127.0.0.1:8080{path}" for arg in args[1:]):
        return READBACK, None
    return None, None


def tool_trace(
    events: list[dict[str, Any]], trial_id: str, staged: dict[str, str], taskset: str, task: str
) -> SkillToolTrace:
    """Correlate observed root tool calls, keeping unobserved exits and components unknown."""
    calls: dict[str, tuple[int, dict[str, Any]]] = {}
    outputs: dict[str, str | None] = {}
    step, timestamp = -1, None
    for event in events:
        payload = event.get("payload")
        if event.get("type") != "response_item" or not isinstance(payload, dict):
            continue
        call_id = payload.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        if payload.get("type") == "function_call":
            name = payload.get("name")
            if name in ("exec_command", "skills.read") or (payload.get("namespace"), name) == ("skills", "read"):
                if call_id in calls:
                    continue
                current = event.get("timestamp") or call_id
                if current != timestamp:
                    timestamp, step = current, step + 1
                calls[call_id] = (step, payload)
        elif payload.get("type") == "function_call_output":
            outputs[call_id] = _function_output_text(payload.get("output")) if call_id not in outputs else None

    normalized = []
    for call_id, (step, call) in calls.items():
        output = outputs.get(call_id)
        succeeded = _command_succeeded(output)
        component, arguments = None, None
        tool = "exec_command" if call["name"] == "exec_command" else "skills.read"
        for name, document in staged.items():
            if _proves_tool_read(call, output, _skill_location(name), document):
                component, succeeded = f"Skill:{name}", True
                break
        if component is None and tool == "exec_command" and succeeded is True and output:
            command = _json_object(call.get("arguments")).get("cmd")
            if isinstance(command, str):
                component, arguments = _confirmed_component(command, output.partition("\nOutput:\n")[2], taskset, task)
        normalized.append(
            SkillToolCall(
                id=call_id, step=step, tool=tool, component=component, arguments=arguments, succeeded=succeeded
            )
        )
    # These root events omit opaque subprocess work, retry links and productivity annotations.
    return SkillToolTrace(trial_id=trial_id, complete=False, calls=normalized)
