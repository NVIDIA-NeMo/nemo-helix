# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF trajectories to SFT chat rows, one row per assistant turn.

Pure functions over plain dicts: no IO, and no dependency on the evals SDK, so the customizer task
image stays small. Inputs are ATIF-v1.x documents as an agent-eval bundle stores them. Outputs are
``SFTPChatDatasetItemSchema`` rows (``messages`` plus ``tools``).

Why one row per assistant turn: training on a whole trajectory weights every turn by the same
conversation, and later turns see ever longer prefixes. A row per turn has the conversation so far as
context and exactly that turn as the target, so each decision is one training example. The cost is
that a trajectory with N agent turns writes N rows that repeat their prefixes.

ATIF to SFT mapping:

- ``system`` / ``user`` step: a ``system`` / ``user`` message.
- ``agent`` step: ``reasoning_content`` becomes its own assistant ``thinking`` message (the schema
  keeps ``thinking`` and ``content`` in separate messages), then one assistant message carrying the
  step ``message`` as ``content`` and ``tool_calls`` with dict ``arguments``.
- ``observation.results`` of an agent step: one ``tool`` message per call, in call order. ATIF keeps
  the result on the step that made the call, so it becomes context for the next turn, never part of
  the target.
- ``agent.tool_definitions``: the row's ``tools``.
- ``subagent_trajectories``: each one is its own conversation. A trajectory with no system or user
  step before its first agent step has no prompt to condition on, so it yields nothing. An orchestrator
  wrapper step that only delegates to a subagent is skipped that way.

Raw completions (preferred when a trial has them)
-------------------------------------------------

ATIF is a summary of what an agent did, and agents differ in how much they keep. Harbor's OpenHands
writes a label such as ``Running command: ls`` as the step message instead of the model's text, drops
``reasoning_content``, and adds notice steps the model never saw. Training on that teaches the student
the labels. The agent's raw model calls (``agent/completions/*.json``, one per request) hold what the
model was actually sent and what it actually answered, so ``convert_completions`` builds rows from them:

- each call is one row: the request's ``messages`` are the context exactly as the model received them
  (the student is deployed behind the same agent, so it will see the same prompts), and the response's
  ``reasoning_content``, text and ``tool_calls`` are the target, mapped like an ATIF agent step;
- ``tools`` come from the request's ``tools``;
- a call that does not extend the previous request starts a new chain;
- a response cut off by the token limit (``finish_reason == "length"``) is not a target.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from dataclasses import dataclass, field
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

TRACE_ATIF_KEY = "trace:atif"
COMPLETIONS_KEY_PREFIX = "artifact:agent/completions/"


class PassFilter(BaseModel):
    """Which trials are good enough to learn from, judged by their score rows.

    Empty (the default) keeps every completed trial. Otherwise a trial passes when at least one of its
    completed score rows matches ``metric_type`` and ``output`` and every matched value is at least
    ``min_value``. Unknown keys are an error so a typo cannot silently train on failures.

    Examples:
        ``{"metric_type": "harbor_reward", "output": "reward", "min_value": 1.0}`` keeps resolved
        Harbor trials.
    """

    model_config = ConfigDict(extra="forbid")

    metric_type: str | None = Field(default=None, description="Only score rows of this metric type.")
    output: str | None = Field(default=None, description="Only outputs with this name.")
    min_value: float | None = Field(
        default=None,
        description="Passing threshold. Defaults to 1.0 once metric_type or output is set.",
    )

    @property
    def active(self) -> bool:
        return self.metric_type is not None or self.output is not None or self.min_value is not None

    def passes(self, score_rows: list[dict[str, Any]]) -> bool:
        if not self.active:
            return True
        threshold = 1.0 if self.min_value is None else self.min_value
        matched: list[float] = []
        for row in score_rows:
            if row.get("status") != "completed":
                continue
            if self.metric_type is not None and row.get("metric_type") != self.metric_type:
                continue
            for item in row.get("outputs") or []:
                value = item.get("value")
                if self.output is not None and item.get("name") != self.output:
                    continue
                # bool is an int subclass; True is not a score of 1.0.
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                matched.append(float(value))
        return bool(matched) and all(value >= threshold for value in matched)


def trace_ref(trial: dict[str, Any]) -> str | None:
    """Bundle-relative ref of a trial's ATIF trajectory, or ``None``.

    Reads the evidence descriptors rather than guessing file names, because each runtime lays its
    traces out differently (Harbor ``traces/*.atif.json``, Fabric ``relay/.../trajectory-*.atif.json``).
    """
    descriptors = (trial.get("evidence") or {}).get("descriptors") or {}
    preferred = descriptors.get(TRACE_ATIF_KEY)
    candidates = [preferred] if isinstance(preferred, dict) else []
    candidates += [d for d in descriptors.values() if d.get("kind") == "trace" and d.get("format") == "atif"]
    candidates += [d for d in descriptors.values() if d.get("kind") == "atif"]
    for descriptor in candidates:
        ref = descriptor.get("ref")
        if isinstance(ref, str) and ref:
            return ref
    return None


def completion_refs(trial: dict[str, Any]) -> list[str]:
    """Bundle-relative refs of a trial's raw model calls (``agent/completions/*.json``), in name order.

    Order of the calls is decided by the timestamp inside each file, not by name.
    """
    descriptors = (trial.get("evidence") or {}).get("descriptors") or {}
    refs = []
    for key, descriptor in descriptors.items():
        if not (isinstance(key, str) and key.startswith(COMPLETIONS_KEY_PREFIX) and key.endswith(".json")):
            continue
        ref = descriptor.get("ref") if isinstance(descriptor, dict) else None
        if isinstance(ref, str) and ref:
            refs.append(ref)
    return sorted(refs)


def join_bundle_path(bundle_dir: str, ref: str) -> str:
    """Resolve a bundle-relative ref inside the fileset, refusing to climb out of the bundle."""
    path = posixpath.normpath(posixpath.join(bundle_dir, ref))
    root = bundle_dir.rstrip("/")
    escaped = path == ".." or path.startswith("../") or (root != "" and not path.startswith(root + "/"))
    if posixpath.isabs(ref) or escaped:
        raise ValueError(f"evidence ref {ref!r} is outside its bundle")
    return path


def held_out_tasks(task_ids: set[str], fraction: float) -> set[str]:
    """Tasks reserved for validation, chosen by a stable hash so reruns split identically.

    Splitting by task, not trial, keeps every turn of a task on one side. With two or more tasks and a
    positive fraction at least one task is held out and at least one stays for training.
    """
    if fraction <= 0 or len(task_ids) < 2:
        return set()
    count = min(len(task_ids) - 1, max(1, round(len(task_ids) * fraction)))
    ranked = sorted(task_ids, key=lambda task: hashlib.sha256(task.encode("utf-8")).hexdigest())
    return set(ranked[:count])


@dataclass
class Converted:
    """Rows from one trajectory document, plus the reasons any conversation in it was skipped.

    ``chains`` keeps the rows of one conversation together, in turn order: each row extends the
    previous one, so a length limit can be applied to the chain as a whole.
    """

    chains: list[list[dict[str, Any]]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return [row for chain in self.chains for row in chain]


def convert_trajectory(trajectory: dict[str, Any]) -> Converted:
    """One ATIF document (and its subagent trajectories) to SFT rows, one per assistant turn."""
    result = Converted()
    for conversation, inherited_tools in _conversations(trajectory, None):
        rows, skipped = _convert_conversation(conversation, inherited_tools)
        if rows:
            result.chains.append(rows)
        if skipped:
            result.skipped.append(skipped)
    return result


def convert_completions(calls: list[dict[str, Any]]) -> Converted:
    """One agent run's raw model calls to SFT rows, one per call (see the module docstring).

    ``calls`` are the parsed ``agent/completions/*.json`` documents, each ``{"messages": [...request],
    "response": <chat completion>, "kwargs": {"tools": [...]}, "timestamp": ...}``. They are put in
    time order here. A malformed document raises ``ValueError``, so the caller can fall back to the
    trial's ATIF trace.
    """
    ordered = sorted(calls, key=lambda call: _timestamp(call))
    result = Converted()
    chain: list[dict[str, Any]] = []
    previous: list[dict[str, Any]] | None = None
    uses_tools = False
    tools_missing = False
    for call in ordered:
        request, message, finish_reason, tools = _completion_parts(call)
        try:
            context = [part for raw in request for part in _request_messages(raw)]
            turn = _assistant_messages(_response_step(message))
        except BadToolArguments:
            return Converted(skipped=["bad_tool_arguments"])
        if previous is None or context[: len(previous)] != previous:
            if chain:
                result.chains.append(chain)
            chain = []
        previous = context
        if not any(m["role"] in ("system", "user") for m in context):
            return Converted(skipped=["no_prompt"])
        if not turn:
            continue
        if finish_reason == "length":
            result.skipped.append("truncated_completion")
            continue
        calls_tools = any("tool_calls" in m for m in [*context, *turn])
        uses_tools = uses_tools or calls_tools
        tools_missing = tools_missing or (calls_tools and not tools)
        row: dict[str, Any] = {"messages": [*context, *turn]}
        if tools:
            row["tools"] = tools
        chain.append(row)
    if chain:
        result.chains.append(chain)
    if uses_tools and tools_missing:
        return Converted(skipped=["no_tool_definitions"])
    return result


def _timestamp(call: dict[str, Any]) -> float:
    value = call.get("timestamp") if isinstance(call, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _completion_parts(
    call: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None, list[dict[str, Any]] | None]:
    if not isinstance(call, dict):
        raise ValueError("completion is not an object")
    request = call.get("messages")
    response = call.get("response")
    choices = response.get("choices") if isinstance(response, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(request, list) or not all(isinstance(m, dict) for m in request) or not isinstance(message, dict):
        raise ValueError("completion has no request messages or response message")
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    kwargs = call.get("kwargs")
    definitions = kwargs.get("tools") if isinstance(kwargs, dict) else None
    tools = _tools({"agent": {"tool_definitions": definitions}}) if isinstance(definitions, list) else None
    return request, message, finish_reason if isinstance(finish_reason, str) else None, tools


def _response_step(message: dict[str, Any]) -> dict[str, Any]:
    """A model response as the ATIF-shaped step that `_assistant_messages` reads."""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return {
        "reasoning_content": reasoning,
        "message": message.get("content"),
        "tool_calls": [_raw_call(call) for call in message.get("tool_calls") or []],
    }


def _raw_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or not function.get("name"):
        raise ValueError("completion tool call has no function name")
    return {"function_name": function["name"], "arguments": function.get("arguments")}


def _request_messages(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """A request message as SFT messages. Roles are kept; ids and tool names are not (the schema has none)."""
    role = raw.get("role")
    if role == "developer":
        role = "system"
    if role in ("system", "user"):
        return [{"role": role, "content": _text(raw.get("content"))}]
    if role == "tool":
        return [{"role": "tool", "content": _text(raw.get("content"))}]
    if role == "assistant":
        return _assistant_messages(_response_step(raw))
    raise ValueError(f"unknown message role {role!r}")


def _conversations(
    trajectory: dict[str, Any], parent_tools: list[dict[str, Any]] | None
) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]] | None]]:
    tools = _tools(trajectory) or parent_tools
    yield trajectory, parent_tools
    for child in trajectory.get("subagent_trajectories") or []:
        if isinstance(child, dict):
            yield from _conversations(child, tools)


def _tools(trajectory: dict[str, Any]) -> list[dict[str, Any]] | None:
    definitions = (trajectory.get("agent") or {}).get("tool_definitions")
    if not definitions:
        return None
    tools = []
    for definition in definitions:
        function = definition.get("function") if definition.get("type") == "function" else definition
        if not isinstance(function, dict) or not function.get("name"):
            continue
        parameters = dict(function.get("parameters") or {})
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        normalized = {k: v for k, v in function.items() if k not in {"name", "description", "parameters"}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": function["name"],
                    "description": function.get("description") or "",
                    "parameters": parameters,
                    **normalized,
                },
            }
        )
    return tools or None


def _convert_conversation(
    trajectory: dict[str, Any], inherited_tools: list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], str | None]:
    tools = _tools(trajectory) or inherited_tools
    context: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    saw_prompt = False
    saw_agent_turn = False
    uses_tools = False
    for step in trajectory.get("steps") or []:
        source = step.get("source")
        if source in ("system", "user"):
            message = _text(step.get("message"))
            if message.strip():
                context.append({"role": source, "content": message})
                saw_prompt = True
            continue
        if source != "agent":
            continue
        try:
            turn = _assistant_messages(step)
        except ValueError:
            return [], "bad_tool_arguments"
        if not turn:
            continue
        saw_agent_turn = True
        if not saw_prompt:
            continue  # a conversation starts at its prompt; an earlier agent turn has nothing to answer
        uses_tools = uses_tools or any("tool_calls" in message for message in turn)
        # Copied context repeats an earlier conversation: it is history to see, not a turn to learn.
        if not step.get("is_copied_context"):
            row: dict[str, Any] = {"messages": [*context, *turn]}
            if tools:
                row["tools"] = tools
            rows.append(row)
        context.extend(turn)
        context.extend(_tool_messages(step))
    if saw_agent_turn and not saw_prompt:
        return [], "no_prompt"
    if uses_tools and not tools:
        return [], "no_tool_definitions"
    return rows, None


def _assistant_messages(step: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    reasoning = step.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        messages.append({"role": "assistant", "thinking": reasoning})
    reply: dict[str, Any] = {"role": "assistant"}
    content = _text(step.get("message"))
    if content.strip():
        reply["content"] = content
    calls = [_tool_call(call) for call in step.get("tool_calls") or []]
    if calls:
        reply["tool_calls"] = calls
    if len(reply) > 1:
        messages.append(reply)
    return messages


class BadToolArguments(ValueError):
    """A tool call whose arguments are not a JSON object."""


def _arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise BadToolArguments("tool-call arguments are not JSON") from exc
    if not isinstance(arguments, dict):
        raise BadToolArguments("tool-call arguments are not an object")
    return arguments


def _tool_call(call: dict[str, Any]) -> dict[str, Any]:
    arguments = _arguments(call.get("arguments"))
    return {"type": "function", "function": {"name": call["function_name"], "arguments": arguments}}


def _tool_messages(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool results in call order. Results that cannot be matched to a call are dropped."""
    calls = step.get("tool_calls") or []
    results = (step.get("observation") or {}).get("results") or []
    if not calls or not results:
        return []
    by_id = {r.get("source_call_id"): r for r in results if r.get("source_call_id")}
    if len(by_id) == len(results):
        ordered = [by_id[c["tool_call_id"]] for c in calls if c.get("tool_call_id") in by_id]
    elif len(results) == len(calls) and not by_id:
        ordered = list(results)  # producer gave no ids; positions are all we have
    else:
        return []
    return [{"role": "tool", "content": _text(result.get("content"))} for result in ordered]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [p if isinstance(p, str) else p.get("text") for p in value if isinstance(p, (str, dict))]
        if all(isinstance(p, str) for p in parts):
            return "".join(parts)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
