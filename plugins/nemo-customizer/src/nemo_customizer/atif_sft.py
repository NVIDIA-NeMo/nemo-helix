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
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from dataclasses import dataclass, field
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

TRACE_ATIF_KEY = "trace:atif"


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


def _tool_call(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments")
    if arguments is None:
        arguments = {}
    elif isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("tool-call arguments are not JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("tool-call arguments are not an object")
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
