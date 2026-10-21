# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure transform: one tau2 SimulationRun → a list of OTLP span dicts.

This is the intermediate representation — plain dicts, no protobuf and no
network — so the transform stays trivially unit-testable. :mod:`evaluation.otlp_ingest`
turns these dicts into an OTLP ``ExportTraceServiceRequest`` and POSTs them.

The span tree mirrors what Intake's ATIF importer produced (so Compass reads
it unchanged), but emitted over the permissive OTLP route so multi-actor traces
(telecom user-driven device tools) ingest without ATIF-v1.7's agent-only 422::

    AGENT       name=<agent_name>        one per sim (the trajectory root)
     ├─ CHAIN   name=user-<step_id>      one per user utterance (tau2.actor="user")
     ├─ LLM     name=agent-<step_id>     one per assistant turn
     │   └─ TOOL name=<function_name>    agent tool call (child of its LLM turn)
     ├─ TOOL    name=<function_name>     user-driven device tool  ← multi-actor
     └─ EVALUATOR name=tau2.verifier     one per sim, oracle (include_rewards) only
"""

import hashlib
import json
import re
import time
from typing import Any

_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")

# tau2 sims carry only ISO timestamps / latencies, not the contiguous nanos OTLP
# wants, so spans get a synthesized monotonic clock seeded from ``base_ns`` (one
# tick per span). The base must be near ingest time: Intake drops spans dated
# outside its retention window, so a fixed past base makes them silently
# un-queryable. Callers pass ``base_ns`` (the adapter passes wall-clock ``now``);
# tests pin it for determinism. Span ids derive from the session id, not the
# clock, so the transform stays deterministic regardless of the base.
_STEP_NS = 1_000_000  # 1ms per span


def _slug(value: object) -> str:
    """Make an id path-safe: collapse any run of disallowed chars to a single '-'."""
    return _SLUG.sub("-", str(value))


def session_id_for(sim: dict[str, Any], *, evaluation_name: str) -> str:
    """The per-run, per-sim session id: ``<evaluation_name>-task<task_id>-t<trial>``.

    Scoped by ``evaluation_name`` (the run id) so every id derived from it — the span
    id, trace id, and the evaluator-results id — is unique per run. Runs that
    re-cover the same task in a shared workspace no longer collide (which would
    cross-link spans and overwrite one run's scores with another's). ``evaluation_name``
    already carries the agent/base name as its prefix, so the agent stays visible,
    and the realistic + oracle twins of one sim in one run still share this id.
    """
    trial = sim.get("trial") or 0
    return f"{_slug(evaluation_name)}-task{_slug(sim.get('task_id', ''))}-t{trial}"


def _span_id(session_id: str, index: int) -> str:
    """A per-sim-unique, deterministic 8-byte span id (hex).

    Derived from the session id so ids are stable across realistic/oracle builds
    of the same sim and unique across sims **and runs** in a workspace (the EVALUATOR span id is
    later the loose target of an evaluator_results row, so cross-sim collisions
    would cross-link rewards)."""
    return hashlib.sha256(f"{session_id}:span:{index}".encode()).digest()[:8].hex()


def _text(content: object) -> str:
    """Normalize a message's content to a string (JSON-encode non-strings)."""
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content)


def sim_to_spans(
    sim: dict[str, Any],
    *,
    agent_name: str,
    agent_version: str,
    session_id: str,
    evaluation_name: str,
    task: dict[str, Any] | None = None,
    include_rewards: bool,
    agent_llm: str | None = None,
    base_ns: int | None = None,
) -> list[dict[str, Any]]:
    """Transform one tau2 SimulationRun into a list of OTLP span dicts.

    Each span dict is ``{span_id, parent_span_id, name, kind, start_ns, end_ns,
    attributes, status, status_message}`` where ``kind`` is one of
    ``AGENT|LLM|TOOL|EVALUATOR``, ``parent_span_id`` is ``None`` for the root, and
    ids/timestamps are hex/int (no protobuf here). Pure: no network, no wall-clock.

    ``include_rewards`` gates the oracle answer key — the EVALUATOR span and the
    task's ``evaluation_criteria`` only appear when it is set (the realistic twin
    withholds them so the eval is unaided).

    ``evaluation_name`` is stamped on every span as ``nemo.evaluation.name`` (with
    the sim's task name as ``nemo.test_case.name``) so a run's spans are queryable
    back via the spans filter ``{"evaluation_name": evaluation_name}`` — the per-run
    scope that lets many runs share one workspace.

    ``base_ns`` seeds the per-span monotonic clock; it must be near ingest time
    (Intake drops spans dated outside its retention window). Defaults to
    wall-clock ``now`` when omitted.
    """
    messages = sim.get("messages", [])
    base = base_ns if base_ns is not None else time.time_ns()
    spans: list[dict[str, Any]] = []
    test_case_id = str(sim.get("task_id", ""))

    def _common() -> dict[str, Any]:
        return {
            "gen_ai.conversation.id": session_id,
            "session.id": session_id,
            "nemo.evaluation.name": evaluation_name,
            "nemo.test_case.name": test_case_id,
        }

    def _add(*, name: str, kind: str, parent: str | None, attributes: dict[str, Any]) -> dict[str, Any]:
        index = len(spans)
        start = base + index * _STEP_NS
        span = {
            "span_id": _span_id(session_id, index),
            "parent_span_id": parent,
            "name": name,
            "kind": kind,
            "start_ns": start,
            "end_ns": start + _STEP_NS,
            "attributes": {"openinference.span.kind": kind, **_common(), **attributes},
            "status": "OK",
            "status_message": None,
        }
        spans.append(span)
        return span

    first_user = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    last_agent = next((_text(m.get("content")) for m in reversed(messages) if m.get("role") == "assistant"), "")
    root_attrs: dict[str, Any] = {
        "gen_ai.agent.name": agent_name,
        "gen_ai.agent.version": agent_version,
        "input.value": first_user,
        "output.value": last_agent,
    }
    if task is not None:
        root_attrs["tau2.task"] = json.dumps(_trim_task(task, include_rewards=include_rewards))
    root = _add(name=agent_name, kind="AGENT", parent=None, attributes=root_attrs)

    # Match tool results back to their TOOL span: by id, else positional FIFO within
    # the same actor (agent results never claim a user tool span, and vice versa).
    tool_span_by_id: dict[str, dict[str, Any]] = {}
    pending: dict[str, list[dict[str, Any]]] = {"assistant": [], "user": []}
    step_id = 0

    def _tool_span(call: dict[str, Any], *, parent: str, actor: str) -> dict[str, Any]:
        name = call.get("name") or ""
        attrs = {"gen_ai.tool.name": name, "input.value": json.dumps(call.get("arguments") or {})}
        if actor == "user":
            attrs["tau2.actor"] = "user"  # not the agent's action; no gen_ai.agent.name
        else:
            attrs["gen_ai.agent.name"] = agent_name
        span = _add(name=name, kind="TOOL", parent=parent, attributes=attrs)
        call_id = call.get("id")
        if call_id:
            tool_span_by_id[call_id] = span
        pending[actor].append(span)
        return span

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            step_id += 1
            content = _text(msg.get("content"))
            calls = msg.get("tool_calls") or []
            output = {
                "content": content,
                "tool_calls": [
                    {"id": c.get("id") or "", "name": c.get("name") or "", "arguments": c.get("arguments") or {}}
                    for c in calls
                ],
            }
            llm_attrs = {
                "output.value": json.dumps(output),
                "output.mime_type": "application/json",
            }
            if agent_llm:
                llm_attrs["gen_ai.request.model"] = agent_llm
            llm = _add(name=f"agent-{step_id}", kind="LLM", parent=root["span_id"], attributes=llm_attrs)
            for call in calls:
                actor = call.get("requestor") or "assistant"
                parent = root["span_id"] if actor == "user" else llm["span_id"]
                _tool_span(call, parent=parent, actor=actor)
        elif role == "user":
            step_id += 1
            content = _text(msg.get("content"))
            if content:
                # The user's utterance as a timeline turn (no LLM span; mark the actor).
                _add(
                    name=f"user-{step_id}",
                    kind="CHAIN",
                    parent=root["span_id"],
                    attributes={"tau2.actor": "user", "output.value": content, "output.mime_type": "text/plain"},
                )
            # A user turn's device tools also hang off the AGENT root.
            for call in msg.get("tool_calls") or []:
                _tool_span(call, parent=root["span_id"], actor=call.get("requestor") or "user")
        elif role == "system":
            step_id += 1
        elif role == "tool":
            _record_tool_result(msg, tool_span_by_id=tool_span_by_id, pending=pending)

    reward = (sim.get("reward_info") or {}).get("reward")
    if include_rewards and reward is not None:
        verifier_result = sim.get("reward_info") or {"score": reward}
        ev_out = {"score": reward, "verifier_result": verifier_result}
        _add(
            name="tau2.verifier",
            kind="EVALUATOR",
            parent=root["span_id"],
            attributes={
                "score": reward,
                "output.value": json.dumps(ev_out),
                "output.mime_type": "application/json",
            },
        )
    root["end_ns"] = max((s["end_ns"] for s in spans), default=root["end_ns"])
    return spans


def _record_tool_result(
    msg: dict[str, Any], *, tool_span_by_id: dict[str, dict[str, Any]], pending: dict[str, list[dict[str, Any]]]
) -> None:
    """Fold a ``role="tool"`` result into its TOOL span's output + error status."""
    actor = msg.get("requestor") or "assistant"
    call_id = msg.get("tool_call_id") or msg.get("id")
    span = tool_span_by_id.pop(call_id, None) if call_id else None
    if span is not None:
        for queue in pending.values():
            if span in queue:
                queue.remove(span)
                break
    elif pending[actor]:
        span = pending[actor].pop(0)
    if span is None:
        return  # orphan tool result with no matching call — nothing to attach to
    content = _text(msg.get("content"))
    span["attributes"]["output.value"] = content
    if msg.get("error"):
        span["status"] = "ERROR"
        span["status_message"] = content or "tool call failed"
        span["attributes"]["exception.message"] = content or "tool call failed"


def _trim_task(task: dict[str, Any], *, include_rewards: bool) -> dict[str, Any]:
    """Keep the task's setup (description, scenario); withhold the gold answer key
    (``evaluation_criteria``) unless ``include_rewards`` — the oracle-gating invariant."""
    trimmed: dict[str, Any] = {"description": task.get("description"), "user_scenario": task.get("user_scenario")}
    if include_rewards:
        trimmed["evaluation_criteria"] = task.get("evaluation_criteria")
    return trimmed
