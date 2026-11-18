# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One tick of a distillation workflow.

The tick mutates the workflow and returns the starts and cancels to perform
after the caller has saved it. A node is ``running``, with its external names
already stored, before anything is created, so a crash retries the same name.

A node is ``succeeded`` only after its output is confirmed. Job nodes that write
their own ``output.json`` are confirmed by reading that file. The other kinds
return the confirmed output from the external API; the tick writes the file and
then the entity. A missing or invalid file stays ``running`` for
:data:`~nemo_distillation_plugin.models.CONFIRM_GRACE_SECONDS`, then ``failed``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from nemo_distillation_plugin.models import (
    CONFIRM_GRACE_SECONDS,
    OWN_OUTPUT_KINDS,
    TERMINAL_NODE_STATUSES,
    Workflow,
)
from nemo_distillation_plugin.projection import project


@dataclass(frozen=True)
class Action:
    type: Literal["start", "cancel"]
    node: str


@dataclass
class Poll:
    phase: Literal["running", "completed", "failed", "cancelled", "missing"]
    outputs: dict[str, Any] | None = None
    error: str | None = None


class NodeAdapter(Protocol):
    async def start(self, workflow: Workflow, node_key: str) -> None: ...

    async def poll(self, workflow: Workflow, node_key: str) -> Poll: ...

    async def cancel(self, workflow: Workflow, node_key: str) -> None: ...


class Ports(Protocol):
    def adapter(self, kind: str) -> NodeAdapter: ...

    async def read_output(self, workflow: Workflow, node_key: str) -> dict[str, Any] | None: ...

    async def write_output(self, workflow: Workflow, node_key: str, payload: dict[str, Any]) -> None: ...


async def tick(workflow: Workflow, ports: Ports, now: float) -> list[Action]:
    """Advance ``workflow`` one step. The caller saves it before performing the actions."""
    actions: list[Action] = []
    cancelling = workflow.status in {"cancelling", "cancelled"}

    for key in list(workflow.nodes):
        runtime = workflow.nodes[key]
        if runtime.status != "running":
            continue
        spec = workflow.graph[key]
        poll = await ports.adapter(spec.kind).poll(workflow, key)
        if poll.phase == "missing":
            if cancelling and not spec.always_run:
                runtime.status = "cancelled"
            else:
                actions.append(Action("start", key))
            continue
        if poll.phase == "running":
            continue
        if poll.phase == "failed":
            runtime.status = "failed"
            runtime.error = poll.error or "failed"
            continue
        if poll.phase == "cancelled":
            runtime.status = "cancelled"
            continue
        if spec.kind in OWN_OUTPUT_KINDS:
            payload = await ports.read_output(workflow, key)
            if payload is None:
                if runtime.confirm_since is None:
                    runtime.confirm_since = now
                elif now - runtime.confirm_since >= CONFIRM_GRACE_SECONDS:
                    runtime.status = "failed"
                    runtime.error = "output.json was missing or invalid"
                continue
            runtime.outputs = payload
        else:
            payload = dict(poll.outputs or {})
            await ports.write_output(workflow, key, payload)
            runtime.outputs = payload
        runtime.status = "succeeded"
        runtime.confirm_since = None
        runtime.error = None

    for key in list(workflow.nodes):
        runtime = workflow.nodes[key]
        if runtime.status != "pending":
            continue
        spec = workflow.graph[key]
        dep_status = [workflow.nodes[dep].status for dep in spec.depends_on]
        if _can_start(spec.always_run, dep_status, cancelling):
            runtime.status = "running"
            runtime.confirm_since = None
            actions.append(Action("start", key))
            continue
        if cancelling and not spec.always_run:
            runtime.status = "cancelled"
            continue
        if not spec.always_run and any(status in {"failed", "cancelled", "skipped"} for status in dep_status):
            runtime.status = "skipped"
            runtime.error = "a dependency did not succeed"

    if cancelling:
        for key, runtime in workflow.nodes.items():
            if runtime.status == "running" and not workflow.graph[key].always_run:
                actions.append(Action("cancel", key))

    project(workflow)
    workflow.status = _workflow_status(workflow, cancelling)
    return actions


def _can_start(always_run: bool, dep_status: Sequence[str], cancelling: bool) -> bool:
    if not dep_status:
        return always_run or not cancelling
    if always_run:
        return all(status in TERMINAL_NODE_STATUSES for status in dep_status)
    if cancelling:
        return False
    return all(status == "succeeded" for status in dep_status)


def _workflow_status(
    workflow: Workflow, cancelling: bool
) -> Literal["pending", "running", "cancelling", "completed", "failed", "cancelled"]:
    statuses = {node.status for node in workflow.nodes.values()}
    if cancelling:
        return "cancelled" if statuses <= TERMINAL_NODE_STATUSES else "cancelling"
    if not statuses or statuses <= {"pending"}:
        return "pending"
    if statuses <= {"succeeded"}:
        return "completed"
    if statuses <= TERMINAL_NODE_STATUSES:
        return "failed"
    return "running"
