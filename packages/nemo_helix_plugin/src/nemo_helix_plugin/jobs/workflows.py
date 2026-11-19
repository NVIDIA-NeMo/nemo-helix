# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow definitions and controller-owned progress shared by Jobs clients."""

from collections import deque
from datetime import datetime
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

WorkflowMode = Literal["run_to_completion", "continuous"]
WorkflowControl = Literal["running", "pausing", "paused", "cancelling", "failing"]
WorkflowScheduling = Literal["ready", "waiting_dependencies", "waiting_capacity", "held", "running", "idle"]


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: WorkflowMode = "run_to_completion"
    input_change_policy: Literal["after_drain"] = "after_drain"


class WorkflowInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fileset_id: str = Field(min_length=1)


class WorkflowInvalidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    steps: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_steps(self) -> Self:
        if len(set(self.steps)) != len(self.steps):
            raise ValueError("Invalidation steps must be distinct")
        return self


class WorkflowSignals(RootModel[dict[str, str]]):
    pass


class WorkflowNodeState(BaseModel):
    requested_run_revision: str = Field(default_factory=lambda: str(uuid4()))
    observed_inputs: dict[str, str] | None = None
    successful_inputs: dict[str, str] | None = None
    successful_revision: str | None = None
    last_attempted_revision: str | None = None
    active_step_id: str | None = None
    active_run_revision: str | None = None
    active_inputs: dict[str, str] | None = None
    latest_step_id: str | None = None
    successful_step_id: str | None = None
    last_outcome: Literal["completed", "error", "cancelled"] | None = None
    last_invalidation: str | None = None
    decision_pass: int = 0
    last_inputs_checked_at: datetime | None = None
    last_poll_attempt_at: datetime | None = None
    last_poll_error: str | None = None
    scheduling_state: WorkflowScheduling = "waiting_dependencies"
    reason: str | None = None
    blocked_by: list[str] = Field(default_factory=list)

    @property
    def freshness(self) -> Literal["clean", "dirty", "unknown"]:
        if self.observed_inputs is None or self.last_poll_error:
            return "unknown"
        if self.successful_revision == self.requested_run_revision and self.successful_inputs == self.observed_inputs:
            return "clean"
        return "dirty"


class WorkflowState(BaseModel):
    nodes: dict[str, WorkflowNodeState]
    pass_number: int = 1
    pass_open: bool = True
    control: WorkflowControl = "running"
    paused_at: datetime | None = None
    stopped_at: datetime | None = None


class WorkflowNodeView(BaseModel):
    name: str
    depends_on: list[str]
    freshness: Literal["clean", "dirty", "unknown"]
    scheduling_state: WorkflowScheduling
    reason: str | None
    blocked_by: list[str]
    requested_run_revision: str
    active_execution_id: str | None
    latest_execution_id: str | None
    successful_execution_id: str | None
    last_outcome: Literal["completed", "error", "cancelled"] | None
    has_pending_run: bool
    last_inputs_checked_at: datetime | None
    last_poll_attempt_at: datetime | None
    last_poll_error: str | None


class WorkflowView(BaseModel):
    mode: WorkflowMode
    pass_number: int
    control: WorkflowControl
    waiting_for_changes: bool
    degraded: bool
    nodes: list[WorkflowNodeView]


def validate_workflow_dependencies(dependencies: dict[str, list[str]]) -> None:
    """Reject unknown/duplicate dependencies and cycles before persistence."""
    for name, parents in dependencies.items():
        if len(parents) != len(set(parents)):
            raise ValueError(f"Duplicate dependencies for workflow step {name}")
        for parent in parents:
            if parent not in dependencies:
                raise ValueError(f"Unknown dependency {parent} for workflow step {name}")

    visited: set[str] = set()
    path: list[str] = []
    positions: dict[str, int] = {}
    for root in dependencies:
        if root in visited:
            continue
        path.append(root)
        positions[root] = 0
        stack = [iter(dependencies[root])]
        while stack:
            parent = next(stack[-1], None)
            if parent is None:
                finished = path.pop()
                visited.add(finished)
                del positions[finished]
                stack.pop()
            elif parent in positions:
                cycle = path[positions[parent] :] + [parent]
                raise ValueError("Workflow cycle: " + " -> ".join(cycle))
            elif parent not in visited:
                positions[parent] = len(path)
                path.append(parent)
                stack.append(iter(dependencies[parent]))


def workflow_order(dependencies: dict[str, list[str]]) -> list[str]:
    """Order an already validated workflow, preserving declaration order for ties."""
    followers: dict[str, list[str]] = {name: [] for name in dependencies}
    remaining: dict[str, int] = {}
    for name, parents in dependencies.items():
        remaining[name] = len(parents)
        for parent in parents:
            followers[parent].append(name)
    ready = deque(name for name, count in remaining.items() if count == 0)
    order: list[str] = []
    while ready:
        name = ready.popleft()
        order.append(name)
        for child in followers[name]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    return order
