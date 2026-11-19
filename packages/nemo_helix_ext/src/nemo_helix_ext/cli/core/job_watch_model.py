# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation state for exploring an Attempt's workflow."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

from nemo_helix_plugin.jobs.watch_types import JobLogEvent, JobWorkflowEvent
from nemo_helix_plugin.jobs.workflows import WorkflowNodeView, WorkflowView


class Pane(IntEnum):
    INPUTS = 0
    NODE = 1
    OUTPUTS = 2


@dataclass(frozen=True)
class Transition:
    observed_at: datetime
    node: str
    description: str


@dataclass(frozen=True)
class BufferedLog:
    sequence: int
    event: JobLogEvent


def phase(node: WorkflowNodeView) -> str:
    if node.reason == "unchanged_inputs" and node.scheduling_state == "idle" and not node.has_pending_run:
        return "reused"
    if node.scheduling_state == "idle" and not node.has_pending_run and node.last_outcome:
        return node.last_outcome
    return node.scheduling_state


def changes(before: WorkflowNodeView, after: WorkflowNodeView) -> str:
    parts: list[str] = []
    if before.scheduling_state != after.scheduling_state or before.last_outcome != after.last_outcome:
        old, new = phase(before), phase(after)
        parts.append(f"{old} → {new}" if old != new else f"outcome: {after.last_outcome or 'none'}")
    if before.requested_run_revision != after.requested_run_revision:
        parts.append("new run requested")
    if before.last_poll_error != after.last_poll_error:
        parts.append("input check: " + (after.last_poll_error or "recovered"))
    if before.freshness != after.freshness:
        parts.append(f"{before.freshness} → {after.freshness}")
    if before.has_pending_run != after.has_pending_run:
        parts.append("rerun pending" if after.has_pending_run else "pending run cleared")
    if before.active_execution_id != after.active_execution_id:
        parts.append(f"execution: {after.active_execution_id or 'finished'}")
    if before.latest_execution_id != after.latest_execution_id and after.latest_execution_id:
        parts.append(f"latest: {after.latest_execution_id}")
    if before.reason != after.reason or before.blocked_by != after.blocked_by:
        parts.append(
            after.reason or ("blocked by " + ", ".join(after.blocked_by) if after.blocked_by else "blockers cleared")
        )
    return "; ".join(parts)


class DagWatchModel:
    def __init__(self, *, emoji: bool = True, step_id: str | None = None) -> None:
        self.emoji = emoji
        self.job_name = ""
        self.job_id = ""
        self.attempt_id = ""
        self.status = "connecting"
        self.workflow: WorkflowView | None = None
        self.nodes: dict[str, WorkflowNodeView] = {}
        self.outputs: dict[str, list[str]] = {}
        self.focus = ""
        self.pane = Pane.NODE
        self.orders: dict[Pane, list[str]] = {Pane.INPUTS: [], Pane.OUTPUTS: []}
        self.selected: dict[Pane, str] = {Pane.INPUTS: "", Pane.OUTPUTS: ""}
        self.history: list[str] = []
        self.activity: dict[str, datetime] = {}
        self.latest: dict[str, Transition] = {}
        self.transitions: deque[Transition] = deque(maxlen=500)
        self.executions: dict[str, list[str]] = {}
        self.execution_selection: dict[str, str] = {}
        self.logs: deque[BufferedLog] = deque(maxlen=5000)
        self.log_sequence = 0
        self.dropped_logs = 0
        self.log_cursor: int | None = None
        self.all_logs = False
        self.last_update: datetime | None = None
        self.notice = ""
        self.initial_step_id = step_id

    @property
    def running(self) -> int:
        return sum(node.scheduling_state == "running" for node in self.nodes.values())

    def apply_workflow(self, event: JobWorkflowEvent, observed_at: datetime) -> None:
        if self.attempt_id and self.attempt_id != event.attempt_id:
            raise ValueError("Watch received a different Attempt; restart watch to select it")
        self.job_name = self.job_name or event.job_name
        self.job_id, self.attempt_id = event.job_id, event.attempt_id
        if self.workflow is not None:
            if self.workflow.control != event.workflow.control:
                self._record("workflow", f"{self.workflow.control} → {event.workflow.control}", observed_at)
            if self.workflow.pass_number != event.workflow.pass_number:
                self._record(
                    "workflow", f"pass {self.workflow.pass_number} → {event.workflow.pass_number}", observed_at
                )
            if self.workflow.waiting_for_changes != event.workflow.waiting_for_changes:
                self._record(
                    "workflow",
                    "waiting for input changes" if event.workflow.waiting_for_changes else "work pending",
                    observed_at,
                )
        for node in event.workflow.nodes:
            previous = self.nodes.get(node.name)
            if previous is not None and (description := changes(previous, node)):
                self._record(node.name, description, observed_at)
            known = self.executions.setdefault(node.name, [])
            for execution in (node.successful_execution_id, node.latest_execution_id, node.active_execution_id):
                if execution and execution not in known:
                    known.append(execution)
        self.workflow = event.workflow.model_copy(deep=True)
        self.nodes = {node.name: node for node in self.workflow.nodes}
        self.outputs = {name: [] for name in self.nodes}
        for node in self.nodes.values():
            for parent in node.depends_on:
                if parent in self.outputs:
                    self.outputs[parent].append(node.name)
        if self.focus not in self.nodes:
            initial = next(
                (
                    node.name
                    for node in self.nodes.values()
                    if self.initial_step_id
                    in (node.active_execution_id, node.latest_execution_id, node.successful_execution_id)
                    and self.initial_step_id is not None
                ),
                "",
            )
            self.focus = initial or next(
                (node.name for node in self.nodes.values() if node.scheduling_state == "running"),
                next(iter(self.nodes), ""),
            )
        self.last_update = observed_at
        self._update_orders()

    def _record(self, name: str, description: str, observed_at: datetime) -> None:
        transition = Transition(observed_at, name, description)
        self.transitions.append(transition)
        self.latest[name] = transition
        self.activity[name] = observed_at

    def _update_orders(self, *, force: bool = False) -> None:
        if not self.focus:
            return
        neighbors = {Pane.INPUTS: self.nodes[self.focus].depends_on, Pane.OUTPUTS: self.outputs[self.focus]}
        for pane, names in neighbors.items():
            available = set(names) & self.nodes.keys()
            ordered = sorted(
                available,
                key=lambda name: (-self.activity[name].timestamp() if name in self.activity else float("inf"), name),
            )
            if pane == self.pane and not force:
                old = [name for name in self.orders[pane] if name in available]
                old_names = set(old)
                ordered = old + [name for name in ordered if name not in old_names]
            self.orders[pane] = ordered
            if self.selected[pane] not in available:
                self.selected[pane] = next(iter(ordered), "")

    def reorder(self) -> None:
        self._update_orders(force=True)

    def choose_pane(self, delta: int) -> None:
        self.pane = Pane((int(self.pane) + delta) % 3)
        self._update_orders()

    def move(self, delta: int) -> None:
        if self.pane == Pane.NODE:
            return
        names = self.orders[self.pane]
        if names:
            index = names.index(self.selected[self.pane])
            self.selected[self.pane] = names[max(0, min(len(names) - 1, index + delta))]

    def focus_node(self, name: str, *, remember: bool = True) -> None:
        if name not in self.nodes or name == self.focus:
            return
        if remember:
            self.history.append(self.focus)
            self.history = self.history[-100:]
        self.focus = name
        self.pane = Pane.NODE
        self.orders = {Pane.INPUTS: [], Pane.OUTPUTS: []}
        self.log_cursor = None
        self._update_orders(force=True)

    def enter_neighbor(self) -> None:
        if self.pane != Pane.NODE:
            self.focus_node(self.selected[self.pane])

    def back(self) -> None:
        if self.history:
            self.focus_node(self.history.pop(), remember=False)

    def search(self, query: str) -> list[str]:
        return sorted(name for name in self.nodes if query.casefold() in name.casefold())

    def execution(self) -> str | None:
        selected = self.execution_selection.get(self.focus)
        if selected:
            return selected
        node = self.nodes.get(self.focus)
        return (node.active_execution_id or node.latest_execution_id or node.successful_execution_id) if node else None

    def cycle_execution(self) -> None:
        known = self.executions.get(self.focus, [])
        if known:
            current = self.execution()
            index = known.index(current) if current in known else -1
            self.execution_selection[self.focus] = known[(index + 1) % len(known)]
            self.log_cursor = None

    def add_log(self, event: JobLogEvent) -> None:
        self.log_sequence += 1
        if len(self.logs) == self.logs.maxlen:
            self.dropped_logs += 1
        self.logs.append(
            BufferedLog(
                self.log_sequence,
                JobLogEvent(
                    kind="log",
                    job_name=event.job_name,
                    timestamp=event.timestamp,
                    step_id=event.step_id,
                    task_id=event.task_id,
                    message=event.message[:4096] + (" [truncated]" if len(event.message) > 4096 else ""),
                ),
            )
        )

    def visible_logs(self) -> list[BufferedLog]:
        execution = self.execution()
        return [
            line for line in self.logs if (self.all_logs or (execution is not None and line.event.step_id == execution))
        ]

    def scroll_logs(self, delta: int) -> None:
        lines = self.visible_logs()
        if not lines:
            return
        index = (
            next((i for i, line in enumerate(lines) if line.sequence >= self.log_cursor), len(lines) - 1)
            if self.log_cursor is not None
            else len(lines) - 1
        )
        self.log_cursor = lines[max(0, min(len(lines) - 1, index + delta))].sequence

    def icon(self, state: str) -> str:
        symbols = {
            "reused": ("♻️", "="),
            "running": ("🏃", ">"),
            "completed": ("✅", "+"),
            "error": ("❌", "!"),
            "cancelled": ("🛑", "x"),
            "ready": ("🚦", ">"),
            "waiting_dependencies": ("⏳", "."),
            "waiting_capacity": ("🚧", "."),
            "held": ("⏸️", "|"),
            "idle": ("💤", "-"),
            "dirty": ("🔄", "*"),
            "unknown": ("❔", "?"),
            "clean": ("✅", "+"),
        }
        pair = symbols.get(state, ("❔", "?"))
        return pair[0 if self.emoji else 1]
