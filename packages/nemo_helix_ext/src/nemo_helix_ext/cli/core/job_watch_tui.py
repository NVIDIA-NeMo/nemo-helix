# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive node-focused Jobs watch using the CLI's terminal and clients."""

import asyncio
import os
import sys
import textwrap
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Literal

from nemo_helix_plugin.jobs.watch_types import (
    JobLogEvent,
    JobStatusEvent,
    JobWarningEvent,
    JobWatchEvent,
    JobWorkflowEvent,
)
from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

from nemo_helix_ext.cli.core.job_watch_model import DagWatchModel, Pane, phase
from nemo_helix_ext.cli.core.job_watch_renderer import JobWatchRenderResult


def interactive_watch_available() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def _literal(value: str) -> str:
    return "".join(char if char.isprintable() or char in "\ufe0f\u200d" else " " for char in value)


def _fit(value: str, width: int) -> str:
    value = _literal(value)
    if get_cwidth(value) <= width:
        return value + " " * (width - get_cwidth(value))
    result = ""
    for char in value:
        if get_cwidth(result + char) > max(0, width - 1):
            break
        result += char
    return result + ("…" if width else "")


class DagWatchApp:
    def __init__(self, model: DagWatchModel, *, input: Input | None = None, output: Output | None = None) -> None:
        self.model = model
        self.mode: Literal["status", "logs", "overview"] = "status"
        self.searching = False
        self.query = ""
        self.overview_index = 0
        self.node_scroll = 0
        self.log_horizontal = 0
        self.page_nodes = 1
        self.page_lines = 1
        self.status_finished = False
        self.logs_finished = False
        self.result = JobWatchRenderResult.INTERRUPTED
        self.application: Application[JobWatchRenderResult] = Application(
            layout=Layout(Window(FormattedTextControl(self._formatted_screen), wrap_lines=False)),
            key_bindings=self._bindings(),
            full_screen=True,
            mouse_support=False,
            refresh_interval=1,
            input=input,
            output=output,
            style=Style.from_dict({"header": "bold ansicyan", "selected": "reverse", "footer": "ansiyellow"}),
        )

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        browsing = Condition(lambda: not self.searching)

        @bindings.add("q", filter=browsing)
        @bindings.add("c-c")
        def quit_watch(event: KeyPressEvent) -> None:
            event.app.exit(result=self.result)

        @bindings.add("tab", filter=browsing)
        def switch(event: KeyPressEvent) -> None:
            self.mode = "status" if self.mode == "logs" else "logs"

        @bindings.add("1", filter=browsing)
        def status(event: KeyPressEvent) -> None:
            self.mode = "status"

        @bindings.add("2", filter=browsing)
        def logs(event: KeyPressEvent) -> None:
            self.mode = "logs"

        @bindings.add("left", filter=browsing)
        @bindings.add("right", filter=browsing)
        def pane(event: KeyPressEvent) -> None:
            delta = -1 if event.key_sequence[0].key == Keys.Left else 1
            if self.mode == "logs":
                self.log_horizontal = max(0, min(4200, self.log_horizontal + delta * 8))
            else:
                self.model.choose_pane(delta)

        @bindings.add("up", filter=browsing)
        @bindings.add("down", filter=browsing)
        @bindings.add("pageup", filter=browsing)
        @bindings.add("pagedown", filter=browsing)
        def move(event: KeyPressEvent) -> None:
            key = event.key_sequence[0].key
            amount = -1 if key in (Keys.Up, Keys.PageUp) else 1
            if self.mode == "logs":
                self.model.scroll_logs(amount * (self.page_lines if key in (Keys.PageUp, Keys.PageDown) else 1))
            elif self.mode == "overview":
                names = self.model.search(self.query)
                self.overview_index = max(
                    0,
                    min(
                        len(names) - 1,
                        self.overview_index + amount * (self.page_nodes if key in (Keys.PageUp, Keys.PageDown) else 1),
                    ),
                )
            elif self.model.pane == Pane.NODE:
                self.node_scroll = max(0, self.node_scroll + amount)
            else:
                self.model.move(amount * (self.page_nodes if key in (Keys.PageUp, Keys.PageDown) else 1))

        @bindings.add("enter")
        def enter(event: KeyPressEvent) -> None:
            if self.searching or self.mode == "overview":
                names = self.model.search(self.query)
                exact = self.query if self.query in names else None
                if names:
                    self.node_scroll = 0
                    self.model.focus_node(exact or names[min(self.overview_index, len(names) - 1)])
                    self.mode = "status"
                self.searching = False
            else:
                self.node_scroll = 0
                self.model.enter_neighbor()

        @bindings.add("backspace", filter=browsing)
        def back(event: KeyPressEvent) -> None:
            self.node_scroll = 0
            self.model.back()
            self.mode = "status"

        @bindings.add("/", filter=browsing)
        def search(event: KeyPressEvent) -> None:
            self.searching = True
            self.query = ""
            self.overview_index = 0
            self.mode = "overview"

        @bindings.add("o", filter=browsing)
        def overview(event: KeyPressEvent) -> None:
            self.mode = "status" if self.mode == "overview" else "overview"
            self.query = ""
            self.overview_index = 0

        @bindings.add("escape")
        def escape(event: KeyPressEvent) -> None:
            self.searching = False
            self.mode = "status"

        @bindings.add("a", filter=browsing)
        def all_or_activity(event: KeyPressEvent) -> None:
            if self.mode == "logs":
                self.model.all_logs = not self.model.all_logs
                self.model.log_cursor = None
            else:
                self.model.reorder()

        @bindings.add("e", filter=browsing)
        def execution(event: KeyPressEvent) -> None:
            self.model.cycle_execution()

        @bindings.add("f", filter=browsing)
        def follow(event: KeyPressEvent) -> None:
            self.model.log_cursor = None

        @bindings.add("backspace", filter=Condition(lambda: self.searching))
        def delete_search(event: KeyPressEvent) -> None:
            self.query = self.query[:-1]
            self.overview_index = 0

        @bindings.add(Keys.Any, filter=Condition(lambda: self.searching))
        def type_search(event: KeyPressEvent) -> None:
            if event.data.isprintable():
                self.query += event.data
                self.overview_index = 0

        return bindings

    def _formatted_screen(self) -> FormattedText:
        size = self.application.output.get_size()
        lines = self.screen(size)
        result = FormattedText()
        for index, line in enumerate(lines):
            style = "class:header" if index < 3 else "class:footer" if index >= len(lines) - 2 else ""
            if line.startswith("> "):
                style = "class:selected"
            result.append((style, _fit(line, size.columns) + ("\n" if index < len(lines) - 1 else "")))
        return result

    def screen(self, size: Size) -> list[str]:
        model = self.model
        width, height = max(1, size.columns), max(1, size.rows)
        workflow = model.workflow
        if height < 8:
            return [f"{model.status} · {model.running} running", "Resize terminal (8+ rows) · q quit"][:height]
        age = max(0, int((datetime.now(timezone.utc) - model.last_update).total_seconds())) if model.last_update else 0
        control = workflow.control if workflow else "connecting"
        if workflow and workflow.waiting_for_changes:
            control += " / waiting for changes"
        if workflow and workflow.degraded:
            control += " / degraded"
        header = [
            f"{'⚡ ' if model.emoji else ''}{model.job_name} · Attempt {model.attempt_id} · {model.status.upper()}",
            f"{len(model.nodes)} nodes · {model.icon('running')} {model.running} running in parallel · Pass {workflow.pass_number if workflow else '?'} · {control} · workflow update {age}s ago",
            f"{'[1 Status]' if self.mode == 'status' else '1 Status'}   {'[2 Logs]' if self.mode == 'logs' else '2 Logs'}   {'[o Overview]' if self.mode == 'overview' else 'o Overview'} · Focus: {model.focus}",
        ]
        if self.status_finished:
            header[0] += " · WATCH FINISHED (q to close)"
            if not self.logs_finished:
                header[0] += " · logs draining"
        body_height = max(1, height - 6)
        self.page_lines = body_height
        if self.mode == "logs":
            body = self._logs(body_height)
        elif self.mode == "overview":
            body = self._overview(body_height)
        else:
            body = self._status(width, body_height)
        body = (body + [""] * body_height)[:body_height]
        footer = [
            model.notice,
            ("Find node: " + self.query + "_ · Enter jump · Esc cancel")
            if self.searching
            else (
                "↑↓ scroll · ←→ pan · a all/focused · e execution · f follow · Tab status · q quit"
                if self.mode == "logs"
                else "←→ pane · ↑↓/PgUp/PgDn select · Enter focus · Backspace back · / search · a reorder · Tab logs · q quit"
            ),
        ]
        return (header + [""] + body + footer)[:height]

    def _status(self, width: int, height: int) -> list[str]:
        if width < 100:
            title = f"Compact view · ←→ choose pane · {self.model.pane.name.lower()}"
            return [title] + self._pane(self.model.pane, width, max(1, height - 1))[: max(0, height - 1)]
        column = max(1, (width - 6) // 3)
        panes = [self._pane(pane, column, height) for pane in Pane]
        return [
            " │ ".join(_fit(lines[index] if index < len(lines) else "", column) for lines in panes)
            for index in range(height)
        ]

    def _pane(self, pane: Pane, width: int, height: int) -> list[str]:
        model = self.model
        active = "> " if model.pane == pane else "  "
        title = (
            {Pane.INPUTS: "📥 INPUTS", Pane.NODE: "🎯 FOCUSED NODE", Pane.OUTPUTS: "📤 OUTPUTS"}[pane]
            if model.emoji
            else pane.name
        )
        if pane == Pane.NODE:
            node = model.nodes.get(model.focus)
            if node is None:
                return [active + title, "Waiting for workflow state"]
            lines = [
                active + title,
                model.focus,
                f"{model.icon(phase(node))} {phase(node)}",
                f"Freshness: {node.freshness}",
                f"Latest outcome: {node.last_outcome or 'none'}",
                f"Pending rerun: {'yes' if node.has_pending_run else 'no'}",
                f"Execution: {model.execution() or 'none'}",
            ]
            if node.reason:
                lines.extend(textwrap.wrap("Reason: " + _literal(node.reason), max(1, width))[:2])
            if node.blocked_by:
                blockers = ", ".join(node.blocked_by[:2])
                if len(node.blocked_by) > 2:
                    blockers += f" (+{len(node.blocked_by) - 2} more upstream)"
                lines.extend(textwrap.wrap("Blocked by: " + blockers, max(1, width)))
            lines += ["", "Observed transitions (session)"]
            recent = [
                transition for transition in reversed(model.transitions) if transition.node in (model.focus, "workflow")
            ]
            for transition in recent:
                lines.append(transition.observed_at.astimezone().strftime("%H:%M:%S") + " " + transition.description)
            if not recent:
                lines.append("No observed changes yet")
            self.node_scroll = min(self.node_scroll, max(0, len(lines) - height))
            return lines[self.node_scroll : self.node_scroll + height]
        names = model.orders[pane]
        self.page_nodes = max(1, (height - 3) // 4)
        index = names.index(model.selected[pane]) if names else 0
        start = max(0, min(index - self.page_nodes // 2, len(names) - self.page_nodes))
        visible = names[start : start + self.page_nodes]
        lines = [
            active + title,
            f"{start + 1 if names else 0}–{start + len(visible)} of {len(names)} · activity first",
            "↑ more" if start else "",
        ]
        for name in visible:
            node = model.nodes[name]
            marker = "> " if name == model.selected[pane] else "  "
            latest = model.latest.get(name)
            lines += [
                marker + model.icon(phase(node)) + " " + name,
                "  " + (latest.description if latest else phase(node)),
                "  "
                + (
                    latest.observed_at.astimezone().strftime("%H:%M:%S") + " observed"
                    if latest
                    else "Activity time unknown"
                ),
                "",
            ]
        if start + len(visible) < len(names):
            if len(lines) >= height:
                lines = lines[: height - 1]
            lines.append(f"↓ {len(names) - start - len(visible)} more")
        if not names:
            lines.append("No upstream nodes" if pane == Pane.INPUTS else "No downstream nodes")
        return lines[:height]

    def _overview(self, height: int) -> list[str]:
        names = self.model.search(self.query)
        self.overview_index = min(self.overview_index, max(0, len(names) - 1))
        self.page_nodes = max(1, (height - 2) // 2)
        start = max(0, self.overview_index - self.page_nodes // 2)
        visible = names[start : start + self.page_nodes]
        lines = [
            f"DAG overview · {len(names)} matching nodes · / search · Enter focus",
            f"{start + 1 if names else 0}–{start + len(visible)} of {len(names)}",
        ]
        for index, name in enumerate(visible, start):
            node = self.model.nodes[name]
            lines += [
                ("> " if index == self.overview_index else "  ")
                + f"{self.model.icon(phase(node))} {name} · {phase(node)}",
                "   ← " + (", ".join(node.depends_on) or "root"),
            ]
        return lines[:height]

    def _logs(self, height: int) -> list[str]:
        model = self.model
        scope = "all executions" if model.all_logs else f"{model.focus} · {model.execution() or 'no execution'}"
        lines = [
            f"Logs: {scope} · {'FOLLOW' if model.log_cursor is None else 'SCROLL'}",
            "Executions: session-known references · e cycle · dropped buffer lines: " + str(model.dropped_logs),
        ]
        buffered = model.visible_logs()
        if model.log_cursor is not None:
            buffered = [line for line in buffered if line.sequence <= model.log_cursor]
        for line in buffered[-max(1, height - 2) :]:
            event = line.event
            stamp = event.timestamp.astimezone().strftime("%H:%M:%S") if event.timestamp else "--:--:--"
            lines.append(
                (f"{stamp} [{event.step_id or '?'}/{event.task_id or '?'}] {_literal(event.message)}")[
                    self.log_horizontal :
                ]
            )
        if len(lines) == 2:
            lines.append("No buffered logs for this execution")
        return lines[:height]

    async def _consume_status(self, events: AsyncIterator[JobWatchEvent]) -> None:
        try:
            async for event in events:
                if isinstance(event, JobWorkflowEvent):
                    self.model.apply_workflow(event, datetime.now(timezone.utc))
                elif isinstance(event, JobStatusEvent):
                    self.model.status = event.status
                    if event.terminal:
                        self.result = (
                            JobWatchRenderResult.SUCCEEDED if event.successful else JobWatchRenderResult.FAILED
                        )
                        self.model.notice = str(event.error_details or "")
                elif isinstance(event, JobWarningEvent):
                    self.model.notice = "Status: " + event.message
                self.application.invalidate()
        except Exception as exc:
            self.result = JobWatchRenderResult.FAILED
            self.model.notice = "Watch failed: " + str(exc)
        finally:
            self.status_finished = True
            self.application.invalidate()

    async def _consume_logs(self, events: AsyncIterator[JobWatchEvent]) -> None:
        try:
            async for event in events:
                if isinstance(event, JobLogEvent):
                    self.model.add_log(event)
                elif isinstance(event, JobWarningEvent):
                    self.model.notice = "Logs: " + event.message
                self.application.invalidate()
        except Exception as exc:
            self.model.notice = "Logs unavailable: " + str(exc)
        finally:
            self.logs_finished = True
            self.application.invalidate()

    async def run(
        self, status_events: AsyncIterator[JobWatchEvent], log_events: AsyncIterator[JobWatchEvent]
    ) -> JobWatchRenderResult:
        tasks = [
            asyncio.create_task(self._consume_status(status_events)),
            asyncio.create_task(self._consume_logs(log_events)),
        ]
        try:
            return await self.application.run_async()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
