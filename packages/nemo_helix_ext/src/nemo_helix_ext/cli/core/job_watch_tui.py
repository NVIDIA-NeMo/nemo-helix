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

from nemo_helix_ext.cli.core.job_watch_graph import BOX_WIDTH, DagLayout, Direction, phase_label, state_style
from nemo_helix_ext.cli.core.job_watch_model import DagWatchModel, Pane, condition_label, node_parents, phase
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


def _transition_summary(description: str) -> str:
    first = description.split("; ", 1)[0]
    before, arrow, after = first.partition(" → ")
    return f"{phase_label(before)} → {phase_label(after)}" if arrow else first


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
        self.graph_layout: DagLayout | None = None
        self.graph_selected = ""
        self.graph_x = self.graph_y = 0
        self.graph_follow = True
        self.graph_width, self.graph_height = 80, 20
        self.body_fragments: dict[int, FormattedText] = {}
        self.pane_styles: dict[Pane, dict[int, str]] = {}
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
            style=Style.from_dict(
                {
                    "header": "bold ansicyan",
                    "selected": "bg:#284b63 #ffffff bold",
                    "footer": "ansiyellow",
                    "running": "ansicyan bold",
                    "completed": "ansigreen",
                    "failed": "ansired bold",
                    "waiting": "ansiyellow",
                    "ready": "ansiblue bold",
                    "reused": "ansimagenta",
                    "muted": "ansibrightblack",
                    "edge": "ansibrightblack",
                    "edge-focus": "ansicyan bold",
                }
            ),
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
            elif self.mode == "overview":
                self._graph_move("left" if delta < 0 else "right")
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
                if key in (Keys.PageUp, Keys.PageDown):
                    self._graph_pan(0, amount * self.graph_height)
                else:
                    self._graph_move("up" if amount < 0 else "down")
            elif self.model.pane == Pane.NODE:
                self.node_scroll = max(0, self.node_scroll + amount)
            else:
                self.model.move(amount * (self.page_nodes if key in (Keys.PageUp, Keys.PageDown) else 1))

        @bindings.add("enter")
        def enter(event: KeyPressEvent) -> None:
            if self.mode == "overview" and not self.searching:
                self.node_scroll = 0
                self.model.focus_node(self.graph_selected)
                self.mode = "status"
            elif self.searching:
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
            self.graph_selected = self.model.focus
            self.graph_follow = True
            self.query = ""
            self.overview_index = 0

        @bindings.add("h", filter=browsing)
        @bindings.add("j", filter=browsing)
        @bindings.add("k", filter=browsing)
        @bindings.add("l", filter=browsing)
        def pan_graph(event: KeyPressEvent) -> None:
            if self.mode == "overview":
                dx, dy = {"h": (-8, 0), "j": (0, 4), "k": (0, -4), "l": (8, 0)}[event.data]
                self._graph_pan(dx, dy)

        @bindings.add("home", filter=browsing)
        def center_graph(event: KeyPressEvent) -> None:
            self.graph_follow = True
            self.node_scroll = 0

        @bindings.add("up", filter=Condition(lambda: self.searching))
        @bindings.add("down", filter=Condition(lambda: self.searching))
        def search_move(event: KeyPressEvent) -> None:
            delta = -1 if event.key_sequence[0].key == Keys.Up else 1
            self.overview_index = max(0, min(len(self.model.search(self.query)) - 1, self.overview_index + delta))

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
            fragments = self.body_fragments.get(index - 4)
            if fragments is not None:
                result.extend(fragments)
                if index < len(lines) - 1:
                    result.append(("", "\n"))
            else:
                result.append((style, _fit(line, size.columns) + ("\n" if index < len(lines) - 1 else "")))
        return result

    def screen(self, size: Size) -> list[str]:
        model = self.model
        self.body_fragments = {}
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
            body = self._overview(width, body_height)
        else:
            body = self._status(width, body_height)
        body = (body + [""] * body_height)[:body_height]
        footer = [
            model.notice,
            ("Find node: " + self.query + "_ · Enter jump · Esc cancel")
            if self.searching
            else (
                "q quit · Tab status · ↑↓ scroll · ←→ pan · a all · e run · f follow"
                if self.mode == "logs"
                else "q quit · o status · arrows select · hjkl pan · Home center · Enter focus · / find"
                if self.mode == "overview"
                else "q quit · Tab logs · ←→ pane · ↑↓ select · Enter focus · / find · o DAG · a sort"
            ),
        ]
        return (header + [""] + body + footer)[:height]

    def _status(self, width: int, height: int) -> list[str]:
        if width < 100:
            title = f"Compact view · ←→ choose pane · {self.model.pane.name.lower()}"
            lines = self._pane(self.model.pane, width, max(1, height - 1))[: max(0, height - 1)]
            for index, line in enumerate(lines):
                self.body_fragments[index + 1] = FormattedText(
                    [(self.pane_styles[self.model.pane].get(index, ""), _fit(line, width))]
                )
            return [title] + lines
        column = max(1, (width - 6) // 3)
        panes = [self._pane(pane, column, height) for pane in Pane]
        result: list[str] = []
        for index in range(height):
            fragments = FormattedText()
            for pane in Pane:
                if pane != Pane.INPUTS:
                    fragments.append(("class:edge", " │ "))
                line = panes[int(pane)][index] if index < len(panes[int(pane)]) else ""
                fragments.append((self.pane_styles[pane].get(index, ""), _fit(line, column)))
            self.body_fragments[index] = fragments
            result.append("".join(fragment[1] for fragment in fragments))
        return result

    def _pane(self, pane: Pane, width: int, height: int) -> list[str]:
        model = self.model
        self.pane_styles[pane] = {0: "class:selected" if model.pane == pane else "class:header"}
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
                f"{model.icon(phase(node))} {phase_label(phase(node))}",
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
                    blockers += f" (+{len(node.blocked_by) - 2} more)"
                lines.extend(textwrap.wrap("Blocked by: " + blockers, max(1, width)))
            lines += ["", "Observed transitions (session)"]
            recent = [
                transition for transition in reversed(model.transitions) if transition.node in (model.focus, "workflow")
            ]
            for transition in recent:
                lines.append(transition.observed_at.astimezone().strftime("%H:%M:%S") + " " + transition.description)
            if not recent:
                lines.append("No observed changes yet")
            if node.depends_on:
                lines.append("Ready when (all):")
                for item in node.depends_on:
                    lines.extend(
                        textwrap.wrap("  " + _literal(condition_label(item)), max(1, width), subsequent_indent="  ")
                    )
            if node.trigger:
                lines.append("Triggered by (any):")
                for item in node.trigger:
                    lines.extend(
                        textwrap.wrap("  " + _literal(condition_label(item)), max(1, width), subsequent_indent="  ")
                    )
            self.node_scroll = min(self.node_scroll, max(0, len(lines) - height))
            styles = {
                0: "class:selected" if model.pane == pane else "class:header",
                1: "bold",
                2: state_style(phase(node)),
            }
            self.pane_styles[pane] = {
                index - self.node_scroll: style for index, style in styles.items() if index >= self.node_scroll
            }
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
            row = len(lines)
            selected = name == model.selected[pane] and model.pane == pane
            style = state_style(phase(node)) + (" reverse bold" if selected else "")
            self.pane_styles[pane].update(
                {row: style, row + 1: state_style(phase(node)), row + 2: "class:muted", row + 3: "class:muted"}
            )
            age = max(0, int((datetime.now(timezone.utc) - latest.observed_at).total_seconds())) if latest else 0
            ago = f"{age}s ago" if age < 60 else f"{age // 60}m ago" if age < 3600 else f"{age // 3600}h ago"
            lines += [
                marker + model.icon(phase(node)) + " " + name,
                "  " + phase_label(phase(node)) + (" · rerun pending" if node.has_pending_run else ""),
                "  " + (_transition_summary(latest.description) if latest else "No changes observed this session"),
                "  Observed " + ago if latest else "",
            ]
        if start + len(visible) < len(names):
            if len(lines) >= height:
                lines = lines[: height - 1]
            lines.append(f"↓ {len(names) - start - len(visible)} more")
        if not names:
            lines.append("No upstream nodes" if pane == Pane.INPUTS else "No downstream nodes")
        return lines[:height]

    def _layout(self) -> DagLayout:
        dependencies = {node.name: tuple(sorted(node_parents(node))) for node in self.model.nodes.values()}
        if self.graph_layout is None or self.graph_layout.dependencies != dependencies:
            self.graph_layout = DagLayout(dependencies)
        if self.graph_selected not in self.graph_layout.boxes:
            self.graph_selected = self.model.focus
        return self.graph_layout

    def _graph_move(self, direction: Direction) -> None:
        self.graph_selected = self._layout().neighbor(self.graph_selected, direction)
        self.graph_follow = True

    def _graph_pan(self, dx: int, dy: int) -> None:
        layout = self._layout()
        self.graph_x = max(0, min(max(0, layout.width - self.graph_width), self.graph_x + dx))
        self.graph_y = max(0, min(max(0, layout.height - self.graph_height), self.graph_y + dy))
        self.graph_follow = False

    def _overview(self, width: int, height: int) -> list[str]:
        if self.searching:
            return self._search_results(height)
        layout = self._layout()
        self.graph_width, self.graph_height = width, max(1, height - 2)
        box = layout.boxes.get(self.graph_selected)
        if self.graph_follow and box:
            self.graph_x = max(0, min(max(0, layout.width - width), box.x - max(0, (width - BOX_WIDTH) // 2)))
            self.graph_y = max(0, min(max(0, layout.height - self.graph_height), box.y - self.graph_height // 2))
        rows = layout.viewport(
            self.model.nodes, self.graph_selected, self.graph_x, self.graph_y, width, self.graph_height
        )
        lines = [
            f"DAG · {len(layout.boxes)} nodes · arrows follow selection · ╳ crossing (no connection)",
            f"Selected: {self.graph_selected} · viewport x={self.graph_x}/{max(0, layout.width - width)} y={self.graph_y}/{max(0, layout.height - self.graph_height)} · hjkl pan · Home center",
        ]
        for index, row in enumerate(rows):
            self.body_fragments[index + 2] = row
            lines.append("".join(fragment[1] for fragment in row))
        return lines[:height]

    def _search_results(self, height: int) -> list[str]:
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
                "   ← " + (", ".join(node_parents(node)) or "root"),
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
