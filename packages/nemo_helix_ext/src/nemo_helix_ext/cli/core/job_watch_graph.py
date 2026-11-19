# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable DAG geometry and rendering restricted to the visible viewport."""

from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Literal

from nemo_helix_plugin.jobs.workflows import WorkflowNodeView
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.utils import get_cwidth

from nemo_helix_ext.cli.core.job_watch_model import phase

BOX_WIDTH = 24
ROW_HEIGHT = 5
Direction = Literal["left", "right", "up", "down"]
Vertex = tuple[str, str]


_GLYPHS = {
    frozenset(key): glyph
    for key, glyph in [
        (("left", "right"), "─"),
        (("up", "down"), "│"),
        (("left", "down"), "┐"),
        (("right", "down"), "┌"),
        (("left", "up"), "┘"),
        (("right", "up"), "└"),
        (("left", "right", "down"), "┬"),
        (("left", "right", "up"), "┴"),
        (("up", "down", "right"), "├"),
        (("up", "down", "left"), "┤"),
        (("left", "right", "up", "down"), "┼"),
    ]
}


def state_style(state: str) -> str:
    return "class:" + {
        "running": "running",
        "completed": "completed",
        "reused": "reused",
        "error": "failed",
        "cancelled": "failed",
        "waiting_dependencies": "waiting",
        "waiting_capacity": "waiting",
        "held": "waiting",
        "ready": "ready",
        "idle": "muted",
    }.get(state, "muted")


def phase_label(state: str) -> str:
    return {
        "waiting_dependencies": "Waiting on dependencies",
        "waiting_capacity": "Waiting for capacity",
        "held": "Held",
        "error": "Failed",
    }.get(state, state.replace("_", " ").capitalize())


@dataclass(frozen=True)
class Box:
    name: str
    x: int
    y: int


@dataclass(frozen=True)
class Segment:
    source: str
    target: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class _Cell:
    glyph: str = " "
    style: str = ""
    directions: set[str] = field(default_factory=set)
    edges: set[tuple[str, str]] = field(default_factory=set)


class DagLayout:
    def __init__(self, dependencies: dict[str, tuple[str, ...]]) -> None:
        self.dependencies = dependencies
        self.boxes: dict[str, Box] = {}
        self.segments: list[Segment] = []
        self.width = self.height = 1
        self._build()

    def _build(self) -> None:
        followers: dict[str, list[str]] = defaultdict(list)
        remaining = {name: len(parents) for name, parents in self.dependencies.items()}
        ranks: dict[str, int] = {}
        ready: list[str] = []
        for name, parents in self.dependencies.items():
            if not parents:
                heappush(ready, name)
            for parent in parents:
                if parent not in self.dependencies:
                    raise ValueError(f"Unknown DAG dependency: {parent}")
                followers[parent].append(name)
        while ready:
            name = heappop(ready)
            ranks[name] = max((ranks[parent] + 1 for parent in self.dependencies[name]), default=0)
            for child in followers[name]:
                remaining[child] -= 1
                if not remaining[child]:
                    heappush(ready, child)
        if len(ranks) != len(self.dependencies):
            raise ValueError("Cannot draw a cyclic DAG")
        layers: dict[int, list[Vertex]] = defaultdict(list)
        routes: list[tuple[str, str, list[tuple[int, Vertex]]]] = []
        for name in sorted(ranks):
            layers[ranks[name]].append((name, ""))
        for target in sorted(self.dependencies):
            for source in sorted(self.dependencies[target]):
                path = [(ranks[source], (source, ""))]
                for rank in range(ranks[source] + 1, ranks[target]):
                    vertex = (source, target)
                    layers[rank].append(vertex)
                    path.append((rank, vertex))
                path.append((ranks[target], (target, "")))
                routes.append((source, target, path))
        gap_counts: dict[int, int] = defaultdict(int)
        for _, _, path in routes:
            for rank, _ in path[:-1]:
                gap_counts[rank] += 1
        positions: dict[tuple[int, Vertex], tuple[int, int]] = {}
        column = 1
        for rank in sorted(layers):
            for index, vertex in enumerate(sorted(layers[rank])):
                y = index * ROW_HEIGHT + 1
                positions[(rank, vertex)] = (column, y + 1)
                if not vertex[1]:
                    self.boxes[vertex[0]] = Box(vertex[0], column, y)
            self.height = max(self.height, len(layers[rank]) * ROW_HEIGHT)
            self.width = column + BOX_WIDTH + 1
            column += BOX_WIDTH + max(8, gap_counts[rank] + 3)
        lanes: dict[int, int] = defaultdict(int)
        for source, target, path in routes:
            for (rank, vertex), (next_rank, next_vertex) in zip(path, path[1:]):
                x, y = positions[(rank, vertex)]
                next_x, next_y = positions[(next_rank, next_vertex)]
                start = x + BOX_WIDTH if not vertex[1] else x
                end = next_x - 1 if not next_vertex[1] else next_x
                lane = x + BOX_WIDTH + 1 + lanes[rank]
                lanes[rank] += 1
                self.segments.extend(
                    [
                        Segment(source, target, start, y, lane, y),
                        Segment(source, target, lane, y, lane, next_y),
                        Segment(source, target, lane, next_y, end, next_y),
                    ]
                )

    def neighbor(self, name: str, direction: Direction) -> str:
        current = self.boxes.get(name)
        if current is None:
            return next(iter(self.boxes), "")
        candidates: list[tuple[int, int, str]] = []
        for other in self.boxes.values():
            dx, dy = other.x - current.x, other.y - current.y
            if direction == "left" and dx < 0 or direction == "right" and dx > 0:
                candidates.append((abs(dx), abs(dy), other.name))
            elif direction == "up" and dy < 0 or direction == "down" and dy > 0:
                candidates.append((abs(dy), abs(dx), other.name))
        return min(candidates)[2] if candidates else name

    def viewport(
        self, nodes: dict[str, WorkflowNodeView], selected: str, x: int, y: int, width: int, height: int
    ) -> list[FormattedText]:
        grid = [[_Cell() for _ in range(width)] for _ in range(height)]
        for segment in self.segments:
            horizontal = segment.y1 == segment.y2
            low = max(min(segment.x1, segment.x2), x) if horizontal else max(min(segment.y1, segment.y2), y)
            high = (
                min(max(segment.x1, segment.x2), x + width - 1)
                if horizontal
                else min(max(segment.y1, segment.y2), y + height - 1)
            )
            fixed = segment.y1 if horizontal else segment.x1
            if horizontal and not y <= fixed < y + height or not horizontal and not x <= fixed < x + width:
                continue
            for position in range(low, high + 1):
                cx, cy = (position - x, fixed - y) if horizontal else (fixed - x, position - y)
                cell = grid[cy][cx]
                first, last = (
                    (min(segment.x1, segment.x2), max(segment.x1, segment.x2))
                    if horizontal
                    else (min(segment.y1, segment.y2), max(segment.y1, segment.y2))
                )
                if position > first:
                    cell.directions.add("left" if horizontal else "up")
                if position < last:
                    cell.directions.add("right" if horizontal else "down")
                cell.edges.add((segment.source, segment.target))
                cell.style = (
                    "class:edge-focus" if selected in (segment.source, segment.target) else cell.style or "class:edge"
                )
        for row in grid:
            for cell in row:
                if cell.directions:
                    crossing = (
                        len(cell.directions) > 2
                        and len({edge[0] for edge in cell.edges}) > 1
                        and len({edge[1] for edge in cell.edges}) > 1
                    )
                    cell.glyph = (
                        "╳"
                        if crossing
                        else _GLYPHS.get(
                            frozenset(cell.directions),
                            "─" if "left" in cell.directions or "right" in cell.directions else "│",
                        )
                    )
        for box in self.boxes.values():
            if box.x + BOX_WIDTH < x or box.x >= x + width or box.y + 3 < y or box.y >= y + height:
                continue
            node = nodes[box.name]
            style = state_style(phase(node)) + (" reverse bold" if box.name == selected else "")
            name = _short(box.name, BOX_WIDTH - 4)
            label = _short(phase_label(phase(node)), BOX_WIDTH - 4)
            rows = [
                "┌" + "─" * (BOX_WIDTH - 2) + "┐",
                "│ " + _pad(name, BOX_WIDTH - 4) + " │",
                "│ " + _pad(label, BOX_WIDTH - 4) + " │",
                "└" + "─" * (BOX_WIDTH - 2) + "┘",
            ]
            for dy, text in enumerate(rows):
                column = box.x
                for char in text:
                    cell_width = get_cwidth(char)
                    if cell_width == 0:
                        if x < column <= x + width and y <= box.y + dy < y + height:
                            previous = column - x - 1
                            while previous > 0 and not grid[box.y + dy - y][previous].glyph:
                                previous -= 1
                            grid[box.y + dy - y][previous].glyph += char
                        continue
                    if x <= column < x + width and y <= box.y + dy < y + height:
                        cell = grid[box.y + dy - y][column - x]
                        cell.glyph, cell.style = (" " if cell_width == 2 and column + 1 >= x + width else char), style
                        if cell_width == 2 and column + 1 < x + width:
                            grid[box.y + dy - y][column + 1 - x].glyph = ""
                            grid[box.y + dy - y][column + 1 - x].style = style
                    column += max(0, cell_width)
        for target, parents in self.dependencies.items():
            box = self.boxes[target]
            if parents and x <= box.x - 1 < x + width and y <= box.y + 1 < y + height:
                cell = grid[box.y + 1 - y][box.x - 1 - x]
                cell.glyph = "▶"
        return [FormattedText([(cell.style, cell.glyph) for cell in row]) for row in grid]


def _short(value: str, width: int) -> str:
    result = ""
    for char in value:
        if not char.isprintable():
            continue
        if get_cwidth(result + char) > width:
            return result[:-1] + "…"
        result += char
    return result


def _pad(value: str, width: int) -> str:
    return value + " " * max(0, width - get_cwidth(value))
