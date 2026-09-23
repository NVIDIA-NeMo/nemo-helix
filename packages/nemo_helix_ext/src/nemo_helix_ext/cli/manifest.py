# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from typing import Literal

PanelName = Literal["Setup", "CLI functions", "Core plugins", "Functional plugins"]
EntryKind = Literal["group", "command"]
EntrySource = Literal["module", "plugin"]

logger = logging.getLogger(__name__)

PANEL_ORDER: tuple[PanelName, ...] = (
    "Setup",
    "CLI functions",
    "Core plugins",
    "Functional plugins",
)

PANEL_DESCRIPTIONS: dict[PanelName, str] = {
    "Setup": "Set up and run local platform components",
    "CLI functions": "Interactive, documentation, and agent-oriented workflows",
    "Core plugins": "Core platform resources",
    "Functional plugins": "Functional service and plugin commands",
}

TOP_LEVEL_COMMAND_ORDER: dict[PanelName, tuple[str, ...]] = {
    "Setup": ("setup", "auth", "services", "skills"),
    "CLI functions": ("chat", "docs", "wait", "agent", "plugins"),
    "Core plugins": ("files", "inference", "jobs", "models", "secrets", "workspaces"),
    "Functional plugins": (
        "agents",
        "data-designer",
        "guardrail",
        "auditor",
        "anonymizer",
        "evaluator",
        "customization",
        "insights",
        "safe-synthesizer",
    ),
}


@dataclass(frozen=True)
class TopLevelEntry:
    import_path: str
    help: str
    name: str
    panel: PanelName
    kind: EntryKind
    source: EntrySource = "module"
    hidden: bool = False


def top_level_entry_sort_key(entry: TopLevelEntry) -> tuple[int, str]:
    panel_commands = TOP_LEVEL_COMMAND_ORDER.get(entry.panel, ())
    try:
        command_index = panel_commands.index(entry.name)
    except ValueError:
        command_index = len(panel_commands)
    return command_index, entry.name


def functional_plugin_entry(
    name: str,
    import_path: str,
    *,
    source: EntrySource = "module",
    hidden: bool = False,
) -> TopLevelEntry:
    return TopLevelEntry(
        import_path=import_path,
        help=f"Plugin commands for {name}.",
        name=name,
        panel="Functional plugins",
        kind="group",
        source=source,
        hidden=hidden,
    )


def build_top_level_entries(
    entries: Iterable[TopLevelEntry],
    plugin_entry_points: Mapping[str, EntryPoint],
    *,
    include_hidden: bool,
) -> tuple[TopLevelEntry, ...]:
    entries_by_panel: dict[PanelName, list[TopLevelEntry]] = {panel: [] for panel in PANEL_ORDER}
    top_level_names: set[str] = set()
    for entry in entries:
        top_level_names.add(entry.name)
        if entry.hidden and not include_hidden:
            continue
        panel_entries = entries_by_panel.get(entry.panel)
        if panel_entries is None:
            logger.warning("Ignoring top-level entry %r with unexpected panel %r", entry.name, entry.panel)
            continue
        panel_entries.append(entry)

    for plugin_name, entry_point in plugin_entry_points.items():
        if plugin_name in top_level_names:
            logger.warning(
                "Ignoring plugin CLI entry point %r because it collides with a top-level command", plugin_name
            )
            continue
        entries_by_panel["Functional plugins"].append(
            functional_plugin_entry(plugin_name, entry_point.value, source="plugin")
        )

    return tuple(
        entry for panel in PANEL_ORDER for entry in sorted(entries_by_panel[panel], key=top_level_entry_sort_key)
    )
