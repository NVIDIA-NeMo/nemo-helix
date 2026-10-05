# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal renderer for usage leaderboard tables.

Rendering is intentionally separate from discovery, normalization, and
ranking so tests can assert on the presentation layer independently and
future output variants can reuse the same ranked entry model.
"""

from __future__ import annotations

from io import StringIO

from nemo_agents_plugin.leaderboard.types import AgentLeaderboard, AgentLeaderboardEntry
from nemo_helix_ext.cli.core.help_formatter import _get_terminal_width
from rich.console import Console
from rich.table import Table

COMPACT_WIDTH_THRESHOLD = 100

# Upper bound for the in-memory console height; large enough to never clip
# leaderboard rows while forcing Rich to honour the explicit width.
_RENDER_HEIGHT = 10_000


def render_entries(
    entries: tuple[AgentLeaderboardEntry, ...],
    *,
    compact: bool | None = None,
    width: int | None = None,
) -> str:
    """Render ranked leaderboard entries as a terminal table."""

    resolved_width = width or _get_terminal_width()
    use_compact = compact if compact is not None else resolved_width < COMPACT_WIDTH_THRESHOLD

    table = _build_table(entries, compact=use_compact)
    # Pin both dimensions so the in-memory render honours the requested width even
    # under a dumb terminal (``TERM=dumb``), where Rich otherwise clamps the
    # console to 80x25 and truncates columns. ``size`` returns the explicit
    # dimensions verbatim when both width and height are set. The height is a
    # generous upper bound; ``print`` does not vertically clip table rows.
    console = Console(
        file=StringIO(),
        width=resolved_width,
        height=_RENDER_HEIGHT,
        record=True,
        legacy_windows=False,
    )
    console.print(table)
    return console.file.getvalue()


def render_leaderboard(
    leaderboard: AgentLeaderboard,
    *,
    compact: bool | None = None,
    width: int | None = None,
) -> str:
    """Render a leaderboard, appending warning messages when present."""

    body = render_entries(leaderboard.entries, compact=compact, width=width).rstrip()
    if not leaderboard.warnings:
        return body

    warning_lines = ["", "Warnings:"]
    warning_lines.extend(
        f"- {warning.message}" if warning.source_path is None else f"- {warning.message} ({warning.source_path})"
        for warning in leaderboard.warnings
    )
    return body + "\n" + "\n".join(warning_lines) + "\n"


def _build_table(entries: tuple[AgentLeaderboardEntry, ...], *, compact: bool) -> Table:
    table = Table(title="Usage Leaderboard", header_style="bold cyan")

    if compact:
        table.add_column("#", justify="right")
        table.add_column("Task")
        table.add_column("Tokens", justify="right")
        table.add_column("Compute Units", justify="right")
    else:
        table.add_column("#", justify="right")
        table.add_column("Task")
        table.add_column("Runtime Image")
        table.add_column("Tokens", justify="right")
        table.add_column("Compute Units", justify="right")
        table.add_column("CU/Token", justify="right")
        table.add_column("Created")

    if not entries:
        table.add_row("—", "No entries", *([""] * (2 if compact else 5)))
        return table

    for index, entry in enumerate(entries, start=1):
        if compact:
            table.add_row(
                str(index),
                entry.task_name,
                _format_tokens(entry.token_count),
                _format_compute_units(entry.compute_units),
            )
        else:
            table.add_row(
                str(index),
                entry.task_name,
                entry.runtime_image or "—",
                _format_tokens(entry.token_count),
                _format_compute_units(entry.compute_units),
                _format_ratio(entry.compute_units_per_token),
                _format_created(entry.created_at),
            )

    return table


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}"


def _format_compute_units(value: float) -> str:
    if 0 < value < 0.005:
        return "<0.01"
    if -0.005 < value < 0:
        return ">-0.01"
    return f"{value:,.2f}"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "0.0000"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _format_created(value: object) -> str:
    if value is None:
        return "—"
    return str(value)
