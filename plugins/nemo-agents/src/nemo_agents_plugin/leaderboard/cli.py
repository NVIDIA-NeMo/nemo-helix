# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents leaderboard`` — rank local usage report outputs.

One subcommand, ``nemo agents leaderboard show <paths...>``, registered
onto the parent ``agents`` Typer app under a ``leaderboard`` group.

The command is intentionally local-path driven for now: users point it at
one or more usage report files or directories, the CLI discovers matching
JSON artifacts, normalizes them into a canonical entry shape, applies the
default ranking rules, and renders a terminal table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Mapping

import typer
from nemo_agents_plugin.leaderboard.discovery import discover_report_paths
from nemo_agents_plugin.leaderboard.load import load_reports
from nemo_agents_plugin.leaderboard.normalize import normalize_reports
from nemo_agents_plugin.leaderboard.rank import rank_entries
from nemo_agents_plugin.leaderboard.render import render_entries
from nemo_agents_plugin.leaderboard.types import AgentLeaderboardEntry
from nemo_helix_ext.cli.core.help_formatter import create_typer_app


def register_leaderboard_commands(app: typer.Typer) -> None:
    """Register `leaderboard` subcommands onto the parent `agents` Typer app."""
    leaderboard_app = create_typer_app(
        name="leaderboard",
        no_args_is_help=True,
        help="""\
Commands for usage leaderboard workflows.

Examples:
# Show leaderboard help.
nemo agents leaderboard --help""",
    )

    @leaderboard_app.command("show")
    def leaderboard_show_command(
        paths: Annotated[
            list[str],
            typer.Argument(
                help="One or more usage report files or directories containing report files.",
            ),
        ],
        compact: Annotated[
            bool,
            typer.Option(
                "--compact",
                help="Render the compact leaderboard table.",
                rich_help_panel="Output Options",
            ),
        ] = False,
    ) -> None:
        """Show a ranked leaderboard from local usage report files.

        Examples:
          nemo agents leaderboard show ./result.json
          nemo agents leaderboard show ./reports/
          nemo agents leaderboard show ./reports/ ./baseline.json --compact
        """
        _show(paths, compact=compact)

    app.add_typer(leaderboard_app, rich_help_panel="Local commands")


# ---------------------------------------------------------------------------
# Internal pipeline: discover → load → normalize → rank → render
# ---------------------------------------------------------------------------


def _show(
    paths: list[str],
    *,
    compact: bool,
) -> None:
    """Resolve input paths, build ranked entries, and render the leaderboard."""
    try:
        ranked_entries = _build_ranked_entries(paths)
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(_render_leaderboard(ranked_entries, compact=compact))


def _build_ranked_entries(paths: list[str]) -> tuple[AgentLeaderboardEntry, ...]:
    """End-to-end pipeline: discover → load → normalize → rank."""
    report_paths = _resolve_report_paths(paths)
    raw_reports = _load_raw_reports(report_paths)
    entries = _normalize_entries(raw_reports, report_paths=report_paths)
    return _rank_entries(entries)


def _resolve_report_paths(paths: list[str]) -> tuple[Path, ...]:
    """Resolve CLI path arguments into a stable sequence of report files."""
    return discover_report_paths(paths)


def _load_raw_reports(report_paths: tuple[Path, ...]) -> tuple[Mapping[str, object], ...]:
    """Load raw leaderboard reports in stable path order."""
    return load_reports(report_paths)


def _normalize_entries(
    raw_reports: tuple[Mapping[str, object], ...],
    *,
    report_paths: tuple[Path, ...],
) -> tuple[AgentLeaderboardEntry, ...]:
    """Normalize raw reports into leaderboard entry objects."""
    return normalize_reports(raw_reports, source_paths=report_paths)


def _rank_entries(entries: tuple[AgentLeaderboardEntry, ...]) -> tuple[AgentLeaderboardEntry, ...]:
    """Apply the default leaderboard ordering."""
    return rank_entries(entries)


def _render_leaderboard(
    ranked_entries: tuple[AgentLeaderboardEntry, ...],
    *,
    compact: bool,
) -> str:
    """Render ranked entries for terminal output."""
    return render_entries(ranked_entries, compact=compact)
