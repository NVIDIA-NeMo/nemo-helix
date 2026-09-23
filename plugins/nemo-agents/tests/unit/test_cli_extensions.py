# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint
from unittest.mock import patch

import typer
from nemo_agents_plugin.cli import AgentsCLI
from nemo_helix_plugin.cli import NemoCLI
from nemo_helix_plugin.discovery import AGENT_CLI_GROUP
from typer.testing import CliRunner


class _AnalystCLI(NemoCLI):
    name = "analyst"

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(help="Analyst agent commands.")

        @app.callback()
        def _root() -> None:
            """Force subcommand dispatch."""

        @app.command()
        def run() -> None:
            typer.echo("analysis complete")

        return app


class _BrokenCLI(NemoCLI):
    name = "broken"

    def get_cli(self) -> typer.Typer:
        raise RuntimeError("broken extension")


def test_broken_plugin_does_not_hide_other_agent_clis(caplog) -> None:
    # Agent CLI extensions are discovered lazily (metadata-only) and only
    # imported when their specific subcommand is resolved, so the entry
    # points here must be real ``module:attr`` paths — not classes — and
    # resolvable from this test module (pytest's default import mode puts
    # this file on ``sys.path`` as a top-level module).
    entry_points = {
        "broken": EntryPoint(name="broken", value="test_cli_extensions:_BrokenCLI", group=AGENT_CLI_GROUP),
        "analyst": EntryPoint(name="analyst", value="test_cli_extensions:_AnalystCLI", group=AGENT_CLI_GROUP),
    }
    with patch("nemo_agents_plugin.cli.discover_entry_points", return_value=entry_points):
        app = AgentsCLI().get_cli()

        analyst_result = CliRunner().invoke(app, ["analyst", "run"])
        assert analyst_result.exit_code == 0, analyst_result.stderr
        assert analyst_result.stdout.strip() == "analysis complete"

        with caplog.at_level(logging.WARNING, logger="nemo_agents_plugin.cli"):
            broken_result = CliRunner().invoke(app, ["broken"])
        assert broken_result.exit_code != 0
        assert "Failed to load agent CLI extension 'broken'; skipping" in caplog.text
