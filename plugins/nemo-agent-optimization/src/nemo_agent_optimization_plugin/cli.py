# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents optimization-strategies`` — list installed agent-optimize strategy jobs."""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_agent_optimization_plugin.discovery import discover_agent_optimize_jobs
from nemo_platform_plugin.cli import NemoCLI


class OptimizationStrategiesCLI(NemoCLI):
    """Contributes ``nemo agents optimization-strategies`` via the ``nemo.cli.agents`` group."""

    name: ClassVar[str] = "optimization-strategies"
    description: ClassVar[str] = "List installed nemo agents optimize strategies."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(name=self.name, help=self.description, no_args_is_help=True)

        @app.callback()
        def _root() -> None:
            """Force subcommand dispatch even when only one verb is registered."""

        @app.command("list")
        def list_strategies() -> None:
            """Print the ``strategy`` name of every installed ``AgentOptimizeJob`` subclass."""
            strategies = discover_agent_optimize_jobs()
            if not strategies:
                typer.echo("No optimization strategies are installed.")
                raise typer.Exit(code=0)
            for name in sorted(strategies):
                typer.echo(name)

        return app
