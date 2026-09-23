# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyst CLI — ``nemo agents analyst ...`` subcommands.

Registered under ``nemo.cli.agents`` and mounted by ``AgentsCLI`` as
``nemo agents analyst <verb>``. Verb bodies live as module-level callbacks in
``nemo_insights_plugin.cli`` so the analyst implementation stays in one place.
Periodic-analysis and job surfaces stay on ``nemo insights``: they manage the
plugin's scheduled runs rather than driving the analyst itself.
"""

from typing import ClassVar

import typer
from nemo_helix_plugin.cli import NemoCLI
from nemo_insights_plugin.cli import analyze, doctor


class AnalystCLI(NemoCLI):
    """``nemo agents analyst ...`` subcommands."""

    name: ClassVar[str] = "analyst"
    description: ClassVar[str] = "Analyze agent telemetry and record what the agent gets wrong."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(help=self.description, no_args_is_help=True)

        @app.callback()
        def _root() -> None:
            """Force subcommand dispatch even when only one verb is registered."""

        app.command("run")(analyze)
        app.command("doctor")(doctor)
        return app
