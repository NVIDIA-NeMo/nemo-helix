# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo intake`` and ``nemo experiments`` plugin CLIs.

Registered under the ``nemo.cli`` entry-point group; the platform CLI mounts
each returned Typer app as a top-level command group.
"""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_platform_plugin.cli import NemoCLI


class IntakeCLI(NemoCLI):
    """Intake command group: traces, spans, sessions, annotations, evaluator results, ingest."""

    name: ClassVar[str] = "intake"
    description: ClassVar[str] = "Intake operations."

    def get_cli(self) -> typer.Typer:
        from nmp.intake.cli_commands.intake import app

        app.info.help = self.description
        return app


class ExperimentsCLI(NemoCLI):
    """Experiments command group."""

    name: ClassVar[str] = "experiments"
    description: ClassVar[str] = "Manage experiments."

    def get_cli(self) -> typer.Typer:
        from nmp.intake.cli_commands.experiments import app

        app.info.help = self.description
        return app
