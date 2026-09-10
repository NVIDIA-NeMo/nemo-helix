# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo guardrail`` plugin CLI.

Registered under the ``nemo.cli`` entry-point group; the platform CLI mounts
the returned Typer app as the top-level ``guardrail`` command group.
"""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_platform_plugin.cli import NemoCLI


class GuardrailCLI(NemoCLI):
    """Guardrail command group: ``check`` plus the nested ``configs`` group."""

    name: ClassVar[str] = "guardrail"
    description: ClassVar[str] = "Manage guardrails."

    def get_cli(self) -> typer.Typer:
        from nemo_guardrails_plugin.cli_commands.guardrail import app

        app.info.help = self.description
        return app
