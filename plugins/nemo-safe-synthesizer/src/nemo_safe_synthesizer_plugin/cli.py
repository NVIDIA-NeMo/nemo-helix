# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe Synthesizer plugin CLI."""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_helix_plugin.cli import NemoCLI


class SafeSynthesizerCLI(NemoCLI):
    """CLI surface for the Safe Synthesizer plugin."""

    name: ClassVar[str] = "safe-synthesizer"
    description: ClassVar[str] = "Safe Synthesizer commands."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(
            name=self.name,
            help=self.description,
            no_args_is_help=True,
        )

        @app.callback()
        def _root() -> None:
            pass

        return app
