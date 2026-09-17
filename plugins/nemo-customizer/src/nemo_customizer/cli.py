# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI router for customization — mounts contributor subgroups."""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.customization_contributor import CustomizationContributorDiscoveryError
from nemo_platform_plugin.discovery import (
    CUSTOMIZATION_CONTRIBUTORS_GROUP,
    discover_customization_contributors,
)

# The router is deliberately backend-neutral: it never names automodel, unsloth or
# rl. Backend-specific text comes from each contributor's get_cli_summary().
_OVERVIEW = """Train a model on your own data.

Choose a backend, write a job JSON for it, and submit it. The platform
creates the job and runs the training on a GPU execution profile. Each backend
trains a different way, and the schema of the job JSON depends on the backend
you choose."""

_NEXT_STEPS = """Run 'nemo customization <backend> --help' for the full description of a
backend, or 'nemo customization <backend> explain' to print its job JSON
schema."""


class CustomizationCLIError(CustomizationContributorDiscoveryError):
    """Raised when the customization CLI cannot start."""


class CustomizationCLI(NemoCLI):
    """``nemo customization`` root command."""

    name: ClassVar[str] = "customization"
    description: ClassVar[str] = "Train a model on your own data with an installed training backend."

    def __init__(self) -> None:
        self._contributors = discover_customization_contributors()
        if not self._contributors:
            raise CustomizationCLIError(
                "Customization CLI is enabled but no contributors were discovered. "
                "Install a backend plugin (e.g. nemo-automodel) and ensure "
                f"'{CUSTOMIZATION_CONTRIBUTORS_GROUP}' entry points are registered.",
            )

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(
            name=self.name,
            help=self._compose_help(),
            no_args_is_help=True,
        )

        for key in sorted(self._contributors.keys()):
            contributor = self._contributors[key]
            subgroup = contributor.get_cli()
            if subgroup is not None:
                app.add_typer(subgroup, name=key)

        return app

    def _compose_help(self) -> str:
        """Overview, one summary per discovered backend in name order, then next steps."""
        blocks = [_OVERVIEW, "Installed backends:"]

        for key in sorted(self._contributors.keys()):
            # Read through getattr so that a contributor without a summary is
            # listed by name instead of breaking --help for the others.
            get_summary = getattr(self._contributors[key], "get_cli_summary", None)
            summary = get_summary() if get_summary is not None else None
            blocks.append(summary.render(key) if summary is not None else key)

        blocks.append(_NEXT_STEPS)
        return "\n\n".join(blocks)
