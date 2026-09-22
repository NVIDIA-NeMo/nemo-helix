# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI router for customization — mounts contributor subgroups."""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_helix_plugin.cli import NemoCLI
from nemo_helix_plugin.customization_contributor import CustomizationContributorDiscoveryError
from nemo_helix_plugin.discovery import (
    CUSTOMIZATION_CONTRIBUTORS_GROUP,
    discover_customization_contributors,
)


class CustomizationCLIError(CustomizationContributorDiscoveryError):
    """Raised when the customization CLI cannot start."""


class CustomizationCLI(NemoCLI):
    """``nemo customization`` root command."""

    name: ClassVar[str] = "customization"
    description: ClassVar[str] = "Customization training backends (Automodel, …)."

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
            help=self.description,
            no_args_is_help=True,
        )

        for key in sorted(self._contributors.keys()):
            contributor = self._contributors[key]
            subgroup = contributor.get_cli()
            if subgroup is not None:
                app.add_typer(subgroup, name=key)

        return app
