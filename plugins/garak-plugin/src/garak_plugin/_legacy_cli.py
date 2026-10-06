# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated ``nemo auditor`` command group; removed one release after the plugin rename."""

from __future__ import annotations

from typing import ClassVar

import typer
from garak_plugin.cli import GarakPluginCLI
from garak_plugin.jobs.scan import ScanJob
from nemo_helix_plugin.cli import NemoCLI
from nemo_helix_plugin.commands import add_job_commands

LEGACY_JOB_VERB = "audit"


class AuditorAliasCLI(NemoCLI):
    """``nemo auditor ...`` forwards to ``nemo garak-plugin ...`` and warns on stderr."""

    name: ClassVar[str] = "auditor"
    description: ClassVar[str] = "Deprecated alias for `nemo garak-plugin`."

    def get_cli(self) -> typer.Typer:
        target = GarakPluginCLI()
        app = target.get_cli()
        # The CLI loader injects job commands only for the group's own name, so the
        # alias mounts the scan job itself and exposes it under its old verb.
        add_job_commands(app, {f"{target.name}.{ScanJob.name}": ScanJob}, cli=target)
        for group in app.registered_groups:
            if group.name == ScanJob.name:
                group.name = LEGACY_JOB_VERB

        @app.callback()
        def _warn_deprecated() -> None:
            typer.echo(
                f"Warning: `nemo {self.name}` is deprecated and will be removed; use `nemo {target.name}` "
                f"(and `{target.name} {ScanJob.name}` instead of `{self.name} {LEGACY_JOB_VERB}`).",
                err=True,
            )

        return app
