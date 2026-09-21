# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer plugin CLI."""

from __future__ import annotations

from typing import ClassVar, Literal

import typer
from nemo_data_designer_plugin.cli.inputs import apply_create_cli_overrides, apply_preview_cli_overrides
from nemo_data_designer_plugin.cli.personas import make_fileset_command
from nemo_data_designer_plugin.cli.renderers import CreateRenderer, PreviewRenderer
from nemo_data_designer_plugin.functions.preview import PreviewFunction
from nemo_data_designer_plugin.jobs.create import CreateJob
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.cli_renderer import CLIRenderer
from nemo_platform_plugin.function import NemoFunction
from nemo_platform_plugin.job import NemoJob


class DataDesignerCLI(NemoCLI):
    name: ClassVar[str] = "data-designer"
    description: ClassVar[str] = "Data Designer: generate synthetic datasets"

    def get_cli(self) -> typer.Typer:
        from data_designer.cli.main import agent_app
        from data_designer.cli.runtime import ensure_cli_default_model_settings
        from nemo_data_designer_plugin.cli.check_models import check_models_command
        from nemo_data_designer_plugin.cli.retrieval import retrieval_app
        from nemo_data_designer_plugin.cli.validate import validate_command

        ensure_cli_default_model_settings()

        app = typer.Typer(name=self.name, help=self.description, no_args_is_help=True)
        app.command("validate")(validate_command)
        app.command("check-models")(check_models_command)

        personas_app = typer.Typer(
            name="personas",
            help="Manage Nemotron Personas datasets",
            no_args_is_help=True,
        )
        personas_app.command("make-fileset")(make_fileset_command)

        app.add_typer(personas_app, name="personas")
        app.add_typer(retrieval_app, name="retrieval")
        app.add_typer(agent_app, name="agent")

        return app

    def update_function_cli(self, fn_cls: type[NemoFunction], group: typer.Typer) -> None:
        if fn_cls is PreviewFunction:
            apply_preview_cli_overrides(group)

    def update_job_cli(self, job_cls: type[NemoJob], group: typer.Typer) -> None:
        if job_cls is CreateJob:
            apply_create_cli_overrides(group)

    def get_function_renderer(
        self,
        fn_cls: type[NemoFunction],
        *,
        verb: Literal["run", "submit"],
    ) -> type[CLIRenderer] | None:
        if fn_cls is PreviewFunction:
            return PreviewRenderer
        return None

    def get_job_renderer(
        self,
        job_cls: type[NemoJob],
        *,
        verb: Literal["run", "submit"],
    ) -> type[CLIRenderer] | None:
        if job_cls is CreateJob:
            return CreateRenderer
        return None
