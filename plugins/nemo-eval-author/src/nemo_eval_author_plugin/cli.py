# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo evaluator author`` -- contributed to the evaluator CLI through ``nemo.cli.evaluator``."""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_eval_author_plugin.jobs.first_eval import FirstEvalJob
from nemo_helix_plugin.cli import NemoCLI
from nemo_helix_plugin.commands import add_job_commands


class EvalAuthorCLI(NemoCLI):
    name: ClassVar[str] = "author"
    description: ClassVar[str] = "Author evaluations for a platform agent with NeMo Eval Author."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(name=self.name, help=self.description, no_args_is_help=True)

        @app.callback()
        def _root() -> None:
            """Force subcommand dispatch."""

        add_job_commands(app, {"eval-author.first-eval": FirstEvalJob}, cli=self)
        return app
