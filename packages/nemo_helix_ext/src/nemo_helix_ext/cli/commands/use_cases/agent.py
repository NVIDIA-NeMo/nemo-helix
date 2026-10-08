# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated ``nemo agent`` group, kept as a hidden alias for ``nemo describe``."""

from __future__ import annotations

import typer

from nemo_helix_ext.cli.commands.use_cases.describe import render_cli_description, render_commands_table
from nemo_helix_ext.cli.core.help_formatter import create_typer_app

app = create_typer_app(
    name="agent",
    no_args_is_help=False,
    help="""\
Deprecated: use 'nemo describe' instead.

Examples:
# Describe the installed CLI, plugins, and skills.
nemo describe""",
)


def _warn_deprecated(replacement: str) -> None:
    typer.echo(f"Warning: 'nemo agent' is deprecated and will be removed; use '{replacement}' instead.", err=True)


@app.callback(invoke_without_command=True)
def agent_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


@app.command("context")
def context_command() -> None:
    """Deprecated: use 'nemo describe' instead."""
    _warn_deprecated("nemo describe")
    typer.echo(render_cli_description())


@app.command("commands")
def commands_command() -> None:
    """Deprecated: use 'nemo describe' instead."""
    _warn_deprecated("nemo describe")
    typer.echo("# NeMo CLI Commands\n")
    typer.echo(render_commands_table())
