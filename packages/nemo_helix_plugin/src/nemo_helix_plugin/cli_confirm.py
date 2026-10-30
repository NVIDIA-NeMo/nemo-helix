# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Confirm a destructive CLI action before it runs.

Pair :func:`confirm_destructive` with :data:`~nemo_helix_plugin.cli_options.YesOption`::

    @app.command("delete")
    @handle_errors
    def delete_widget(ctx: typer.Context, name: str, yes: YesOption = False) -> None:
        confirm_destructive(f"Delete widget '{name}'?", yes=yes)
        ...

On a terminal the user is asked to confirm. The prompt goes to stderr so it
never mixes into output a caller is parsing. Without a terminal there is no one to
ask, so the command fails with a usage error naming ``--yes`` rather than acting
on a destructive request nobody approved.
"""

from __future__ import annotations

import click
import typer
from nemo_helix_plugin.cli_input import is_stdin_available


def confirm_destructive(message: str, *, yes: bool) -> None:
    """Ask the user to confirm *message* unless ``--yes`` was passed.

    Raises:
        click.UsageError: stdin is not a terminal and ``--yes`` was not passed (exit 2).
        click.Abort: the user declined (exit 1).
    """
    if yes:
        return
    if is_stdin_available():
        raise click.UsageError(f"{message} Pass --yes to confirm when not running interactively.")
    typer.confirm(message, abort=True, err=True)
