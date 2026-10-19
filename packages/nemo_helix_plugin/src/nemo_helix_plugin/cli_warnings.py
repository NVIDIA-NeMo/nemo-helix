# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect warnings during a CLI command and print them to stderr when it finishes.

Helpers such as :func:`~nemo_helix_plugin.cli_output.format_output` and the
pagination helpers call :func:`add_warning` (for example "Use --no-truncate to
see full values."). Decorate the command with :func:`collect_warnings` so those
warnings are printed after the command's output::

    @app.command("list")
    @collect_warnings
    def list_widgets(ctx: typer.Context) -> None:
        ...
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import ParamSpec, TypeVar

from rich.console import Console

_warnings_context: ContextVar[list[str | None]] = ContextVar("warnings")

_P = ParamSpec("_P")
_R = TypeVar("_R")


def print_warnings(warnings: list[str | None] | None = None) -> None:
    """
    Print warnings as a bullet list to stderr.

    Args:
        warnings: List of warning messages to display (None values are filtered out)
    """
    if not warnings:
        return

    # Filter out None values
    warnings = [w for w in warnings if w]
    if not warnings:
        return

    error_console = Console(stderr=True)
    error_console.print()
    error_console.print("[bold yellow]Warnings:[/]")
    for warning in warnings:
        error_console.print(f"  • {warning}", style="yellow")


def collect_warnings(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Decorator that collects warnings and prints them at the end.

    Usage:
        @collect_warnings
        def my_command():
            add_warning("some warning")
            # ... warnings are automatically printed when the function returns
    """

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        warnings: list[str | None] = []
        token = _warnings_context.set(warnings)
        try:
            return func(*args, **kwargs)
        finally:
            _warnings_context.reset(token)
            print_warnings(warnings)

    return wrapper


def add_warning(warning: str | list[str | None] | None) -> None:
    """
    Add one or more warnings to the current warnings collection.

    Must be called within a function decorated with `@collect_warnings`.
    If called outside such a function, the warning(s) are silently ignored.

    Args:
        warning: A single warning message, a list of warning messages, or None.
                 None values are allowed and filtered later when printing.
    """
    try:
        warnings = _warnings_context.get()
        if isinstance(warning, list):
            warnings.extend(warning)
        else:
            warnings.append(warning)
    except LookupError:
        # Not inside a collect_warnings context, ignore
        pass
