# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a command path in the live CLI tree and describe the command it names.

Resolution goes through the same ``get_command`` calls that running the command
would, so built-in groups, plugin ``nemo.cli`` groups (including ones that
replace a built-in), and generated job/function commands are described exactly
as they run. Only the named subtree is loaded.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import click
from rich.errors import MarkupError
from rich.text import Text

# Bump when a key is removed or changes meaning; adding keys is compatible.
SCHEMA_VERSION = "v1"

# Generated body-parameter help ends with this suffix when the value is required
# but may also come from an input file, so click itself treats it as optional.
_REQUIRED_SUFFIX = " (required)"

ROOT_COMMAND_NAME = "nemo"


@dataclass(frozen=True)
class ResolvedCommand:
    """A command found by walking a path from the root group."""

    command: click.Command
    context: click.Context
    ignored_args: list[str]


def resolve_command_path(root: click.Command, tokens: Sequence[str]) -> ResolvedCommand:
    """Follow ``tokens`` through subcommands, starting at ``root``.

    Resolution stops at the first token that is an option or that follows a
    leaf command. The remaining tokens are returned as ``ignored_args``, so a
    full command line resolves to the command it would run.

    Raises:
        click.UsageError: A token names no subcommand of the current group.
    """
    context = click.Context(root, info_name=ROOT_COMMAND_NAME)
    command = root
    for index, token in enumerate(tokens):
        if not isinstance(command, click.Group) or token.startswith("-"):
            return ResolvedCommand(command, context, list(tokens[index:]))
        subcommand = command.get_command(context, token)
        if subcommand is None:
            hint = " ".join(["nemo", "describe", *tokens[:index]])
            raise click.UsageError(
                f"No such command '{token}' in '{_command_name(context)}'. Run '{hint}' to list its subcommands."
            )
        context = click.Context(subcommand, parent=context, info_name=token)
        command = subcommand
    return ResolvedCommand(command, context, [])


def describe_command(
    resolved: ResolvedCommand,
    *,
    subcommand_summaries: Callable[[], list[tuple[str, str]]] | None = None,
) -> dict[str, Any]:
    """Describe a resolved command as JSON-serializable data.

    Args:
        resolved: The command and its context from :func:`resolve_command_path`.
        subcommand_summaries: ``(name, short help)`` pairs to report instead of
            loading the group's children. The root group passes its manifest
            here so describing ``nemo`` does not import every plugin.
    """
    command = resolved.command
    context = resolved.context
    if isinstance(command, click.Group):
        subcommands = subcommand_summaries() if subcommand_summaries else _subcommand_summaries(command, context)
    else:
        subcommands = []

    return {
        "schema_version": SCHEMA_VERSION,
        "command": _command_name(context),
        "kind": "group" if isinstance(command, click.Group) else "command",
        "help": _plain_text(inspect.cleandoc(command.help or "")),
        # Click's command path matches `--help`: it includes the arguments a
        # parent group takes before the subcommand name.
        "usage": " ".join([context.command_path, *command.collect_usage_pieces(context)]),
        "deprecated": bool(command.deprecated),
        "hidden": command.hidden,
        "arguments": [_describe_argument(param) for param in command.params if isinstance(param, click.Argument)],
        "options": [
            _describe_option(param)
            for param in command.params
            if isinstance(param, click.Option) and not param.hidden and param.name != "help"
        ],
        "subcommands": [{"name": name, "help": help_text} for name, help_text in subcommands],
        "ignored_args": resolved.ignored_args,
    }


def _command_name(context: click.Context) -> str:
    """Return the command names from the root to ``context``, without arguments."""
    names: list[str] = []
    current: click.Context | None = context
    while current is not None:
        names.append(current.info_name or "")
        current = current.parent
    return " ".join(reversed(names))


def _subcommand_summaries(group: click.Group, context: click.Context) -> list[tuple[str, str]]:
    summaries: list[tuple[str, str]] = []
    for name in group.list_commands(context):
        subcommand = group.get_command(context, name)
        if subcommand is None or subcommand.hidden:
            continue
        summaries.append((name, _plain_text(subcommand.get_short_help_str(limit=200))))
    return summaries


def _describe_argument(param: click.Argument) -> dict[str, Any]:
    # Typer stores argument help on its click.Argument subclass.
    help_text, required = _help_and_required(param, getattr(param, "help", None))
    return {
        "name": param.human_readable_name,
        **_describe_value(param, required=required),
        "help": help_text,
    }


def _describe_option(param: click.Option) -> dict[str, Any]:
    help_text, required = _help_and_required(param, param.help)
    return {
        "name": param.name,
        "flags": [*param.opts, *param.secondary_opts],
        **_describe_value(param, required=required),
        "is_flag": param.is_flag,
        "envvar": param.envvar,
        "help": help_text,
    }


def _help_and_required(param: click.Parameter, help_text: str | None) -> tuple[str, bool]:
    text = _plain_text(help_text or "")
    if text.endswith(_REQUIRED_SUFFIX):
        return text.removesuffix(_REQUIRED_SUFFIX), True
    return text, param.required


def _plain_text(text: str) -> str:
    """Drop Rich markup that the terminal help renderer would style."""
    try:
        return Text.from_markup(text).plain
    except MarkupError:
        return text


def _describe_value(param: click.Parameter, *, required: bool) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": param.type.name,
        "required": required,
        "multiple": param.multiple,
        "nargs": param.nargs,
        "default": _json_value(param.default),
    }
    choices = getattr(param.type, "choices", None)
    if choices:
        data["choices"] = [_json_value(choice) for choice in choices]
    return data


def _json_value(value: object) -> Any:
    """Convert a parameter default or choice to plain JSON data.

    Callable defaults are computed at run time, so they are reported as unset
    rather than evaluated.
    """
    if value is None or callable(value) or value is getattr(click.core, "UNSET", None):
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return str(value)
