# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical CLI option definitions shared by every plugin command surface.

This module exists next to :mod:`nemo_helix_plugin.cli_state` on purpose:
the option and the resolver that consumes it belong together. Commands import
:data:`WorkspaceOption` and :func:`~nemo_helix_plugin.cli_state.resolve_cli_workspace`
from adjacent modules, and neither is usable wrongly on its own.

That pairing is the point. The bug this replaced was hand-written per command::

    workspace: str = typer.Option("default", "--workspace", "-w")

A concrete Typer default makes an omitted flag indistinguishable from an
explicit ``--workspace default``, so the command can never consult the active
CLI context and silently acts on the wrong workspace. An alias that *cannot*
be written with a literal default removes that failure mode structurally
rather than by convention.

Two shapes are offered because the CLI has two construction styles:

- :data:`WorkspaceOption` — an ``Annotated`` alias for hand-written commands,
  used as ``workspace: WorkspaceOption = None``.
- :func:`workspace_option` — the underlying ``typer.Option``, for the
  generated ``NemoJob`` / ``NemoFunction`` verbs in
  :mod:`nemo_helix_plugin.commands`, which assemble their signatures
  programmatically via ``kw()`` and cannot consume an ``Annotated`` alias.

Both are built from the same call, so the flag names and help text cannot
drift apart.
"""

from typing import Annotated, Any, Literal, Optional

import typer

# Kept as its own constant so per-command help can lead with a
# command-specific sentence and still document one resolution order.
#
# Note $NHX_WORKSPACE outranks the context's configured workspace: the SDK
# ``Config`` treats the env var as an override of the configured value, so it
# wins whenever a context loads at all. Getting this backwards has been a
# recurring bug in this area -- if you edit it, check
# ``nemo_helix_ext.config.config.Config.resolve`` first.
WORKSPACE_RESOLUTION = (
    "Resolution order: "
    "(1) this --workspace flag; "
    "(2) NHX_WORKSPACE; "
    "(3) the active CLI context's workspace (`nemo config use-context`); "
    "(4) 'default'."
)

WORKSPACE_HELP = f"Target workspace. {WORKSPACE_RESOLUTION}"


def workspace_help(lead: str) -> str:
    """Compose per-command workspace help that keeps one resolution order.

    *lead* is the command-specific sentence — e.g. "Workspace path segment
    used in the submit URL." — and the shared resolution order is appended.
    """
    return f"{lead} {WORKSPACE_RESOLUTION}"


# Flag names live here so the two construction shapes below cannot disagree.
WORKSPACE_FLAGS = ("--workspace", "-w")


def workspace_option(*, help: str = WORKSPACE_HELP, rich_help_panel: str | None = None) -> Any:
    """Build the canonical ``--workspace`` option, default included.

    For signatures assembled programmatically (the generated verbs' ``kw()``
    calls), where Typer takes the default as the first positional argument.
    The default is always ``None``: callers resolve with
    :func:`~nemo_helix_plugin.cli_state.resolve_cli_workspace` on the first
    line of the command body, and there is deliberately no way to pass a
    literal default through this factory.
    """
    return typer.Option(None, *WORKSPACE_FLAGS, help=help, rich_help_panel=rich_help_panel)


# Reusable ``--workspace`` for hand-written commands: ``workspace: WorkspaceOption = None``.
#
# The ``Annotated`` form must *omit* the default -- Typer reads it from the
# ``= None`` in the signature, and a value passed here would be parsed as
# another flag name. That asymmetry with :func:`workspace_option` is why the
# flag names are a shared constant rather than one shape calling the other.
WorkspaceOption = Annotated[
    Optional[str],
    typer.Option(*WORKSPACE_FLAGS, help=WORKSPACE_HELP),
]


# Output formats. ``OutputFormat`` is what a user can store as a context
# preference (``nemo config set``); commands add ``code`` on top, which prints
# the equivalent typed-client Python instead of calling the API.
OutputFormat = Literal["table", "json", "yaml", "markdown", "csv", "raw"]
ListOutputFormat = Literal[OutputFormat, "code"]
EntityOutputFormat = Literal["json", "yaml", "raw", "code"]
ConfigOutputFormat = Literal["json", "yaml", "raw"]
TimestampFormat = Literal["relative", "iso8601"]

# Every command's format flag answers to the same names; per-command aliases
# are how ``nemo agents`` drifted to ``--format``/``-o``.
OUTPUT_FORMAT_FLAGS = ("--output-format", "--output", "-f")


def _output_format_option(help: str) -> Any:
    return typer.Option(
        *OUTPUT_FORMAT_FLAGS,
        show_choices=True,
        help=help,
        rich_help_panel="Output Options",
    )


# Resolve these with :func:`~nemo_helix_plugin.cli_state.resolve_output_format`
# so an omitted flag falls through to the global ``--output-format`` and
# context preference instead of a per-command default.
ListOutputFormatOption = Annotated[
    Optional[ListOutputFormat],
    _output_format_option("Output format for the list of results."),
]

EntityOutputFormatOption = Annotated[
    Optional[EntityOutputFormat],
    _output_format_option("Output format for an entity."),
]

ConfigOutputFormatOption = Annotated[
    Optional[ConfigOutputFormat],
    _output_format_option("Output format for config."),
]

StreamOutputOption = Annotated[
    bool,
    typer.Option(
        "--stream",
        help="Emit newline-delimited JSON, one record per line. Requires JSON or raw output.",
        rich_help_panel="Output Options",
    ),
]

NoTruncateOption = Annotated[
    Optional[bool],
    typer.Option(
        "--no-truncate",
        help="Don't truncate long values in table/markdown/csv output.",
        rich_help_panel="Output Options",
    ),
]

TimestampFormatOption = Annotated[
    Optional[TimestampFormat],
    typer.Option(
        help="Timestamp format for table/markdown/csv output. Overrides global.",
        show_choices=True,
        rich_help_panel="Output Options",
    ),
]

OutputColumnsOption = Annotated[
    Optional[str],
    typer.Option(
        "--output-columns",
        "-c",
        help="Columns to display: 'default', 'all', or comma-separated names. Only affects table/csv/markdown formats.",
        rich_help_panel="Output Options",
    ),
]

AllPagesOption = Annotated[
    bool,
    typer.Option(
        "--all-pages",
        help="Fetch all pages",
        rich_help_panel="Pagination Options",
    ),
]
