# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type definitions for the NeMo CLI."""

from __future__ import annotations

from typing import Annotated, Literal

import typer

from nemo_helix_ext.cli.core.autocomplete import autocomplete_workspace
from nemo_helix_ext.config.types import OutputFormat as SDKOutputFormat
from nemo_helix_ext.config.types import TimestampFormat as SDKTimestampFormat

# Output format type
ListOutputFormat = Literal[SDKOutputFormat, "code"]
EntityOutputFormat = Literal["json", "yaml", "raw", "code"]
ConfigOutputFormat = Literal["json", "yaml", "raw"]

# Timestamp format type
TimestampFormat = SDKTimestampFormat

# Typer annotated types for use in commands
ListOutputFormatOption = Annotated[
    ListOutputFormat | None,
    typer.Option(
        "--output-format",
        "--output",
        "-f",
        show_choices=True,
        help="Output format for the list of results.",
        rich_help_panel="Output Options",
    ),
]

EntityOutputFormatOption = Annotated[
    EntityOutputFormat | None,
    typer.Option(
        "--output-format",
        "--output",
        "-f",
        show_choices=True,
        help="Output format for an entity.",
        rich_help_panel="Output Options",
    ),
]
ConfigOutputFormatOption = Annotated[
    ConfigOutputFormat | None,
    typer.Option(
        "--output-format",
        "--output",
        "-f",
        show_choices=True,
        help="Output format for config.",
        rich_help_panel="Output Options",
    ),
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
    bool | None,
    typer.Option(
        "--no-truncate",
        help="Don't truncate long values in table/markdown/csv output.",
        rich_help_panel="Output Options",
    ),
]

TimestampFormatOption = Annotated[
    TimestampFormat | None,
    typer.Option(
        help="Timestamp format for table/markdown/csv output. Overrides global.",
        show_choices=True,
        rich_help_panel="Output Options",
    ),
]

OutputColumnsOption = Annotated[
    str | None,
    typer.Option(
        "--output-columns",
        "-c",
        help="Columns to display: 'default', 'all', or comma-separated names. Only affects table/csv/markdown formats.",
        rich_help_panel="Output Options",
    ),
]

# Common pagination options
AllPagesOption = Annotated[
    bool,
    typer.Option(
        "--all-pages",
        help="Fetch all pages",
        rich_help_panel="Pagination Options",
    ),
]

WorkspaceFilterOption = Annotated[
    str | None,
    typer.Option(
        "--filter.workspace",
        help="Filter by workspace",
        autocompletion=autocomplete_workspace,
    ),
]
