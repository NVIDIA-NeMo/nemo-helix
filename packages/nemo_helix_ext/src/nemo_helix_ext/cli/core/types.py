# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type definitions for the NeMo CLI.

The output options are shared with plugin commands and live in
:mod:`nemo_helix_plugin.cli_options`; they are re-exported here for core commands.
"""

from __future__ import annotations

from typing import Annotated

import typer
from nemo_helix_plugin.cli_options import (
    AllPagesOption,
    ConfigOutputFormat,
    ConfigOutputFormatOption,
    EntityOutputFormat,
    EntityOutputFormatOption,
    ListOutputFormat,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
    TimestampFormat,
    TimestampFormatOption,
)

from nemo_helix_ext.cli.core.autocomplete import autocomplete_workspace

__all__ = [
    "AllPagesOption",
    "ConfigOutputFormat",
    "ConfigOutputFormatOption",
    "EntityOutputFormat",
    "EntityOutputFormatOption",
    "ListOutputFormat",
    "ListOutputFormatOption",
    "NoTruncateOption",
    "OutputColumnsOption",
    "StreamOutputOption",
    "TimestampFormat",
    "TimestampFormatOption",
    "WorkspaceFilterOption",
]

WorkspaceFilterOption = Annotated[
    str | None,
    typer.Option(
        "--filter.workspace",
        help="Filter by workspace",
        autocompletion=autocomplete_workspace,
    ),
]
