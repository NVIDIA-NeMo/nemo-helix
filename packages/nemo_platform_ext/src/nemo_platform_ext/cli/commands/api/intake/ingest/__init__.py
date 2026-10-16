# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from importlib import import_module as _importlib_import_module

from nemo_platform_ext.cli.core.help_formatter import create_typer_app

_cli_child_atif = _importlib_import_module("nemo_platform_ext.cli.commands.api.intake.ingest.atif")
_cli_child_chat_completions = _importlib_import_module(
    "nemo_platform_ext.cli.commands.api.intake.ingest.chat_completions"
)
_cli_child_spans = _importlib_import_module("nemo_platform_ext.cli.commands.api.intake.ingest.spans")

app = create_typer_app(name="ingest", help="Ingest operations")

app.add_typer(_cli_child_atif.app, name="atif")
app.add_typer(_cli_child_chat_completions.app, name="chat-completions")
app.add_typer(_cli_child_spans.app, name="spans")
