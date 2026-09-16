# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from importlib import import_module as _importlib_import_module

from nemo_helix_ext.cli.core.help_formatter import create_typer_app

_cli_child_v1 = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.gateway.openai.v1")

app = create_typer_app(name="openai", help="Openai operations")

app.add_typer(_cli_child_v1.app, name="v1")
