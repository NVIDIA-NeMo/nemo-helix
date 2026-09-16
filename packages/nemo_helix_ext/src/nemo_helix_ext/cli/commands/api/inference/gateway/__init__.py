# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from importlib import import_module as _importlib_import_module

from nemo_helix_ext.cli.core.help_formatter import create_typer_app

_cli_child_model = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.gateway.model")
_cli_child_openai = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.gateway.openai")
_cli_child_provider = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.gateway.provider")

app = create_typer_app(name="gateway", help="Gateway operations")

app.add_typer(_cli_child_model.app, name="model")
app.add_typer(_cli_child_openai.app, name="openai")
app.add_typer(_cli_child_provider.app, name="provider")
