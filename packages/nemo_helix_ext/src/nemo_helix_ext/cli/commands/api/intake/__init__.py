# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from importlib import import_module as _importlib_import_module

from nemo_helix_ext.cli.core.help_formatter import create_typer_app

_cli_child_annotations = _importlib_import_module("nemo_helix_ext.cli.commands.api.intake.annotations")
_cli_child_evaluator_results = _importlib_import_module("nemo_helix_ext.cli.commands.api.intake.evaluator_results")
_cli_child_ingest = _importlib_import_module("nemo_helix_ext.cli.commands.api.intake.ingest")
_cli_child_sessions = _importlib_import_module("nemo_helix_ext.cli.commands.api.intake.sessions")
_cli_child_spans = _importlib_import_module("nemo_helix_ext.cli.commands.api.intake.spans")
_cli_child_traces = _importlib_import_module("nemo_helix_ext.cli.commands.api.intake.traces")

app = create_typer_app(name="intake", help="Intake operations")

app.add_typer(_cli_child_annotations.app, name="annotations")
app.add_typer(_cli_child_evaluator_results.app, name="evaluator-results")
app.add_typer(_cli_child_ingest.app, name="ingest")
app.add_typer(_cli_child_sessions.app, name="sessions")
app.add_typer(_cli_child_spans.app, name="spans")
app.add_typer(_cli_child_traces.app, name="traces")
