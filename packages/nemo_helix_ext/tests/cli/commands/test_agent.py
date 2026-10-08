# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deprecated nemo agent alias."""

from __future__ import annotations

from unittest.mock import patch

from nemo_helix_ext.cli.app import app
from nemo_helix_ext.quickstart.config import QuickstartConfig
from typer.testing import CliRunner

runner = CliRunner()

_qs_no_auth = QuickstartConfig(auth_enabled=False)


def _invoke(*args: str):
    with patch("nemo_helix_ext.quickstart.QuickstartConfig.load", return_value=_qs_no_auth):
        return runner.invoke(app, list(args))


def test_agent_is_hidden_from_root_help():
    result = _invoke("--help")
    assert result.exit_code == 0
    assert "describe" in result.stdout
    assert not any(line.split()[:1] == ["agent"] for line in result.stdout.splitlines())


def test_agent_no_args_shows_deprecation_help():
    result = _invoke("agent")
    assert result.exit_code == 0
    assert "Deprecated: use 'nemo describe' instead." in result.stdout


def test_agent_context_matches_describe_and_warns():
    describe = _invoke("describe")
    alias = _invoke("agent", "context")
    assert alias.exit_code == 0
    assert alias.stdout == describe.stdout
    assert "'nemo agent' is deprecated" in alias.stderr


def test_agent_commands_prints_table_and_warns():
    result = _invoke("agent", "commands")
    assert result.exit_code == 0
    assert "# NeMo CLI Commands" in result.stdout
    assert "| nemo describe |" in result.stdout
    assert "'nemo agent' is deprecated" in result.stderr
