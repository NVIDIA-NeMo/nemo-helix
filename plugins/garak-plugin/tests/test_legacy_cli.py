# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deprecated ``nemo auditor`` alias group."""

from __future__ import annotations

from garak_plugin._legacy_cli import AuditorAliasCLI
from garak_plugin.cli import GarakPluginCLI
from typer.main import get_command
from typer.testing import CliRunner


def test_alias_exposes_the_same_commands_plus_the_old_job_verb() -> None:
    alias = get_command(AuditorAliasCLI().get_cli()).commands  # type: ignore[attr-defined]
    current = get_command(GarakPluginCLI().get_cli()).commands  # type: ignore[attr-defined]

    assert set(current) <= set(alias)
    assert "audit" in alias
    assert "scan" not in alias


def test_alias_warns_on_stderr_and_still_runs() -> None:
    result = CliRunner().invoke(AuditorAliasCLI().get_cli(), ["info"])

    assert result.exit_code == 0
    assert "deprecated" in result.output
    assert "nemo garak-plugin" in result.output


def test_old_job_verb_resolves() -> None:
    result = CliRunner().invoke(AuditorAliasCLI().get_cli(), ["audit", "--help"])

    assert result.exit_code == 0
    assert "deprecated" in result.output
