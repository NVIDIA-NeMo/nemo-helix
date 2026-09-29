# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for ``nemo agents leaderboard`` tests."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from nemo_agents_plugin.cli import AgentsCLI
from typer.testing import CliRunner

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """CLI runner for ``nemo agents`` command tests."""
    # The CLI sizes the table from the terminal; pin a wide one so output doesn't depend on the host.
    monkeypatch.setenv("COLUMNS", "122")
    return CliRunner()


@pytest.fixture
def app():
    """Build the actual ``nemo agents`` Typer app (with ``leaderboard`` registered)."""
    return AgentsCLI().get_cli()


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the directory holding hand-written leaderboard report fixtures."""
    return FIXTURES_DIR


@pytest.fixture
def sample_reports_dir(tmp_path: Path, fixtures_dir: Path) -> Path:
    """A directory containing two valid leaderboard report fixtures.

    Layout:

    .. code-block:: text

        <tmp_path>/reports/
            report-alpha.json
            report-beta.json
    """
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    for fixture_name in ("report-alpha.json", "report-beta.json"):
        shutil.copy(fixtures_dir / fixture_name, reports_dir / fixture_name)
    return reports_dir


@pytest.fixture
def write_report(tmp_path: Path) -> Callable[[str, str], Path]:
    """Write an arbitrary report file into *tmp_path* and return its path."""

    def _write_report(filename: str, contents: str) -> Path:
        report = tmp_path / filename
        report.write_text(contents)
        return report

    return _write_report
