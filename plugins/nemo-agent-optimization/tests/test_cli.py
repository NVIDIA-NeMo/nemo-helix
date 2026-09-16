# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_agent_optimization_plugin.cli import OptimizationStrategiesCLI
from typer.testing import CliRunner


class _FakeStrategy:
    name = "fake"

    def validate_config(self, config, *, agent):
        del config, agent

    def run(self, **kwargs):
        del kwargs
        return {}


def test_lists_installed_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_agent_optimization_plugin.cli.discover_agent_optimize_jobs",
        lambda: {"fake": _FakeStrategy()},
    )
    runner = CliRunner()
    result = runner.invoke(OptimizationStrategiesCLI().get_cli(), ["list"])
    assert result.exit_code == 0
    assert "fake" in result.stdout
