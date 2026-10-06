# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_eval_author_plugin.cli import EvalAuthorCLI
from nemo_evaluator import cli as evaluator_cli
from typer.testing import CliRunner


def test_author_cli_exposes_first_eval_with_an_agent_flag() -> None:
    result = CliRunner().invoke(EvalAuthorCLI().get_cli(), ["first-eval", "--help"])

    assert result.exit_code == 0
    assert "--agent" in result.output


def test_evaluator_cli_mounts_the_contributed_author_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator_cli, "discover", lambda group: {"author": EvalAuthorCLI})

    result = CliRunner().invoke(evaluator_cli.EvaluatorPluginCLI().get_cli(), ["author", "first-eval", "--help"])

    assert result.exit_code == 0
    assert "--agent" in result.output
