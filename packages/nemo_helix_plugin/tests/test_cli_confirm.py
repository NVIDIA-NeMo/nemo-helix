# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the destructive-action confirmation helper and ``--yes`` option."""

from __future__ import annotations

import pytest
import typer
from nemo_helix_plugin.cli_confirm import confirm_destructive
from nemo_helix_plugin.cli_options import YesOption
from typer.testing import CliRunner

runner = CliRunner()


def _app(ran: list[str]) -> typer.Typer:
    app = typer.Typer()

    @app.command()
    def delete(name: str, yes: YesOption = False) -> None:
        confirm_destructive(f"Delete '{name}'?", yes=yes)
        ran.append(name)

    return app


@pytest.mark.parametrize("flag", ["--yes", "-y"])
def test_yes_skips_the_prompt(flag: str) -> None:
    ran: list[str] = []

    result = runner.invoke(_app(ran), ["thing", flag])

    assert result.exit_code == 0, result.output
    assert "Delete" not in result.output
    assert ran == ["thing"]


def test_off_a_terminal_without_yes_is_a_usage_error() -> None:
    ran: list[str] = []

    result = runner.invoke(_app(ran), ["thing"])

    assert result.exit_code == 2
    assert "Delete 'thing'? Pass --yes to confirm when not running interactively." in result.output
    assert ran == []


def test_on_a_terminal_confirming_runs_the_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_helix_plugin.cli_confirm.is_stdin_available", lambda: False)
    ran: list[str] = []

    result = runner.invoke(_app(ran), ["thing"], input="y\n")

    assert result.exit_code == 0, result.output
    assert "Delete 'thing'? [y/N]" in result.stderr
    assert "Delete" not in result.stdout
    assert ran == ["thing"]


@pytest.mark.parametrize("answer", ["n\n", "\n"])
def test_on_a_terminal_declining_aborts(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    monkeypatch.setattr("nemo_helix_plugin.cli_confirm.is_stdin_available", lambda: False)
    ran: list[str] = []

    result = runner.invoke(_app(ran), ["thing"], input=answer)

    assert result.exit_code == 1
    assert "Aborted" in result.output
    assert ran == []
