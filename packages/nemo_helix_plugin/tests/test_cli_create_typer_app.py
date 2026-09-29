# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``create_typer_app`` gives plugin command groups the ``nemo`` CLI's defaults."""

from __future__ import annotations

import typer
from nemo_helix_plugin.cli import HELP_OPTION_NAMES, create_typer_app
from typer.main import get_command
from typer.testing import CliRunner

runner = CliRunner()


def _group(**kwargs: object) -> typer.Typer:
    app = create_typer_app(help="Widgets.", **kwargs)

    @app.command("list")
    def list_widgets() -> None:
        typer.echo("listed")

    @app.command("get")
    def get_widget() -> None:  # pragma: no cover - only listed in help
        typer.echo("got")

    return app


def test_a_bare_group_prints_its_help() -> None:
    result = runner.invoke(_group(), [])

    # Click reports no-args help as a usage exit; under ``nemo`` the host exits 0.
    assert result.exit_code == 2
    assert "Widgets." in result.output
    assert "list" in result.output


def test_short_help_flag() -> None:
    result = runner.invoke(_group(), ["-h"])

    assert result.exit_code == 0
    assert "Widgets." in result.output


def test_shell_completion_is_left_to_the_root_app() -> None:
    assert [param.name for param in get_command(_group()).params] == []


def test_keyword_arguments_override_the_defaults() -> None:
    result = runner.invoke(_group(no_args_is_help=False, invoke_without_command=True), [])

    assert "Widgets." not in result.output


def test_help_names_merge_into_existing_context_settings() -> None:
    app = create_typer_app(context_settings={"max_content_width": 120})

    assert app.info.context_settings == {"max_content_width": 120, "help_option_names": list(HELP_OPTION_NAMES)}
