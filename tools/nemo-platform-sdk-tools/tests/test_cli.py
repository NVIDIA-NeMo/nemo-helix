# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import click
from nemo_platform_sdk_tools.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_main_help_lists_preserved_command_groups() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "generate-cli" not in result.output
    assert "license" in result.output
    assert "openapi-stainless" in result.output
    assert "publish" not in result.output
    assert "vendor" in result.output


def test_license_help_is_registered() -> None:
    result = runner.invoke(app, ["license", "--help"])

    assert result.exit_code == 0
    assert "generate" in result.output
    assert "find-missing" in result.output


def test_license_generate_help_includes_output_option() -> None:
    result = runner.invoke(app, ["license", "generate", "--help"])

    assert result.exit_code == 0
    assert "--output" in click.unstyle(result.output)


def test_representative_sdk_help_is_registered() -> None:
    result = runner.invoke(app, ["openapi-stainless", "--help"])

    assert result.exit_code == 0
    assert "sync-methods" in result.output
    assert "sync-models" in result.output
