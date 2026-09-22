# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test various format combinations to ensure they work."""

import json

from nemo_helix_ext.cli.app import app
from typer.testing import CliRunner

from ..utils import assert_exit_code


def test_global_table_format_with_command_json_override(runner: CliRunner, random_workspace):
    """Test global --output-format table with command-level --output-format json override."""
    for i in range(2):
        result = runner.invoke(
            app,
            f"projects create --workspace {random_workspace} project-{i:02d} --description 'Project {i}'",
        )
        assert_exit_code(result, 0)

    # Run command with global table format but json override
    result = runner.invoke(
        app,
        [
            "--output-format",
            "table",
            "projects",
            "list",
            "--workspace",
            random_workspace,
            "--output-format",
            "json",
        ],
    )

    # Should succeed and output JSON (not table)
    assert_exit_code(result, 0)
    assert "project-01" in result.stdout
    # Should be JSON format
    data = json.loads(result.stdout)
    assert data["data"][0]["workspace"] == random_workspace


def test_markdown_format(runner: CliRunner, random_workspace):
    for i in range(2):
        result = runner.invoke(
            app,
            f"projects create --workspace {random_workspace} project-{i:02d} --description 'Project {i}'",
        )
        assert_exit_code(result, 0)

    result = runner.invoke(app, f"projects list --workspace {random_workspace} --output-format markdown")

    assert_exit_code(result, 0)
    assert "|" in result.stdout
    assert "---" in result.stdout
    assert "project-00" in result.stdout
    assert "project-01" in result.stdout
