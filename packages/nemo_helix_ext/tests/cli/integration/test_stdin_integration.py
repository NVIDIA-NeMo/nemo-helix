# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for stdin functionality with CLI commands."""

from __future__ import annotations

import json
import uuid

from nemo_helix_ext.cli.app import app
from typer.testing import CliRunner

from ..utils import assert_exit_code

# Basic runner for tests that check error handling before API is called
basic_runner = CliRunner(env={"NHX_BASE_URL": "http://127.0.0.1"})


class TestWorkspaceCreateStdin:
    """Test workspace create command with stdin input."""

    def test_create_workspace_from_stdin(self, runner: CliRunner) -> None:
        """Test creating a workspace from stdin JSON."""
        workspace = f"test-ws-{uuid.uuid4().hex[:8]}"

        json_input = json.dumps({"name": workspace, "description": "Test workspace"})

        result = runner.invoke(
            app,
            ["workspaces", "create", workspace, "--input-file", "-"],
            input=json_input,
        )

        assert_exit_code(result, 0)

        output = json.loads(result.stdout)
        assert output["name"] == workspace
        assert output["description"] == "Test workspace"

    def test_create_workspace_stdin_with_cli_override(self, runner: CliRunner) -> None:
        """Test that CLI options override stdin data."""
        workspace = f"test-ws-{uuid.uuid4().hex[:8]}"

        json_input = json.dumps({"name": "foo-ws", "description": "From stdin"})

        result = runner.invoke(
            app,
            [
                "workspaces",
                "create",
                workspace,
                "--input-file",
                "-",
                "--description",
                "From CLI",
            ],
            input=json_input,
        )

        assert_exit_code(result, 0)

        output = json.loads(result.stdout)
        assert output["name"] == workspace
        # CLI description should override stdin description
        assert output["description"] == "From CLI"

    def test_create_workspace_invalid_json(self, runner: CliRunner) -> None:
        """Test error handling for invalid JSON/YAML in stdin."""
        # Use syntax that is invalid in both JSON and YAML
        json_input = "{{{{{"

        result = runner.invoke(app, ["workspaces", "create", "--input-file", "-"], input=json_input)

        # Exit code 1 = data error (invalid JSON/YAML), exit code 2 = usage error
        assert_exit_code(result, 1)
        # Check for data error message (YAML parser error)
        combined_output = result.stderr + result.stdout
        assert "Invalid data" in combined_output or "error" in combined_output.lower()

    def test_create_workspace_without_stdin(self, runner: CliRunner) -> None:
        """Test creating workspace with only CLI options (no stdin)."""
        result = runner.invoke(app, ["workspaces", "create", "cli-ws"])

        assert_exit_code(result, 0)

        output = json.loads(result.stdout)
        assert output["name"] == "cli-ws"


class TestWorkspaceUpdateStdin:
    """Test workspace update command with stdin input."""

    def test_update_workspace_from_stdin_with_positional_name(self, runner: CliRunner, random_workspace: str) -> None:
        """Test updating a workspace from stdin with name as positional argument."""
        json_input = json.dumps({"description": "Updated description"})

        result = runner.invoke(
            app,
            ["workspaces", "update", random_workspace, "--input-file", "-"],
            input=json_input,
        )

        assert_exit_code(result, 0)

        output = json.loads(result.stdout)
        assert output["name"] == random_workspace
        assert output["description"] == "Updated description"

    def test_update_workspace_cli_flags_override_stdin(self, runner: CliRunner, random_workspace: str) -> None:
        """Test that CLI flags override stdin data."""
        json_input = json.dumps({"description": "From stdin"})

        result = runner.invoke(
            app,
            [
                "workspaces",
                "update",
                random_workspace,
                "--input-file",
                "-",
                "--description",
                "From CLI",
            ],
            input=json_input,
        )

        assert_exit_code(result, 0)

        output = json.loads(result.stdout)
        # CLI flag should override stdin description
        assert output["description"] == "From CLI"

    def test_update_workspace_missing_id(self) -> None:
        """Test error when workspace ID is missing (positional arg required)."""
        json_input = json.dumps({"description": "Updated description"})

        # Use basic_runner since this test checks CLI argument validation before any API call
        result = basic_runner.invoke(app, ["workspaces", "update", "--input-file", "-"], input=json_input)

        # Should fail because name is now a required positional argument
        assert_exit_code(result, 2)
        assert "Missing required argument" in result.stderr or "NAME" in result.stderr
