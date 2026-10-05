# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for CLI error handling with SDK errors."""

from nemo_helix_ext.cli.app import app
from typer.testing import CliRunner

from ..utils import assert_exit_code


class TestMissingWorkspaceError:
    """Test that SDK missing workspace error is converted to CLI-friendly message."""

    def test_missing_workspace_shows_cli_friendly_message(self, runner: CliRunner):
        """When workspace is not set, show CLI-specific guidance instead of SDK message."""

        # Call a command that requires workspace without providing it
        result = runner.invoke(app, "secrets list")

        assert_exit_code(result, 2)

        # Should show CLI-friendly guidance
        assert "Missing workspace" in result.stderr
        assert "config set --workspace <name>" in result.stderr
        assert "--workspace" in result.stderr
