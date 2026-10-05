# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for --agent-mode flag, env var activation, and agent helpers."""

from __future__ import annotations

from unittest.mock import patch

from nemo_helix_ext.cli.app import app
from nemo_helix_ext.cli.core.agent_helpers import AGENT_HELPERS, get_agent_helpers
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.quickstart.config import QuickstartConfig
from typer.testing import CliRunner

runner = CliRunner()

qs_no_auth = QuickstartConfig(auth_enabled=False)


def _invoke(*args: str, env: dict[str, str] | None = None):
    """Invoke the CLI with auth disabled."""
    with patch("nemo_helix_ext.quickstart.QuickstartConfig.load", return_value=qs_no_auth):
        return runner.invoke(app, list(args), env=env)


class TestAgentModeFlag:
    def test_agent_mode_shows_in_help(self):
        result = _invoke("--help")
        assert "--agent-mode" in result.stdout

    def test_agent_mode_flag_sets_context(self):
        """--agent-mode should set agent_mode=True on CLIContext."""
        captured_ctx = {}

        original_init = CLIContext.__init__

        def patched_init(self, *a, **kw):
            original_init(self, *a, **kw)
            captured_ctx["obj"] = self

        with patch.object(CLIContext, "__init__", patched_init):
            _invoke("--agent-mode", "docs", "--help")

        assert captured_ctx["obj"].agent_mode is True

    def test_no_agent_mode_by_default(self):
        """Without --agent-mode, agent_mode should be False."""
        captured_ctx = {}

        original_init = CLIContext.__init__

        def patched_init(self, *a, **kw):
            original_init(self, *a, **kw)
            captured_ctx["obj"] = self

        with patch.object(CLIContext, "__init__", patched_init):
            _invoke("docs", "--help")

        assert captured_ctx["obj"].agent_mode is False

    def test_agent_mode_defaults_output_to_markdown(self):
        """--agent-mode should set output_format override to markdown."""
        captured_ctx = {}

        original_init = CLIContext.__init__

        def patched_init(self, *a, **kw):
            original_init(self, *a, **kw)
            captured_ctx["obj"] = self

        with patch.object(CLIContext, "__init__", patched_init):
            _invoke("--agent-mode", "docs", "--help")

        assert captured_ctx["obj"].overrides.get("output_format") == "markdown"

    def test_explicit_format_overrides_agent_mode(self):
        """--output-format should take precedence over agent mode default."""
        captured_ctx = {}

        original_init = CLIContext.__init__

        def patched_init(self, *a, **kw):
            original_init(self, *a, **kw)
            captured_ctx["obj"] = self

        with patch.object(CLIContext, "__init__", patched_init):
            _invoke("--agent-mode", "--output-format", "json", "docs", "--help")

        assert captured_ctx["obj"].overrides.get("output_format") == "json"


class TestAgentModeEnvVar:
    def _capture_agent_mode(self, *args: str, env: dict[str, str] | None = None) -> bool:
        captured_ctx = {}

        original_init = CLIContext.__init__

        def patched_init(self, *a, **kw):
            original_init(self, *a, **kw)
            captured_ctx["obj"] = self

        with patch.object(CLIContext, "__init__", patched_init):
            _invoke(*args, env=env)

        return captured_ctx["obj"].agent_mode

    def test_env_var_1_activates(self):
        assert self._capture_agent_mode("docs", "--help", env={"NHX_AGENT_MODE": "1"}) is True

    def test_env_var_true_activates(self):
        assert self._capture_agent_mode("docs", "--help", env={"NHX_AGENT_MODE": "true"}) is True

    def test_env_var_yes_activates(self):
        assert self._capture_agent_mode("docs", "--help", env={"NHX_AGENT_MODE": "yes"}) is True

    def test_env_var_false_does_not_activate(self):
        assert self._capture_agent_mode("docs", "--help", env={"NHX_AGENT_MODE": "false"}) is False

    def test_env_var_empty_does_not_activate(self):
        assert self._capture_agent_mode("docs", "--help", env={"NHX_AGENT_MODE": ""}) is False


class TestAgentHelpers:
    def test_get_helpers_exact_match(self):
        helpers = get_agent_helpers("workspaces")
        assert any("workspaces" in h.lower() for h in helpers)

    def test_get_helpers_prefix_fallback(self):
        helpers = get_agent_helpers("workspaces list")
        assert any("workspaces" in h.lower() for h in helpers)

    def test_get_helpers_global_fallback(self):
        helpers = get_agent_helpers("nonexistent-command")
        assert any("nemo docs" in h for h in helpers)

    def test_all_helpers_are_strings(self):
        for key, helpers in AGENT_HELPERS.items():
            assert isinstance(helpers, list), f"Helpers for '{key}' should be a list"
            for h in helpers:
                assert isinstance(h, str), f"Helper in '{key}' should be a string"

    def test_agent_hints_shown_on_help_page(self):
        """--agent-mode should show AGENT HINTS in help output."""
        result = _invoke("--agent-mode", "workspaces", "--help")
        assert "AGENT HINTS" in result.stdout
        assert "nemo docs" in result.stdout

    def test_no_agent_hints_without_flag(self):
        """Without --agent-mode, no agent hints should appear."""
        result = _invoke("workspaces", "--help")
        assert "AGENT HINTS" not in result.stdout
