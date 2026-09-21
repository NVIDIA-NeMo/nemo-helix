# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo agent context and commands."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nemo_platform_ext.cli.app import app
from nemo_platform_ext.quickstart.config import QuickstartConfig
from typer.testing import CliRunner

runner = CliRunner()

_qs_no_auth = QuickstartConfig(auth_enabled=False)


def _invoke(*args: str):
    with patch("nemo_platform_ext.quickstart.QuickstartConfig.load", return_value=_qs_no_auth):
        return runner.invoke(app, list(args))


class TestAgentHelp:
    def test_agent_help(self):
        result = _invoke("agent", "--help")
        assert result.exit_code == 0
        assert "context" in result.stdout
        assert "commands" in result.stdout

    def test_agent_no_args_shows_help(self):
        result = _invoke("agent")
        assert result.exit_code == 0
        assert "context" in result.stdout


class TestAgentContext:
    def test_context_maps_job_entry_points_to_plugin_surface(self):
        with patch(
            "nemo_platform_plugin.discovery.discover_entry_points",
            side_effect=lambda group: {"test-plugin.some-job": object()} if group == "nemo.jobs" else {},
        ):
            result = _invoke("agent", "context")
        assert result.exit_code == 0
        assert "| test-plugin |" in result.stdout
        assert "Tasks" in result.stdout
        assert "test-plugin.some-job" in result.stdout

    def test_context_sections_present(self):
        result = _invoke("agent", "context")
        assert result.exit_code == 0
        assert "## Installed Plugins" in result.stdout
        assert "## Available CLI Commands" in result.stdout
        assert "## Entry-Point Catalog" in result.stdout
        assert "## Agent Skills" in result.stdout
        assert "## Quick Reference" in result.stdout

    def test_context_no_plugins_fallback(self):
        with (
            patch(
                "nemo_platform_ext.cli.commands.use_cases.agent._build_plugin_surfaces",
                return_value={},
            ),
            patch.dict("sys.modules", {"nemo_platform_plugin": None, "nemo_platform_plugin.discovery": None}),
        ):
            result = _invoke("agent", "context")
        assert result.exit_code == 0
        assert "_No plugins installed._" in result.stdout

    def test_context_includes_known_commands(self):
        result = _invoke("agent", "context")
        assert result.exit_code == 0
        assert "nemo agent" in result.stdout
        assert "nemo docs" in result.stdout

    def test_context_with_plugins(self):
        mock_manifest = MagicMock()
        mock_manifest.version = "1.2.3"
        mock_manifest.description = "A test plugin"

        with (
            patch(
                "nemo_platform_ext.cli.commands.use_cases.agent._build_plugin_surfaces",
                return_value={"test-plugin": ["CLI", "Tasks"]},
            ),
            patch(
                "nemo_platform_plugin.discovery.discover_manifests",
                return_value={"test-plugin": mock_manifest},
            ),
        ):
            result = _invoke("agent", "context")
        assert result.exit_code == 0
        assert "test-plugin" in result.stdout
        assert "1.2.3" in result.stdout
        assert "A test plugin" in result.stdout

    def test_context_normalizes_manifest_cells(self):
        mock_manifest = MagicMock()
        mock_manifest.version = None
        mock_manifest.description = "Pipe | and\nnewline"

        with (
            patch(
                "nemo_platform_ext.cli.commands.use_cases.agent._build_plugin_surfaces",
                return_value={"test-plugin": []},
            ),
            patch(
                "nemo_platform_plugin.discovery.discover_manifests",
                return_value={"test-plugin": mock_manifest},
            ),
        ):
            result = _invoke("agent", "context")
        assert result.exit_code == 0
        assert "Pipe \\| and newline" in result.stdout

    def test_commands_ignores_plugin_cli_discovery_failure(self):
        with patch("nemo_platform_plugin.discovery.discover_entry_points", side_effect=RuntimeError("boom")):
            result = _invoke("agent", "commands")
        assert result.exit_code == 0
        assert "nemo docs" in result.stdout

    def test_context_skills_unavailable_fallback(self):
        with patch(
            "nemo_platform_ext.cli.commands.skills.registry.load_skills",
            side_effect=ImportError("no skills"),
        ):
            result = _invoke("agent", "context")
        assert result.exit_code == 0
        assert "_Skills unavailable._" in result.stdout

    def test_context_skills_value_error_fallback(self):
        with patch(
            "nemo_platform_ext.cli.commands.skills.registry.load_skills",
            side_effect=ValueError("bad frontmatter"),
        ):
            result = _invoke("agent", "context")
        assert result.exit_code == 0
        assert "_Skills unavailable._" in result.stdout

    def test_context_quick_reference_content(self):
        result = _invoke("agent", "context")
        assert result.exit_code == 0
        assert "nemo docs" in result.stdout
        assert "nemo agent commands" in result.stdout
        assert "nemo agent context" in result.stdout


class TestAgentCommands:
    def test_commands_header(self):
        result = _invoke("agent", "commands")
        assert result.exit_code == 0
        assert "# NeMo CLI Commands" in result.stdout

    def test_commands_table_format(self):
        result = _invoke("agent", "commands")
        assert result.exit_code == 0
        assert "| Command | Panel | Description |" in result.stdout

    def test_commands_includes_known_entry(self):
        result = _invoke("agent", "commands")
        assert result.exit_code == 0
        assert "nemo agent" in result.stdout
        assert "nemo docs" in result.stdout

    def test_commands_uses_visible_command_order_with_discovered_plugins(self):
        plugin_entry_points = {
            "data-designer": SimpleNamespace(value="fake.module:DataDesignerCLI"),
            "anonymizer": SimpleNamespace(value="fake.module:AnonymizerCLI"),
            "experiments": SimpleNamespace(value="nmp.intake.cli:ExperimentsCLI"),
            "intake": SimpleNamespace(value="nmp.intake.cli:IntakeCLI"),
            "guardrail": SimpleNamespace(value="nemo_guardrails_plugin.cli:GuardrailCLI"),
        }

        with patch("nemo_platform_plugin.discovery.discover_entry_points", return_value=plugin_entry_points):
            result = _invoke("agent", "commands")

        assert result.exit_code == 0
        command_rows = [line for line in result.stdout.splitlines() if line.startswith("| nemo ")]
        assert command_rows == [
            "| nemo setup | Setup | Set up NeMo Platform: connect or start services, configure a provider, install skills. |",
            "| nemo auth | Setup | Manage authentication for NeMo Platform. |",
            "| nemo services | Setup | Run platform services locally. |",
            "| nemo skills | Setup | Install AI agent skill files for Nemo. |",
            "| nemo chat | CLI functions | Start an interactive chat session with a model. |",
            "| nemo docs | CLI functions | Read NeMo Platform documentation. |",
            "| nemo wait | CLI functions | Wait for resources to reach a desired status. |",
            "| nemo agent | CLI functions | Commands for AI agent context and capability discovery. |",
            "| nemo plugins | CLI functions | Commands for plugin discovery. |",
            "| nemo files | Core plugins | Manage files. |",
            "| nemo inference | Core plugins | Inference operations. |",
            "| nemo jobs | Core plugins | Manage jobs. |",
            "| nemo models | Core plugins | Manage models. |",
            "| nemo secrets | Core plugins | Manage secrets. |",
            "| nemo workspaces | Core plugins | Manage workspaces. |",
            "| nemo data-designer | Functional plugins | Plugin commands for data-designer. |",
            "| nemo guardrail | Functional plugins | Plugin commands for guardrail. |",
            "| nemo anonymizer | Functional plugins | Plugin commands for anonymizer. |",
            "| nemo safe-synthesizer | Functional plugins | Plugin commands for safe-synthesizer. |",
            "| nemo experiments | Functional plugins | Plugin commands for experiments. |",
            "| nemo intake | Functional plugins | Plugin commands for intake. |",
        ]
