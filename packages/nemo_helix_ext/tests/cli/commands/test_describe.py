# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo describe."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nemo_helix_ext.cli.app import app
from nemo_helix_ext.quickstart.config import QuickstartConfig
from typer.testing import CliRunner

runner = CliRunner()

_qs_no_auth = QuickstartConfig(auth_enabled=False)


def _invoke(*args: str):
    with patch("nemo_helix_ext.quickstart.QuickstartConfig.load", return_value=_qs_no_auth):
        return runner.invoke(app, list(args))


class TestDescribe:
    def test_maps_job_entry_points_to_plugin_surface(self):
        with patch(
            "nemo_helix_plugin.discovery.discover_entry_points",
            side_effect=lambda group: {"test-plugin.some-job": object()} if group == "nemo.jobs" else {},
        ):
            result = _invoke("describe")
        assert result.exit_code == 0
        assert "| test-plugin |" in result.stdout
        assert "Tasks" in result.stdout
        assert "test-plugin.some-job" in result.stdout

    def test_sections_present(self):
        result = _invoke("describe")
        assert result.exit_code == 0
        assert "## Installed Plugins" in result.stdout
        assert "## Available CLI Commands" in result.stdout
        assert "## Entry-Point Catalog" in result.stdout
        assert "## Agent Skills" in result.stdout
        assert "## Quick Reference" in result.stdout

    def test_no_plugins_fallback(self):
        with (
            patch(
                "nemo_helix_ext.cli.commands.use_cases.describe._build_plugin_surfaces",
                return_value={},
            ),
            # Hide only discovery: the CLI itself still imports other nemo_helix_plugin modules.
            patch.dict("sys.modules", {"nemo_helix_plugin.discovery": None}),
        ):
            result = _invoke("describe")
        assert result.exit_code == 0
        assert "_No plugins installed._" in result.stdout

    def test_includes_known_commands(self):
        result = _invoke("describe")
        assert result.exit_code == 0
        assert "| nemo describe |" in result.stdout
        assert "| nemo docs |" in result.stdout
        assert "| nemo agent |" not in result.stdout

    def test_with_plugins(self):
        mock_manifest = MagicMock()
        mock_manifest.version = "1.2.3"
        mock_manifest.description = "A test plugin"

        with (
            patch(
                "nemo_helix_ext.cli.commands.use_cases.describe._build_plugin_surfaces",
                return_value={"test-plugin": ["CLI", "Tasks"]},
            ),
            patch(
                "nemo_helix_plugin.discovery.discover_manifests",
                return_value={"test-plugin": mock_manifest},
            ),
        ):
            result = _invoke("describe")
        assert result.exit_code == 0
        assert "test-plugin" in result.stdout
        assert "1.2.3" in result.stdout
        assert "A test plugin" in result.stdout

    def test_normalizes_manifest_cells(self):
        mock_manifest = MagicMock()
        mock_manifest.version = None
        mock_manifest.description = "Pipe | and\nnewline"

        with (
            patch(
                "nemo_helix_ext.cli.commands.use_cases.describe._build_plugin_surfaces",
                return_value={"test-plugin": []},
            ),
            patch(
                "nemo_helix_plugin.discovery.discover_manifests",
                return_value={"test-plugin": mock_manifest},
            ),
        ):
            result = _invoke("describe")
        assert result.exit_code == 0
        assert "Pipe \\| and newline" in result.stdout

    def test_ignores_plugin_cli_discovery_failure(self):
        with patch("nemo_helix_plugin.discovery.discover_entry_points", side_effect=RuntimeError("boom")):
            result = _invoke("describe")
        assert result.exit_code == 0
        assert "nemo docs" in result.stdout

    def test_skills_unavailable_fallback(self):
        with patch(
            "nemo_helix_ext.cli.commands.skills.registry.load_skills",
            side_effect=ImportError("no skills"),
        ):
            result = _invoke("describe")
        assert result.exit_code == 0
        assert "_Skills unavailable._" in result.stdout

    def test_skills_value_error_fallback(self):
        with patch(
            "nemo_helix_ext.cli.commands.skills.registry.load_skills",
            side_effect=ValueError("bad frontmatter"),
        ):
            result = _invoke("describe")
        assert result.exit_code == 0
        assert "_Skills unavailable._" in result.stdout

    def test_quick_reference_content(self):
        result = _invoke("describe")
        assert result.exit_code == 0
        assert "nemo docs" in result.stdout
        assert "nemo describe" in result.stdout
        assert "nemo agent context" not in result.stdout
        assert "nemo agent commands" not in result.stdout

    def test_removed_agent_command_is_unknown(self):
        result = _invoke("agent", "context")
        assert result.exit_code == 2
        assert "No such command 'agent'" in result.output


class TestCommandsTable:
    def test_uses_visible_command_order_with_discovered_plugins(self):
        plugin_entry_points = {
            "data-designer": SimpleNamespace(value="fake.module:DataDesignerCLI"),
            "anonymizer": SimpleNamespace(value="fake.module:AnonymizerCLI"),
            "experiments": SimpleNamespace(value="nhx.intake.cli:ExperimentsCLI"),
            "intake": SimpleNamespace(value="nhx.intake.cli:IntakeCLI"),
            "guardrail": SimpleNamespace(value="nemo_guardrails_plugin.cli:GuardrailCLI"),
        }

        with patch("nemo_helix_plugin.discovery.discover_entry_points", return_value=plugin_entry_points):
            result = _invoke("describe")

        assert result.exit_code == 0
        command_rows = [line for line in result.stdout.splitlines() if line.startswith("| nemo ")]
        assert command_rows == [
            "| nemo setup | Setup | Set up NeMo Helix: connect or start services, configure a provider, install skills. |",
            "| nemo auth | Setup | Manage authentication for NeMo Helix. |",
            "| nemo config | Setup | Manage NeMo Helix CLI configuration. |",
            "| nemo services | Setup | Run Helix services locally. |",
            "| nemo skills | Setup | Install AI agent skill files for Nemo. |",
            "| nemo chat | CLI functions | Start an interactive chat session with a model. |",
            "| nemo docs | CLI functions | Read NeMo Helix documentation. |",
            "| nemo describe | CLI functions | Describe the NeMo Helix CLI or any command in it. |",
            "| nemo wait | CLI functions | Wait for resources to reach a desired status. |",
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


class TestDescribeCommandPath:
    def test_describes_a_leaf_command_as_markdown(self):
        result = _invoke("describe", "models", "create")
        assert result.exit_code == 0
        assert result.stdout.startswith("# nemo models create\n")
        assert "Usage: `nemo models create [OPTIONS] [NAME]`" in result.stdout
        assert "## Arguments" in result.stdout
        assert "| --description | text |" in result.stdout

    def test_strips_rich_markup_from_help(self):
        result = _invoke("describe", "models", "create")
        assert result.exit_code == 0
        assert "[bold" not in result.stdout
        assert "[/]" not in result.stdout
        assert "Required fields: name" in result.stdout

    def test_describes_a_leaf_command_as_json(self):
        result = _invoke("describe", "-f", "json", "models", "create")
        assert result.exit_code == 0
        description = json.loads(result.stdout)
        assert description["schema_version"] == "v1"
        assert description["command"] == "nemo models create"
        assert description["kind"] == "command"
        assert description["ignored_args"] == []
        options = {option["name"]: option for option in description["options"]}
        assert options["description"]["flags"] == ["--description"]
        assert options["output_format"]["choices"] == ["json", "yaml", "raw", "code"]
        assert "help" not in options

    def test_generated_required_suffix_marks_argument_required(self):
        result = _invoke("describe", "-f", "json", "models", "create")
        assert result.exit_code == 0
        (argument,) = json.loads(result.stdout)["arguments"]
        assert argument["name"] == "NAME"
        assert argument["required"] is True
        assert not argument["help"].endswith("(required)")

    def test_ignores_the_described_commands_own_arguments_and_options(self):
        result = _invoke(
            "describe", "-f", "json", "models", "create", "my-model", "--spec-file", "model.yaml", "-f", "yaml"
        )
        assert result.exit_code == 0
        description = json.loads(result.stdout)
        assert description["command"] == "nemo models create"
        assert description["ignored_args"] == ["my-model", "--spec-file", "model.yaml", "-f", "yaml"]

    def test_ignored_arguments_are_reported_in_markdown(self):
        result = _invoke("describe", "models", "list", "--page-size", "2")
        assert result.exit_code == 0
        assert "_Ignored arguments: `--page-size 2`_" in result.stdout

    def test_describes_a_group_with_its_subcommands(self):
        result = _invoke("describe", "-f", "json", "models")
        assert result.exit_code == 0
        description = json.loads(result.stdout)
        assert description["kind"] == "group"
        names = [subcommand["name"] for subcommand in description["subcommands"]]
        assert {"create", "list", "get"} <= set(names)

    def test_describes_hidden_commands_by_name(self):
        result = _invoke("describe", "-f", "json", "projects")
        assert result.exit_code == 0
        description = json.loads(result.stdout)
        assert description["hidden"] is True
        assert {"create", "list", "get"} <= {subcommand["name"] for subcommand in description["subcommands"]}

    def test_unknown_command_is_a_usage_error(self):
        result = _invoke("describe", "models", "nope")
        assert result.exit_code == 2
        assert "No such command 'nope' in 'nemo models'" in result.stderr
        assert "nemo describe models" in result.stderr

    def test_root_json_lists_top_level_commands_without_loading_them(self):
        from nemo_helix_ext.cli.core import lazy_load

        loaded: list[str] = []
        original = lazy_load.build_lazy_loader

        def _record(entry):
            loaded.append(entry.name)
            return original(entry)

        with patch("nemo_helix_ext.cli.core.lazy_load.build_lazy_loader", side_effect=_record):
            result = _invoke("describe", "-f", "json")

        assert result.exit_code == 0
        description = json.loads(result.stdout)
        assert description["command"] == "nemo"
        names = [subcommand["name"] for subcommand in description["subcommands"]]
        assert "describe" in names
        assert "projects" not in names
        assert loaded == ["describe"]

    def test_bare_markdown_is_the_cli_overview(self):
        result = _invoke("describe")
        assert result.exit_code == 0
        assert result.stdout.startswith("# NeMo Helix CLI\n")
