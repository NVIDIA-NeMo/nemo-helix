# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import click
import nemo_platform
import pytest
import typer
from click.testing import CliRunner as ClickCliRunner
from nemo_platform_ext.cli.app import app
from nemo_platform_ext.cli.commands.api import API_TOP_LEVEL_ENTRIES
from nemo_platform_ext.cli.commands.manifest_registry import TOP_LEVEL_ENTRIES
from nemo_platform_ext.cli.core.lazy_load import (
    ManifestBackedNmpGroup,
    attach_lazy_entries,
    lazy_command_loader,
    lazy_group_loader,
    lazy_plugin_loader,
)
from nemo_platform_ext.cli.manifest import TopLevelEntry, build_top_level_entries
from nemo_platform_ext.quickstart.config import QuickstartConfig
from nemo_platform_plugin.cli import NemoCLI
from typer.testing import CliRunner


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag(flag):
    """--version and -V both show version and exit."""
    runner = CliRunner()
    result = runner.invoke(app, [flag])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"nemo version {nemo_platform.__version__}"


def test_version_flag_before_command():
    """--version should work even when placed before a command (is_eager=True)."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version", "projects", "list"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"nemo version {nemo_platform.__version__}"


def test_help_includes_getting_started():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Getting started:" in result.stdout
    assert "nemo docs --list" in result.stdout
    assert "nemo services run --help" in result.stdout
    assert "Exit codes:" in result.stdout
    assert "3: Remote/API error" in result.stdout
    # Help panel truncation depends on terminal width; match a stable prefix.
    assert "Set up NeMo Platform" in result.stdout
    assert "--output-format, --output, -f" in result.stdout
    assert "--help, -h" in result.stdout
    assert "nemo auth login --base-url" not in result.stdout
    assert "nemo quickstart configure" not in result.stdout


def test_generated_list_help_includes_stream_option():
    runner = CliRunner()
    qs_config = QuickstartConfig(auth_enabled=False)

    with patch("nemo_platform_ext.quickstart.QuickstartConfig.load", return_value=qs_config):
        result = runner.invoke(app, ["models", "list", "--help"])

    assert result.exit_code == 0
    assert "--stream" in result.stdout
    assert "--output-format, --output, -f" in result.stdout


def test_generated_list_validates_stream_output_before_client_setup():
    runner = CliRunner()
    qs_config = QuickstartConfig(auth_enabled=False)

    with (
        patch("nemo_platform_ext.quickstart.QuickstartConfig.load", return_value=qs_config),
        patch("nemo_platform_ext.cli.core.context.CLIContext.get_client") as mock_get_client,
    ):
        result = runner.invoke(app, ["workspaces", "list", "--output", "table", "--stream"])

    assert result.exit_code == 2
    assert "--stream requires --output json or --output raw" in result.stderr
    mock_get_client.assert_not_called()


def test_generated_api_manifest_excludes_source_owned_auth_commands():
    entry_names = {entry.name for entry in API_TOP_LEVEL_ENTRIES}

    assert {"auth", "access-keys", "iam"}.isdisjoint(entry_names)


@pytest.mark.parametrize(
    ("argv", "expected_text"),
    [
        ([], "Command-line interface for NeMo Platform."),
        (["agent"], "Commands for AI agent context and capability discovery."),
        (["skills"], "Install AI agent skill files for Nemo."),
        (["services"], "Run platform services locally."),
    ],
)
def test_no_arg_help_exits_successfully(argv: list[str], expected_text: str):
    runner = CliRunner()
    result = runner.invoke(app, argv)

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert expected_text in result.stdout


def test_root_no_arg_help_includes_active_context_and_workspace(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
current_context: production
clusters:
  - name: production
    base_url: https://api.example.com
users:
  - name: production
    type: no-auth
contexts:
  - name: production
    cluster: production
    user: production
    workspace: prod-ns
"""
    )
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))

    runner = CliRunner()
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "Active context: production (workspace: prod-ns)" in result.stdout
    assert result.stdout.count("Active context:") == 1
    assert result.stdout.index("Active context:") < result.stdout.index("Usage:")


def test_root_help_includes_active_context_and_workspace(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
current_context: production
clusters:
  - name: production
    base_url: https://api.example.com
users:
  - name: production
    type: no-auth
contexts:
  - name: production
    cluster: production
    user: production
    workspace: prod-ns
"""
    )
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Active context: production (workspace: prod-ns)" in result.stdout
    assert result.stdout.count("Active context:") == 1
    assert result.stdout.index("Active context:") < result.stdout.index("Usage:")


def test_root_no_arg_help_honors_context_env_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
current_context: production
clusters:
  - name: production
    base_url: https://api.example.com
  - name: staging
    base_url: https://staging.example.com
users:
  - name: production
    type: no-auth
  - name: staging
    type: no-auth
contexts:
  - name: production
    cluster: production
    user: production
    workspace: prod-ns
  - name: staging
    cluster: staging
    user: staging
    workspace: staging-ns
"""
    )
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("NMP_CURRENT_CONTEXT", "staging")

    runner = CliRunner()
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "Active context: staging (workspace: staging-ns)" in result.stdout
    assert "Active context: production" not in result.stdout


def test_root_no_arg_help_skips_active_context_when_config_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("NMP_CONFIG_FILE", str(tmp_path / "missing.yaml"))

    runner = CliRunner()
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "Active context:" not in result.stdout


def test_generated_api_group_no_arg_help_exits_successfully():
    runner = CliRunner()
    qs_config = QuickstartConfig(auth_enabled=False)

    with patch("nemo_platform_ext.quickstart.QuickstartConfig.load", return_value=qs_config):
        result = runner.invoke(app, ["files"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "Manage files" in result.stdout


def test_root_help_includes_lazy_api_commands():
    runner = CliRunner()
    sys.modules.pop("nemo_platform_ext.cli.commands.api.entities", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.api.experiments", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.api.files", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.api.intake", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.auth", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.use_cases.chat", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.config", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.skills.cli", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.quickstart.cli", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.services.cli", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.use_cases.wait", None)
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "experiments" in result.stdout
    assert "Manage experiments." in result.stdout
    assert "files" in result.stdout
    assert "Manage files" in result.stdout
    assert "intake" in result.stdout
    assert "Intake operations." in result.stdout
    assert "entities" not in result.stdout
    assert "nemo_platform_ext.cli.commands.api.entities" not in sys.modules
    assert "nemo_platform_ext.cli.commands.api.experiments" not in sys.modules
    assert "nemo_platform_ext.cli.commands.api.files" not in sys.modules
    assert "nemo_platform_ext.cli.commands.api.intake" not in sys.modules
    assert "nemo_platform_ext.cli.commands.auth" not in sys.modules
    assert "nemo_platform_ext.cli.commands.use_cases.chat" not in sys.modules
    assert "nemo_platform_ext.cli.commands.config" not in sys.modules
    assert "nemo_platform_ext.cli.commands.skills.cli" not in sys.modules
    assert "nemo_platform_ext.cli.commands.quickstart.cli" not in sys.modules
    assert "nemo_platform_ext.cli.commands.services.cli" not in sys.modules
    assert "nemo_platform_ext.cli.commands.use_cases.wait" not in sys.modules


def test_entities_api_command_is_not_registered():
    runner = CliRunner()
    sys.modules.pop("nemo_platform_ext.cli.commands.api.entities", None)

    result = runner.invoke(app, ["entities", "--help"])

    assert result.exit_code != 0
    assert "No such command 'entities'" in result.stderr
    assert "nemo_platform_ext.cli.commands.api.entities" not in sys.modules


def test_members_api_command_is_nested_under_workspaces():
    runner = CliRunner()
    sys.modules.pop("nemo_platform_ext.cli.commands.members", None)
    sys.modules.pop("nemo_platform_ext.cli.commands.workspaces", None)

    result = runner.invoke(app, ["workspaces", "members", "--help"])

    assert result.exit_code == 0
    assert "Manage members" in result.stdout
    assert "nemo_platform_ext.cli.commands.workspaces" in sys.modules
    assert "nemo_platform_ext.cli.commands.members" not in sys.modules


def test_members_api_command_is_not_registered_at_top_level():
    runner = CliRunner()
    sys.modules.pop("nemo_platform_ext.cli.commands.members", None)

    result = runner.invoke(app, ["members", "--help"])

    assert result.exit_code != 0
    assert "No such command 'members'" in result.stderr
    assert "nemo_platform_ext.cli.commands.members" not in sys.modules


def test_jobs_watch_command_is_registered():
    runner = CliRunner()
    result = runner.invoke(app, ["jobs", "watch", "--help"])

    assert result.exit_code == 0
    assert "Watch a platform job until it reaches a terminal status." in result.stdout
    assert "--history" in result.stdout
    assert "--no-history" in result.stdout


def test_jobs_tail_command_is_registered():
    runner = CliRunner()
    result = runner.invoke(app, ["jobs", "tail", "--help"])

    assert result.exit_code == 0
    assert "Print the newest lines from a platform job log." in result.stdout
    assert "--lines" in result.stdout
    assert "-n" in result.stdout


def test_jobs_create_exposes_wait_and_watch_flags():
    runner = CliRunner()
    result = runner.invoke(app, ["jobs", "create", "--help"])

    assert result.exit_code == 0
    assert "--watch" in result.stdout
    assert "--wait" in result.stdout


def test_inference_deployments_create_exposes_wait_and_watch_flags():
    runner = CliRunner()
    result = runner.invoke(app, ["inference", "deployments", "create", "--help"])

    assert result.exit_code == 0
    assert "--watch" in result.stdout
    assert "--wait" in result.stdout
    assert "up and running" in result.stdout
    assert "until it is stable" in result.stdout


def test_root_help_excludes_hidden_commands_and_context_option():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--context" not in result.stdout
    assert "\n  auth" in result.stdout
    for hidden_command in ("config", "quickstart", "cluster-info", "adapters", "projects"):
        assert f"\n  {hidden_command}" not in result.stdout


def test_auth_command_and_hidden_context_option_remain_invokable():
    runner = CliRunner()
    qs_config = QuickstartConfig(auth_enabled=False)

    with patch("nemo_platform_ext.quickstart.QuickstartConfig.load", return_value=qs_config):
        result = runner.invoke(app, ["--context", "dev", "auth", "--help"])

    assert result.exit_code == 0
    assert "Manage authentication for NeMo Platform" in result.stdout


def test_lazy_api_group_help_loads_on_demand():
    runner = CliRunner()
    sys.modules.pop("nemo_platform_ext.cli.commands.api.files", None)
    result = runner.invoke(app, ["files", "--help"])

    assert result.exit_code == 0
    assert "Manage files" in result.stdout
    assert "filesets" in result.stdout
    assert "nemo_platform_ext.cli.commands.api.files" in sys.modules


@pytest.mark.parametrize(
    ("argv", "module_name", "expected_text"),
    [
        (["auth", "--help"], "nemo_platform_ext.cli.commands.auth", "Manage authentication for NeMo Platform"),
        (["config", "--help"], "nemo_platform_ext.cli.commands.config", "Manage NeMo Platform CLI configuration"),
        (["skills", "--help"], "nemo_platform_ext.cli.commands.skills.cli", "Install AI agent skill files for Nemo"),
        (
            ["quickstart", "--help"],
            "nemo_platform_ext.cli.commands.quickstart.cli",
            "Quickstart commands for managing the NeMo Platform container",
        ),
        (["services", "--help"], "nemo_platform_ext.cli.commands.services.cli", "Run platform services locally"),
        (
            ["cluster-info", "--help"],
            "nemo_platform_ext.cli.commands.quickstart.cli",
            "Show information about the connected platform cluster",
        ),
        (
            ["wait", "--help"],
            "nemo_platform_ext.cli.commands.use_cases.wait",
            "Wait for resources to reach a desired status",
        ),
    ],
)
def test_lazy_manual_group_help_loads_on_demand(argv: list[str], module_name: str, expected_text: str):
    runner = CliRunner()
    sys.modules.pop(module_name, None)

    result = runner.invoke(app, argv)

    assert result.exit_code == 0
    assert expected_text in result.stdout
    assert module_name in sys.modules


def test_lazy_top_level_command_help_loads_on_demand():
    runner = CliRunner()
    sys.modules.pop("nemo_platform_ext.cli.commands.use_cases.chat", None)

    result = runner.invoke(app, ["chat", "--help"])

    assert result.exit_code == 0
    assert "Start an interactive chat session with a model" in result.stdout
    assert "Options:" in result.stdout
    assert "nemo_platform_ext.cli.commands.use_cases.chat" in sys.modules


def test_lazy_top_level_command_preserves_loaded_hidden_flag():
    def callback() -> None:
        pass

    attach_lazy_entries(
        callback,
        (
            TopLevelEntry(
                import_path="fake.module:hidden_command",
                name="hidden-command",
                help="Hidden command.",
                panel="CLI functions",
                kind="command",
            ),
        ),
    )
    group = ManifestBackedNmpGroup(name="nemo", callback=callback)
    loaded_command = click.Command("hidden-command", hidden=True)

    with patch("nemo_platform_ext.cli.core.lazy_load.build_lazy_loader", return_value=lambda: loaded_command):
        command = group.get_command(click.Context(group), "hidden-command")

    assert command is not None
    assert command.hidden is True


def test_lazy_group_help_uses_loaded_group_help():
    runner = CliRunner()
    result = runner.invoke(app, ["docs", "--help"])

    assert result.exit_code == 0
    assert "Read NeMo Platform documentation." in result.stdout
    assert "Read NeMo Platform documentation from the CLI." not in result.stdout
    assert "--list" in result.stdout


def test_build_top_level_lazy_entries_prefers_plugin_over_api_name_collision():
    from nemo_platform_ext.cli.app import _build_top_level_lazy_entries

    plugin_entry_points = {
        "custom-plugin": SimpleNamespace(value="nemo_custom_plugin.cli:CustomPluginCLI"),
    }
    api_entries = (
        TopLevelEntry(
            import_path="nemo_platform_ext.cli.commands.api.custom_plugin:app",
            name="custom-plugin",
            help="Custom plugin operations.",
            panel="Functional plugins",
            kind="group",
        ),
        TopLevelEntry(
            import_path="nemo_platform_ext.cli.commands.api.files:app",
            name="files",
            help="Manage files.",
            panel="Core plugins",
            kind="group",
        ),
    )

    with (
        patch("nemo_platform_ext.cli.app.TOP_LEVEL_ENTRIES", ()),
        patch("nemo_platform_ext.cli.app.API_TOP_LEVEL_ENTRIES", api_entries),
        patch(
            "nemo_platform_ext.cli.app._installed_plugin_command_entry_points",
            return_value=plugin_entry_points,
        ),
    ):
        entries = _build_top_level_lazy_entries()

    by_name = {entry.name: entry for entry in entries}
    assert set(by_name) == {"files", "custom-plugin"}
    assert by_name["files"].source == "module"
    assert by_name["custom-plugin"].source == "plugin"


def test_plugin_entry_point_name_collision_is_skipped(caplog):
    entries = (
        TopLevelEntry(
            import_path="nemo_platform_ext.cli.commands.api.files:app",
            name="files",
            help="Manage files.",
            panel="Core plugins",
            kind="group",
        ),
    )
    plugin_entry_points = {
        "files": SimpleNamespace(value="plugin.module:FilesCLI"),
        "example": SimpleNamespace(value="plugin.module:ExampleCLI"),
    }

    result = build_top_level_entries(entries, plugin_entry_points, include_hidden=True)

    assert [(entry.name, entry.source) for entry in result] == [("files", "module"), ("example", "plugin")]
    assert result[1].hidden is False
    assert "collides with a top-level command" in caplog.text


def test_example_plugin_entry_point_is_visible_when_installed():
    plugin_entry_points = {
        "example": SimpleNamespace(value="plugin.module:ExampleCLI"),
        "anonymizer": SimpleNamespace(value="plugin.module:AnonymizerCLI"),
    }

    visible_entries = build_top_level_entries((), plugin_entry_points, include_hidden=False)
    all_entries = build_top_level_entries((), plugin_entry_points, include_hidden=True)

    assert [(entry.name, entry.hidden) for entry in visible_entries] == [("anonymizer", False), ("example", False)]
    assert [(entry.name, entry.hidden) for entry in all_entries] == [("anonymizer", False), ("example", False)]


def test_evaluator_plugin_entry_point_has_deliberate_order_before_unknown_plugins():
    plugin_entry_points = {
        "aardvark": SimpleNamespace(value="plugin.module:AardvarkCLI"),
        "evaluator": SimpleNamespace(value="plugin.module:EvaluatorCLI"),
        "zeta": SimpleNamespace(value="plugin.module:ZetaCLI"),
    }

    visible_entries = build_top_level_entries((), plugin_entry_points, include_hidden=False)

    assert [entry.name for entry in visible_entries] == ["evaluator", "aardvark", "zeta"]


@pytest.mark.parametrize(
    "entry",
    [e for e in TOP_LEVEL_ENTRIES if e.source != "plugin"],
    ids=lambda entry: entry.name,
)
def test_manifest_help_matches_loaded_manual_entry(entry):
    loader = lazy_group_loader(entry.import_path) if entry.kind == "group" else lazy_command_loader(entry.import_path)

    loaded = loader()

    assert loaded.help == entry.help


def test_plugin_loader_registers_unavailable_command_for_broken_jobs():
    plugin_app = typer.Typer(help="Plugin help")

    class _PluginCLI(NemoCLI):
        name = "example"

        def get_cli(self) -> typer.Typer:
            return plugin_app

    good_job = MagicMock()
    good_job.load.return_value = object()
    bad_job = MagicMock()
    bad_job.load.side_effect = RuntimeError("boom")
    bad_job.value = "fake.module:BadJob"
    entry_point = MagicMock()
    entry_point.load.return_value = _PluginCLI

    with (
        patch("nemo_platform_ext.cli.app._add_plugin_job_commands") as mock_add_jobs,
        patch(
            "nemo_platform_ext.cli.app._discover_plugin_job_entry_points",
            return_value={"example.bad": bad_job, "example.good": good_job, "other.job": MagicMock()},
        ),
        patch("nemo_platform_ext.cli.core.lazy_load.resolve_name", return_value=_PluginCLI),
    ):
        loaded = lazy_plugin_loader("example", "fake.module:PluginCLI")()

    assert isinstance(loaded, click.Command)
    mock_add_jobs.assert_called_once()
    assert mock_add_jobs.call_args[0][1] == {"example.good": good_job.load.return_value}
    result = ClickCliRunner().invoke(loaded, ["bad"])
    assert result.exit_code != 0
    assert "Plugin job command 'bad' is unavailable due to import error: boom" in result.output
    result = ClickCliRunner().invoke(loaded, ["bad", "run", "--spec", "{}"])
    assert result.exit_code != 0
    assert "Plugin job command 'bad' is unavailable due to import error: boom" in result.output
    assert "No such option: --spec" not in result.output


def test_plugin_loader_surfaces_broken_cli_error():
    with patch("nemo_platform_ext.cli.core.lazy_load.resolve_name", side_effect=RuntimeError("broken")):
        with pytest.raises(click.ClickException, match="Failed to load plugin commands for 'example': broken"):
            lazy_plugin_loader("example", "fake.module:PluginCLI")()


def test_plugin_loader_surfaces_customization_contributor_discovery_error():
    from nemo_platform_plugin.customization_contributor import CustomizationContributorDiscoveryError

    class _BrokenCustomizationCLI(NemoCLI):
        name = "customization"

        def __init__(self) -> None:
            raise CustomizationContributorDiscoveryError("no contributors were discovered")

        def get_cli(self) -> typer.Typer:
            return typer.Typer()

    with patch("nemo_platform_ext.cli.core.lazy_load.resolve_name", return_value=_BrokenCustomizationCLI):
        with pytest.raises(click.ClickException, match="no contributors were discovered"):
            lazy_plugin_loader("customization", "fake.module:BrokenCustomizationCLI")()


def test_token_refresh_skipped_when_quickstart_auth_disabled():
    """Token refresh should not run when the quickstart config has auth disabled."""
    runner = CliRunner()
    qs_config = QuickstartConfig(auth_enabled=False)

    with (
        patch("nemo_platform_ext.quickstart.QuickstartConfig.load", return_value=qs_config),
        patch("nemo_platform_ext.cli.commands.auth.ensure_valid_token") as mock_ensure,
    ):
        runner.invoke(app, ["workspaces", "--help"])

    mock_ensure.assert_not_called()


def test_token_refresh_runs_when_quickstart_auth_enabled():
    """Token refresh should run when the quickstart config has auth enabled."""
    runner = CliRunner()
    qs_config = QuickstartConfig(auth_enabled=True)

    with (
        patch("nemo_platform_ext.quickstart.QuickstartConfig.load", return_value=qs_config),
        patch("nemo_platform_ext.cli.commands.auth.ensure_valid_token", return_value=True) as mock_ensure,
        patch("nemo_platform_ext.cli.core.context.CLIContext.get_sdk_context", return_value=MagicMock()),
        patch("nemo_platform_ext.cli.core.context.CLIContext.reset_sdk_context"),
    ):
        runner.invoke(app, ["workspaces", "--help"])

    mock_ensure.assert_called_once()
