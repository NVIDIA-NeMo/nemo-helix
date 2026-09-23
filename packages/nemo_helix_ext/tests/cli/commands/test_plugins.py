# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo plugins commands."""

from __future__ import annotations

from unittest.mock import patch

from nemo_helix_ext.cli.app import app
from nemo_helix_ext.quickstart.config import QuickstartConfig
from nemo_helix_plugin.interface import PluginManifest
from typer.testing import CliRunner

from ..utils import assert_exit_code

runner = CliRunner()

_qs_no_auth = QuickstartConfig(auth_enabled=False)
_qs_auth = QuickstartConfig(auth_enabled=True)


def _invoke(*args: str):
    with patch("nemo_helix_ext.quickstart.QuickstartConfig.load", return_value=_qs_no_auth):
        return runner.invoke(app, list(args))


class TestPluginsHelp:
    def test_plugins_help(self):
        result = _invoke("plugins", "--help")
        assert_exit_code(result, 0)
        assert "Commands for plugin discovery." in result.stdout
        assert "list" in result.stdout

    def test_plugins_no_args_shows_help(self):
        result = _invoke("plugins")
        assert_exit_code(result, 0)
        assert "Commands for plugin discovery." in result.stdout
        assert "list" in result.stdout

    def test_plugins_list_help(self):
        result = _invoke("plugins", "list", "--help")
        assert_exit_code(result, 0)
        assert "List installed plugins." in result.stdout
        assert "--output-format" in result.stdout
        assert "--output" in result.stdout
        assert "--stream" in result.stdout


class TestPluginsList:
    def test_list_shows_installed_plugins(self):
        manifests = {
            "zeta": PluginManifest(name="zeta", version="2.0.0", description="Zeta plugin"),
            "alpha": PluginManifest(name="alpha", version="1.0.0", description="Alpha plugin"),
        }

        with patch("nemo_helix_plugin.discovery.discover_manifests", return_value=manifests):
            result = _invoke("plugins", "list")

        assert_exit_code(result, 0)
        assert "alpha" in result.stdout
        assert "zeta" in result.stdout
        assert "1.0.0" in result.stdout
        assert "Zeta plugin" in result.stdout

    def test_list_sorts_by_plugin_name(self):
        manifests = {
            "zeta": PluginManifest(name="zeta", version="2.0.0", description="Zeta plugin"),
            "alpha": PluginManifest(name="alpha", version="1.0.0", description="Alpha plugin"),
            "beta": PluginManifest(name="beta", version="1.5.0", description="Beta plugin"),
        }

        with patch("nemo_helix_plugin.discovery.discover_manifests", return_value=manifests):
            result = _invoke("plugins", "list", "-f", "json")

        assert_exit_code(result, 0)
        assert result.stdout.index('"name": "alpha"') < result.stdout.index('"name": "beta"')
        assert result.stdout.index('"name": "beta"') < result.stdout.index('"name": "zeta"')

    def test_list_json_output(self):
        manifests = {
            "example": PluginManifest(name="example", version="0.1.0", description="Example plugin"),
        }

        with patch("nemo_helix_plugin.discovery.discover_manifests", return_value=manifests):
            result = _invoke("plugins", "list", "-f", "json")

        assert_exit_code(result, 0)
        assert '"name": "example"' in result.stdout
        assert '"version": "0.1.0"' in result.stdout
        assert '"description": "Example plugin"' in result.stdout

    def test_list_accepts_output_alias(self):
        manifests = {
            "example": PluginManifest(name="example", version="0.1.0", description="Example plugin"),
        }

        with patch("nemo_helix_plugin.discovery.discover_manifests", return_value=manifests):
            result = _invoke("plugins", "list", "--output", "json")

        assert_exit_code(result, 0)
        assert '"name": "example"' in result.stdout

    def test_list_streams_json_lines(self):
        manifests = {
            "beta": PluginManifest(name="beta", version="1.5.0", description="Beta plugin"),
            "alpha": PluginManifest(name="alpha", version="1.0.0", description="Alpha plugin"),
        }

        with patch("nemo_helix_plugin.discovery.discover_manifests", return_value=manifests):
            result = _invoke("plugins", "list", "--output", "json", "--stream")

        assert_exit_code(result, 0)
        assert result.stdout.splitlines() == [
            '{"name":"alpha","version":"1.0.0","description":"Alpha plugin"}',
            '{"name":"beta","version":"1.5.0","description":"Beta plugin"}',
        ]

    def test_stream_requires_json_output(self):
        with patch("nemo_helix_plugin.discovery.discover_manifests") as mock_discover:
            result = _invoke("plugins", "list", "--output", "table", "--stream")

        assert_exit_code(result, 2)
        assert "--stream requires --output json or --output raw" in result.stderr
        mock_discover.assert_not_called()

    def test_list_empty_result(self):
        with patch("nemo_helix_plugin.discovery.discover_manifests", return_value={}):
            result = _invoke("plugins", "list", "-f", "json")

        assert_exit_code(result, 0)
        assert result.stdout.strip() == "[]"

    def test_list_output_columns_for_table(self):
        manifests = {
            "example": PluginManifest(name="example", version="0.1.0", description="Example plugin"),
        }

        with patch("nemo_helix_plugin.discovery.discover_manifests", return_value=manifests):
            result = _invoke("plugins", "list", "--output-columns", "name")

        assert_exit_code(result, 0)
        assert "example" in result.stdout
        assert "0.1.0" not in result.stdout
        assert "Example plugin" not in result.stdout

    def test_list_output_columns_warns_for_json(self):
        manifests = {
            "example": PluginManifest(name="example", version="0.1.0", description="Example plugin"),
        }

        with patch("nemo_helix_plugin.discovery.discover_manifests", return_value=manifests):
            result = _invoke("plugins", "list", "-f", "json", "--output-columns", "name")

        assert_exit_code(result, 0)
        assert "Note: --output-columns is not used with `--output-format json`." in result.stderr


class TestPluginsAuthBehavior:
    def test_plugins_list_skips_token_refresh_when_auth_enabled(self):
        with (
            patch("nemo_helix_ext.quickstart.QuickstartConfig.load", return_value=_qs_auth),
            patch("nemo_helix_ext.cli.commands.auth.ensure_valid_token") as mock_ensure,
            patch("nemo_helix_plugin.discovery.discover_manifests", return_value={}),
        ):
            result = runner.invoke(app, ["plugins", "list", "-f", "json"])

        assert_exit_code(result, 0)
        mock_ensure.assert_not_called()

    def test_plugins_help_skips_token_refresh_when_auth_enabled(self):
        with (
            patch("nemo_helix_ext.quickstart.QuickstartConfig.load", return_value=_qs_auth),
            patch("nemo_helix_ext.cli.commands.auth.ensure_valid_token") as mock_ensure,
        ):
            result = runner.invoke(app, ["plugins", "--help"])

        assert_exit_code(result, 0)
        mock_ensure.assert_not_called()
