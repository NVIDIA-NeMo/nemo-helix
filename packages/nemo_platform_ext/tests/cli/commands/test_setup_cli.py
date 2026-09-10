# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI integration tests for ``nemo setup`` connection selection."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from nemo_platform_ext.cli.app import app
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.config.config import Config
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from typer.testing import CliRunner

SETUP_MOD = "nemo_platform_ext.cli.commands.setup"


class _StubWorkspaces:
    """Workspaces typed-client stub whose calls always succeed."""

    def get_workspace(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(data=lambda: object())

    def create_workspace(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(data=lambda: object())


def _typed_client(client_cls: type, timeout: float = 60.0) -> object:
    """Stand in for ``CLIContext.typed_client``: real workspace stub, mocks for everything else."""
    if client_cls is WorkspacesClient:
        return _StubWorkspaces()
    return MagicMock()


def test_setup_help_documents_resume_flag() -> None:
    result = CliRunner().invoke(app, ["setup", "--help"])

    assert result.exit_code == 0, result.output
    assert "--resume" in result.output
    assert "Retry an interrupted setup using the normal idempotent setup path." in result.output


def test_remote_choice_retries_and_persists_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))

    with (
        patch(f"{SETUP_MOD}.is_interactive", return_value=True),
        patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
        patch(f"{SETUP_MOD}.prompt_choice", return_value="remote"),
        patch(
            f"{SETUP_MOD}.prompt_text",
            side_effect=["https://unreachable.example.com", "https://remote.example.com/"],
        ),
        patch(
            f"{SETUP_MOD}._check_platform_reachable_with_retries",
            side_effect=[False, True, True],
        ),
        patch(f"{SETUP_MOD}._ensure_platform_auth") as ensure_auth,
        patch(f"{SETUP_MOD}._start_services_background") as start_services,
        patch.object(CLIContext, "typed_client", side_effect=_typed_client),
        patch(f"{SETUP_MOD}._run_interactive_mode") as run_interactive,
    ):
        result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 0, result.output
    assert "Unable to connect to NeMo Platform at https://unreachable.example.com" in result.output
    context = Config.load(config_path=config_path).resolve()
    assert str(context.cluster.base_url) == "https://remote.example.com/"
    start_services.assert_not_called()
    ensure_auth.assert_called_once()
    assert run_interactive.call_args.args[3] == "https://remote.example.com"


def test_local_choice_starts_services_and_keeps_local_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))

    process = MagicMock(pid=1234)

    with (
        patch(f"{SETUP_MOD}.is_interactive", return_value=True),
        patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
        patch(f"{SETUP_MOD}.prompt_choice", return_value="yes"),
        patch(f"{SETUP_MOD}._prompt_data_dir", return_value="/tmp/nemo-demo"),
        patch(f"{SETUP_MOD}.importlib.util.find_spec", return_value=MagicMock()),
        patch(f"{SETUP_MOD}._ensure_port_available_for_start"),
        patch(f"{SETUP_MOD}._start_services_background", return_value=process) as start_services,
        patch(f"{SETUP_MOD}._wait_for_platform", return_value=True),
        patch(f"{SETUP_MOD}._check_platform_reachable_with_retries", return_value=True),
        patch.object(CLIContext, "typed_client", side_effect=_typed_client),
        patch(f"{SETUP_MOD}._run_interactive_mode") as run_interactive,
    ):
        result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 0, result.output
    context = Config.load(config_path=config_path).resolve()
    assert str(context.cluster.base_url) == "http://localhost:8080/"
    start_services.assert_called_once()
    assert run_interactive.call_args.args[3].rstrip("/") == "http://localhost:8080"


def test_start_myself_exits_without_starting_or_mutating_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))

    with (
        patch(f"{SETUP_MOD}.is_interactive", return_value=True),
        patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
        patch(f"{SETUP_MOD}.prompt_choice", return_value="manual"),
        patch(f"{SETUP_MOD}._start_services_background") as start_services,
    ):
        result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 1
    assert "Start the platform first" in result.output
    assert config_path.read_text() == ""
    start_services.assert_not_called()
