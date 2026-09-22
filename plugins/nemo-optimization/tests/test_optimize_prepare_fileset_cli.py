# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents optimize prepare-fileset`` — staging a bundle for remote submit."""

from __future__ import annotations

import json
import tomllib
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import typer
import yaml
from nemo_optimization.optimize_cli import register_prepare_fileset_command
from typer.testing import CliRunner

CONFIG: dict[str, Any] = {
    "schema_version": "fabric.agent/v1alpha1",
    "metadata": {"name": "hermes-optimize-chatonly"},
    "harness": {"adapter_id": "nvidia.fabric.hermes"},
    "models": {"default": {"provider": "nvidia", "model": "nvidia/meta/llama-3.1-8b-instruct"}},
    "optimizer": {
        "numeric": {"enabled": True, "n_trials": 2},
        "search_space": {"temperature": {"type": "fabric", "path": "models.default.temperature", "values": [0.0]}},
    },
    "eval": {
        "general": {"dataset": {"file_path": "dataset.json"}},
        "fabric": {"base_dir": "."},
    },
}


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    (tmp_path / "optimize.yml").write_text(yaml.safe_dump(CONFIG))
    (tmp_path / "dataset.json").write_text(json.dumps([{"question": "q", "answer": "a"}]))
    return tmp_path


@pytest.fixture
def app() -> typer.Typer:
    """The shared ``optimize`` group, carrying the verb this plugin contributes to it."""
    group = typer.Typer(name="optimize")

    @group.callback()
    def _root() -> None:
        """Force subcommand dispatch."""

    register_prepare_fileset_command(group)
    return group


class _StubSDK:
    pass


def _patch_upload(record: dict[str, Any]) -> ExitStack:
    class _StubManager:
        def validate_storage(self) -> None:
            record["validated"] = True

        def upload(self, *, local_path: Path, remote_path: str) -> None:
            record.update(local_path=local_path, remote_path=remote_path)

    def manager(_files_client: object, *, workspace: str, fileset: str, ensure_fileset_exists: bool) -> _StubManager:
        record.update(
            fileset=fileset,
            workspace=workspace,
            auto_create=ensure_fileset_exists,
        )
        return _StubManager()

    stack = ExitStack()
    stack.enter_context(patch("nemo_optimization.optimize_cli._platform_sdk", return_value=_StubSDK()))
    stack.enter_context(patch("nemo_agents_plugin.jobs.fileset_io.client_from_platform", return_value=object()))
    stack.enter_context(patch("nemo_agents_plugin.jobs.fileset_io._fileset_manager", side_effect=manager))
    return stack


def test_uploads_the_bundle_and_prints_the_submit_command(app: typer.Typer, bundle: Path) -> None:
    record: dict[str, Any] = {}
    with _patch_upload(record):
        result = CliRunner().invoke(
            app,
            [
                "prepare-fileset",
                "--source",
                str(bundle),
                "--optimize-config",
                "optimize.yml",
                "--fileset",
                "my-opt-fs",
                "--no-check-models",
            ],
        )

    assert result.exit_code == 0, result.output
    assert record["fileset"] == "my-opt-fs"
    assert record["workspace"] == "default"
    assert record["auto_create"] is True
    assert record["validated"] is True
    assert record["local_path"] == bundle
    assert record["remote_path"] == ""
    assert "nemo agents optimize run-strategy" in result.output
    assert "--strategy nat" in result.output
    assert "--optimize-config-fileset default/my-opt-fs" in result.output
    assert "--optimize-config optimize.yml" in result.output


def test_honours_a_workspace_qualified_fileset_ref(app: typer.Typer, bundle: Path) -> None:
    record: dict[str, Any] = {}
    with _patch_upload(record):
        result = CliRunner().invoke(
            app,
            [
                "prepare-fileset",
                "--source",
                str(bundle),
                "--optimize-config",
                "optimize.yml",
                "--fileset",
                "team-a/my-opt-fs",
                "--workspace",
                "default",
                "--no-check-models",
            ],
        )

    assert result.exit_code == 0, result.output
    assert (record["workspace"], record["fileset"]) == ("team-a", "my-opt-fs")


def test_refuses_to_upload_a_bundle_that_fails_preflight(app: typer.Typer, bundle: Path) -> None:
    broken = dict(CONFIG)
    broken["eval"] = {"general": {"dataset": {"file_path": "/Users/me/dataset.json"}}}
    (bundle / "optimize.yml").write_text(yaml.safe_dump(broken))

    def _no_sdk(_base_url: str) -> Any:
        raise AssertionError("preflight must fail before the platform is contacted")

    with patch("nemo_optimization.optimize_cli._platform_sdk", side_effect=_no_sdk):
        result = CliRunner().invoke(
            app,
            [
                "prepare-fileset",
                "--source",
                str(bundle),
                "--optimize-config",
                "optimize.yml",
                "--fileset",
                "my-opt-fs",
                "--no-check-models",
            ],
        )

    assert result.exit_code == 1
    assert "eval.general.dataset is an absolute path" in result.output


def test_dry_run_validates_without_uploading(app: typer.Typer, bundle: Path) -> None:
    def _no_sdk(_base_url: str) -> Any:
        raise AssertionError("--dry-run must not contact the platform")

    with patch("nemo_optimization.optimize_cli._platform_sdk", side_effect=_no_sdk):
        result = CliRunner().invoke(
            app,
            [
                "prepare-fileset",
                "--source",
                str(bundle),
                "--optimize-config",
                "optimize.yml",
                "--fileset",
                "my-opt-fs",
                "--no-check-models",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Would upload" in result.output


def test_this_plugin_contributes_only_prepare_fileset() -> None:
    """What the optimize group gets when it calls this plugin's registrar: exactly one verb."""
    group = typer.Typer(name="optimize")
    register_prepare_fileset_command(group)
    assert [command.name for command in group.registered_commands] == ["prepare-fileset"]


def test_the_entry_point_points_at_the_registrar() -> None:
    """Discovery calls whatever this key names; a stale path would drop the verb silently."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        entry_points = tomllib.load(f)["project"]["entry-points"]["nemo.cli.agents.optimize"]

    assert entry_points == {"prepare-fileset": "nemo_optimization.optimize_cli:register_prepare_fileset_command"}
