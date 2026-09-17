# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `--benign-suite` wiring on the SDK `run` method and the CLI `run` command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _doubles import make_async_sdk, make_sdk
from nemo_agent_hardener_plugin.cli import _shared, war_game
from typer.testing import CliRunner


def _write_suite(path: Path) -> Path:
    path.write_text("tool,payload,label,rationale,persona\nt,p,benign,r,pe\n", encoding="utf-8")
    return path


def test_sdk_run_uploads_benign_suite_into_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured: dict[str, Any] = {}

    class _Scheduler:
        def run_local(self, _job: Any, spec: dict, **kwargs: Any) -> dict:
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return {"status": "completed"}

    monkeypatch.setattr(sdk_module, "NemoJobScheduler", _Scheduler)
    monkeypatch.setattr(
        sdk_module, "upload_file_to_fileset", lambda _sdk, path, *, workspace: f"{workspace}/uploaded-{path.name}"
    )

    suite = _write_suite(tmp_path / "suite.csv")
    resource = sdk_module.AgentHardenerPluginResource(make_sdk())
    resource.run(config="agent-hardener.yaml", benign_suite=str(suite), workspace="ws1")

    assert captured["spec"]["benign_suite_fileset"] == "ws1/uploaded-suite.csv"
    assert captured["kwargs"]["workspace"] == "ws1"


def test_sdk_run_without_benign_suite_omits_fileset(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured: dict[str, Any] = {}

    class _Scheduler:
        def run_local(self, _job: Any, spec: dict, **kwargs: Any) -> dict:
            captured["spec"] = spec
            return {"status": "completed"}

    monkeypatch.setattr(sdk_module, "NemoJobScheduler", _Scheduler)

    resource = sdk_module.AgentHardenerPluginResource(make_sdk())
    resource.run(config="agent-hardener.yaml")

    assert "benign_suite_fileset" not in captured["spec"]


def test_async_sdk_run_builds_sync_client_and_uploads_benign_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured: dict[str, Any] = {}
    sync_client = SimpleNamespace(name="sync-sdk")

    class _Scheduler:
        def run_local(self, _job: Any, spec: dict, **kwargs: Any) -> dict:
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return {"status": "completed"}

    def _fake_make_sdk(base: str) -> Any:
        captured["base"] = base
        return sync_client

    monkeypatch.setattr(sdk_module, "NemoJobScheduler", _Scheduler)
    monkeypatch.setattr(sdk_module, "make_sdk", _fake_make_sdk)
    monkeypatch.setattr(
        sdk_module, "upload_file_to_fileset", lambda sdk, path, *, workspace: f"{workspace}/uploaded-{path.name}"
    )

    suite = _write_suite(tmp_path / "suite.csv")
    async_platform = make_async_sdk(base_url="http://localhost:8080/")
    resource = sdk_module.AsyncAgentHardenerPluginResource(async_platform)
    asyncio.run(resource.run(config="agent-hardener.yaml", benign_suite=str(suite), workspace="ws1"))

    assert captured["base"] == "http://localhost:8080/"  # sync client targets the async client's base URL
    assert captured["spec"]["benign_suite_fileset"] == "ws1/uploaded-suite.csv"
    assert captured["kwargs"]["workspace"] == "ws1"
    assert captured["kwargs"]["sdk"] is sync_client  # job runs against the sync client, not async_sdk


def _patch_cli(cli_main: Any, monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> Any:
    fake_sdk = SimpleNamespace(
        agent_hardener=SimpleNamespace(run=lambda **kwargs: captured.update(kwargs) or {"status": "completed"})
    )
    monkeypatch.setattr(_shared.checks, "require_preflight", lambda _c: None)
    monkeypatch.setattr(_shared, "make_sdk", lambda _u: fake_sdk)
    monkeypatch.setattr(_shared, "base_url", lambda: "http://localhost:8080")
    monkeypatch.setattr(war_game, "missing_secrets", lambda _p, env_files: [])
    monkeypatch.setattr(
        _shared.AgentHardenerConfig,
        "get",
        classmethod(lambda _cls: SimpleNamespace(default_workspace="default", operator_env_file=Path(".env"))),
    )
    return cli_main.AgentHardenerCLI().get_cli()


def test_cli_run_forwards_benign_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    config = tmp_path / "agent-hardener.yaml"
    config.write_text("agent: {}\n", encoding="utf-8")
    suite = _write_suite(tmp_path / "suite.csv")

    captured: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, captured)
    result = CliRunner().invoke(app, ["run", "--config", str(config), "--benign-suite", str(suite)])

    assert result.exit_code == 0, result.output
    assert captured["benign_suite"] == str(suite)


def test_cli_run_rejects_missing_benign_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    config = tmp_path / "agent-hardener.yaml"
    config.write_text("agent: {}\n", encoding="utf-8")

    captured: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, captured)
    result = CliRunner().invoke(app, ["run", "--config", str(config), "--benign-suite", str(tmp_path / "nope.csv")])

    assert result.exit_code == 1
    assert "not found" in result.output
    assert captured == {}  # bailed before invoking the SDK


def test_sdk_run_puts_per_run_overrides_in_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-launch overrides reach the job spec so a run can deviate without editing the manifest."""
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured: dict[str, Any] = {}

    class _Scheduler:
        def run_local(self, _job: Any, spec: dict, **kwargs: Any) -> dict:
            captured["spec"] = spec
            return {"status": "completed"}

    monkeypatch.setattr(sdk_module, "NemoJobScheduler", _Scheduler)

    resource = sdk_module.AgentHardenerPluginResource(make_sdk())
    resource.run(
        manifest_id="finance",
        rounds=3,
        port=9000,
        defenders=["guardrails"],
        attack_intensity="thorough",
        replay_hitlog_fileset="ws/hits",
    )

    spec = captured["spec"]
    assert spec["rounds"] == 3
    assert spec["port"] == 9000
    assert spec["defenders"] == ["guardrails"]
    assert spec["attack_intensity"] == "thorough"
    assert spec["replay_hitlog_fileset"] == "ws/hits"


def test_sdk_run_omits_unset_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset override must stay absent, so the manifest's stored default still wins."""
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured: dict[str, Any] = {}

    class _Scheduler:
        def run_local(self, _job: Any, spec: dict, **kwargs: Any) -> dict:
            captured["spec"] = spec
            return {"status": "completed"}

    monkeypatch.setattr(sdk_module, "NemoJobScheduler", _Scheduler)
    sdk_module.AgentHardenerPluginResource(make_sdk()).run(manifest_id="finance")

    for key in ("rounds", "port", "defenders", "attack_intensity", "replay_hitlog_fileset"):
        assert key not in captured["spec"]


def test_sdk_run_rejects_manifest_overrides_with_local_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """These overlay a materialized manifest, so with --config the job would silently discard them."""
    from nemo_agent_hardener_plugin import sdk as sdk_module

    class _Scheduler:
        def run_local(self, _job: Any, spec: dict, **kwargs: Any) -> dict:  # pragma: no cover - must not run
            raise AssertionError("launch should have been rejected before scheduling")

    monkeypatch.setattr(sdk_module, "NemoJobScheduler", _Scheduler)
    resource = sdk_module.AgentHardenerPluginResource(make_sdk())

    with pytest.raises(ValueError, match="cannot be combined with a local 'config' manifest"):
        resource.run(config="agent-hardener.yaml", port=9000)


def test_sdk_run_allows_rounds_with_local_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """rounds is an `agent-hardener run` argument, not a manifest field, so it applies to both paths."""
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured: dict[str, Any] = {}

    class _Scheduler:
        def run_local(self, _job: Any, spec: dict, **kwargs: Any) -> dict:
            captured["spec"] = spec
            return {"status": "completed"}

    monkeypatch.setattr(sdk_module, "NemoJobScheduler", _Scheduler)
    sdk_module.AgentHardenerPluginResource(make_sdk()).run(config="agent-hardener.yaml", rounds=5)

    assert captured["spec"]["rounds"] == 5


def test_cli_run_forwards_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, captured)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--manifest-id",
            "finance",
            "--rounds",
            "3",
            "--defender",
            "guardrails",
            "--defender",
            "openshell",
            "--attack-intensity",
            "thorough",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["rounds"] == 3
    assert captured["defenders"] == ["guardrails", "openshell"]
    assert captured["attack_intensity"] == "thorough"


def test_cli_run_rejects_unknown_attack_intensity(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown preset is read as 'standard' downstream, so it must not reach the job."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, captured)
    result = CliRunner().invoke(app, ["run", "--manifest-id", "finance", "--attack-intensity", "heavy"])

    assert result.exit_code == 1
    assert "light, standard, thorough" in result.output
    assert captured == {}


def test_cli_run_rejects_unknown_defender(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown defender key leaves agent-hardener's full default set in place — never silently."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, captured)
    result = CliRunner().invoke(app, ["run", "--manifest-id", "finance", "--defender", "bogus"])

    assert result.exit_code == 1
    assert "bogus" in result.output
    assert captured == {}


def test_cli_run_reports_override_config_conflict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK's ValueError surfaces as a clean CLI error, not a traceback."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    config = tmp_path / "agent-hardener.yaml"
    config.write_text("agent: {}\n", encoding="utf-8")

    def _raise(**_kwargs: Any) -> dict:
        raise ValueError("port cannot be combined with a local 'config' manifest")

    captured: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, captured)
    monkeypatch.setattr(_shared, "make_sdk", lambda _u: SimpleNamespace(agent_hardener=SimpleNamespace(run=_raise)))
    result = CliRunner().invoke(app, ["run", "--config", str(config), "--port", "9000"])

    assert result.exit_code == 1
    assert "cannot be combined" in result.output


def test_cli_run_forwards_model_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, captured)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--manifest-id",
            "finance",
            "--attack-model",
            "atk/model",
            "--attack-base-url",
            "https://atk/v1",
            "--attack-key-secret",
            "atk-key",
            "--analysis-model",
            "ana/model",
            "--safety-model",
            "guard-1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["models"] == {
        "attack": {"model": "atk/model", "base_url": "https://atk/v1", "api_key_secret": "atk-key"},
        "analysis": {"model": "ana/model"},
        "safety": {"model": "guard-1"},
    }


def test_cli_run_omits_models_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty selection must stay absent so the manifest's stored models still apply."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, captured)
    result = CliRunner().invoke(app, ["run", "--manifest-id", "finance"])

    assert result.exit_code == 0, result.output
    assert captured["models"] is None
