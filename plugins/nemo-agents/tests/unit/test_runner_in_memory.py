# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``InMemoryRunnerBackend`` log/config layout and lifecycle.

Pin contracts that out-of-process callers (notably the ``nemo agents logs``
CLI) rely on:

- The log file path is deterministic (``<system_dir>/<name>.log``), exposed
  on ``DeploymentInfo.log_path``, and matches the path returned by
  ``log_path_for(name)`` — so the CLI can locate logs without round-trip
  through the API.
- Subprocess exit (``proc.poll() is not None``) is surfaced through
  ``get_deployment_status`` with the exit code in ``DeploymentInfo.error``,
  so the controller can mark deployments failed without waiting for the
  health-check timeout.
- The system dir lives under the configured ``workspace_dir`` (default:
  ``nmp_user_data_dir() / "agents"``), not the plugin source tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest
import yaml
from nemo_agents_plugin.config import AgentsConfig, ControllerConfig
from nemo_agents_plugin.runner.backend import DeploymentInfo
from nemo_agents_plugin.runner.fabric_artifact_staging import FabricArtifactStagingError
from nemo_agents_plugin.runner.in_memory import InMemoryRunnerBackend, _resolve_nat_bin
from nemo_platform_plugin.config import Configuration, nmp_user_data_dir


def _backend(workspace_dir: Path) -> InMemoryRunnerBackend:
    cfg = ControllerConfig(workspace_dir=workspace_dir)
    return InMemoryRunnerBackend(cfg)


# ---------------------------------------------------------------------------
# Default workspace_dir resolves through nmp_user_data_dir()
# ---------------------------------------------------------------------------


def test_default_workspace_dir_is_under_user_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default workspace_dir resolves to ``nmp_user_data_dir() / 'agents'``.

    Earlier versions defaulted to the plugin source root, which leaked
    runtime state into the source tree (and was undocumented).  Artifacts
    now route through the standard NMP user-data location so they survive
    ``/tmp/`` cleanup and live in a well-known place.
    """
    monkeypatch.setenv("NMP_DATA_DIR", str(tmp_path))
    Configuration.clear_cache()
    try:
        cfg = AgentsConfig.get()
        # workspace_dir is computed relative to the user-data root.
        assert cfg.controller.workspace_dir == nmp_user_data_dir() / "agents"
        assert cfg.controller.workspace_dir == tmp_path / "agents"
    finally:
        Configuration.clear_cache()


def test_workspace_dir_follows_xdg_data_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``XDG_DATA_HOME`` shifts the workspace_dir alongside the rest of NMP state."""
    monkeypatch.delenv("NMP_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    Configuration.clear_cache()
    try:
        cfg = AgentsConfig.get()
        assert cfg.controller.workspace_dir == tmp_path / "nemo" / "agents"
    finally:
        Configuration.clear_cache()


def test_workspace_dir_passed_directly_to_controller_config(tmp_path: Path) -> None:
    """Constructing ``ControllerConfig`` with an explicit ``workspace_dir`` honours it."""
    custom = tmp_path / "custom"
    cfg = ControllerConfig(workspace_dir=custom)
    backend = InMemoryRunnerBackend(cfg)
    assert backend.output_base_dir == custom.resolve()
    assert backend.system_dir == custom.resolve() / "system"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_log_path_for_is_deterministic(tmp_path: Path) -> None:
    """``log_path_for`` returns ``<system_dir>/<workspace>/<name>.log``."""
    backend = _backend(tmp_path)
    assert backend.log_path_for("ws", "react-agent-abcd1234") == tmp_path / "system" / "ws" / "react-agent-abcd1234.log"


def test_log_path_and_config_path_share_basename(tmp_path: Path) -> None:
    """Config and log files share a deterministic basename so they pair up."""
    backend = _backend(tmp_path)
    name = "calc-1"
    assert backend.log_path_for("ws", name).stem == backend.config_path_for("ws", name).stem
    assert backend.log_path_for("ws", name).suffix == ".log"
    assert backend.config_path_for("ws", name).suffix == ".yaml"


def test_log_path_separates_workspaces(tmp_path: Path) -> None:
    """Same-named deployments in two workspaces resolve to distinct files."""
    backend = _backend(tmp_path)
    a = backend.log_path_for("ws-a", "foo")
    b = backend.log_path_for("ws-b", "foo")
    assert a != b
    assert a.parent != b.parent


def test_sanitize_name_strips_unsafe_chars(tmp_path: Path) -> None:
    """Pathological deployment names are coerced to a safe filename component."""
    backend = _backend(tmp_path)
    log = backend.log_path_for("ws", "../oops/../escape")
    # No path traversal: the resolved path stays under system_dir.
    assert backend.system_dir in log.resolve().parents
    # Slashes are replaced (so the path can't escape system_dir).
    assert "/" not in log.name


def test_system_dir_is_under_workspace_dir(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    assert backend.system_dir == tmp_path / "system"
    assert backend.output_base_dir == tmp_path


# ---------------------------------------------------------------------------
# _write_config writes deterministically under system_dir
# ---------------------------------------------------------------------------


def test_write_config_uses_deterministic_path(tmp_path: Path) -> None:
    """Repeated writes target the same file (no random suffix)."""
    backend = _backend(tmp_path)
    config = {"workflow": {"_type": "react_agent"}}
    p1 = backend._write_config("ws", "calc", config)
    p2 = backend._write_config("ws", "calc", config)
    assert p1 == p2
    assert p1 == backend.config_path_for("ws", "calc")
    # Round-trip the YAML to confirm contents are correct.
    assert yaml.safe_load(p1.read_text()) == config


def test_write_config_creates_system_dir(tmp_path: Path) -> None:
    backend = _backend(tmp_path / "fresh")
    backend._write_config("ws", "calc", {"a": 1})
    # The workspace-namespaced subdirectory exists; ``system/`` is its parent.
    assert (tmp_path / "fresh" / "system" / "ws").is_dir()


# ---------------------------------------------------------------------------
# create_deployment populates DeploymentInfo.log_path and writes the log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_deployment_records_log_path(tmp_path: Path) -> None:
    """``DeploymentInfo.log_path`` is populated and matches ``log_path_for``."""
    backend = _backend(tmp_path)

    # Replace the spawn step with a no-op fake process that immediately
    # "starts" (poll returns None) so the test doesn't depend on ``nat``.
    class _FakeProc:
        pid = 4242
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    fake = _FakeProc()

    def _fake_spawn(self_, name, config_path, log_path, port):  # noqa: ANN001
        # Touch the log file so callers can locate it — mirrors real spawn.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        return fake

    with patch.object(InMemoryRunnerBackend, "_spawn", _fake_spawn):
        info = await backend.create_deployment("ws", "calc-1", {"workflow": {}}, port=49200)

    assert info.log_path == str(backend.log_path_for("ws", "calc-1"))
    assert Path(info.log_path).exists()
    assert info.pid == 4242
    assert info.status == "starting"


@pytest.mark.asyncio
async def test_create_deployment_validates_platform_agent_config(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "default/test-model",
                "base_url": "http://platform/apis/inference-gateway/v2/workspaces/ws/openai/-/v1",
                "api_key_env": "NVIDIA_API_KEY",
            }
        },
    }
    validation_calls: list[Any] = []

    async def _validate_platform_agent_config(config_: dict[str, Any], *, base_dir: Path) -> Any:
        validation_calls.append({"config": config_, "base_dir": base_dir})
        return SimpleNamespace(agent_config=SimpleNamespace(name="fabric-agent"))

    fake_process = SimpleNamespace(pid=4242, returncode=None, poll=lambda: None)

    def _spawn_fabric(self_, name, config_path, log_path, port, credential_env=None):  # noqa: ANN001
        del self_, name, config_path, port
        assert credential_env == {"NVIDIA_API_KEY": "not-used"}
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        return fake_process

    with (
        patch("nemo_agents_plugin.runner.in_memory.validate_platform_agent_config", _validate_platform_agent_config),
        patch.object(InMemoryRunnerBackend, "_spawn_fabric", _spawn_fabric),
    ):
        info = await backend.create_deployment("ws", "fabric-dep", config, port=49210)

    assert info.status == "starting"
    assert info.endpoint == "http://127.0.0.1:49210"
    assert info.port == 49210
    assert info.pid == 4242
    assert Path(info.log_path).exists()
    assert validation_calls == [{"config": config, "base_dir": tmp_path / "system" / "ws" / "fabric-dep-fabric"}]
    assert yaml.safe_load((Path(info.extra["base_dir"]) / "agent.yaml").read_text()) == config
    status = await backend.get_deployment_status("ws", "fabric-dep")
    assert status is info


@pytest.mark.asyncio
async def test_delete_deployment_removes_fabric_deployment(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
    }

    async def _validate_platform_agent_config(config_: dict[str, Any], *, base_dir: Path) -> Any:
        del config_, base_dir
        return SimpleNamespace(agent_config=SimpleNamespace(name="fabric-agent"))

    fake_process = SimpleNamespace(pid=4242, returncode=None, poll=lambda: None)
    terminate_calls: list[tuple[str, Any]] = []

    def _spawn_fabric(self_, name, config_path, log_path, port, credential_env=None):  # noqa: ANN001
        del self_, name, config_path, port, credential_env
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        return fake_process

    def _terminate(self_, name, proc):  # noqa: ANN001
        del self_
        terminate_calls.append((name, proc))

    with (
        patch("nemo_agents_plugin.runner.in_memory.validate_platform_agent_config", _validate_platform_agent_config),
        patch.object(InMemoryRunnerBackend, "_spawn_fabric", _spawn_fabric),
        patch.object(InMemoryRunnerBackend, "_terminate", _terminate),
    ):
        info = await backend.create_deployment("ws", "fabric-dep", config, port=49211)
        base_dir = Path(info.extra["base_dir"])
        assert base_dir.exists()
        cleaned = await backend.delete_deployment("ws", "fabric-dep")

    assert cleaned is True
    assert terminate_calls == [("fabric-dep", fake_process)]
    assert not base_dir.exists()
    assert await backend.get_deployment_status("ws", "fabric-dep") is None


@pytest.mark.asyncio
async def test_create_deployment_cleans_fabric_base_dir_on_validation_failure(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
    }
    base_dir = tmp_path / "system" / "ws" / "fabric-dep-fabric"

    async def _validate_platform_agent_config(config_: dict[str, Any], *, base_dir: Path) -> Any:
        del config_
        (base_dir / "validation.txt").write_text("created during validation")
        raise ValueError("bad fabric config")

    with patch("nemo_agents_plugin.runner.in_memory.validate_platform_agent_config", _validate_platform_agent_config):
        with pytest.raises(ValueError, match="bad fabric config"):
            await backend.create_deployment("ws", "fabric-dep", config, port=0)

    assert not base_dir.exists()
    assert await backend.get_deployment_status("ws", "fabric-dep") is None


def test_spawn_fabric_uses_current_python_and_platform_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(tmp_path)
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("name: test-agent\n")
    log_path = tmp_path / "agent.log"
    process = SimpleNamespace()
    monkeypatch.setenv("NVIDIA_API_KEY", "real-controller-key")

    with patch("nemo_agents_plugin.runner.in_memory.subprocess.Popen", return_value=process) as popen:
        spawned = backend._spawn_fabric(
            "fabric-dep",
            config_path,
            log_path,
            49212,
            {"NVIDIA_API_KEY": "not-used"},
        )

    assert spawned is process
    popen.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "nemo_agents_plugin.fabric.server",
            "--agent-config",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            "49212",
        ],
        stdout=ANY,
        stderr=subprocess.STDOUT,
        env=ANY,
    )
    assert popen.call_args.kwargs["env"]["NVIDIA_API_KEY"] == "not-used"
    assert os.environ["NVIDIA_API_KEY"] == "real-controller-key"


@pytest.mark.asyncio
async def test_shutdown_removes_fabric_deployment_directory(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    base_dir = backend._fabric_base_dir_for("ws", "fabric-dep")
    base_dir.mkdir(parents=True)
    backend._deployments[("ws", "fabric-dep")] = DeploymentInfo(
        name="fabric-dep",
        extra={"base_dir": str(base_dir)},
    )

    await backend.shutdown()

    assert not base_dir.exists()


# ---------------------------------------------------------------------------
# get_deployment_status surfaces subprocess exit code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_deployment_status_reports_dead_subprocess(tmp_path: Path) -> None:
    """A dead subprocess is reported as ``failed`` with the OS exit code.

    Without this contract the deployment could sit in ``starting`` until
    the health-check timeout elapsed; the controller relies on this to
    fail fast when the spawned process exits during startup.
    """
    backend = _backend(tmp_path)

    class _FakeProc:
        pid = 7777
        returncode = 1

        def poll(self) -> int | None:
            return 1  # always reports exited with code 1

    fake = _FakeProc()

    def _fake_spawn(self_, name, config_path, log_path, port):  # noqa: ANN001
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        return fake

    with patch.object(InMemoryRunnerBackend, "_spawn", _fake_spawn):
        info = await backend.create_deployment("ws", "calc-1", {}, port=49201)
    assert info.status == "starting"

    status = await backend.get_deployment_status("ws", "calc-1")
    assert status is not None
    assert status.status == "failed"
    assert "exited with code 1" in status.error


@pytest.mark.asyncio
async def test_get_deployment_status_alive_subprocess_unchanged(tmp_path: Path) -> None:
    """Backward behaviour: an alive process keeps its ``starting`` status."""
    backend = _backend(tmp_path)

    class _FakeProc:
        pid = 1111
        returncode: int | None = None

        def poll(self) -> int | None:
            return None

    fake = _FakeProc()

    def _fake_spawn(self_, name, config_path, log_path, port):  # noqa: ANN001
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        return fake

    with patch.object(InMemoryRunnerBackend, "_spawn", _fake_spawn):
        await backend.create_deployment("ws", "calc-1", {}, port=49202)

    status = await backend.get_deployment_status("ws", "calc-1")
    assert status is not None
    assert status.status == "starting"
    assert status.error == ""


# ---------------------------------------------------------------------------
# End-to-end: real subprocess that exits 1 — log path is reachable.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_subprocess_exit_writes_log_at_recorded_path(tmp_path: Path) -> None:
    """Spawn a real (trivial) subprocess and verify the log file is at the
    location recorded in ``DeploymentInfo.log_path``.

    Catches regressions where the log path advertised by the backend
    drifts from where the file is actually written — which would
    silently break the ``nemo agents logs`` post-mortem flow.
    """
    backend = _backend(tmp_path)

    # Patch _spawn to run a tiny python one-liner that prints and exits 1.
    def _spawn_python(self_, name, config_path, log_path, port):  # noqa: ANN001
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w")
        try:
            return subprocess.Popen(
                [sys.executable, "-c", "import sys; print('agent boot failed'); sys.exit(1)"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        finally:
            log_file.close()

    with patch.object(InMemoryRunnerBackend, "_spawn", _spawn_python):
        info = await backend.create_deployment("ws", "calc-1", {}, port=49203)

    # Wait briefly for the subprocess to exit.
    proc = backend._processes[("ws", "calc-1")]
    proc.wait(timeout=10)

    log_path = Path(info.log_path)
    assert log_path == backend.log_path_for("ws", "calc-1")
    assert log_path.exists()
    assert "agent boot failed" in log_path.read_text()

    status = await backend.get_deployment_status("ws", "calc-1")
    assert status is not None
    assert status.status == "failed"
    assert "exited with code 1" in status.error


# ---------------------------------------------------------------------------
# _resolve_nat_bin: how the runner finds the `nat` executable.
#
# The runner spawns `nat start fastapi` per deployment.  Resolution must work
# under every supported install path (activated venv, agentic container,
# `uv tool install`, and explicit override) without requiring users to massage
# PATH themselves.
# ---------------------------------------------------------------------------


def test_resolve_nat_bin_uses_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """`shutil.which` is preferred — covers activated venvs and the container."""
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/local/bin/{name}")
    assert _resolve_nat_bin() == "/usr/local/bin/nat"


def test_resolve_nat_bin_uses_sibling_of_sys_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When `nat` is not on PATH, look next to `sys.executable`.

    This is the `uv tool install nemo-platform` case: the tool venv contains
    `nat` (it's co-installed with `nemo` via the `[services]` chain), but the
    venv's `bin/` is not prepended to PATH, so `shutil.which` returns None.
    """
    monkeypatch.setattr("shutil.which", lambda _: None)

    fake_venv_bin = tmp_path / "bin"
    fake_venv_bin.mkdir()
    fake_python = fake_venv_bin / "python"
    fake_python.touch()
    fake_nat = fake_venv_bin / "nat"
    fake_nat.write_text("#!/bin/sh\necho nat-stub\n")
    monkeypatch.setattr(sys, "executable", str(fake_python))

    assert _resolve_nat_bin() == str(fake_nat)


def test_resolve_nat_bin_falls_back_to_container_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When nothing else matches, return the container path. This preserves
    backwards-compatible behavior in the agentic container even if the PATH
    lookup somehow fails inside it."""
    monkeypatch.setattr("shutil.which", lambda _: None)

    # sys.executable points somewhere with no sibling `nat`.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.touch()
    monkeypatch.setattr(sys, "executable", str(fake_python))

    assert _resolve_nat_bin() == "/app/.venv/bin/nat"


@pytest.mark.asyncio
async def test_create_deployment_stages_ethos_fileset_into_base_dir(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
    }
    staged: list[dict[str, Any]] = []

    async def _stage_fabric_ethos_dir(
        *,
        workspace: str,
        agent_name: str,
        agent_config: dict[str, Any],
        base_dir: Path,
        files_client: Any | None,
    ) -> None:
        del files_client
        staged.append({"workspace": workspace, "agent_name": agent_name, "base_dir": base_dir})
        assert not (base_dir / "agent.yaml").exists()
        (base_dir / "mcps").mkdir()
        (base_dir / "mcps" / "calculator.py").write_text("print(1)\n")
        assert agent_config == config

    async def _validate_platform_agent_config(config_: dict[str, Any], *, base_dir: Path) -> Any:
        del config_, base_dir
        return SimpleNamespace(agent_config=SimpleNamespace(name="fabric-agent"))

    fake_process = SimpleNamespace(pid=4343, returncode=None, poll=lambda: None)

    def _spawn_fabric(self_, name, config_path, log_path, port, credential_env=None):  # noqa: ANN001
        del self_, name, config_path, port, credential_env
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        return fake_process

    with (
        patch("nemo_agents_plugin.runner.in_memory.validate_platform_agent_config", _validate_platform_agent_config),
        patch("nemo_agents_plugin.runner.in_memory.stage_fabric_ethos_dir", _stage_fabric_ethos_dir),
        patch("nemo_agents_plugin.runner.in_memory.get_async_platform_sdk", MagicMock()),
        patch("nemo_agents_plugin.runner.in_memory.client_from_platform", return_value=MagicMock()),
        patch.object(InMemoryRunnerBackend, "_spawn_fabric", _spawn_fabric),
    ):
        info = await backend.create_deployment("ws", "fabric-dep", config, port=49212, agent="fabric-agent")

    base_dir = Path(info.extra["base_dir"])
    assert staged == [{"workspace": "ws", "agent_name": "fabric-agent", "base_dir": base_dir}]
    assert (base_dir / "mcps" / "calculator.py").exists()
    assert yaml.safe_load((base_dir / "agent.yaml").read_text()) == config


@pytest.mark.asyncio
async def test_create_deployment_cleans_base_dir_when_staging_fails(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
    }
    base_dir = tmp_path / "system" / "ws" / "fabric-dep-fabric"

    async def _stage_fabric_ethos_dir(**kwargs: Any) -> None:
        del kwargs
        raise FabricArtifactStagingError("skills/review missing")

    with (
        patch("nemo_agents_plugin.runner.in_memory.stage_fabric_ethos_dir", _stage_fabric_ethos_dir),
        patch("nemo_agents_plugin.runner.in_memory.get_async_platform_sdk", MagicMock()),
        patch("nemo_agents_plugin.runner.in_memory.client_from_platform", return_value=MagicMock()),
    ):
        with pytest.raises(FabricArtifactStagingError, match="skills/review"):
            await backend.create_deployment("ws", "fabric-dep", config, port=0, agent="fabric-agent")

    assert not base_dir.exists()
    assert await backend.get_deployment_status("ws", "fabric-dep") is None


@pytest.mark.asyncio
async def test_redeploy_after_crash_does_not_merge_previous_fileset(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "fabric-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
    }

    def _files_client_serving(skill_name: str) -> SimpleNamespace:
        async def _download(*, local_path: str, fileset: str | None = None, workspace: str | None = None) -> None:
            del fileset, workspace
            skills = Path(local_path) / "skills"
            skills.mkdir(parents=True, exist_ok=True)
            (skills / skill_name).write_text("# skill\n")

        return SimpleNamespace(download=_download)

    async def _download_fileset(
        files_client: Any,
        *,
        workspace: str,
        fileset_name: str,
        local_path: Path,
    ) -> None:
        await files_client.download(local_path=str(local_path), fileset=fileset_name, workspace=workspace)

    async def _validate(config_: dict[str, Any], *, base_dir: Path) -> Any:
        del config_, base_dir
        return SimpleNamespace(agent_config=SimpleNamespace(name="fabric-agent"))

    fake_process = SimpleNamespace(pid=5150, returncode=None, poll=lambda: None)

    def _spawn_fabric(self_, name, config_path, log_path, port, credential_env=None):  # noqa: ANN001
        del self_, name, config_path, port, credential_env
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        return fake_process

    base_dirs: list[Path] = []
    for skill_name in ("old.md", "new.md"):
        files_client = _files_client_serving(skill_name)
        with (
            patch("nemo_agents_plugin.runner.in_memory.validate_platform_agent_config", _validate),
            patch(
                "nemo_agents_plugin.runner.in_memory.get_async_platform_sdk",
                return_value=MagicMock(),
            ),
            patch("nemo_agents_plugin.runner.in_memory.client_from_platform", return_value=files_client),
            patch("nemo_agents_plugin.runner.fabric_artifact_staging._download_fileset", _download_fileset),
            patch.object(InMemoryRunnerBackend, "_spawn_fabric", _spawn_fabric),
        ):
            info = await backend.create_deployment("ws", "dep", config, port=49300, agent="fabric-agent")
        base_dirs.append(Path(info.extra["base_dir"]))
        # Crash case: in-memory state disappears without delete_deployment's rmtree.
        backend._processes.pop(("ws", "dep"), None)
        backend._deployments.pop(("ws", "dep"), None)

    assert base_dirs[0] == base_dirs[1]
    assert sorted(path.name for path in (base_dirs[1] / "skills").iterdir()) == ["new.md"]
