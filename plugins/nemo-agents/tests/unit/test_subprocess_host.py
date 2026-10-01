# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from nemo_agents_plugin.config import AgentsConfig, ControllerConfig
from nemo_agents_plugin.runner.in_memory import InMemoryRunnerBackend
from nemo_agents_plugin.runner.registry import RunnerBackendRegistry
from pydantic import ValidationError


def _backend(tmp_path: Path, host: str | None = None) -> InMemoryRunnerBackend:
    config = ControllerConfig(workspace_dir=tmp_path)
    return InMemoryRunnerBackend(config) if host is None else InMemoryRunnerBackend(config, host=host)


def _spawned_host(cmd: list[str]) -> str:
    return cmd[cmd.index("--host") + 1]


async def _create_nat_deployment(backend: InMemoryRunnerBackend, port: int) -> str:
    def _fake_spawn(self_, name, config_path, log_path, port):  # noqa: ANN001
        return SimpleNamespace(pid=1, poll=lambda: None)

    with patch.object(InMemoryRunnerBackend, "_spawn", _fake_spawn):
        info = await backend.create_deployment("ws", "dep", {"workflow": {}}, port=port)
    return info.endpoint


def test_subprocess_host_defaults_to_loopback() -> None:
    assert AgentsConfig().subprocess_host == "127.0.0.1"


def test_subprocess_host_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_AGENTS_SUBPROCESS_HOST", "10.1.2.3")

    assert AgentsConfig().subprocess_host == "10.1.2.3"


def test_subprocess_deployments_are_enabled_by_default() -> None:
    assert AgentsConfig().subprocess_enabled is True


def test_subprocess_deployments_can_be_turned_off_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_AGENTS_SUBPROCESS_ENABLED", "false")

    assert AgentsConfig().subprocess_enabled is False


@pytest.mark.parametrize("host", ["0.0.0.0", "::"])
def test_subprocess_host_rejects_wildcard_addresses(host: str) -> None:
    with pytest.raises(ValidationError, match="reachable address"):
        AgentsConfig(subprocess_host=host)


def test_subprocess_host_accepts_hostnames() -> None:
    assert AgentsConfig(subprocess_host="controller-0.example").subprocess_host == "controller-0.example"


@pytest.mark.asyncio
async def test_endpoint_defaults_to_loopback(tmp_path: Path) -> None:
    assert await _create_nat_deployment(_backend(tmp_path), 49200) == "http://127.0.0.1:49200"


@pytest.mark.asyncio
async def test_endpoint_uses_configured_host(tmp_path: Path) -> None:
    assert await _create_nat_deployment(_backend(tmp_path, "10.1.2.3"), 49200) == "http://10.1.2.3:49200"


@pytest.mark.asyncio
async def test_endpoint_brackets_ipv6_hosts(tmp_path: Path) -> None:
    assert await _create_nat_deployment(_backend(tmp_path, "fd00::5"), 49200) == "http://[fd00::5]:49200"


def test_nat_agent_binds_to_configured_host(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "10.1.2.3")

    with (
        patch("nemo_agents_plugin.runner.in_memory._resolve_nat_bin", return_value="nat"),
        patch("nemo_agents_plugin.runner.in_memory.subprocess.Popen") as popen,
    ):
        backend._spawn("dep", tmp_path / "config.yaml", tmp_path / "dep.log", 49200)

    assert _spawned_host(popen.call_args.args[0]) == "10.1.2.3"


def test_fabric_agent_binds_to_configured_host(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "10.1.2.3")

    with patch("nemo_agents_plugin.runner.in_memory.subprocess.Popen") as popen:
        backend._spawn_fabric("dep", tmp_path / "agent.yaml", tmp_path / "dep.log", 49200)

    assert _spawned_host(popen.call_args.args[0]) == "10.1.2.3"


def test_port_probe_checks_configured_host(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "10.1.2.3")

    with patch.object(InMemoryRunnerBackend, "_is_port_free", return_value=True) as probe:
        backend.allocate_port()

    probe.assert_called_once_with(49152, "10.1.2.3")


def test_registry_passes_subprocess_host_to_runner(tmp_path: Path) -> None:
    config = AgentsConfig(subprocess_host="10.1.2.3", controller=ControllerConfig(workspace_dir=tmp_path))

    registry = RunnerBackendRegistry(config)

    assert registry.backend_for("subprocess")._endpoint(49200) == "http://10.1.2.3:49200"
