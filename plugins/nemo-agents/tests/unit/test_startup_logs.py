# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup diagnostics survive runtime cleanup and controller replacement."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_agents_plugin.api.v2 import deployment_logs
from nemo_agents_plugin.config import ControllerConfig
from nemo_agents_plugin.entities import AgentDeployment
from nemo_agents_plugin.runner import startup_logs
from nemo_agents_plugin.runner.backend import DeploymentInfo, LocalLog
from nemo_agents_plugin.runner.controller import AgentDeploymentController
from nemo_agents_plugin.runner.startup_logs import STARTUP_LOG_BYTES, capture_startup_logs


@pytest.mark.asyncio
async def test_capture_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(startup_logs, "STARTUP_LOG_TIMEOUT", 0.01)
    backend = AsyncMock()

    async def blocked(**kwargs: object) -> None:
        await asyncio.Event().wait()

    backend.get_logs.side_effect = blocked
    with pytest.raises(TimeoutError):
        await capture_startup_logs(backend, "workspace", "name")


def _controller(backend: AsyncMock) -> AgentDeploymentController:
    controller = AgentDeploymentController()
    controller._registry = MagicMock()
    controller._registry.backend_for.return_value = backend
    controller._entities = AsyncMock()
    controller._controller_config = ControllerConfig(health_check_timeout_seconds=0)
    return controller


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_failed", [True, False])
async def test_capture_is_persisted_before_cleanup_and_specific_failure_wins(backend_failed: bool) -> None:
    backend = AsyncMock()
    backend.get_deployment_status.return_value = DeploymentInfo(
        name="broken", status="failed" if backend_failed else "starting", error="Unsupported harness kind"
    )
    backend.get_logs.return_value = SimpleNamespace(lines=["traceback", "Unsupported harness kind"])
    controller = _controller(backend)
    dep = AgentDeployment(name="broken", workspace="isolated", status="starting", deployment_mode="docker")

    async def deleted(*args: object) -> bool:
        assert isinstance(controller.entities.update, AsyncMock)
        controller.entities.update.assert_awaited_once()
        assert dep.startup_logs.endswith("Unsupported harness kind")
        return True

    backend.delete_deployment.side_effect = deleted
    await controller._check_health(dep)
    backend.delete_deployment.assert_awaited_once_with("isolated", "broken")
    assert dep.status == "failed"
    assert dep.error == ("Unsupported harness kind" if backend_failed else "Health check timed out after 0s.")
    # Round-trip the durable entity; no controller or backend state is needed.
    restored = AgentDeployment.model_validate_json(dep.model_dump_json())
    assert restored.startup_logs == "traceback\nUnsupported harness kind"


@pytest.mark.asyncio
async def test_capture_error_does_not_prevent_cleanup() -> None:
    backend = AsyncMock()
    backend.get_deployment_status.return_value = DeploymentInfo(name="broken", status="failed", error="exit 1")
    backend.get_logs.side_effect = RuntimeError("log channel down")
    controller = _controller(backend)
    dep = AgentDeployment(name="broken", workspace="default", status="starting", deployment_mode="docker")
    await controller._check_health(dep)
    assert dep.error == "exit 1"
    assert dep.startup_logs == ""
    backend.delete_deployment.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_persistence_leaves_runtime_intact() -> None:
    backend = AsyncMock()
    backend.get_deployment_status.return_value = None
    backend.get_logs.return_value = SimpleNamespace(lines=["diagnostic"])
    controller = _controller(backend)
    controller._save = AsyncMock(side_effect=RuntimeError("database unavailable"))
    dep = AgentDeployment(name="broken", workspace="default", status="starting")
    with pytest.raises(RuntimeError, match="database unavailable"):
        await controller._check_health(dep)
    backend.delete_deployment.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("lines", [[str(i) for i in range(1000)], ["🦙" * STARTUP_LOG_BYTES]])
async def test_external_capture_bounds(lines: list[str]) -> None:
    backend = AsyncMock()
    backend.get_logs.return_value = SimpleNamespace(lines=lines)
    tail = await capture_startup_logs(backend, "workspace", "name")
    assert len(tail.encode("utf-8")) <= STARTUP_LOG_BYTES
    assert len(tail.splitlines()) <= 500
    assert tail.endswith(lines[-1][-100:])
    if len(lines) > 500:
        assert tail.splitlines() == lines[-500:]


@pytest.mark.asyncio
async def test_local_capture_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "agent.log"
    path.write_text("old\n" * STARTUP_LOG_BYTES + "final failure\n")
    backend = MagicMock(spec=["get_log_location"])
    backend.get_log_location.return_value = LocalLog(path=path)
    tail = await capture_startup_logs(backend, "workspace", "name")
    assert len(tail.encode()) <= STARTUP_LOG_BYTES
    assert len(tail.splitlines()) == 500
    assert tail.endswith("final failure")


@pytest.mark.asyncio
@pytest.mark.parametrize("tail, expected", [(0, []), (1, ["failure"]), (500, ["traceback", "failure"])])
async def test_retained_logs_after_restart(monkeypatch: pytest.MonkeyPatch, tail: int, expected: list[str]) -> None:
    restored = AgentDeployment.model_validate_json(
        AgentDeployment(
            name="broken",
            workspace="private",
            status="failed",
            deployment_mode="docker",
            startup_logs="traceback\nfailure",
        ).model_dump_json()
    )
    entities = AsyncMock()
    entities.get.return_value = restored
    backend = AsyncMock()
    backend.get_logs.side_effect = AssertionError("runtime was deleted")
    registry = MagicMock()
    registry.backend_for.return_value = backend
    monkeypatch.setattr(deployment_logs, "get_runner_registry", lambda: registry)
    result = await deployment_logs._resolve_external_logs("private", "broken", tail, entities)
    assert result is not None
    assert [line.message for line in result.data] == expected
    entities.get.assert_awaited_with(AgentDeployment, name="broken", workspace="private")
    backend.get_logs.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_stream_getter_switches_to_retained_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    dep = AgentDeployment(name="broken", workspace="private", deployment_mode="docker")
    entities = AsyncMock()
    entities.get.return_value = dep
    backend = AsyncMock()
    backend.get_logs.return_value = SimpleNamespace(lines=["starting"])
    registry = MagicMock()
    registry.backend_for.return_value = backend
    monkeypatch.setattr(deployment_logs, "get_runner_registry", lambda: registry)
    getter = await deployment_logs._resolve_external_log_getter("private", "broken", entities)
    assert (await getter(workspace="private", name="broken", tail=500)).lines == ["starting"]
    dep.startup_logs = "starting\nfailure"
    assert (await getter(workspace="private", name="broken", tail=500)).lines == ["starting", "failure"]
    assert backend.get_logs.await_count == 1
