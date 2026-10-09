# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AgentDeploymentController health checks and state transitions."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_agents_plugin.config import ControllerConfig
from nemo_agents_plugin.entities import AgentDeployment, DeploymentMode, DeploymentStatus, Endpoint
from nemo_agents_plugin.runner.backend import DeploymentInfo
from nemo_agents_plugin.runner.controller import AgentDeploymentController


def _make_deployment(
    name: str = "test-dep",
    workspace: str = "default",
    agent: str = "test-agent",
    status: DeploymentStatus = "pending",
    port: int = 0,
    pid: int = 0,
    endpoint: str = "",
    deployment_mode: DeploymentMode = "subprocess",
    endpoints: list[Endpoint] | None = None,
    image: str = "",
) -> AgentDeployment:
    return AgentDeployment(
        name=name,
        workspace=workspace,
        agent=agent,
        status=status,
        port=port,
        pid=pid,
        endpoint=endpoint,
        deployment_mode=deployment_mode,
        endpoints=endpoints or [],
        image=image,
    )


def _make_controller(
    interval_seconds: int = 2,
    health_check_timeout_seconds: int = 10,
    health_check_interval_seconds: int = 1,
) -> Any:
    """Return a controller with mocked registry/backend and entities."""
    ctrl = AgentDeploymentController()
    backend = AsyncMock()
    backend.allocate_port = MagicMock(return_value=0)
    backend.delete_deployment = AsyncMock(return_value=True)
    registry = MagicMock()
    registry.backend = backend
    registry.backend_for = MagicMock(return_value=backend)
    ctrl._registry = registry
    ctrl._entities = AsyncMock()
    ctrl._controller_config = ControllerConfig(
        interval_seconds=interval_seconds,
        health_check_timeout_seconds=health_check_timeout_seconds,
        health_check_interval_seconds=health_check_interval_seconds,
    )
    ctrl._interval_seconds = float(interval_seconds)
    return ctrl


def _key(dep: AgentDeployment) -> tuple[str, str]:
    return (dep.workspace, dep.name)


class TestDefaultInterval:
    def test_default_interval_is_2s(self) -> None:
        cfg = ControllerConfig()
        assert cfg.interval_seconds == 2

    def test_default_health_check_interval_is_2s(self) -> None:
        cfg = ControllerConfig()
        assert cfg.health_check_interval_seconds == 2


class TestStartDeployment:
    @pytest.mark.asyncio
    async def test_start_transitions_to_starting(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="pending")
        ctrl.backend.allocate_port.return_value = 50000
        ctrl.backend.create_deployment.return_value = DeploymentInfo(
            name="test-dep",
            status="starting",
            port=50000,
            pid=12345,
            endpoint="http://127.0.0.1:50000",
        )

        await ctrl._start_deployment(dep)

        assert dep.status == "starting"
        assert dep.port == 50000
        ctrl.backend.allocate_port.assert_not_called()
        assert ctrl.backend.create_deployment.await_args.kwargs["port"] == 0
        assert dep.pid == 12345
        assert dep.endpoint == "http://127.0.0.1:50000"
        assert _key(dep) in ctrl._starting_since

    @pytest.mark.asyncio
    async def test_start_container_mode_clears_loopback_endpoint(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="pending", deployment_mode="docker", image="agent:latest")
        ctrl.backend.allocate_port.return_value = 0
        ctrl.backend.create_deployment.return_value = DeploymentInfo(
            name="test-dep",
            status="starting",
            endpoints=[],
        )

        await ctrl._start_deployment(dep)

        assert dep.status == "starting"
        assert dep.endpoint == ""
        assert dep.plugin_deployment == "test-dep"
        ctrl.backend.create_deployment.assert_awaited_once()
        kwargs = ctrl.backend.create_deployment.await_args.kwargs
        assert kwargs["deployment_mode"] == "docker"
        assert kwargs["image"] == "agent:latest"

    @pytest.mark.asyncio
    async def test_start_accepts_backend_running_status(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="pending")
        ctrl.backend.create_deployment.return_value = DeploymentInfo(
            name="test-dep",
            status="running",
        )

        await ctrl._start_deployment(dep)

        assert dep.status == "running"
        assert dep.endpoint == ""
        assert _key(dep) not in ctrl._starting_since

    @pytest.mark.asyncio
    async def test_start_failure_transitions_to_failed(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="pending")
        ctrl.backend.allocate_port.return_value = 50000
        ctrl.backend.create_deployment.side_effect = RuntimeError("spawn failed")

        await ctrl._start_deployment(dep)

        assert dep.status == "failed"
        assert "spawn failed" in dep.error


class TestCheckHealth:
    @pytest.mark.asyncio
    async def test_healthy_on_first_check(self) -> None:
        ctrl = _make_controller(health_check_interval_seconds=1)
        dep = _make_deployment(
            status="starting",
            port=50000,
            pid=123,
            endpoint="http://127.0.0.1:50000",
        )
        ctrl._starting_since[_key(dep)] = time.monotonic()

        ctrl.backend.get_deployment_status.return_value = DeploymentInfo(
            name="test-dep",
            status="starting",
        )
        ctrl.backend.health_check.return_value = True

        await ctrl._check_health(dep)

        assert dep.status == "running"
        assert _key(dep) not in ctrl._starting_since

    @pytest.mark.asyncio
    async def test_container_ready_projects_endpoints_without_health_check(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="starting", deployment_mode="k8s")
        ctrl._starting_since[_key(dep)] = time.monotonic()
        endpoints = [Endpoint(name="http", url="http://svc.ns.svc.cluster.local:8080")]
        ctrl.backend.get_deployment_status.return_value = DeploymentInfo(
            name="test-dep",
            status="running",
            endpoints=endpoints,
        )

        await ctrl._check_health(dep)

        assert dep.status == "running"
        assert dep.endpoint == ""
        assert dep.endpoints == endpoints
        ctrl.backend.health_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_healthy_stays_starting(self) -> None:
        """Single-shot: one failed probe leaves status as 'starting' for the next cycle."""
        ctrl = _make_controller()
        dep = _make_deployment(
            status="starting",
            port=50000,
            pid=123,
            endpoint="http://127.0.0.1:50000",
        )
        ctrl._starting_since[_key(dep)] = time.monotonic()

        ctrl.backend.get_deployment_status.return_value = DeploymentInfo(
            name="test-dep",
            status="starting",
        )
        ctrl.backend.health_check.return_value = False

        await ctrl._check_health(dep)

        assert dep.status == "starting"
        assert _key(dep) in ctrl._starting_since
        ctrl.backend.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_exit_during_health_check(self) -> None:
        ctrl = _make_controller(health_check_interval_seconds=0)
        dep = _make_deployment(
            status="starting",
            port=50000,
            pid=123,
            endpoint="http://127.0.0.1:50000",
        )
        ctrl._starting_since[_key(dep)] = time.monotonic()

        ctrl.backend.get_deployment_status.return_value = DeploymentInfo(
            name="test-dep",
            status="failed",
            error="Process exited with code 1",
        )

        await ctrl._check_health(dep)

        assert dep.status == "failed"
        assert "exited" in dep.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_transitions_to_failed(self) -> None:
        ctrl = _make_controller(
            health_check_timeout_seconds=0,
            health_check_interval_seconds=0,
        )
        dep = _make_deployment(
            status="starting",
            port=50000,
            pid=123,
            endpoint="http://127.0.0.1:50000",
        )
        ctrl._starting_since[_key(dep)] = 0  # started long ago

        ctrl.backend.get_deployment_status.return_value = DeploymentInfo(
            name="test-dep",
            status="starting",
        )
        ctrl.backend.health_check.return_value = False

        await ctrl._check_health(dep)

        assert dep.status == "failed"
        assert "timed out" in dep.error.lower()
        ctrl.backend.delete_deployment.assert_called_once_with("default", "test-dep")

    @pytest.mark.asyncio
    async def test_no_endpoint_skips_health_check(self) -> None:
        """If endpoint is empty, health_check should not be called."""
        ctrl = _make_controller(
            health_check_timeout_seconds=0,
            health_check_interval_seconds=0,
        )
        dep = _make_deployment(status="starting", endpoint="")
        ctrl._starting_since[_key(dep)] = 0

        ctrl.backend.get_deployment_status.return_value = DeploymentInfo(
            name="test-dep",
            status="starting",
        )
        ctrl.backend.health_check.return_value = False

        await ctrl._check_health(dep)

        assert dep.status == "failed"
        ctrl.backend.health_check.assert_not_called()


class TestReconcileOne:
    @pytest.mark.asyncio
    async def test_pending_triggers_start(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="pending")
        ctrl.backend.allocate_port.return_value = 50000
        ctrl.backend.create_deployment.return_value = DeploymentInfo(
            name="test-dep",
            status="starting",
            port=50000,
            pid=99,
            endpoint="http://127.0.0.1:50000",
        )

        await ctrl._reconcile_one(dep)

        assert dep.status == "starting"

    @pytest.mark.asyncio
    async def test_running_verify_process_gone_resets_to_pending(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="running")
        ctrl.backend.get_deployment_status.return_value = None

        await ctrl._reconcile_one(dep)

        assert dep.status == "pending"

    @pytest.mark.asyncio
    async def test_running_verify_container_gone_marks_failed(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="running", deployment_mode="docker")
        ctrl.backend.get_deployment_status.return_value = None

        await ctrl._reconcile_one(dep)

        assert dep.status == "failed"
        assert "not found" in dep.error.lower()

    @pytest.mark.asyncio
    async def test_container_failed_startup_deletes_plugin_entities(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="starting", deployment_mode="k8s")
        ctrl._starting_since[_key(dep)] = time.monotonic()
        ctrl.backend.get_deployment_status.return_value = DeploymentInfo(
            name="test-dep",
            status="failed",
            error="ImagePullBackOff",
        )

        await ctrl._check_health(dep)

        assert dep.status == "failed"
        ctrl.backend.delete_deployment.assert_awaited_once_with("default", "test-dep")

    @pytest.mark.asyncio
    async def test_deleting_removes_deployment(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="deleting")

        await ctrl._reconcile_one(dep)

        ctrl.backend.delete_deployment.assert_called_once_with("default", "test-dep")
        ctrl.entities.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deleting_keeps_entity_when_teardown_incomplete(self) -> None:
        ctrl = _make_controller()
        dep = _make_deployment(status="deleting", deployment_mode="docker")
        ctrl.backend.delete_deployment.return_value = False

        await ctrl._reconcile_one(dep)

        ctrl.entities.delete.assert_not_called()
        assert dep.status == "deleting"
        ctrl.entities.update.assert_awaited()

    @pytest.mark.asyncio
    async def test_health_timeout_saves_failed_even_if_cleanup_raises(self) -> None:
        ctrl = _make_controller(health_check_timeout_seconds=0, health_check_interval_seconds=0)
        dep = _make_deployment(status="starting", deployment_mode="docker")
        ctrl._starting_since[_key(dep)] = 0
        ctrl.backend.delete_deployment.side_effect = RuntimeError("cleanup boom")

        await ctrl._check_health(dep)

        assert dep.status == "failed"
        assert "timed out" in dep.error.lower()
        ctrl.entities.update.assert_awaited()
