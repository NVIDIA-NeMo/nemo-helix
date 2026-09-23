# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from helpers import list_response, make_deployment, make_deployment_config, make_volume
from nemo_deployments_plugin.backends.registry import ExecutorRegistry
from nemo_deployments_plugin.config import ControllerConfig
from nemo_deployments_plugin.controller import DeploymentsController, _orphan_protected_ids
from nemo_deployments_plugin.entities import Deployment, DeploymentConfig, Prerequisite, StatusEvent
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntityConflictError


def _stub_registry() -> ExecutorRegistry:
    return cast(ExecutorRegistry, type("R", (), {"all_backends": lambda self: []})())


@pytest.mark.asyncio
async def test_controller_reconcile_runs_volumes_then_deployments() -> None:
    ctrl = DeploymentsController()
    dep = make_deployment()
    vol = make_volume()

    mock_entities = AsyncMock()
    mock_entities.list.side_effect = [
        list_response([dep]),
        list_response([vol]),
    ]
    mock_entities.get.return_value = AsyncMock()
    mock_entities.update = AsyncMock(side_effect=lambda e: e)

    mock_registry = _stub_registry()

    mock_dep_reconciler = AsyncMock()
    mock_vol_reconciler = AsyncMock()
    mock_dep_reconciler.set_config_cache = lambda configs: None

    ctrl._entities = mock_entities
    ctrl._registry = mock_registry
    ctrl._controller_config = ControllerConfig(orphan_cleanup_interval_seconds=0)
    ctrl._deployment_reconciler = mock_dep_reconciler
    ctrl._volume_reconciler = mock_vol_reconciler

    await ctrl.reconcile()

    mock_vol_reconciler.reconcile_one.assert_awaited_once_with(vol)
    mock_dep_reconciler.reconcile_one.assert_awaited_once()
    call_kwargs = mock_dep_reconciler.reconcile_one.await_args.kwargs
    assert "volumes_by_name" in call_kwargs


@pytest.mark.asyncio
async def test_controller_swallows_conflict_on_deployment() -> None:
    ctrl = DeploymentsController()
    dep = make_deployment()

    mock_entities = AsyncMock()
    mock_entities.list.side_effect = [list_response([dep]), list_response([])]

    mock_dep_reconciler = AsyncMock()
    mock_dep_reconciler.reconcile_one.side_effect = NemoEntityConflictError("conflict")
    mock_dep_reconciler.set_config_cache = lambda configs: None

    ctrl._entities = mock_entities
    ctrl._registry = _stub_registry()
    ctrl._controller_config = ControllerConfig(orphan_cleanup_interval_seconds=0)
    ctrl._deployment_reconciler = mock_dep_reconciler
    ctrl._volume_reconciler = AsyncMock()

    await ctrl.reconcile()  # should not raise


@pytest.mark.asyncio
@patch("nemo_deployments_plugin.controller.client_from_platform")
@patch("nemo_deployments_plugin.controller.get_async_platform_sdk")
@patch("nemo_deployments_plugin.config.DeploymentsConfig.get")
async def test_controller_on_startup(
    mock_config_get: AsyncMock,
    mock_sdk: AsyncMock,
    mock_adapter: AsyncMock,
) -> None:
    from nemo_deployments_plugin.config import DeploymentsConfig

    mock_config_get.return_value = DeploymentsConfig()
    mock_sdk.return_value = AsyncMock()

    ctrl = DeploymentsController()
    await ctrl.on_startup()

    mock_adapter.assert_called_once_with(mock_sdk.return_value, AsyncEntitiesClient)
    assert ctrl._entities is not None
    assert ctrl._registry is not None
    assert ctrl.interval_seconds == 5.0


@pytest.mark.asyncio
async def test_controller_unhealthy_after_list_failure() -> None:
    ctrl = DeploymentsController()
    ctrl._entities = AsyncMock()
    ctrl._entities.list.side_effect = RuntimeError("entity store down")

    deployments = await ctrl._list_deployments()
    assert deployments == []
    assert not ctrl.is_healthy


@pytest.mark.asyncio
async def test_controller_unhealthy_when_deployments_fail_volumes_ok() -> None:
    ctrl = DeploymentsController()
    ctrl._entities = AsyncMock()
    ctrl._entities.list.side_effect = RuntimeError("deployments down")

    await ctrl._list_deployments()
    assert not ctrl._deployments_list_ok

    ctrl._entities.list.side_effect = None
    ctrl._entities.list.return_value = list_response([])
    await ctrl._list_volumes()
    assert ctrl._volumes_list_ok
    assert not ctrl.is_healthy


@pytest.mark.asyncio
@patch("nemo_deployments_plugin.controller.reconcile_orphans")
async def test_orphan_cleanup_skipped_when_terminal_list_fails(mock_orphans: AsyncMock) -> None:
    ctrl = DeploymentsController()
    dep = make_deployment()

    mock_entities = AsyncMock()
    mock_entities.list.side_effect = [
        list_response([dep]),
        list_response([]),
        RuntimeError("terminal list failed"),
    ]

    mock_registry = _stub_registry()
    mock_dep_reconciler = AsyncMock()
    mock_dep_reconciler.set_config_cache = lambda configs: None

    ctrl._entities = mock_entities
    ctrl._registry = mock_registry
    ctrl._controller_config = ControllerConfig(orphan_cleanup_interval_seconds=10)
    ctrl._deployment_reconciler = mock_dep_reconciler
    ctrl._volume_reconciler = AsyncMock()

    await ctrl.reconcile()
    await ctrl.reconcile()
    mock_orphans.assert_not_awaited()


@pytest.mark.asyncio
@patch("nemo_deployments_plugin.controller.reconcile_orphans")
async def test_orphan_cleanup_skipped_when_deployments_list_fails(mock_orphans: AsyncMock) -> None:
    ctrl = DeploymentsController()
    ctrl._entities = AsyncMock()
    ctrl._entities.list.side_effect = RuntimeError("list failed")

    ctrl._registry = _stub_registry()
    ctrl._controller_config = ControllerConfig(orphan_cleanup_interval_seconds=5)
    ctrl._deployment_reconciler = AsyncMock()
    ctrl._deployment_reconciler.set_config_cache = lambda configs: None
    ctrl._volume_reconciler = AsyncMock()

    await ctrl.reconcile()
    mock_orphans.assert_not_awaited()


@pytest.mark.asyncio
@patch("nemo_deployments_plugin.controller.reconcile_orphans")
async def test_controller_runs_orphan_cleanup_on_interval(mock_orphans: AsyncMock) -> None:
    ctrl = DeploymentsController()
    dep = make_deployment()

    mock_entities = AsyncMock()
    mock_entities.list.side_effect = [
        list_response([dep]),
        list_response([]),
        list_response([dep]),
        list_response([]),
        list_response([]),
    ]

    mock_registry = _stub_registry()
    mock_dep_reconciler = AsyncMock()
    mock_dep_reconciler.set_config_cache = lambda configs: None

    ctrl._entities = mock_entities
    ctrl._registry = mock_registry
    ctrl._controller_config = ControllerConfig(orphan_cleanup_interval_seconds=10)
    ctrl._deployment_reconciler = mock_dep_reconciler
    ctrl._volume_reconciler = AsyncMock()

    await ctrl.reconcile()
    mock_orphans.assert_not_awaited()

    await ctrl.reconcile()
    mock_orphans.assert_awaited_once()


@pytest.mark.asyncio
async def test_controller_swallows_volume_conflict() -> None:
    ctrl = DeploymentsController()
    vol = make_volume()

    mock_entities = AsyncMock()
    mock_entities.list.side_effect = [list_response([]), list_response([vol])]

    mock_vol_reconciler = AsyncMock()
    mock_vol_reconciler.reconcile_one.side_effect = NemoEntityConflictError("conflict")

    ctrl._entities = mock_entities
    ctrl._registry = _stub_registry()
    ctrl._controller_config = ControllerConfig(orphan_cleanup_interval_seconds=0)
    ctrl._deployment_reconciler = AsyncMock()
    ctrl._deployment_reconciler.set_config_cache = lambda configs: None
    ctrl._volume_reconciler = mock_vol_reconciler

    await ctrl.reconcile()  # should not raise


@pytest.mark.asyncio
async def test_controller_fetches_terminal_prerequisite_for_dag() -> None:
    """Puller SUCCEEDED is not in the active list but must unblock server create."""
    ctrl = DeploymentsController()
    puller = make_deployment("puller")
    puller.deployment_config = "puller"
    puller.status = "SUCCEEDED"
    puller.exit_code = 0
    server = make_deployment("server")
    server.deployment_config = "server"
    server.status = "PENDING"
    server.prerequisites = [Prerequisite(deployment_name="puller", condition="succeeded")]
    server_cfg = make_deployment_config("server")

    mock_entities = AsyncMock()
    mock_entities.list.side_effect = [
        list_response([server]),
        list_response([]),
    ]

    async def mock_get(entity_cls, name: str, workspace: str = "default", **_kwargs: object):
        if entity_cls is DeploymentConfig:
            if name == "server":
                return server_cfg
            return make_deployment_config(name, workspace)
        if entity_cls is Deployment and name == "puller":
            return puller
        raise AssertionError(f"unexpected get: {entity_cls} {name}")

    mock_entities.get.side_effect = mock_get

    mock_dep_reconciler = AsyncMock()
    mock_dep_reconciler.set_config_cache = lambda configs: None

    ctrl._entities = mock_entities
    ctrl._registry = _stub_registry()
    ctrl._controller_config = ControllerConfig(orphan_cleanup_interval_seconds=0)
    ctrl._deployment_reconciler = mock_dep_reconciler
    ctrl._volume_reconciler = AsyncMock()

    await ctrl.reconcile()

    mock_entities.get.assert_awaited()
    call_args = mock_dep_reconciler.reconcile_one.await_args
    by_name = call_args.kwargs["deployments_by_name"]
    puller_key = ("default", "puller")
    assert puller_key in by_name
    assert by_name[puller_key].status == "SUCCEEDED"


@pytest.mark.asyncio
async def test_controller_fetches_prerequisite_by_deployment_name() -> None:
    """Prerequisite references Deployment entity name, not DeploymentConfig name."""
    ctrl = DeploymentsController()
    puller = make_deployment("puller-run-1")
    puller.deployment_config = "puller"
    puller.status = "SUCCEEDED"
    puller.exit_code = 0
    server = make_deployment("server")
    server.deployment_config = "server"
    server.status = "PENDING"
    server.prerequisites = [Prerequisite(deployment_name="puller-run-1", condition="succeeded")]
    server_cfg = make_deployment_config("server")

    mock_entities = AsyncMock()
    mock_entities.list.side_effect = [
        list_response([server]),
        list_response([]),
    ]

    async def mock_get(entity_cls, name: str, workspace: str = "default", **_kwargs: object):
        if entity_cls is DeploymentConfig:
            if name == "server":
                return server_cfg
            return make_deployment_config(name, workspace)
        if entity_cls is Deployment and name == "puller-run-1":
            return puller
        raise AssertionError(f"unexpected get: {entity_cls} {name}")

    mock_entities.get.side_effect = mock_get

    mock_dep_reconciler = AsyncMock()
    mock_dep_reconciler.set_config_cache = lambda configs: None

    ctrl._entities = mock_entities
    ctrl._registry = _stub_registry()
    ctrl._controller_config = ControllerConfig(orphan_cleanup_interval_seconds=0)
    ctrl._deployment_reconciler = mock_dep_reconciler
    ctrl._volume_reconciler = AsyncMock()

    await ctrl.reconcile()

    call_args = mock_dep_reconciler.reconcile_one.await_args
    by_name = call_args.kwargs["deployments_by_name"]
    puller_key = ("default", "puller-run-1")
    assert puller_key in by_name
    assert by_name[puller_key].name == "puller-run-1"


@pytest.mark.asyncio
async def test_controller_stops_mid_reconcile_when_stop_requested() -> None:
    ctrl = DeploymentsController()
    dep1 = make_deployment("dep1")
    dep2 = make_deployment("dep2")
    stop = threading.Event()

    mock_entities = AsyncMock()
    mock_entities.list.side_effect = [
        list_response([dep1, dep2]),
        list_response([]),
    ]

    mock_dep_reconciler = AsyncMock()
    mock_dep_reconciler.set_config_cache = lambda configs: None

    async def reconcile_and_stop(deployment: object, **kwargs: object) -> None:
        stop.set()

    mock_dep_reconciler.reconcile_one.side_effect = reconcile_and_stop

    ctrl.set_stop_signal(stop)
    ctrl._entities = mock_entities
    ctrl._registry = _stub_registry()
    ctrl._controller_config = ControllerConfig(orphan_cleanup_interval_seconds=0)
    ctrl._deployment_reconciler = mock_dep_reconciler
    ctrl._volume_reconciler = AsyncMock()

    await ctrl.reconcile()

    assert mock_dep_reconciler.reconcile_one.await_count == 1


def test_orphan_protected_ids_includes_terminal_within_grace() -> None:
    active = make_deployment("active")
    terminal = make_deployment("done")
    terminal.status = "SUCCEEDED"
    recent = datetime.now(timezone.utc) - timedelta(seconds=30)
    terminal.status_history = [StatusEvent(status="SUCCEEDED", timestamp=recent.isoformat())]

    known = _orphan_protected_ids([active], [terminal], grace_seconds=3600)
    assert "default/active" in known
    assert "default/done" in known


def test_orphan_protected_ids_excludes_terminal_after_grace() -> None:
    terminal = make_deployment("done")
    terminal.status = "SUCCEEDED"
    old = datetime.now(timezone.utc) - timedelta(seconds=7200)
    terminal.status_history = [StatusEvent(status="SUCCEEDED", timestamp=old.isoformat())]

    known = _orphan_protected_ids([], [terminal], grace_seconds=3600)
    assert "default/done" not in known


def test_orphan_protected_ids_includes_terminal_without_history() -> None:
    terminal = make_deployment("done")
    terminal.status = "SUCCEEDED"
    terminal.status_history = []

    known = _orphan_protected_ids([], [terminal], grace_seconds=3600)
    assert "default/done" in known
