# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``AgentDeploymentController`` startup / health-check transitions.

Pin the contracts callers depend on:

- ``_start_deployment`` writes the runtime fields (status, pid, port,
  endpoint) onto the entity but does NOT propagate the runner's host-bound
  ``log_path`` — the entity is backend-agnostic and the path is meaningful
  only on the platform host.
- ``_check_health`` gives precedence to subprocess-exit over a successful
  health probe, so a dead process is reported as failed even if a stale
  ``/health`` response would otherwise pass.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from nemo_agents_plugin.config import AgentsConfig, ControllerConfig
from nemo_agents_plugin.entities import (
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    AgentDeployment,
    AgentSession,
    SessionStatus,
)
from nemo_agents_plugin.runner.backend import DeploymentInfo
from nemo_agents_plugin.runner.controller import AgentDeploymentController
from nemo_agents_plugin.spec_revision import SpecRevision
from nemo_platform_plugin.auth import AuthContext
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.entity_client import NemoEntityConflictError

EXPIRATION_NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def test_agent_deployment_controller_declares_entities_dependency() -> None:
    assert AgentDeploymentController.dependencies == ["entities"]


@pytest.mark.asyncio
async def test_controller_startup_adapts_sdk_to_typed_entities_client() -> None:
    sdk = MagicMock()
    typed_client = MagicMock()
    entity_client = MagicMock()
    empty_page = MagicMock(data=[], pagination=None)
    entity_client.list = AsyncMock(return_value=empty_page)

    with (
        patch("nemo_agents_plugin.config.AgentsConfig.get", return_value=AgentsConfig()),
        patch("nemo_platform_plugin.sdk_provider.get_async_platform_sdk", return_value=sdk),
        patch(
            "nemo_platform_plugin.client.adapter.client_from_platform",
            return_value=typed_client,
        ) as mock_adapter,
        patch("nemo_platform_plugin.entities.EntityClient", return_value=entity_client) as mock_entity_client,
    ):
        controller = AgentDeploymentController()
        await controller.on_startup()

    mock_adapter.assert_called_once_with(sdk, AsyncEntitiesClient)
    mock_entity_client.assert_called_once_with(typed_client)
    assert controller._entities is entity_client
    assert controller._startup_sessions_reconciled is True


@pytest.mark.asyncio
async def test_startup_session_reconciliation_retries_after_entity_list_failure() -> None:
    ctrl, _ = _make_controller()
    empty_page = MagicMock(data=[], pagination=None)
    ctrl.entities.list = AsyncMock(side_effect=[RuntimeError("unavailable"), empty_page])
    ctrl._startup_sessions_reconciled = False

    await ctrl._reconcile_sessions_after_controller_start()
    assert ctrl._startup_sessions_reconciled is False
    reconciliation_time = ctrl._startup_session_reconciliation_at

    await ctrl._reconcile_sessions_after_controller_start()
    assert ctrl._startup_sessions_reconciled is True
    assert ctrl._startup_session_reconciliation_at == reconciliation_time


@pytest.mark.asyncio
async def test_startup_retry_preserves_activity_from_current_runtime() -> None:
    ctrl, _ = _make_controller()
    current_generation = _make_session(
        first_active_at=EXPIRATION_NOW,
        last_active_at=EXPIRATION_NOW,
        expires_at=EXPIRATION_NOW + timedelta(minutes=30),
    )
    active_page = MagicMock(data=[current_generation], pagination=None)
    ctrl.entities.list = AsyncMock(side_effect=[RuntimeError("unavailable"), active_page])
    ctrl.entities.update = AsyncMock()
    ctrl._startup_sessions_reconciled = False
    ctrl._startup_session_reconciliation_at = EXPIRATION_NOW

    await ctrl._reconcile_sessions_after_controller_start()
    await ctrl._reconcile_sessions_after_controller_start()

    assert ctrl._startup_sessions_reconciled is True
    assert current_generation.status is SessionStatus.ACTIVE
    ctrl.entities.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_continues_deployments_while_startup_session_retry_is_pending() -> None:
    ctrl, _ = _make_controller()
    ctrl._startup_sessions_reconciled = False
    ctrl._reconcile_sessions_after_controller_start = AsyncMock()  # type: ignore[method-assign]
    ctrl._reconcile_deployments = AsyncMock()  # type: ignore[method-assign]
    ctrl._retry_pending_restart_reconciliations = AsyncMock()  # type: ignore[method-assign]
    ctrl._reconcile_expired_sessions = AsyncMock()  # type: ignore[method-assign]

    await ctrl.reconcile()

    ctrl._reconcile_sessions_after_controller_start.assert_awaited_once_with()
    ctrl._reconcile_deployments.assert_awaited_once_with()
    ctrl._retry_pending_restart_reconciliations.assert_awaited_once_with()
    ctrl._reconcile_expired_sessions.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_reconcile_stops_after_startup_session_retry_when_stop_is_requested() -> None:
    ctrl, _ = _make_controller()
    ctrl._startup_sessions_reconciled = False
    ctrl._reconcile_sessions_after_controller_start = AsyncMock()  # type: ignore[method-assign]
    ctrl._reconcile_deployments = AsyncMock()  # type: ignore[method-assign]
    ctrl.stop_requested = MagicMock(return_value=True)  # type: ignore[method-assign]

    await ctrl.reconcile()

    ctrl._reconcile_sessions_after_controller_start.assert_awaited_once_with()
    ctrl._reconcile_deployments.assert_not_awaited()


def _make_controller() -> tuple[AgentDeploymentController, Any]:
    """Build a controller with stubbed backend / entities / save.

    Returns the controller plus an ``Any``-typed alias of its backend mock,
    so tests can attach :class:`AsyncMock` attributes without fighting the
    typed ``RunnerBackend`` protocol.
    """
    ctrl = AgentDeploymentController()
    backend = MagicMock()
    backend.delete_deployment = AsyncMock()
    registry = MagicMock()
    registry.backend = backend
    registry.backend_for = MagicMock(return_value=backend)
    # Bypass on_startup() — wire stubs directly.
    ctrl._registry = registry
    ctrl._entities = MagicMock()
    ctrl._controller_config = ControllerConfig(health_check_timeout_seconds=120)
    ctrl._startup_sessions_reconciled = True
    ctrl._save = AsyncMock()  # type: ignore[method-assign]
    return ctrl, cast(Any, backend)


def _mock_runtime_cleanup(ctrl: AgentDeploymentController) -> MagicMock:
    cleanup = MagicMock()
    ctrl._schedule_runtime_cleanup = cleanup  # type: ignore[method-assign]
    return cleanup


def _make_session(
    *,
    name: str = "session-one",
    deployment_id: str = "deployment-id",
    first_active_at: datetime | None = None,
    last_active_at: datetime | None = None,
    expires_at: datetime | None = None,
    status: SessionStatus = SessionStatus.ACTIVE,
) -> AgentSession:
    session = AgentSession(
        name=name,
        workspace="default",
        deployment_id=deployment_id,
        status=status,
        first_active_at=first_active_at,
        last_active_at=last_active_at,
        expires_at=expires_at,
    )
    session._id = f"{name}-id"
    return session


def _make_fabric_deployment(
    *,
    name: str = "fabric-deployment",
    status: str = "running",
    deployment_id: str = "deployment-id",
) -> AgentDeployment:
    deployment = AgentDeployment(
        name=name,
        workspace="default",
        agent="calc",
        status=cast(Any, status),
        endpoint="http://localhost:9001",
        config={"config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT},
    )
    deployment._id = deployment_id
    return deployment


@pytest.mark.asyncio
async def test_reconcile_runs_expiration_after_isolated_deployment_failures() -> None:
    ctrl, _ = _make_controller()
    first = AgentDeployment(name="first", workspace="default", agent="calc")
    second = AgentDeployment(name="second", workspace="default", agent="calc")
    ctrl.list_objects = AsyncMock(return_value=[first, second])  # type: ignore[method-assign]
    ctrl.reconcile_one = AsyncMock(side_effect=[RuntimeError("failed"), None])  # type: ignore[method-assign]
    ctrl._reconcile_expired_sessions = AsyncMock()  # type: ignore[method-assign]

    await ctrl.reconcile()

    assert ctrl.reconcile_one.await_args_list == [call(first), call(second)]
    ctrl._reconcile_expired_sessions.assert_awaited_once_with()


# ---------------------------------------------------------------------------
# Runtime restart reconciliation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_reconciliation_expires_loses_and_preserves_never_invoked_sessions() -> None:
    ctrl, _ = _make_controller()
    lost = _make_session(
        name="lost",
        first_active_at=EXPIRATION_NOW - timedelta(minutes=10),
        last_active_at=EXPIRATION_NOW - timedelta(minutes=5),
        expires_at=EXPIRATION_NOW + timedelta(minutes=25),
    )
    expired = _make_session(
        name="expired",
        first_active_at=EXPIRATION_NOW - timedelta(hours=1),
        last_active_at=EXPIRATION_NOW - timedelta(minutes=31),
        expires_at=EXPIRATION_NOW - timedelta(minutes=1),
    )
    never_invoked = _make_session(name="never-invoked")
    ctrl._list_active_sessions = AsyncMock(  # type: ignore[method-assign]
        return_value=[lost, expired, never_invoked]
    )
    ctrl.entities.update = AsyncMock(side_effect=lambda entity: entity)
    cleanup = _mock_runtime_cleanup(ctrl)

    reconciled = await ctrl._reconcile_sessions_after_restart(at=EXPIRATION_NOW)

    assert reconciled is True
    assert lost.status is SessionStatus.LOST
    assert expired.status is SessionStatus.EXPIRED
    assert never_invoked.status is SessionStatus.ACTIVE
    assert ctrl.entities.update.await_args_list == [call(lost), call(expired)]
    assert cleanup.call_args_list == [call(lost), call(expired)]


@pytest.mark.asyncio
async def test_restart_wins_activity_conflict_for_refetched_invoked_session() -> None:
    ctrl, _ = _make_controller()
    stale = _make_session(
        last_active_at=EXPIRATION_NOW - timedelta(minutes=2),
        expires_at=EXPIRATION_NOW + timedelta(minutes=28),
    )
    refreshed = _make_session(
        last_active_at=EXPIRATION_NOW - timedelta(seconds=1),
        expires_at=EXPIRATION_NOW + timedelta(minutes=30) - timedelta(seconds=1),
    )
    ctrl.entities.update = AsyncMock(side_effect=[NemoEntityConflictError("conflict"), refreshed])
    ctrl.entities.get_by_id = AsyncMock(return_value=refreshed)
    cleanup = _mock_runtime_cleanup(ctrl)

    result = await ctrl._transition_session_after_restart(stale, at=EXPIRATION_NOW)

    assert result is refreshed
    assert refreshed.status is SessionStatus.LOST
    assert ctrl.entities.update.await_count == 2
    cleanup.assert_called_once_with(refreshed)


@pytest.mark.asyncio
async def test_activity_from_current_runtime_wins_restart_reconciliation() -> None:
    ctrl, _ = _make_controller()
    current_generation = _make_session(
        first_active_at=EXPIRATION_NOW,
        last_active_at=EXPIRATION_NOW,
        expires_at=EXPIRATION_NOW + timedelta(minutes=30),
    )
    cleanup = _mock_runtime_cleanup(ctrl)

    result = await ctrl._transition_session_after_restart(current_generation, at=EXPIRATION_NOW)

    assert result is None
    assert current_generation.status is SessionStatus.ACTIVE
    cast(Any, ctrl.entities.update).assert_not_called()
    cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_activity_between_runtime_start_and_controller_observation_stays_active() -> None:
    ctrl, _ = _make_controller()
    runtime_started_at = EXPIRATION_NOW
    first_activity_at = runtime_started_at + timedelta(seconds=1)
    controller_observed_at = runtime_started_at + timedelta(seconds=2)
    current_generation = _make_session(
        first_active_at=first_activity_at,
        last_active_at=first_activity_at,
        expires_at=first_activity_at + timedelta(minutes=30),
    )
    cleanup = _mock_runtime_cleanup(ctrl)

    result = await ctrl._transition_session_after_restart(current_generation, at=runtime_started_at)

    assert first_activity_at < controller_observed_at
    assert result is None
    assert current_generation.status is SessionStatus.ACTIVE
    cast(Any, ctrl.entities.update).assert_not_called()
    cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_new_runtime_activity_does_not_rebind_old_session() -> None:
    ctrl, _ = _make_controller()
    runtime_started_at = EXPIRATION_NOW
    old_session = _make_session(
        first_active_at=runtime_started_at - timedelta(minutes=5),
        last_active_at=runtime_started_at + timedelta(seconds=1),
        expires_at=runtime_started_at + timedelta(minutes=30),
    )
    ctrl.entities.update = AsyncMock(side_effect=lambda entity: entity)
    cleanup = _mock_runtime_cleanup(ctrl)

    result = await ctrl._transition_session_after_restart(old_session, at=runtime_started_at)

    assert result is old_session
    assert old_session.status is SessionStatus.LOST
    ctrl.entities.update.assert_awaited_once_with(old_session)
    cleanup.assert_called_once_with(old_session)


@pytest.mark.asyncio
async def test_restart_reconciliation_does_not_wait_for_runtime_cleanup() -> None:
    ctrl, _ = _make_controller()
    session = _make_session(
        last_active_at=EXPIRATION_NOW - timedelta(minutes=1),
        expires_at=EXPIRATION_NOW + timedelta(minutes=29),
    )
    ctrl.entities.update = AsyncMock(side_effect=lambda entity: entity)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def blocking_cleanup(*_args: object) -> None:
        cleanup_started.set()
        await release_cleanup.wait()

    with patch("nemo_agents_plugin.session_lifecycle.cleanup_fabric_runtime", side_effect=blocking_cleanup):
        result = await ctrl._transition_session_after_restart(session, at=EXPIRATION_NOW)
        assert result is session
        assert session.status is SessionStatus.LOST

        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        cleanup_tasks = list(ctrl._runtime_cleanup_tasks)
        assert len(cleanup_tasks) == 1
        release_cleanup.set()
        await asyncio.gather(*cleanup_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", [SessionStatus.EXPIRED, SessionStatus.LOST, SessionStatus.CLOSED])
async def test_terminal_state_wins_restart_conflict(terminal_status: SessionStatus) -> None:
    ctrl, _ = _make_controller()
    stale = _make_session(
        last_active_at=EXPIRATION_NOW - timedelta(minutes=2),
        expires_at=EXPIRATION_NOW + timedelta(minutes=28),
    )
    terminal = _make_session(status=terminal_status)
    ctrl.entities.update = AsyncMock(side_effect=NemoEntityConflictError("conflict"))
    ctrl.entities.get_by_id = AsyncMock(return_value=terminal)
    cleanup = _mock_runtime_cleanup(ctrl)

    result = await ctrl._transition_session_after_restart(stale, at=EXPIRATION_NOW)

    assert result is terminal
    assert terminal.status is terminal_status
    ctrl.entities.update.assert_awaited_once_with(stale)
    cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_runtime_instance_change_reconciles_bound_sessions() -> None:
    ctrl, _ = _make_controller()
    deployment = _make_fabric_deployment()
    runtime_started_at = EXPIRATION_NOW - timedelta(seconds=2)
    ctrl._read_runtime_instance = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            ("runtime-1", EXPIRATION_NOW - timedelta(minutes=10)),
            ("runtime-1", EXPIRATION_NOW - timedelta(minutes=10)),
            ("runtime-2", runtime_started_at),
        ]
    )
    ctrl._reconcile_deployment_sessions_after_restart = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )

    await ctrl._observe_runtime_instance(deployment)
    await ctrl._observe_runtime_instance(deployment)
    await ctrl._observe_runtime_instance(deployment)

    assert ctrl._runtime_instance_ids[(deployment.workspace, deployment.name)] == "runtime-2"
    ctrl._reconcile_deployment_sessions_after_restart.assert_awaited_once_with(
        deployment,
        at=runtime_started_at,
    )


@pytest.mark.asyncio
async def test_runtime_instance_change_is_recorded_before_retry() -> None:
    ctrl, _ = _make_controller()
    deployment = _make_fabric_deployment()
    key = (deployment.workspace, deployment.name)
    ctrl._runtime_instance_ids[key] = "runtime-1"
    runtime_started_at = EXPIRATION_NOW
    ctrl._read_runtime_instance = AsyncMock(  # type: ignore[method-assign]
        return_value=("runtime-2", runtime_started_at)
    )
    ctrl._reconcile_deployment_sessions_after_restart = AsyncMock(  # type: ignore[method-assign]
        return_value=False
    )

    await ctrl._observe_runtime_instance(deployment)
    assert ctrl._runtime_instance_ids[key] == "runtime-2"

    await ctrl._observe_runtime_instance(deployment)
    assert ctrl._runtime_instance_ids[key] == "runtime-2"
    ctrl._reconcile_deployment_sessions_after_restart.assert_awaited_once_with(
        deployment,
        at=runtime_started_at,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"runtime_instance_id": "runtime-1", "runtime_started_at": EXPIRATION_NOW.isoformat()},
            ("runtime-1", EXPIRATION_NOW),
        ),
        ({"runtime_instance_id": "", "runtime_started_at": EXPIRATION_NOW.isoformat()}, None),
        ({"runtime_instance_id": 1, "runtime_started_at": EXPIRATION_NOW.isoformat()}, None),
        ({"runtime_instance_id": "runtime-1", "runtime_started_at": "not-a-timestamp"}, None),
        ({"runtime_instance_id": "runtime-1", "runtime_started_at": "2026-08-26T12:00:00"}, None),
        ({}, None),
    ],
)
async def test_read_runtime_instance_validates_health_response(
    payload: dict[str, object],
    expected: tuple[str, datetime] | None,
) -> None:
    ctrl, _ = _make_controller()
    deployment = _make_fabric_deployment()
    response = MagicMock()
    response.json.return_value = payload
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    ctrl._runtime_health_client = client

    assert await ctrl._read_runtime_instance(deployment) == expected
    client.get.assert_awaited_once_with("http://localhost:9001/health")
    response.raise_for_status.assert_called_once_with()


@pytest.mark.asyncio
async def test_failed_restart_reconciliation_is_queued_and_retried() -> None:
    ctrl, _ = _make_controller()
    deployment = _make_fabric_deployment()
    ctrl._reconcile_sessions_after_restart = AsyncMock(  # type: ignore[method-assign]
        side_effect=[False, True]
    )

    reconciled = await ctrl._reconcile_deployment_sessions_after_restart(deployment, at=EXPIRATION_NOW)
    assert reconciled is False
    assert ctrl._pending_restart_reconciliations[deployment.id] == EXPIRATION_NOW

    await ctrl._retry_pending_restart_reconciliations()
    assert deployment.id not in ctrl._pending_restart_reconciliations
    assert ctrl._reconcile_sessions_after_restart.await_args_list == [
        call(deployment_id=deployment.id, at=EXPIRATION_NOW),
        call(deployment_id=deployment.id, at=EXPIRATION_NOW),
    ]


@pytest.mark.asyncio
async def test_newer_runtime_replacement_supersedes_pending_reconciliation_cutoff() -> None:
    ctrl, _ = _make_controller()
    deployment = _make_fabric_deployment()
    newer_runtime_started_at = EXPIRATION_NOW + timedelta(minutes=1)
    ctrl._pending_restart_reconciliations[deployment.id] = EXPIRATION_NOW
    ctrl._reconcile_sessions_after_restart = AsyncMock(return_value=False)  # type: ignore[method-assign]

    reconciled = await ctrl._reconcile_deployment_sessions_after_restart(
        deployment,
        at=newer_runtime_started_at,
    )

    assert reconciled is False
    assert ctrl._pending_restart_reconciliations[deployment.id] == newer_runtime_started_at
    ctrl._reconcile_sessions_after_restart.assert_awaited_once_with(
        deployment_id=deployment.id,
        at=newer_runtime_started_at,
    )


@pytest.mark.asyncio
async def test_backend_runtime_disappearance_reconciles_sessions_before_restart() -> None:
    ctrl, backend = _make_controller()
    deployment = _make_fabric_deployment()
    backend.get_deployment_status = AsyncMock(return_value=None)
    ctrl._reconcile_deployment_sessions_after_restart = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )

    await ctrl._verify_running(deployment)

    ctrl._reconcile_deployment_sessions_after_restart.assert_awaited_once_with(deployment)
    assert deployment.status == "pending"


# ---------------------------------------------------------------------------
# _start_deployment writes runtime fields without leaking the host log path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_deployment_writes_runtime_fields_to_entity() -> None:
    """Status / pid / port / endpoint are copied; ``log_path`` is NOT.

    The entity is the public, backend-agnostic schema served by the agents
    API.  The runner backend's ``log_path`` is a host-bound implementation
    detail of the in-memory backend and must not appear on the entity.
    The CLI computes the path itself from a shared convention.
    """
    ctrl, backend = _make_controller()
    backend.allocate_port = MagicMock(return_value=49200)
    backend.create_deployment = AsyncMock(
        return_value=DeploymentInfo(
            name="dep-1",
            status="starting",
            port=49200,
            pid=4242,
            endpoint="http://127.0.0.1:49200",
            log_path="/var/data/nemo/agents/system/dep-1.log",
        )
    )
    dep = AgentDeployment(name="dep-1", workspace="default", agent="calc", status="pending")

    await ctrl._start_deployment(dep)

    assert dep.status == "starting"
    assert dep.pid == 4242
    assert dep.port == 49200
    assert dep.endpoint == "http://127.0.0.1:49200"
    assert dep.error == ""
    # The entity must remain free of host-bound fields.
    assert not hasattr(dep, "log_path") or getattr(dep, "log_path", "") == ""
    # Startup timer is keyed by ``(workspace, name)``; ``_check_health``
    # reads from the same tuple. Asserting the key shape here catches the
    # silent string-vs-tuple drift that the prior pass surfaced.
    starting_key = ("default", "dep-1")
    assert starting_key in ctrl._starting_since


@pytest.mark.asyncio
async def test_start_deployment_records_the_spec_revision_the_backend_staged() -> None:
    ctrl, backend = _make_controller()
    backend.allocate_port = MagicMock(return_value=0)
    backend.create_deployment = AsyncMock(
        return_value=DeploymentInfo(
            name="dep-1",
            status="starting",
            staged_spec=SpecRevision(revision="2" * 40, tracked_revision="main"),
        )
    )
    dep = AgentDeployment(
        name="dep-1",
        workspace="default",
        agent="calc",
        status="pending",
        spec_revision="1" * 40,
        spec_tracked_revision="main",
    )

    await ctrl._start_deployment(dep)

    assert dep.spec_revision == "2" * 40
    assert dep.spec_tracked_revision == "main"


@pytest.mark.asyncio
async def test_start_deployment_keeps_the_spec_revision_when_nothing_was_staged() -> None:
    ctrl, backend = _make_controller()
    backend.allocate_port = MagicMock(return_value=0)
    backend.create_deployment = AsyncMock(return_value=DeploymentInfo(name="dep-1", status="starting"))
    dep = AgentDeployment(
        name="dep-1",
        workspace="default",
        agent="calc",
        status="pending",
        spec_revision="1" * 40,
        spec_tracked_revision="main",
    )

    await ctrl._start_deployment(dep)

    assert dep.spec_revision == "1" * 40
    assert dep.spec_tracked_revision == "main"


@pytest.mark.asyncio
async def test_start_deployment_forwards_image_entrypoint_mode() -> None:
    ctrl, backend = _make_controller()
    backend.allocate_port = MagicMock(return_value=0)
    backend.create_deployment = AsyncMock(return_value=DeploymentInfo(name="dep-1", status="starting"))
    dep = AgentDeployment(
        name="dep-1",
        workspace="default",
        agent="calc",
        status="pending",
        deployment_mode="docker",
        image="hand-built-agent:latest",
        use_image_entrypoint=True,
    )

    await ctrl._start_deployment(dep)

    await_args = backend.create_deployment.await_args
    assert await_args is not None
    kwargs = await_args.kwargs
    assert kwargs["image"] == "hand-built-agent:latest"
    assert kwargs["deployment_mode"] == "docker"
    assert kwargs["use_image_entrypoint"] is True
    assert dep.endpoint == ""
    assert dep.plugin_deployment == "dep-1"


@pytest.mark.asyncio
async def test_start_deployment_forwards_auth_context_to_backend() -> None:
    ctrl, backend = _make_controller()
    backend.allocate_port = MagicMock(return_value=0)
    backend.create_deployment = AsyncMock(return_value=DeploymentInfo(name="dep-1", status="starting"))
    auth_context = AuthContext(principal_id="user:alice", principal_groups=["research"])
    dep = AgentDeployment(
        name="dep-1",
        workspace="default",
        agent="calc",
        status="pending",
        deployment_mode="docker",
    ).with_auth_context(auth_context)

    await ctrl._start_deployment(dep)

    await_args = backend.create_deployment.await_args
    assert await_args is not None
    assert await_args.kwargs["auth_context"] == auth_context


# ---------------------------------------------------------------------------
# _check_health: dead process takes precedence over health probe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_health_marks_failed_when_subprocess_exited() -> None:
    """If ``get_deployment_status`` reports a dead subprocess, mark failed.

    Pins the short-circuit: even if the health probe would succeed (e.g. a
    stale port that briefly opened before the process exited), a dead
    subprocess wins.  Without this contract a deploy could be reported
    ``running`` while the process has actually died.
    """
    ctrl, backend = _make_controller()
    backend.get_deployment_status = AsyncMock(
        return_value=DeploymentInfo(
            name="dep-1",
            status="failed",
            error="Process exited with code 1",
        )
    )
    backend.health_check = AsyncMock(return_value=True)  # would otherwise lie
    dep = AgentDeployment(
        name="dep-1",
        workspace="default",
        agent="calc",
        status="starting",
        endpoint="http://127.0.0.1:49200",
    )
    ctrl._starting_since[("default", "dep-1")] = time.monotonic()

    await ctrl._check_health(dep)

    assert dep.status == "failed"
    assert "exited with code 1" in dep.error
    # health_check should not have been queried — the dead process check
    # short-circuits the function before reaching it.
    backend.health_check.assert_not_called()


@pytest.mark.asyncio
async def test_check_health_marks_running_when_healthy() -> None:
    """Backward behaviour: a healthy process flips to ``running``."""
    ctrl, backend = _make_controller()
    info = DeploymentInfo(name="dep-1", status="starting")
    backend.get_deployment_status = AsyncMock(return_value=info)
    backend.health_check = AsyncMock(return_value=True)
    dep = AgentDeployment(
        name="dep-1",
        workspace="default",
        agent="calc",
        status="starting",
        endpoint="http://127.0.0.1:49200",
    )
    ctrl._starting_since[("default", "dep-1")] = time.monotonic()

    await ctrl._check_health(dep)

    assert dep.status == "running"
    assert info.status == "running"

    ctrl._reconcile_deployment_sessions_after_restart = AsyncMock()  # type: ignore[method-assign]
    ctrl._observe_runtime_instance = AsyncMock()  # type: ignore[method-assign]
    await ctrl._verify_running(dep)
    ctrl._reconcile_deployment_sessions_after_restart.assert_not_awaited()


# ---------------------------------------------------------------------------
# Persisted session expiration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expire_session_transitions_due_session_and_cleans_runtime() -> None:
    ctrl, _ = _make_controller()
    session = _make_session(expires_at=EXPIRATION_NOW)
    ctrl.entities.update = AsyncMock(side_effect=lambda entity: entity)
    cleanup = _mock_runtime_cleanup(ctrl)

    result = await ctrl._expire_session_if_due(session, at=EXPIRATION_NOW)

    assert result is session
    assert session.status is SessionStatus.EXPIRED
    ctrl.entities.update.assert_awaited_once_with(session)
    cleanup.assert_called_once_with(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_at", [None, EXPIRATION_NOW + timedelta(seconds=1)])
async def test_expire_session_leaves_session_before_deadline_active(expires_at: datetime | None) -> None:
    ctrl, _ = _make_controller()
    session = _make_session(expires_at=expires_at)
    cleanup = _mock_runtime_cleanup(ctrl)

    result = await ctrl._expire_session_if_due(
        session,
        at=EXPIRATION_NOW,
    )

    assert result is None
    assert session.status is SessionStatus.ACTIVE
    cast(Any, ctrl.entities.update).assert_not_called()
    cleanup.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("refreshed_status", "deadline_delta_seconds", "expected_updates", "expected_cleanup"),
    [
        pytest.param(SessionStatus.ACTIVE, 30 * 60, 1, False, id="renewed-activity-wins"),
        pytest.param(SessionStatus.ACTIVE, -1, 2, True, id="still-due-retries"),
        pytest.param(SessionStatus.CLOSED, -1, 1, False, id="closed-wins"),
        pytest.param(SessionStatus.EXPIRED, -1, 1, True, id="concurrent-expiration-is-idempotent"),
    ],
)
async def test_expiration_conflict_resolves_from_refetched_session(
    refreshed_status: SessionStatus,
    deadline_delta_seconds: int,
    expected_updates: int,
    expected_cleanup: bool,
) -> None:
    ctrl, _ = _make_controller()
    stale = _make_session(expires_at=EXPIRATION_NOW - timedelta(seconds=2))
    refreshed = _make_session(
        expires_at=EXPIRATION_NOW + timedelta(seconds=deadline_delta_seconds),
        status=refreshed_status,
    )
    ctrl.entities.update = AsyncMock(side_effect=[NemoEntityConflictError("conflict"), refreshed])
    ctrl.entities.get_by_id = AsyncMock(return_value=refreshed)
    cleanup = _mock_runtime_cleanup(ctrl)

    result = await ctrl._expire_session_if_due(stale, at=EXPIRATION_NOW)

    assert (result is refreshed) is expected_cleanup
    expected_status = (
        SessionStatus.EXPIRED
        if refreshed_status is SessionStatus.ACTIVE and deadline_delta_seconds <= 0
        else refreshed_status
    )
    assert refreshed.status is expected_status
    assert ctrl.entities.update.await_count == expected_updates
    ctrl.entities.get_by_id.assert_awaited_once_with(AgentSession, stale.id)
    if expected_cleanup:
        cleanup.assert_called_once_with(refreshed)
    else:
        cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_active_session_listing_follows_pagination() -> None:
    ctrl, _ = _make_controller()
    first = MagicMock(data=[_make_session(name="one")])
    first.pagination = MagicMock(total_pages=2)
    second = MagicMock(data=[_make_session(name="two")])
    second.pagination = MagicMock(total_pages=2)
    ctrl.entities.list = AsyncMock(side_effect=[first, second])

    sessions = await ctrl._list_active_sessions()

    assert [session.name for session in sessions] == ["one", "two"]
    assert ctrl.entities.list.await_args_list == [
        call(
            AgentSession,
            workspace="-",
            filter_obj={"status": SessionStatus.ACTIVE.value},
            page=1,
            page_size=100,
        ),
        call(
            AgentSession,
            workspace="-",
            filter_obj={"status": SessionStatus.ACTIVE.value},
            page=2,
            page_size=100,
        ),
    ]
