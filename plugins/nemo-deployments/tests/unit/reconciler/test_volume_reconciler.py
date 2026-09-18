# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest
from helpers import make_volume
from nemo_deployments_plugin.backends.base import VolumeStatusUpdate
from nemo_deployments_plugin.reconciler.volume_reconciler import VolumeReconciler
from nemo_platform_plugin.entity_client import NemoEntityConflictError
from reconciler.conftest import MockDeploymentBackend


@pytest.mark.asyncio
async def test_pending_volume_becomes_bound(
    volume_reconciler: VolumeReconciler,
    mock_backend: MockDeploymentBackend,
    mock_entities: AsyncMock,
) -> None:
    vol = make_volume()
    mock_backend.volume_create_status = VolumeStatusUpdate(status="BOUND", status_message="ready")

    await volume_reconciler.reconcile_one(vol)

    assert vol.status == "BOUND"
    mock_entities.update.assert_awaited()


@pytest.mark.asyncio
async def test_volume_create_failure(
    volume_reconciler: VolumeReconciler,
    mock_backend: MockDeploymentBackend,
) -> None:
    vol = make_volume()

    async def fail(**kwargs: object) -> VolumeStatusUpdate:
        raise RuntimeError("docker unavailable")

    mock_backend.create_volume = fail  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    await volume_reconciler.reconcile_one(vol)

    assert vol.status == "FAILED"
    assert "docker unavailable" in vol.status_message


@pytest.mark.asyncio
async def test_deleting_volume_removes_backend_then_entity(
    volume_reconciler: VolumeReconciler,
    mock_backend: MockDeploymentBackend,
    mock_entities: AsyncMock,
) -> None:
    vol = make_volume()
    vol.status = "DELETING"

    await volume_reconciler.reconcile_one(vol)

    assert mock_backend.volume_delete_calls == [("default", "vol1")]
    mock_entities.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_deleting_volume_keeps_entity_when_backend_delete_fails(
    volume_reconciler: VolumeReconciler,
    mock_backend: MockDeploymentBackend,
    mock_entities: AsyncMock,
) -> None:
    vol = make_volume()
    vol.status = "DELETING"
    mock_backend.delete_volume = AsyncMock(
        return_value=VolumeStatusUpdate(
            status="FAILED",
            status_message="PVC deletion failed",
            error_details={"reason": "forbidden"},
        )
    )

    await volume_reconciler.reconcile_one(vol)

    mock_backend.delete_volume.assert_awaited_once()
    assert vol.status == "DELETING"
    assert vol.status_message == "PVC deletion failed"
    assert vol.error_details == {"reason": "forbidden"}
    mock_entities.update.assert_awaited_once_with(vol)
    mock_entities.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleting_volume_delete_conflict_is_handled_for_retry(
    volume_reconciler: VolumeReconciler,
    mock_backend: MockDeploymentBackend,
    mock_entities: AsyncMock,
) -> None:
    vol = make_volume()
    vol.status = "DELETING"
    mock_entities.delete.side_effect = NemoEntityConflictError("conflict")

    await volume_reconciler.reconcile_one(vol)

    assert mock_backend.volume_delete_calls == [("default", "vol1")]
    mock_entities.delete.assert_awaited_once_with(
        type(vol),
        name="vol1",
        workspace="default",
        expected_db_version=vol.db_version,
    )


@pytest.mark.asyncio
async def test_deleting_volume_waits_for_executor(
    volume_reconciler: VolumeReconciler,
    mock_entities: AsyncMock,
) -> None:
    from nemo_deployments_plugin.backends.registry import ExecutorRegistry

    empty_registry = ExecutorRegistry({}, default_executor=None)
    reconciler = VolumeReconciler(mock_entities, empty_registry)
    vol = make_volume()
    vol.status = "DELETING"

    await reconciler.reconcile_one(vol)

    mock_entities.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_volume_update_conflict_is_handled_for_retry(
    volume_reconciler: VolumeReconciler,
    mock_entities: AsyncMock,
) -> None:
    vol = make_volume()
    mock_entities.update.side_effect = NemoEntityConflictError("conflict")

    await volume_reconciler.reconcile_one(vol)

    mock_entities.update.assert_awaited()
