# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
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

    mock_backend.create_volume = fail  # type: ignore[method-assign]

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
async def test_deleting_volume_backend_failure_preserves_entity_for_retry(
    volume_reconciler: VolumeReconciler,
    mock_backend: MockDeploymentBackend,
    mock_entities: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A backend delete that reports FAILED must NOT delete the entity row.

    Backends signal a failed delete by returning a FAILED VolumeStatusUpdate
    rather than raising. Deleting the entity anyway orphans the underlying PVC
    while the model reports DELETED. The entity must survive so the next
    reconcile cycle retries the backend delete.
    """
    vol = make_volume()
    vol.status = "DELETING"
    mock_backend.volume_delete_status = VolumeStatusUpdate(status="FAILED", status_message="Failed to delete PVC: boom")

    with caplog.at_level(logging.WARNING, logger="nemo_deployments_plugin.reconciler.volume_reconciler"):
        await volume_reconciler.reconcile_one(vol)

    # Backend was asked to delete (so a retry occurs next cycle) ...
    assert mock_backend.volume_delete_calls == [("default", "vol1")]
    # ... but the entity row survives rather than orphaning the PVC.
    mock_entities.delete.assert_not_awaited()
    # ... and the retry is logged with the backend's reported status interpolated
    # (the %s renders the status word, so the message reads sensibly end to end).
    rendered = caplog.text
    assert "did not succeed (reported FAILED)" in rendered
    assert "will retry: Failed to delete PVC: boom" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_status", ["FAILED", "PENDING", "BOUND", "DELETING"])
async def test_deleting_volume_non_released_status_preserves_entity(
    volume_reconciler: VolumeReconciler,
    mock_backend: MockDeploymentBackend,
    mock_entities: AsyncMock,
    delete_status: str,
) -> None:
    """ANY non-RELEASED backend delete status must preserve the entity for retry.

    The reconciler guards on ``status != "RELEASED"`` rather than ``== "FAILED"``,
    so the entity survives for every non-success status — pinning the
    backend-agnostic contract against a future regression to a FAILED-only check.
    Only RELEASED (see ``test_deleting_volume_removes_backend_then_entity``) may
    remove the entity row.
    """
    vol = make_volume()
    vol.status = "DELETING"
    mock_backend.volume_delete_status = VolumeStatusUpdate(status=delete_status)  # type: ignore[arg-type]

    await volume_reconciler.reconcile_one(vol)

    assert mock_backend.volume_delete_calls == [("default", "vol1")]
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
