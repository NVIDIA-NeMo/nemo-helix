# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.models.client import AsyncModelsClient
from nhx.core.entities.app.repository.sqlalchemy.models import DBWorkspace
from nhx.core.entities.app.repository.sqlalchemy.workspace import SQLAlchemyWorkspaceRepository
from nhx.core.entities.controllers.workspace_cleanup import WorkspaceCleanup
from nhx.core.entities.entities import WorkspaceDeletionStage
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class _EmptyPage:
    async def items(self):
        if False:
            yield None


@contextmanager
def _empty_cleanup_clients():
    jobs_client = MagicMock()
    jobs_client.list_jobs = AsyncMock(return_value=_EmptyPage())
    models_client = MagicMock()
    models_client.list_deployments = AsyncMock(return_value=_EmptyPage())
    models_client.list_adapters = AsyncMock(return_value=_EmptyPage())
    models_client.list_models = AsyncMock(return_value=_EmptyPage())
    files_client = MagicMock()
    files_client.list_filesets = AsyncMock(return_value=_EmptyPage())
    with (
        patch.object(AsyncJobsClient, "from_client", return_value=jobs_client),
        patch.object(AsyncModelsClient, "from_client", return_value=models_client),
        patch.object(AsyncFilesClient, "from_client", return_value=files_client),
    ):
        yield


async def _seed_deletion_stage(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    name: str,
    stage: str,
    updated_at: datetime,
) -> None:
    async with session_maker() as session:
        await session.execute(
            update(DBWorkspace).where(DBWorkspace.name == name).values(deletion_stage=stage, updated_at=updated_at)
        )
        await session.commit()


@pytest.mark.asyncio
async def test_claims_deleting_workspace_after_lease_expires(
    workspace_repo: SQLAlchemyWorkspaceRepository,
    session_maker: async_sessionmaker[AsyncSession],
) -> None:
    await workspace_repo.create_workspace(name="stale")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    stale_timestamp = now - timedelta(minutes=2)
    await _seed_deletion_stage(
        session_maker,
        name="stale",
        stage=WorkspaceDeletionStage.DELETING,
        updated_at=stale_timestamp,
    )

    claimed = await workspace_repo.claim_workspace_for_deletion(
        name="stale",
        expected_stage=WorkspaceDeletionStage.DELETING,
        expected_updated_at=stale_timestamp,
        lease_expired_before=now - timedelta(minutes=1),
    )

    assert claimed is not None
    assert claimed._deletion_stage == WorkspaceDeletionStage.DELETING
    assert claimed.updated_at > stale_timestamp


@pytest.mark.asyncio
async def test_does_not_claim_deleting_workspace_inside_lease(
    workspace_repo: SQLAlchemyWorkspaceRepository,
    session_maker: async_sessionmaker[AsyncSession],
) -> None:
    await workspace_repo.create_workspace(name="fresh")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    await _seed_deletion_stage(
        session_maker,
        name="fresh",
        stage=WorkspaceDeletionStage.DELETING,
        updated_at=now,
    )

    claimed = await workspace_repo.claim_workspace_for_deletion(
        name="fresh",
        expected_stage=WorkspaceDeletionStage.DELETING,
        expected_updated_at=now,
        lease_expired_before=now - timedelta(minutes=1),
    )

    assert claimed is None
    persisted = await workspace_repo.get_workspace_by_name(name="fresh")
    assert persisted is not None
    assert persisted.updated_at == now


@pytest.mark.asyncio
async def test_only_one_replica_claims_same_workspace(
    workspace_repo: SQLAlchemyWorkspaceRepository,
    session_maker: async_sessionmaker[AsyncSession],
) -> None:
    await workspace_repo.create_workspace(name="contended")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    stale_timestamp = now - timedelta(minutes=2)
    await _seed_deletion_stage(
        session_maker,
        name="contended",
        stage=WorkspaceDeletionStage.DELETING,
        updated_at=stale_timestamp,
    )

    claims = await asyncio.gather(
        *(
            workspace_repo.claim_workspace_for_deletion(
                name="contended",
                expected_stage=WorkspaceDeletionStage.DELETING,
                expected_updated_at=stale_timestamp,
                lease_expired_before=now - timedelta(minutes=1),
            )
            for _ in range(2)
        )
    )

    assert sum(claim is not None for claim in claims) == 1


@pytest.mark.asyncio
async def test_renewal_fences_stale_holder_from_failure_transition(
    workspace_repo: SQLAlchemyWorkspaceRepository,
) -> None:
    workspace = await workspace_repo.create_workspace(name="renewed")
    await workspace_repo.mark_workspace_for_deletion(
        name=workspace.name,
        deletion_stage=WorkspaceDeletionStage.PENDING,
    )
    pending = await workspace_repo.get_workspace_by_name(name=workspace.name)
    assert pending is not None

    claimed = await workspace_repo.claim_workspace_for_deletion(
        name=workspace.name,
        expected_stage=WorkspaceDeletionStage.PENDING,
        expected_updated_at=pending.updated_at,
    )
    assert claimed is not None
    renewed = await workspace_repo.renew_workspace_deletion_lease(
        name=workspace.name,
        expected_updated_at=claimed.updated_at,
    )
    assert renewed is not None

    stale_renewal = await workspace_repo.renew_workspace_deletion_lease(
        name=workspace.name,
        expected_updated_at=claimed.updated_at,
    )
    stale_failure = await workspace_repo.fail_workspace_deletion_if_claimed(
        name=workspace.name,
        expected_updated_at=claimed.updated_at,
        deletion_error="stale holder",
    )

    assert stale_renewal is None
    assert stale_failure is False
    assert await workspace_repo.fail_workspace_deletion_if_claimed(
        name=workspace.name,
        expected_updated_at=renewed.updated_at,
        deletion_error="current holder",
    )
    failed = await workspace_repo.get_workspace_by_name(name=workspace.name)
    assert failed is not None
    assert failed._deletion_stage == WorkspaceDeletionStage.FAILED
    assert failed._deletion_error == "current holder"


@pytest.mark.asyncio
async def test_delete_is_fenced_by_current_lease(
    workspace_repo: SQLAlchemyWorkspaceRepository,
) -> None:
    workspace = await workspace_repo.create_workspace(name="delete-fence")
    await workspace_repo.mark_workspace_for_deletion(
        name=workspace.name,
        deletion_stage=WorkspaceDeletionStage.PENDING,
    )
    pending = await workspace_repo.get_workspace_by_name(name=workspace.name)
    assert pending is not None
    claimed = await workspace_repo.claim_workspace_for_deletion(
        name=workspace.name,
        expected_stage=WorkspaceDeletionStage.PENDING,
        expected_updated_at=pending.updated_at,
    )
    assert claimed is not None
    renewed = await workspace_repo.renew_workspace_deletion_lease(
        name=workspace.name,
        expected_updated_at=claimed.updated_at,
    )
    assert renewed is not None

    assert not await workspace_repo.delete_workspace_if_claimed(
        name=workspace.name,
        expected_updated_at=claimed.updated_at,
    )
    assert await workspace_repo.delete_workspace_if_claimed(
        name=workspace.name,
        expected_updated_at=renewed.updated_at,
    )
    assert await workspace_repo.get_workspace_by_name(name=workspace.name) is None


@pytest.mark.asyncio
async def test_controller_reclaims_expired_deletion_with_real_repository(
    workspace_repo: SQLAlchemyWorkspaceRepository,
    session_maker: async_sessionmaker[AsyncSession],
) -> None:
    await workspace_repo.create_workspace(name="restart-recovery")
    stale_timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=2)
    await _seed_deletion_stage(
        session_maker,
        name="restart-recovery",
        stage=WorkspaceDeletionStage.DELETING,
        updated_at=stale_timestamp,
    )
    controller = WorkspaceCleanup(
        client=MagicMock(),
        workspace_repository=workspace_repo,
    )

    with _empty_cleanup_clients():
        await controller._async_step()

    assert await workspace_repo.get_workspace_by_name(name="restart-recovery") is None
    recreated = await workspace_repo.create_workspace(name="restart-recovery")
    assert recreated.name == "restart-recovery"


@pytest.mark.asyncio
async def test_controller_leaves_recent_deletion_with_real_repository(
    workspace_repo: SQLAlchemyWorkspaceRepository,
    session_maker: async_sessionmaker[AsyncSession],
) -> None:
    await workspace_repo.create_workspace(name="active-cleaner")
    recent_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
    await _seed_deletion_stage(
        session_maker,
        name="active-cleaner",
        stage=WorkspaceDeletionStage.DELETING,
        updated_at=recent_timestamp,
    )
    controller = WorkspaceCleanup(
        client=MagicMock(),
        workspace_repository=workspace_repo,
    )

    with _empty_cleanup_clients():
        await controller._async_step()

    persisted = await workspace_repo.get_workspace_by_name(name="active-cleaner")
    assert persisted is not None
    assert persisted.updated_at == recent_timestamp


@pytest.mark.asyncio
async def test_failed_cleanup_stores_error_and_retry_claim_keeps_it(
    workspace_repo: SQLAlchemyWorkspaceRepository,
) -> None:
    workspace = await workspace_repo.create_workspace(name="blocked-fileset")
    await workspace_repo.mark_workspace_for_deletion(
        name=workspace.name,
        deletion_stage=WorkspaceDeletionStage.PENDING,
    )
    pending = await workspace_repo.get_workspace_by_name(name=workspace.name)
    assert pending is not None
    claimed = await workspace_repo.claim_workspace_for_deletion(
        name=workspace.name,
        expected_stage=WorkspaceDeletionStage.PENDING,
        expected_updated_at=pending.updated_at,
    )
    assert claimed is not None

    reason = "shared-weights: HTTP 409: Cannot delete fileset because 1 model entity reference(s) still use it"
    assert await workspace_repo.fail_workspace_deletion_if_claimed(
        name=workspace.name,
        expected_updated_at=claimed.updated_at,
        deletion_error=reason + (" x" * 2000),
    )
    failed = await workspace_repo.get_workspace_by_name(name=workspace.name)
    assert failed is not None
    assert failed._deletion_stage == WorkspaceDeletionStage.FAILED
    assert failed._deletion_error is not None
    assert failed._deletion_error.startswith(reason)
    assert len(failed._deletion_error) == 2000
    assert failed._deletion_error.endswith("...")

    retried = await workspace_repo.claim_workspace_for_deletion(
        name=workspace.name,
        expected_stage=WorkspaceDeletionStage.FAILED,
        expected_updated_at=failed.updated_at,
    )
    assert retried is not None
    assert retried._deletion_stage == WorkspaceDeletionStage.DELETING
    assert retried._deletion_error == failed._deletion_error

    assert not await workspace_repo.fail_workspace_deletion_if_claimed(
        name=workspace.name,
        expected_updated_at=failed.updated_at,
        deletion_error="stale holder must not replace the reason",
    )
    unchanged = await workspace_repo.get_workspace_by_name(name=workspace.name)
    assert unchanged is not None
    assert unchanged._deletion_error == failed._deletion_error

    assert await workspace_repo.fail_workspace_deletion_if_claimed(
        name=workspace.name,
        expected_updated_at=retried.updated_at,
        deletion_error="shared-weights: HTTP 409: still referenced",
    )
    replaced = await workspace_repo.get_workspace_by_name(name=workspace.name)
    assert replaced is not None
    assert replaced._deletion_stage == WorkspaceDeletionStage.FAILED
    assert replaced._deletion_error == "shared-weights: HTTP 409: still referenced"
