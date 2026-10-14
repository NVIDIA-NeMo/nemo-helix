# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from nhx.core.entities.app.repository.sqlalchemy.models import DBWorkspace
from nhx.core.entities.app.repository.sqlalchemy.workspace import SQLAlchemyWorkspaceRepository
from nhx.core.entities.entities import WorkspaceDeletionStage
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


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
    )

    assert stale_renewal is None
    assert stale_failure is False
    assert await workspace_repo.fail_workspace_deletion_if_claimed(
        name=workspace.name,
        expected_updated_at=renewed.updated_at,
    )


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
