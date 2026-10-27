# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label-style JSON keys (``prefix/name``) through the real SQLite repository.

Agents filter on ``data.labels.<key>``. A key may hold ``/`` and ``-`` (but no
dots, which split the path), so check those characters survive JSON path access.
"""

from collections.abc import AsyncIterator

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator
from nhx.core.entities.app.repository.sqlalchemy.base import Base
from nhx.core.entities.app.repository.sqlalchemy.entity import SQLAlchemyEntityRepository
from nhx.core.entities.app.repository.sqlalchemy.models import DBWorkspace
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture
async def repository(tmp_path) -> AsyncIterator[SQLAlchemyEntityRepository]:
    """Yield a real repository over an isolated SQLite database in tmp_path."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'labels.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        session.add(DBWorkspace(id="workspace-default", name="default"))
        await session.commit()
    try:
        yield SQLAlchemyEntityRepository(sessions)
    finally:
        await engine.dispose()


async def _names(repository: SQLAlchemyEntityRepository, filter_op: ComparisonOperation) -> set[str]:
    rows, _ = await repository.list_entities(workspace="default", entity_type="agent", filter_op=filter_op)
    return {row.name for row in rows}


async def test_label_key_with_prefix_and_dashes_matches(repository):
    await repository.create_entity(
        workspace="default",
        entity_type="agent",
        name="analyst",
        data={"labels": {"helix-compass/target-agent": "default/support-bot"}},
    )
    await repository.create_entity(
        workspace="default",
        entity_type="agent",
        name="other-analyst",
        data={"labels": {"helix-compass/target-agent": "default/other-bot"}},
    )
    await repository.create_entity(workspace="default", entity_type="agent", name="unlabeled", data={})

    assert await _names(
        repository,
        ComparisonOperation(
            operator=FilterOperator.EQ, field="data.labels.helix-compass/target-agent", value="default/support-bot"
        ),
    ) == {"analyst"}
    assert await _names(
        repository,
        ComparisonOperation(
            operator=FilterOperator.IN,
            field="data.labels.helix-compass/target-agent",
            value=["default/support-bot", "default/other-bot"],
        ),
    ) == {"analyst", "other-analyst"}


async def test_workspace_qualified_ref_value_matches_with_in(repository):
    """Execute jobs list by agent with ``spec.agent_ref`` ``$in`` over ``workspace/name`` refs."""
    for name, ref in [("job-a", "research/x"), ("job-b", "default/y"), ("job-c", "research/y"), ("job-d", None)]:
        await repository.create_entity(
            workspace="default", entity_type="agent", name=name, data={"spec": {"agent_ref": ref}}
        )

    assert await _names(
        repository,
        ComparisonOperation(operator=FilterOperator.IN, field="data.spec.agent_ref", value=["research/x", "default/y"]),
    ) == {"job-a", "job-b"}
