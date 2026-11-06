# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON write/query safety on SQLite and optionally a real PostgreSQL database.

Set NHX_TEST_POSTGRES_URL to a disposable postgresql+asyncpg URL to run the
PostgreSQL cases. Each case creates and drops only its own random schema.
"""

import copy
import json
import os
from uuid import uuid4

import pytest
import pytest_asyncio
from nhx.common.api.filter import ComparisonOperation, FilterOperator
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository, SQLAlchemyWorkspaceRepository
from nhx.core.entities.app.repository.exceptions import InvalidFilterError
from nhx.core.entities.app.repository.sqlalchemy.base import Base
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest_asyncio.fixture(params=["sqlite", "postgresql"])
async def json_repository(request, session_maker):
    """Use existing SQLite fixtures or an isolated schema on an opted-in PG server."""
    engine = None
    schema = None
    try:
        if request.param == "postgresql":
            url = os.environ.get("NHX_TEST_POSTGRES_URL")
            if not url:
                pytest.skip("Set NHX_TEST_POSTGRES_URL to run real PostgreSQL JSON safety tests")
            schema = "json_safety_" + uuid4().hex
            engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
            async with engine.begin() as connection:
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                await connection.run_sync(Base.metadata.create_all)
            session_maker = async_sessionmaker(engine, expire_on_commit=False)
        await SQLAlchemyWorkspaceRepository(session_maker).create_workspace(name="json-safety")
        yield SQLAlchemyEntityRepository(session_maker)
    finally:
        if engine is not None:
            try:
                async with engine.begin() as connection:
                    await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            finally:
                await engine.dispose()


@pytest.mark.parametrize(
    "data",
    [
        {"nested": [{"value": "a\x00b"}]},
        {"nested": [{"a\x00b": "value"}]},
    ],
    ids=["nested-value", "nested-key"],
)
async def test_nul_create_rejected_without_persistence(json_repository, data):
    before = copy.deepcopy(data)
    with pytest.raises(ValueError, match=r"NUL character \(U\+0000\)"):
        await json_repository.create_entity(workspace="json-safety", entity_type="sample", name="bad", data=data)
    assert data == before
    rows, total = await json_repository.list_entities(workspace="json-safety", entity_type="sample")
    assert rows == [] and total == 0


@pytest.mark.parametrize("by_name", [False, True], ids=["update-id", "update-name"])
async def test_nul_update_preserves_stored_data_and_name(json_repository, by_name):
    original = await json_repository.create_entity(
        workspace="json-safety", entity_type="sample", name="original", data={"value": "safe"}
    )
    data = {"nested": [None, 1, True, {"value": "a\x00b"}]}
    before = copy.deepcopy(data)
    with pytest.raises(ValueError, match=r"NUL character \(U\+0000\)"):
        if by_name:
            await json_repository.update_entity_by_name(
                workspace="json-safety", entity_type="sample", name="original", new_name="changed", data=data
            )
        else:
            await json_repository.update_entity(entity_id=original.id, name="changed", data=data)
    assert data == before
    stored = await json_repository.get_entity_by_id(entity_id=original.id)
    assert stored is not None
    assert (stored.name, stored.data, stored.db_version) == (original.name, original.data, original.db_version)


async def test_safe_json_roundtrips_and_remains_filterable_and_sortable(json_repository):
    data = {"nested": [None, 1, True, {r"a\u0000b": r"a\u0000b"}], "key": "safe"}
    seed = await json_repository.create_entity(workspace="json-safety", entity_type="sample", name="safe", data=data)
    rows, total = await json_repository.list_entities(
        workspace="json-safety",
        entity_type="sample",
        sort="data.key",
        filter_op=ComparisonOperation(field="data.key", operator=FilterOperator.EQ, value="safe"),
    )
    assert total == 1
    assert [row.id for row in rows] == [seed.id]
    assert json.dumps(rows[0].data, sort_keys=True) == json.dumps(data, sort_keys=True)


@pytest.mark.parametrize(
    "operator,value",
    [(FilterOperator.EQ, "1"), (FilterOperator.EQ, None), (FilterOperator.IN, ["1"])],
)
async def test_bare_json_filter_is_explicitly_unsupported(json_repository, operator, value):
    with pytest.raises(InvalidFilterError, match="use dotted data paths"):
        await json_repository.list_entities(
            workspace="json-safety",
            entity_type="sample",
            filter_op=ComparisonOperation(field="data", operator=operator, value=value),
        )
