# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditional ID mutations across independent processes and database connections."""

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository, SQLAlchemyWorkspaceRepository
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

pytestmark = pytest.mark.integration

# Keep a stale ORM identity across commit, then let another process change the row.
# The repository's initial version check sees the stale identity; commit must still
# reject the write using the database version, without any shared Python lock.
WORKER = """
import asyncio,json,sys
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository
from nhx.core.entities.app.repository.exceptions import EntityVersionConflictError
from nhx.core.entities.app.repository.sqlalchemy.models import DBEntity
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker,create_async_engine

async def main():
    args=json.loads(sys.stdin.readline())
    engine=create_async_engine(args["url"])
    sessions=async_sessionmaker(engine,expire_on_commit=False)
    repo=SQLAlchemyEntityRepository(sessions)
    try:
        async with sessions() as session:
            row=(await session.execute(select(DBEntity).where(DBEntity.id==args["id"]))).scalar_one()
            version=row.db_version
            await session.commit()
            print("ready",flush=True)
            await asyncio.to_thread(sys.stdin.readline)
            try:
                if args["operation"]=="update":
                    await repo.update_entity(entity_id=row.id,data={"value":"stale"},expected_db_version=version,session=session)
                else:
                    await repo.delete_entity(entity_id=row.id,expected_db_version=version,session=session)
            except EntityVersionConflictError:
                print("conflict",flush=True)
            else:
                print("incorrect-success",flush=True)
    finally:
        await engine.dispose()
asyncio.run(main())
"""


@pytest.fixture(params=["sqlite", "postgresql"])
async def id_repository(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "sqlite":
        url = f"sqlite+aiosqlite:///{tmp_path / 'entities.db'}"
    else:
        url = os.environ.get("P123_TEST_POSTGRES_URL")
        if not url:
            pytest.skip("Set P123_TEST_POSTGRES_URL for the isolated PostgreSQL integration run")
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    config.set_main_option(
        "sqlalchemy.url", url.replace("sqlite+aiosqlite", "sqlite").replace("postgresql+asyncpg", "postgresql")
    )
    await asyncio.to_thread(command.upgrade, config, "head")
    engine = create_async_engine(url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    workspace = f"id-cas-{uuid4().hex}"
    await SQLAlchemyWorkspaceRepository(sessions).create_workspace(name=workspace)
    try:
        yield SQLAlchemyEntityRepository(sessions), workspace, url
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["update", "delete"])
async def test_id_mutation_rejects_a_stale_process(id_repository, operation: str) -> None:
    repo, workspace, url = id_repository
    original = await repo.create_entity(
        workspace=workspace, entity_type="widget", name="same-name", data={"value": "original"}
    )
    worker = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        WORKER,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert worker.stdin is not None and worker.stdout is not None
    worker.stdin.write((json.dumps({"url": url, "id": original.id, "operation": operation}) + "\n").encode())
    await worker.stdin.drain()
    try:
        assert await asyncio.wait_for(worker.stdout.readline(), 20) == b"ready\n"
        updated = await repo.update_entity(
            entity_id=original.id, data={"value": "fresh"}, expected_db_version=original.db_version
        )
        worker.stdin.write(b"continue\n")
        await worker.stdin.drain()
        stdout, stderr = await asyncio.wait_for(worker.communicate(), 20)
        assert worker.returncode == 0, stderr.decode()
        assert stdout.strip() == b"conflict", stderr.decode()
        current = await repo.get_entity_by_id(entity_id=original.id)
        assert current is not None
        assert current.db_version == updated.db_version
        assert current.data == {"value": "fresh"}
    finally:
        if worker.returncode is None:
            worker.kill()
            await worker.wait()
