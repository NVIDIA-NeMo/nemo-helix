# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditional ID mutations across independent processes and database connections."""

import asyncio
import json
import os
import sys
from concurrent.futures import Future
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from nemo_helix_plugin.entities import EntityBase, EntityConflictError
from nhx.core.entities.api.dependencies import dep_entity_repository_with_session
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository, SQLAlchemyWorkspaceRepository
from nhx.core.entities.app.repository.exceptions import EntityVersionConflictError
from nhx.testing import ClientContext, create_test_client
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

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
    engine = create_async_engine(url, poolclass=NullPool)
    if request.param == "sqlite":

        @event.listens_for(engine.sync_engine, "connect")
        def enable_foreign_keys(connection, record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

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


@pytest.mark.asyncio
async def test_identical_update_advances_the_conditional_version(id_repository) -> None:
    repo, workspace, _ = id_repository
    original = await repo.create_entity(
        workspace=workspace, entity_type="widget", name="fence", data={"state": "provisional"}
    )
    fenced = await repo.update_entity(
        entity_id=original.id, data=original.data, expected_db_version=original.db_version
    )
    assert fenced.db_version == original.db_version + 1
    other = SQLAlchemyEntityRepository(repo.session_maker)
    with pytest.raises(EntityVersionConflictError):
        await other.update_entity(
            entity_id=original.id, data={"state": "created"}, expected_db_version=original.db_version
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
async def test_delayed_write_response_keeps_its_own_committed_version(id_repository, operation: str) -> None:
    repo, workspace, _ = id_repository
    original = await repo.create_entity(
        workspace=workspace, entity_type="widget", name="receipt", data={"state": "provisional"}
    )
    committed = asyncio.Event()
    deliver = asyncio.Event()
    async with repo.session_maker() as session:
        commit = session.commit

        async def delayed_commit() -> None:
            await commit()
            committed.set()
            await deliver.wait()

        with patch.object(session, "commit", side_effect=delayed_commit):
            if operation == "create":
                write = asyncio.create_task(
                    repo.create_entity(
                        workspace=workspace,
                        entity_type="widget",
                        name="creator",
                        data={"state": "provisional"},
                        session=session,
                    )
                )
                name = "creator"
                version = 1
            else:
                write = asyncio.create_task(
                    repo.update_entity(
                        entity_id=original.id,
                        data={"state": "provisional", "receipt": "owned"},
                        expected_db_version=original.db_version,
                        session=session,
                    )
                )
                name = original.name
                version = original.db_version + 1
            try:
                await asyncio.wait_for(committed.wait(), 10)
                other = SQLAlchemyEntityRepository(repo.session_maker)
                current = await other.get_entity_by_name(workspace=workspace, entity_type="widget", name=name)
                assert current is not None
                fenced = await other.update_entity(
                    entity_id=current.id, data={**current.data, "cleanup": True}, expected_db_version=current.db_version
                )
            finally:
                deliver.set()
            response = await asyncio.wait_for(write, 10)
    assert response.db_version == version
    assert "cleanup" not in response.data
    assert fenced.db_version == version + 1
    with pytest.raises(EntityVersionConflictError):
        await repo.update_entity(
            entity_id=response.id, data={"state": "created"}, expected_db_version=response.db_version
        )


@pytest.mark.asyncio
async def test_failed_commit_never_returns_a_snapshot(id_repository) -> None:
    repo, workspace, _ = id_repository
    async with repo.session_maker() as session:
        with patch.object(session, "commit", side_effect=RuntimeError("commit failed")):
            with pytest.raises(RuntimeError, match="commit failed"):
                await repo.create_entity(
                    workspace=workspace, entity_type="widget", name="uncommitted", data={}, session=session
                )
        await session.rollback()
    assert await repo.get_entity_by_name(workspace=workspace, entity_type="widget", name="uncommitted") is None


class PublicationOwner(EntityBase):
    __entity_type__: ClassVar[str] = "publication_owner"
    state: str = "provisional"
    cleanup: bool = False


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
async def test_committed_snapshot_survives_http_and_typed_entity_client(id_repository, operation: str) -> None:
    repo, workspace, _ = id_repository
    with create_test_client(
        client_type=ClientContext,
        workspaces=[workspace],
        projects=[],
        dependency_overrides={dep_entity_repository_with_session: lambda: repo},
    ) as ctx:
        store = ctx.entity_client
        initial = await store.create(PublicationOwner(name="original", workspace=workspace))
        committed: Future[None] = Future()
        deliver: Future[None] = Future()
        commit = AsyncSession.commit
        intercepted = False

        async def delayed_commit(session: AsyncSession) -> None:
            nonlocal intercepted
            await commit(session)
            if not intercepted:
                intercepted = True
                committed.set_result(None)
                await asyncio.wrap_future(deliver)

        with patch.object(AsyncSession, "commit", new=delayed_commit):
            owner = PublicationOwner(name="new-owner", workspace=workspace) if operation == "create" else initial
            creator = asyncio.create_task(store.create(owner) if operation == "create" else store.update(owner))
            try:
                await asyncio.wait_for(asyncio.wrap_future(committed), 10)
                other = SQLAlchemyEntityRepository(repo.session_maker)
                observed = await other.get_entity_by_name(
                    workspace=workspace, entity_type="publication_owner", name=owner.name
                )
                assert observed is not None
                fenced = await other.update_entity(
                    entity_id=observed.id,
                    data={**observed.data, "cleanup": True},
                    expected_db_version=observed.db_version,
                )
            finally:
                deliver.set_result(None)
            response = await asyncio.wait_for(creator, 10)
        assert response.cleanup is False
        assert response.db_version + 1 == fenced.db_version
        response.state = "created"
        with pytest.raises(EntityConflictError):
            await store.update(response)


@pytest.mark.asyncio
async def test_late_setup_children_cascade_or_fail_the_parent_foreign_key(id_repository) -> None:
    repo, workspace, _ = id_repository
    job = await repo.create_entity(
        workspace=workspace, entity_type="publication_owner", name="setup-owner", data={"state": "provisional"}
    )
    fenced = await repo.update_entity(entity_id=job.id, data=job.data, expected_db_version=job.db_version)
    other = SQLAlchemyEntityRepository(repo.session_maker)
    attempt = await other.create_entity(
        workspace=workspace, entity_type="attempt", name="late-attempt", parent=job.id, data={}
    )
    step = await other.create_entity(
        workspace=workspace, entity_type="step", name="late-step", parent=attempt.id, data={}
    )
    assert await repo.delete_entity(entity_id=job.id, expected_db_version=fenced.db_version) == 1
    assert await other.get_entity_by_id(entity_id=attempt.id) is None
    assert await other.get_entity_by_id(entity_id=step.id) is None
    with pytest.raises(IntegrityError):
        await other.create_entity(
            workspace=workspace, entity_type="attempt", name="after-removal", parent=job.id, data={}
        )
