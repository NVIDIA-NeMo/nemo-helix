# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for direct SQLAlchemy entity repository tests."""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository, SQLAlchemyWorkspaceRepository
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def _entities_alembic_ini_path() -> Path:
    """Path to entities service alembic.ini (repository/ -> tests/ -> service root)."""
    return Path(__file__).resolve().parent.parent.parent / "alembic.ini"


def _run_migrations_on_url(url: str, alembic_ini_path: Path | None = None) -> None:
    """Run Alembic upgrade head against the given database URL (test use only)."""
    ini = alembic_ini_path or _entities_alembic_ini_path()
    if not ini.is_file():
        raise RuntimeError(f"alembic.ini not found at {ini}; cannot run migrations")
    sync_url = url.replace("sqlite+aiosqlite", "sqlite").replace("postgresql+asyncpg", "postgresql")
    cfg = Config(str(ini))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    command.upgrade(cfg, "head")


def _enable_sqlite_fks(dbapi_conn, connection_record):
    """Enable FK constraints for SQLite connections."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.close()


@pytest_asyncio.fixture
async def session_maker():
    """Create an async session maker with SQLite DB and Alembic migrations applied."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    sync_url = f"sqlite:///{db_path}"
    _run_migrations_on_url(sync_url, _entities_alembic_ini_path())
    async_url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(async_url, echo=False)
    event.listen(engine.sync_engine, "connect", _enable_sqlite_fks)
    session_maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    try:
        yield session_maker
    finally:
        await engine.dispose()
        Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def workspace_repo(session_maker):
    return SQLAlchemyWorkspaceRepository(session_maker)


@pytest.fixture
def entity_repo(session_maker):
    return SQLAlchemyEntityRepository(session_maker)


@pytest_asyncio.fixture
async def setup_workspaces(workspace_repo: SQLAlchemyWorkspaceRepository):
    """Create test workspaces."""
    await workspace_repo.create_workspace(name="workspace-1")
    await workspace_repo.create_workspace(name="workspace-2")
    await workspace_repo.create_workspace(name="workspace-3")
