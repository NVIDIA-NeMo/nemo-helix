# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database initialization helpers for Entity Store."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from nmp.common.entities.constants import DEFAULT_WORKSPACE, SYSTEM_WORKSPACE
from nmp.core.entities.app.repository.sqlalchemy.models import DBWorkspace
from nmp.core.entities.utils.identifiers import generate_entity_id
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

WORKSPACE_DESCRIPTIONS = {
    DEFAULT_WORKSPACE: "General-purpose workspace (all users have write access)",
    SYSTEM_WORKSPACE: "Platform-provided resources (read-only for users)",
}


async def ensure_default_workspace(
    session: AsyncSession, workspace_name: str = DEFAULT_WORKSPACE
) -> Optional[DBWorkspace]:
    """Ensure a default workspace exists.

    Args:
        session: SQLAlchemy async session
        workspace_name: Workspace name (default: "default")

    Returns:
        The default workspace (existing or newly created)
    """
    result = await session.execute(select(DBWorkspace).filter(DBWorkspace.name == workspace_name))
    workspace = result.scalar_one_or_none()

    if workspace is None:
        description = WORKSPACE_DESCRIPTIONS.get(workspace_name, f"{workspace_name} workspace")
        logger.info(f"Creating default workspace: {workspace_name}")
        workspace = DBWorkspace(
            id=generate_entity_id("workspace"),
            name=workspace_name,
            description=description,
        )
        try:
            session.add(workspace)
            await session.commit()
            logger.info(f"✓ Created default workspace: {workspace_name}")
        except IntegrityError:
            await session.rollback()
            result = await session.execute(select(DBWorkspace).filter(DBWorkspace.name == workspace_name))
            workspace = result.scalar_one_or_none()
            logger.info(f"Default workspace already exists: {workspace_name}")
    else:
        logger.debug(f"Default workspace already exists: {workspace_name}")

    return workspace


def _find_alembic_ini() -> Optional[Path]:
    """Return path to alembic.ini, or None if not found.

    Prefer installed package data path first, then repo service root path.
    """
    package_root = Path(__file__).resolve().parent
    candidate_paths = [
        # Installed package path (wheel force-includes alembic into nmp/core/entities).
        package_root / "alembic.ini",
        # Repo layout path (services/core/entities/alembic.ini).
        package_root.parent.parent.parent.parent / "alembic.ini",
    ]
    for ini in candidate_paths:
        if ini.is_file():
            return ini
    return None


def _run_alembic_upgrade(alembic_ini_path: Path, database_url: str) -> None:
    """Run alembic upgrade head (sync, for use in thread).

    Converts async driver URLs to sync (e.g. sqlite+aiosqlite -> sqlite) for Alembic.
    """
    sync_url = database_url.replace("sqlite+aiosqlite", "sqlite").replace("postgresql+asyncpg", "postgresql")
    cfg = Config(str(alembic_ini_path))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    command.upgrade(cfg, "head")


def _engine_url_for_alembic(engine) -> str:
    """Return a non-masked URL string for Alembic.

    SQLAlchemy URL.__str__ masks passwords as ***. Alembic needs the real URL.
    """
    return engine.url.render_as_string(hide_password=False)


async def initialize_database(run_migrations: bool = True) -> None:
    """Initialize database: run migrations, then ensure default workspace.

    Uses the already-initialized async engine from initialize_async_engine().

    Args:
        run_migrations: If True, run Alembic upgrade head at startup.
    """
    from nmp.core.entities.app.repository import get_async_engine, get_async_session_maker

    logger.info("Initializing entity store database...")

    engine = await get_async_engine()

    if run_migrations:
        alembic_ini = _find_alembic_ini()
        if alembic_ini:
            logger.info("Running database migrations (alembic upgrade head)...")
            try:
                await asyncio.to_thread(_run_alembic_upgrade, alembic_ini, _engine_url_for_alembic(engine))
                logger.info("✓ Database migrations applied")
            except Exception:
                logger.exception("Database migrations failed")
                raise
        else:
            logger.warning("alembic.ini not found; skipping migrations")

    session_maker = await get_async_session_maker()
    async with session_maker() as session:
        await ensure_default_workspace(session, DEFAULT_WORKSPACE)
        await ensure_default_workspace(session, SYSTEM_WORKSPACE)

    logger.info("✓ Entity store database initialized")
