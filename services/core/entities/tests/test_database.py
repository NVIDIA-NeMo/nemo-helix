# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the entities database engine."""

import asyncio
from pathlib import Path

import pytest
from nhx.common.config.base import DatabaseConfig
from nhx.core.entities.app.database import create_async_engine_for_entities
from nhx.core.entities.config import EntitiesConfig


@pytest.mark.asyncio
async def test_close_completes_when_force_stop_lands_first(tmp_path: Path) -> None:
    """A graceful close must finish even if SQLAlchemy's force-close stops the worker first.

    When a connection is invalidated during shutdown, SQLAlchemy runs aiosqlite's ``close()`` in a
    shielded task and, if that wait is cancelled, falls back to ``stop()``. aiosqlite's worker thread
    exits after the first stop it handles, so the stop queued by ``close()``'s ``finally`` would
    never resolve and the close task would hang teardown.
    """
    config = EntitiesConfig(database_config=DatabaseConfig(url=f"sqlite:///{tmp_path}/entities.db"))
    engine = create_async_engine_for_entities(config)
    try:
        async with engine.connect() as conn:
            raw = await conn.get_raw_connection()
            driver = raw.driver_connection
            assert driver is not None

            close_task = asyncio.create_task(driver.close())
            await asyncio.sleep(0)  # close() has queued sqlite3's close on the worker thread
            driver.stop()  # SQLAlchemy's _terminate_force_close(), queued behind it

            await asyncio.wait_for(close_task, timeout=5)
            await conn.invalidate()
    finally:
        await engine.dispose()


def test_postgres_pool_size_follows_connections_limit() -> None:
    """The pool keeps ``connections_limit`` connections open rather than SQLAlchemy's default of 5."""
    config = EntitiesConfig(
        database_config=DatabaseConfig(url="postgresql+asyncpg://user:pw@localhost:5432/entities", connections_limit=12)
    )
    engine = create_async_engine_for_entities(config)
    assert engine.sync_engine.pool.size() == 12


POSTGRES_URL = "postgresql+asyncpg://user:pw@localhost:5432/entities"


@pytest.mark.parametrize(
    ("configured", "argument", "expected"),
    [
        (None, None, False),  # off by default: a ping costs about three round trips per checkout
        (True, None, True),
        (False, True, True),  # an explicit argument wins over the config
    ],
)
def test_pre_ping_follows_config_unless_overridden(
    configured: bool | None, argument: bool | None, expected: bool
) -> None:
    database_config = (
        DatabaseConfig(url=POSTGRES_URL)
        if configured is None
        else DatabaseConfig(url=POSTGRES_URL, pool_pre_ping=configured)
    )
    engine = create_async_engine_for_entities(EntitiesConfig(database_config=database_config), pool_pre_ping=argument)
    assert engine.sync_engine.pool._pre_ping is expected


@pytest.mark.parametrize(("recycle_seconds", "expected"), [(1800, 1800), (600, 600), (0, -1)])
def test_postgres_pool_recycle_follows_config(recycle_seconds: int, expected: int) -> None:
    """0 disables recycling, which SQLAlchemy records as -1."""
    config = EntitiesConfig(database_config=DatabaseConfig(url=POSTGRES_URL, pool_recycle_seconds=recycle_seconds))
    engine = create_async_engine_for_entities(config)
    assert engine.sync_engine.pool._recycle == expected
