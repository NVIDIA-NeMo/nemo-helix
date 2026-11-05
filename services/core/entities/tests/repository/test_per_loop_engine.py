# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-event-loop async engine tests.

In local single-process mode (``nemo services run``) the uvicorn lifespan and the
entities controller thread each run their own event loop. The engine registry keys
engines by ``asyncio.get_running_loop()`` so each loop gets its own connection pool
instead of sharing a process-global engine whose pool queue is bound to whichever
loop initialized it first. Sharing one engine made a pool-saturated loop raise
``RuntimeError: Queue is bound to a different event loop`` and pinned the
workspace_cleanup controller unhealthy (GitHub issue #1230).
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from alembic import command
from alembic.config import Config
from nhx.common.config.base import DatabaseConfig
from nhx.core.entities.app.repository import (
    SQLAlchemyWorkspaceRepository,
    dispose_async_engine,
    get_async_engine,
    get_async_session_maker,
    initialize_async_engine,
)
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.controllers.workspace_cleanup import WorkspaceCleanup
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import AsyncEngine


def _entities_config(db_path: Path) -> EntitiesConfig:
    return EntitiesConfig(database_config=DatabaseConfig(url=f"sqlite:///{db_path}"))


def _run_migrations(sync_url: str) -> None:
    """Apply entities migrations so workspace queries work (repository conftest equivalent)."""
    alembic_ini = Path(__file__).resolve().parent.parent.parent / "alembic.ini"
    assert alembic_ini.is_file(), f"alembic.ini not found at {alembic_ini}"
    cfg = Config(str(alembic_ini))
    # Alembic's ConfigParser treats "%" as interpolation; escape URL-encoded characters.
    cfg.set_main_option("sqlalchemy.url", sync_url.replace("%", "%%"))
    command.upgrade(cfg, "head")


class _LoopHarness:
    """One private event loop plus the engine it owns, like one service loop in a process."""

    def __init__(self, config: EntitiesConfig) -> None:
        self.loop = asyncio.new_event_loop()
        self.run(initialize_async_engine(config))

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def engine(self) -> AsyncEngine:
        return self.run(get_async_engine())

    def close(self) -> None:
        try:
            self.run(dispose_async_engine())
        finally:
            self.loop.close()


async def _select_one(session_maker: Any) -> None:
    async with session_maker() as session:
        await session.execute(text("SELECT 1"))


async def _hold_every_pooled_connection(engine: AsyncEngine) -> tuple[list[asyncio.Task], asyncio.Event]:
    """Check out every connection the engine's pool can provide and hold it until released."""
    pool = engine.pool
    size = getattr(pool, "size", None)
    max_overflow = getattr(pool, "_max_overflow", 0)
    assert callable(size), f"unexpected pool without a size(): {pool!r}"
    count = size() + max_overflow
    assert count >= 1, f"unexpected pool without capacity: {pool!r}"
    release = asyncio.Event()
    ready = asyncio.Event()
    held = 0

    async def _hold() -> None:
        nonlocal held
        async with engine.connect() as connection:
            await connection.exec_driver_sql("SELECT 1")
            held += 1
            if held == count:
                ready.set()
            await release.wait()

    tasks = [asyncio.ensure_future(_hold()) for _ in range(count)]
    await ready.wait()
    return tasks, release


async def _release_held_connections(tasks: list[asyncio.Task], release: asyncio.Event) -> None:
    release.set()
    await asyncio.gather(*tasks)


def test_a_second_loop_gets_its_own_engine_and_queries_succeed_while_the_first_loop_holds_every_connection(
    tmp_path: Path,
):
    config = _entities_config(tmp_path / "entities.db")
    first = _LoopHarness(config)
    second = _LoopHarness(config)  # same database URL, different event loop
    try:
        first_engine = first.engine()
        second_engine = second.engine()
        assert first_engine is not second_engine

        held_tasks, release = first.run(_hold_every_pooled_connection(first_engine))

        # Before per-loop engines the second loop would share the first loop's saturated
        # pool and this query would raise "Queue is bound to a different event loop".
        second_maker = second.run(get_async_session_maker())
        second.run(_select_one(second_maker))

        first.run(_release_held_connections(held_tasks, release))
    finally:
        first.close()
        second.close()


def test_the_same_loop_and_the_same_database_url_stay_idempotent(tmp_path: Path):
    config = _entities_config(tmp_path / "entities.db")
    harness = _LoopHarness(config)
    try:
        engine_before = harness.engine()
        maker_before = harness.run(get_async_session_maker())

        harness.run(initialize_async_engine(config))  # same loop, same URL: no-op

        assert harness.engine() is engine_before
        assert harness.run(get_async_session_maker()) is maker_before
    finally:
        harness.close()


def test_a_different_database_url_on_a_loop_is_refused_until_that_loops_engine_is_disposed(tmp_path: Path):
    first_config = _entities_config(tmp_path / "first.db")
    second_config = _entities_config(tmp_path / "second.db")
    harness = _LoopHarness(first_config)
    try:
        with pytest.raises(RuntimeError, match="already initialized for a different database"):
            harness.run(initialize_async_engine(second_config))

        harness.run(dispose_async_engine())
        harness.run(initialize_async_engine(second_config))

        assert harness.engine().url == make_url(second_config.get_async_sqlalchemy_url())
    finally:
        harness.close()


def test_disposing_one_loop_does_not_close_another_loops_engine(tmp_path: Path):
    config = _entities_config(tmp_path / "entities.db")
    first = _LoopHarness(config)
    second = _LoopHarness(config)
    try:
        second_engine = second.engine()

        first.run(dispose_async_engine())

        with pytest.raises(RuntimeError, match="not initialized"):
            first.engine()

        # The other loop's engine is untouched and still serves queries.
        second_maker = second.run(get_async_session_maker())
        second.run(_select_one(second_maker))
        assert second.engine() is second_engine
    finally:
        first.close()
        second.close()


def test_workspace_cleanup_step_is_healthy_when_another_loop_already_holds_a_saturated_pool(tmp_path: Path):
    db_path = tmp_path / "entities.db"
    _run_migrations(f"sqlite:///{db_path}")
    config = _entities_config(db_path)
    server_loop = _LoopHarness(config)  # the uvicorn lifespan loop wins the startup race
    controller_loop = _LoopHarness(config)  # the workspace_cleanup controller thread's loop
    try:
        server_engine = server_loop.engine()
        held_tasks, release = server_loop.run(_hold_every_pooled_connection(server_engine))

        session_maker = controller_loop.run(get_async_session_maker())
        repository = SQLAlchemyWorkspaceRepository(session_maker)
        controller = WorkspaceCleanup(client=MagicMock(), workspace_repository=repository, loop=controller_loop.loop)

        controller.step()

        assert controller.is_healthy

        server_loop.run(_release_held_connections(held_tasks, release))
    finally:
        server_loop.close()
        controller_loop.close()
