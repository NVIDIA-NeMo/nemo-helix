# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for per-event-loop engine initialization (NVBug 6588975 / #1230).

The service process runs at least two event loops: the uvicorn lifespan loop
(service.py) and the controller worker loop (controllers/main.py daemon
thread).  asyncpg pools bind their internal waiters to the loop that first
uses them, so a process-global engine singleton makes cross-loop acquires fail
with "RuntimeError: <Queue> is bound to a different event loop" once the pool
saturates.  These tests pin the per-loop contract without requiring postgres:
each loop must get its own engine/session maker, and disposal on one loop must
not affect another.
"""

import asyncio
import tempfile
import threading

import pytest
from nmp.core.entities.app.repository import (
    dispose_async_engine,
    get_async_session_maker,
    initialize_async_engine,
    ping_database,
)
from nmp.core.entities.config import EntitiesConfig


def _sqlite_config() -> EntitiesConfig:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    return EntitiesConfig(database_url=f"sqlite+aiosqlite:///{db_path}")


def _run_in_fresh_loop(coro_factory):
    """Run a coroutine in a brand-new event loop on a separate thread.

    Mirrors how controllers/main.py runs: a daemon thread with its own loop.
    Returns the coroutine result (or raises its exception).
    """
    result: dict = {}

    def runner():
        loop = asyncio.new_event_loop()
        try:
            result["value"] = loop.run_until_complete(coro_factory())
        except BaseException as exc:  # noqa: BLE001 - propagate to caller
            result["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=60)
    if "error" in result:
        raise result["error"]
    return result.get("value")


@pytest.mark.asyncio
async def test_each_loop_gets_its_own_session_maker():
    """Two loops initializing the same config must not share a session maker."""
    config = _sqlite_config()

    await initialize_async_engine(config)
    maker_service_loop = await get_async_session_maker()

    async def controller_side():
        await initialize_async_engine(config)
        return await get_async_session_maker()

    maker_controller_loop = _run_in_fresh_loop(controller_side)

    assert maker_controller_loop is not None
    assert maker_service_loop is not maker_controller_loop, (
        "session maker leaked across event loops — this is the exact sharing "
        "that produces 'Queue is bound to a different event loop' (6588975)"
    )
    await dispose_async_engine()


@pytest.mark.asyncio
async def test_second_init_on_same_loop_is_noop():
    """The once-per-loop contract: re-initializing on one loop keeps the maker."""
    config = _sqlite_config()
    await initialize_async_engine(config)
    first = await get_async_session_maker()
    await initialize_async_engine(config)
    second = await get_async_session_maker()
    assert first is second
    await dispose_async_engine()


@pytest.mark.asyncio
async def test_uninitialized_loop_raises_clear_error():
    """A loop that never initialized must get the explicit RuntimeError."""
    config = _sqlite_config()
    await initialize_async_engine(config)

    async def never_initialized():
        try:
            await get_async_session_maker()
        except RuntimeError as exc:
            return str(exc)
        return None

    message = _run_in_fresh_loop(never_initialized)
    assert message is not None and "not initialized on this event loop" in message
    await dispose_async_engine()


@pytest.mark.asyncio
async def test_dispose_only_affects_current_loop():
    """Controller shutdown must not yank the service loop's engine."""
    config = _sqlite_config()
    await initialize_async_engine(config)
    assert await ping_database() is True

    async def controller_lifecycle():
        await initialize_async_engine(config)
        assert await ping_database() is True
        await dispose_async_engine()  # controller shuts down its own engine
        return await ping_database()  # controller loop: gone

    controller_ping_after_dispose = _run_in_fresh_loop(controller_lifecycle)
    assert controller_ping_after_dispose is False

    # service loop unaffected
    assert await ping_database() is True
    await dispose_async_engine()


@pytest.mark.asyncio
async def test_cross_loop_usage_does_not_touch_foreign_pool():
    """The 6588975 scenario shape: loop B working while loop A's pool exists.

    With per-loop engines the controller loop's queries run on its own pool,
    so no cross-loop waiter binding is possible by construction.
    """
    config = _sqlite_config()
    await initialize_async_engine(config)

    async def controller_query_burst():
        await initialize_async_engine(config)
        results = []
        for _ in range(10):
            results.append(await ping_database())
        await dispose_async_engine()
        return all(results)

    assert _run_in_fresh_loop(controller_query_burst) is True
    assert await ping_database() is True
    await dispose_async_engine()
