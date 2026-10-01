# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for asyncio teardown diagnostics."""

import asyncio
import asyncio.runners
import io
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.testing.asyncio_debug import (
    AsyncioTeardownError,
    bounded_event_loop_teardown,
    dump_event_loop_tasks,
    format_task_stack,
)


async def _swallow_cancellation(started: asyncio.Event) -> None:
    """Mimic a background loop that catches CancelledError and keeps waiting."""
    started.set()
    while True:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            continue


async def _inner_wait(release: asyncio.Event) -> None:
    await release.wait()


async def _outer_wait(release: asyncio.Event) -> None:
    await _inner_wait(release)


async def test_format_task_stack_follows_the_await_chain():
    release = asyncio.Event()
    task = asyncio.create_task(_outer_wait(release), name="waiter")
    await asyncio.sleep(0)

    rendered = format_task_stack(task)

    assert rendered.startswith("Task 'waiter' (pending):")
    assert rendered.index("_outer_wait") < rendered.index("_inner_wait")
    assert rendered.rstrip().splitlines()[-2].endswith("in Event.wait")
    release.set()
    await task


def test_dump_event_loop_tasks_reports_tasks_on_another_threads_loop():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    async def _main() -> None:
        ready.set()
        await asyncio.sleep(3600)

    def _run() -> None:
        try:
            loop.run_until_complete(_main())
        except asyncio.CancelledError:
            pass

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        assert ready.wait(5)
        out = io.StringIO()
        dump_event_loop_tasks(out)
        assert "_main" in out.getvalue()
    finally:
        for task in asyncio.all_tasks(loop):
            loop.call_soon_threadsafe(task.cancel)
        thread.join(5)
        loop.close()


def test_bounded_teardown_names_task_that_ignores_cancellation():
    async def _main() -> None:
        started = asyncio.Event()
        asyncio.get_running_loop().create_task(_swallow_cancellation(started), name="stubborn")
        await started.wait()

    with pytest.raises(AsyncioTeardownError, match="1 asyncio task") as exc_info:
        with bounded_event_loop_teardown(timeout=0.2):
            asyncio.run(_main())

    assert "'stubborn'" in str(exc_info.value)
    assert "_swallow_cancellation" in str(exc_info.value)


def test_bounded_teardown_is_silent_when_tasks_cancel_cleanly():
    async def _main() -> None:
        asyncio.get_running_loop().create_task(asyncio.sleep(3600))
        await asyncio.sleep(0)

    with bounded_event_loop_teardown(timeout=5):
        asyncio.run(_main())


def test_bounded_teardown_restores_the_stock_hook():
    original = getattr(asyncio.runners, "_cancel_all_tasks")

    with bounded_event_loop_teardown():
        assert getattr(asyncio.runners, "_cancel_all_tasks") is not original

    assert getattr(asyncio.runners, "_cancel_all_tasks") is original


def test_bounded_teardown_unblocks_starlette_test_client():
    """The CI hang: a lifespan-spawned task survives shutdown and blocks the TestClient portal."""

    @asynccontextmanager
    async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
        asyncio.get_running_loop().create_task(_swallow_cancellation(asyncio.Event()), name="stubborn")
        yield

    app = FastAPI(lifespan=_lifespan)

    with pytest.raises(AsyncioTeardownError, match="'stubborn'"):
        with bounded_event_loop_teardown(timeout=0.2), TestClient(app):
            pass
