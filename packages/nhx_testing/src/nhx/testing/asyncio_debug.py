# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics for asyncio tasks that hang test teardown.

Thread stack dumps show an event loop idling in ``select()`` but not which task it is waiting on.
These helpers name that task: by its full ``await`` chain, and by failing fast when it ignores
cancellation instead of letting the test run into ``--timeout``.
"""

from __future__ import annotations

import asyncio
import asyncio.runners
import gc
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, TextIO

#: How long ``asyncio.Runner.close()`` may wait for cancelled tasks before giving up on them.
DEFAULT_TEARDOWN_TIMEOUT_SECONDS = 10.0

#: Guards against a cycle in the await chain (e.g. a malformed custom awaitable).
_MAX_AWAIT_DEPTH = 200


class AsyncioTeardownError(RuntimeError):
    """An event loop's leftover tasks did not finish after being cancelled."""


def format_task_stack(task: asyncio.Task[Any]) -> str:
    """Render ``task`` and its full ``await`` chain, outermost frame first.

    ``Task.get_stack()`` returns only the task's own coroutine frame; following ``cr_await`` shows
    the innermost frame actually blocked, and the future it is blocked on.
    """
    lines = [f"Task {task.get_name()!r} ({'cancelling' if task.cancelling() else 'pending'}):"]
    awaitable: Any = task.get_coro()
    for _ in range(_MAX_AWAIT_DEPTH):
        if awaitable is None:
            break
        if isinstance(awaitable, asyncio.Future):
            lines.append(f"  awaiting {awaitable!r}")
            break
        frame = getattr(awaitable, "cr_frame", None) or getattr(awaitable, "gi_frame", None)
        if frame is None:
            # A bare future's ``__await__`` iterator (or other leaf awaitable); the frame above is the
            # innermost one blocked.
            if not asyncio.iscoroutine(awaitable):
                lines.append(f"  awaiting {type(awaitable).__qualname__}")
            break
        code = frame.f_code
        lines.append(f'  File "{code.co_filename}", line {frame.f_lineno}, in {code.co_qualname}')
        awaitable = getattr(awaitable, "cr_await", None) or getattr(awaitable, "gi_yieldfrom", None)
    return "\n".join(lines)


def dump_event_loop_tasks(file: TextIO) -> None:
    """Write the await chain of every pending task on every open event loop to ``file``.

    Safe to call from a watchdog thread: it only reads task state.
    """
    loops = [obj for obj in gc.get_objects() if isinstance(obj, asyncio.AbstractEventLoop) and not obj.is_closed()]
    for loop in loops:
        try:
            tasks = asyncio.all_tasks(loop)
        except RuntimeError as exc:  # The task set changed under us; report and move on.
            file.write(f"--- asyncio loop {loop!r}: could not list tasks: {exc}\n")
            continue
        thread = getattr(loop, "_thread_id", None)
        file.write(f"--- asyncio loop {loop!r} (thread {thread}), {len(tasks)} pending task(s)\n")
        for task in tasks:
            file.write(format_task_stack(task) + "\n\n")
    file.flush()


def _make_bounded_cancel_all_tasks(timeout: float, stragglers: list[str], lock: threading.Lock):
    def _bounded_cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
        """Like ``asyncio.runners._cancel_all_tasks``, but stop waiting after ``timeout`` seconds."""
        to_cancel = asyncio.all_tasks(loop)
        if not to_cancel:
            return
        for task in to_cancel:
            task.cancel()
        _, pending = loop.run_until_complete(asyncio.wait(to_cancel, timeout=timeout))
        if pending:
            # Record now, while the frames are still live; the loop is about to close.
            with lock:
                stragglers.extend(format_task_stack(task) for task in pending)
        for task in to_cancel - pending:
            if not task.cancelled() and task.exception() is not None:
                loop.call_exception_handler(
                    {
                        "message": "unhandled exception during asyncio.run() shutdown",
                        "exception": task.exception(),
                        "task": task,
                    }
                )

    return _bounded_cancel_all_tasks


@contextmanager
def bounded_event_loop_teardown(timeout: float = DEFAULT_TEARDOWN_TIMEOUT_SECONDS) -> Generator[None, None, None]:
    """Fail fast, naming the task, when an event loop closed inside this block has an uncancellable task.

    ``asyncio.Runner.close()`` (used by anyio, and so by Starlette's ``TestClient`` portal) cancels
    leftover tasks and waits for them with no timeout. A task that swallows the cancellation hangs
    teardown until pytest-timeout kills the worker. Inside this block that wait is bounded; any task
    still pending is abandoned and reported as an :class:`AsyncioTeardownError` on exit.
    """
    original = getattr(asyncio.runners, "_cancel_all_tasks", None)
    if original is None:  # A future Python moved the private hook; degrade to stock behavior.
        yield
        return

    stragglers: list[str] = []
    lock = threading.Lock()
    setattr(asyncio.runners, "_cancel_all_tasks", _make_bounded_cancel_all_tasks(timeout, stragglers, lock))
    try:
        yield
    finally:
        setattr(asyncio.runners, "_cancel_all_tasks", original)
    if stragglers:
        details = "\n\n".join(stragglers)
        raise AsyncioTeardownError(
            f"{len(stragglers)} asyncio task(s) ignored cancellation for {timeout:g}s while the event loop "
            f"was closing. A background task likely swallows CancelledError or awaits something that never "
            f"completes:\n\n{details}"
        )
