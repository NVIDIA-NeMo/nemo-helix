# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Destroy every sandbox for a job when the owning process is cancelled or exits.

The actor that created a sandbox cannot be relied on to destroy it. Ray kills that
worker on cancel without running its teardown, and a failure in the actor leaves the
BatchSandbox until ``ttl_s``. The orchestrator arms the sweep on the process the
container runtime signals, so this module only knows how to destroy the sandboxes.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import Callable, Coroutine, Mapping
from typing import Any, TypeVar

from sandboxed_gym.host.provider import get_host_provider
from sandboxed_gym.run_sync import run_coro_sync

LOGGER = logging.getLogger(__name__)

DEFAULT_REAP_TIMEOUT_S = 30.0

T = TypeVar("T")


class _InlineExecutor(concurrent.futures.ThreadPoolExecutor):
    """Run work on the calling thread.

    ``set_default_executor`` accepts only a ``ThreadPoolExecutor``, hence the subclass. Nothing
    bounds a call it runs, so a stalled hostname lookup outlasts ``asyncio.wait_for``.
    """

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> concurrent.futures.Future[Any]:
        """Run ``fn`` now and return a future already holding its result or exception."""
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def _thread_pools_accept_work() -> bool:
    """Return False once interpreter shutdown has begun, which is before ``atexit`` hooks run."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(int).result()
    except RuntimeError:
        return False
    return True


def _run_without_threads(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` on a new loop that never starts a thread.

    Thread pools refuse work during ``atexit``, and the loop's default pool is what resolves
    hostnames. ``asyncio.run`` is avoided because it shuts the default executor down on a new
    thread, which CPython 3.12.0 refuses to start during ``atexit``; ``loop.close`` does not.
    """
    loop = asyncio.new_event_loop()
    loop.set_default_executor(_InlineExecutor())
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


def reap_job_sandboxes(
    job_id: str,
    *,
    host_provider: str,
    host_provider_options: Mapping[str, Any] | None = None,
    timeout_s: float = DEFAULT_REAP_TIMEOUT_S,
) -> tuple[str, ...]:
    """Destroy every sandbox labeled with ``job_id``. Failures are logged, not raised.

    Raising from an ``atexit`` hook or a signal handler would skip the rest of
    process teardown. A sandbox that is already gone is not a failure of the exit.

    ``timeout_s`` bounds the sweep. A caller reaping inside a container's termination
    grace period passes less than the default so the rest of its own teardown still runs.
    """
    options = dict(host_provider_options or {})

    async def _destroy() -> tuple[str, ...]:
        """Destroy the job's sandboxes through the host provider."""
        provider = get_host_provider(host_provider, options)
        return await asyncio.wait_for(provider.destroy_job_sandboxes(job_id), timeout=timeout_s)

    try:
        # The inline path is only for shutdown: elsewhere a lookup must stay on a pool so the
        # timeout can fire inside a caller's termination grace period.
        removed = run_coro_sync(_destroy()) if _thread_pools_accept_work() else _run_without_threads(_destroy())
    except Exception:
        LOGGER.exception("failed to reap sandboxes for job %s", job_id)
        return ()
    if removed:
        LOGGER.warning(
            "reaped %d sandbox(es) for job %s: %s",
            len(removed),
            job_id,
            ", ".join(removed),
        )
    return removed
