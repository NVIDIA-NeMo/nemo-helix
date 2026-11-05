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
from collections.abc import Callable, Mapping
from typing import Any

from sandboxed_gym.host.provider import get_host_provider
from sandboxed_gym.run_sync import run_coro_sync

LOGGER = logging.getLogger(__name__)


class _InlineExecutor(concurrent.futures.ThreadPoolExecutor):
    """Run work on the calling thread.

    Thread pools refuse new work once ``atexit`` hooks start, and the loop's default pool is
    what resolves the control plane's hostname. ``set_default_executor`` accepts only a
    ``ThreadPoolExecutor``, hence the subclass.
    """

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> concurrent.futures.Future[Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def reap_job_sandboxes(
    job_id: str,
    *,
    host_provider: str,
    host_provider_options: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Destroy every sandbox labeled with ``job_id``. Failures are logged, not raised.

    Raising from an ``atexit`` hook or a signal handler would skip the rest of
    process teardown. A sandbox that is already gone is not a failure of the exit.
    """
    options = dict(host_provider_options or {})

    async def _destroy() -> tuple[str, ...]:
        asyncio.get_running_loop().set_default_executor(_InlineExecutor())
        provider = get_host_provider(host_provider, options)
        return await asyncio.wait_for(provider.destroy_job_sandboxes(job_id), timeout=30)

    try:
        removed = run_coro_sync(_destroy())
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
