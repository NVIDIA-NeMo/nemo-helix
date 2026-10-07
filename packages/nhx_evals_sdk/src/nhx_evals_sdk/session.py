# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-scoped cache shared by everything participating in one evaluation."""

import asyncio
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

_CACHE: ContextVar[dict[Hashable, Any] | None] = ContextVar("evaluation_session_cache", default=None)
_LOCK: ContextVar[asyncio.Lock | None] = ContextVar("evaluation_session_lock", default=None)


@asynccontextmanager
async def begin_evaluation_session() -> AsyncIterator[None]:
    """Scope cached evaluation state to one run.

    Entries live only inside this boundary, which is what makes caching a failed result safe: a
    transient failure is retried by the next run instead of persisting for the life of the process.
    Outside a session nothing is cached and every caller recomputes.

    Re-entrant. Entry points nest -- a backend opens a session, then the metric executor opens
    another -- and a fresh inner cache would recompute what the run already resolved.
    """
    if _CACHE.get() is not None:
        yield
        return

    cache_token = _CACHE.set({})
    lock_token = _LOCK.set(asyncio.Lock())
    try:
        yield
    finally:
        _CACHE.reset(cache_token)
        _LOCK.reset(lock_token)


def session_cache() -> dict[Hashable, Any] | None:
    """Return the active run's cache, or None outside a session."""
    return _CACHE.get()


def session_lock() -> asyncio.Lock:
    """Return the active run's lock, or a throwaway one outside a session."""
    return _LOCK.get() or asyncio.Lock()
