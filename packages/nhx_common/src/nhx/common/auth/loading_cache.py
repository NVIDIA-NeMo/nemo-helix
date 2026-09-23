# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Coroutine, Hashable
from typing import Any, Generic, TypeVar, cast

KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")

_MISSING = object()


class AsyncLoadingCache(Generic[KeyT, ValueT]):
    """Async cache that guards access and serializes cache misses."""

    def __init__(self) -> None:
        self._values: dict[KeyT, ValueT] = {}
        self._lock = asyncio.Lock()

    async def clear(self) -> None:
        async with self._lock:
            self._values.clear()

    async def get_or_load(self, key: KeyT, loader: Callable[[], Awaitable[ValueT]]) -> ValueT:
        async with self._lock:
            cached = self._values.get(key, _MISSING)
            if cached is not _MISSING:
                return cast(ValueT, cached)

            value = await loader()
            self._values[key] = value
            return value


class AsyncCoalescingLoader(Generic[ValueT]):
    """Share one in-flight async load among concurrent callers."""

    def __init__(self, *, min_interval_seconds: float = 0.0) -> None:
        self._min_interval_seconds = min_interval_seconds
        self._last_load_time = 0.0
        self._task: asyncio.Task[ValueT] | None = None
        self._lock = asyncio.Lock()

    async def clear(self) -> None:
        async with self._lock:
            self._last_load_time = 0.0
            self._task = None

    async def load(
        self,
        loader: Callable[[], Coroutine[Any, Any, ValueT]],
        *,
        rate_limited_value: Callable[[], ValueT] | None = None,
    ) -> ValueT:
        async with self._lock:
            task = self._task
            if task is None:
                now = time.monotonic()
                if (
                    rate_limited_value is not None
                    and self._min_interval_seconds > 0
                    and self._last_load_time > 0
                    and now - self._last_load_time < self._min_interval_seconds
                ):
                    return rate_limited_value()

                self._last_load_time = now
                task = asyncio.create_task(loader())
                task.add_done_callback(self._schedule_forget_task)
                self._task = task

        return await asyncio.shield(task)

    def _schedule_forget_task(self, task: asyncio.Task[ValueT]) -> None:
        asyncio.create_task(self._forget_task(task))

    async def _forget_task(self, task: asyncio.Task[ValueT]) -> None:
        async with self._lock:
            if self._task is task:
                self._task = None
        if not task.cancelled():
            task.exception()
