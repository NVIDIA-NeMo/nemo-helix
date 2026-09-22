# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable, Coroutine, Hashable
from typing import Any, Generic, TypeVar

KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")

_LOOP_CACHE_ID = itertools.count()


class _InFlightLoad(Generic[ValueT]):
    def __init__(self, task: asyncio.Future[ValueT], generation: int) -> None:
        self.task = task
        self.generation = generation


class _LoopLoadingCacheState(Generic[KeyT, ValueT]):
    def __init__(self) -> None:
        self.values: dict[KeyT, ValueT] = {}
        self.loads: dict[KeyT, _InFlightLoad[ValueT]] = {}
        self.generation = 0


class AsyncLoadingCache(Generic[KeyT, ValueT]):
    """Async cache with state scoped to the current event loop."""

    def __init__(self) -> None:
        self._loop_state_attribute = f"_nhx_async_loading_cache_{next(_LOOP_CACHE_ID)}"

    async def clear(self) -> None:
        state = self._state_for_current_loop()
        state.generation += 1
        state.values.clear()
        state.loads.clear()

    async def get_or_load(self, key: KeyT, loader: Callable[[], Awaitable[ValueT]]) -> ValueT:
        state = self._state_for_current_loop()
        if key in state.values:
            return state.values[key]

        load = state.loads.get(key)
        if load is None:
            task = asyncio.ensure_future(loader())
            load = _InFlightLoad(task, state.generation)
            state.loads[key] = load
            task.add_done_callback(lambda _: self._complete_load(state, key, load))

        value = await asyncio.shield(load.task)
        self._complete_load(state, key, load)
        return value

    def _state_for_current_loop(self) -> _LoopLoadingCacheState[KeyT, ValueT]:
        loop = asyncio.get_running_loop()
        # asyncio has no typed loop-local storage; keep tasks and cached values on
        # the loop so they never cross event-loop ownership boundaries.
        state: _LoopLoadingCacheState[KeyT, ValueT] | None = getattr(loop, self._loop_state_attribute, None)
        if state is None:
            state = _LoopLoadingCacheState()
            setattr(loop, self._loop_state_attribute, state)
        return state

    def _complete_load(
        self,
        state: _LoopLoadingCacheState[KeyT, ValueT],
        key: KeyT,
        load: _InFlightLoad[ValueT],
    ) -> None:
        if state.loads.get(key) is not load:
            _discard_task_exception(load.task)
            return
        if not load.task.done():
            return

        del state.loads[key]
        if load.task.cancelled():
            return

        try:
            value = load.task.result()
        except BaseException:
            return
        if load.generation == state.generation:
            state.values[key] = value


def _discard_task_exception(task: asyncio.Future[ValueT]) -> None:
    if task.done() and not task.cancelled():
        task.exception()


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
