# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest
from nhx.common.auth.loading_cache import AsyncCoalescingLoader, AsyncLoadingCache


async def test_async_coalescing_loader_shares_in_flight_load() -> None:
    loader: AsyncCoalescingLoader[object] = AsyncCoalescingLoader()
    load_count = 0

    async def load() -> object:
        nonlocal load_count
        load_count += 1
        await asyncio.sleep(0.01)
        return object()

    values = await asyncio.gather(*(loader.load(load) for _ in range(20)))

    assert all(value is values[0] for value in values)
    assert load_count == 1


async def test_async_loading_cache_loads_value_once_for_same_key() -> None:
    cache: AsyncLoadingCache[str, object] = AsyncLoadingCache()
    load_count = 0

    async def load() -> object:
        nonlocal load_count
        load_count += 1
        return object()

    first = await cache.get_or_load("key", load)
    second = await cache.get_or_load("key", load)

    assert first is second
    assert load_count == 1


async def test_async_loading_cache_serializes_concurrent_misses() -> None:
    cache: AsyncLoadingCache[str, object] = AsyncLoadingCache()
    load_count = 0

    async def load() -> object:
        nonlocal load_count
        load_count += 1
        await asyncio.sleep(0.01)
        return object()

    values = await asyncio.gather(*(cache.get_or_load("key", load) for _ in range(20)))

    assert all(value is values[0] for value in values)
    assert load_count == 1


def test_async_loading_cache_state_is_scoped_to_event_loop_after_contention() -> None:
    cache: AsyncLoadingCache[str, str] = AsyncLoadingCache()
    load_count = 0

    async def load_with_waiter() -> None:
        nonlocal load_count
        loader_started = asyncio.Event()
        release_loader = asyncio.Event()

        async def load() -> str:
            nonlocal load_count
            load_count += 1
            loader_started.set()
            await release_loader.wait()
            return "loaded"

        first = asyncio.create_task(cache.get_or_load("key", load))
        await loader_started.wait()
        second = asyncio.create_task(cache.get_or_load("key", load))
        await asyncio.sleep(0)
        release_loader.set()

        assert await first == "loaded"
        assert await second == "loaded"

    async def load_from_new_loop() -> str:
        nonlocal load_count
        load_count += 1
        return "new-loop"

    asyncio.run(load_with_waiter())

    assert asyncio.run(cache.get_or_load("key", load_from_new_loop)) == "new-loop"
    assert load_count == 2


async def test_async_loading_cache_does_not_cache_loader_failure() -> None:
    cache: AsyncLoadingCache[str, str] = AsyncLoadingCache()
    load_count = 0

    async def load() -> str:
        nonlocal load_count
        load_count += 1
        if load_count == 1:
            raise ValueError("failed")
        return "loaded"

    with pytest.raises(ValueError, match="failed"):
        await cache.get_or_load("key", load)

    assert await cache.get_or_load("key", load) == "loaded"
    assert load_count == 2


async def test_async_coalescing_loader_does_not_cache_completed_loads() -> None:
    loader: AsyncCoalescingLoader[object] = AsyncCoalescingLoader()
    load_count = 0

    async def load() -> object:
        nonlocal load_count
        load_count += 1
        return object()

    first = await loader.load(load)
    second = await loader.load(load)

    assert second is not first
    assert load_count == 2


async def test_async_coalescing_loader_returns_rate_limited_value_after_recent_load() -> None:
    loader: AsyncCoalescingLoader[str] = AsyncCoalescingLoader(min_interval_seconds=60.0)
    load_count = 0

    async def load() -> str:
        nonlocal load_count
        load_count += 1
        return "fresh"

    assert await loader.load(load, rate_limited_value=lambda: "cached") == "fresh"
    assert await loader.load(load, rate_limited_value=lambda: "cached") == "cached"
    assert load_count == 1


async def test_async_coalescing_loader_zero_interval_disables_rate_limit_window() -> None:
    loader: AsyncCoalescingLoader[str] = AsyncCoalescingLoader(min_interval_seconds=0)
    load_count = 0

    async def load() -> str:
        nonlocal load_count
        load_count += 1
        return f"fresh-{load_count}"

    assert await loader.load(load, rate_limited_value=lambda: "cached") == "fresh-1"
    assert await loader.load(load, rate_limited_value=lambda: "cached") == "fresh-2"
    assert load_count == 2


async def test_async_coalescing_loader_clear_resets_rate_limit_window() -> None:
    loader: AsyncCoalescingLoader[str] = AsyncCoalescingLoader(min_interval_seconds=60.0)
    load_count = 0

    async def load() -> str:
        nonlocal load_count
        load_count += 1
        return f"fresh-{load_count}"

    assert await loader.load(load, rate_limited_value=lambda: "cached") == "fresh-1"
    assert await loader.load(load, rate_limited_value=lambda: "cached") == "cached"

    await loader.clear()

    assert await loader.load(load, rate_limited_value=lambda: "cached") == "fresh-2"
    assert load_count == 2


async def test_async_coalescing_loader_does_not_cache_loader_failure() -> None:
    loader: AsyncCoalescingLoader[str] = AsyncCoalescingLoader()
    load_count = 0

    async def load() -> str:
        nonlocal load_count
        load_count += 1
        if load_count == 1:
            raise ValueError("failed")
        return "loaded"

    with pytest.raises(ValueError, match="failed"):
        await loader.load(load)

    assert await loader.load(load) == "loaded"
    assert load_count == 2
