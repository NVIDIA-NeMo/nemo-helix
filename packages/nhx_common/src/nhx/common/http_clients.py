# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP clients for SDK and internal requests.

Ideally, most consumers get their HTTP client from DependencyProvider, which
manages per-service client lifecycle. However, code often constructs SDKs in
isolation (tasks, controllers, background jobs) without access to a
DependencyProvider. These shared clients prevent each SDK instantiation from
creating a new HTTP client, avoiding connection pool and SSL context overhead.

Used by:
- get_platform_sdk() / get_async_platform_sdk() for tasks, controllers, and
  other code outside DependencyProvider context
- Platform shutdown cleanup via close_cached_http_clients()

The wrapper classes make close()/aclose() a no-op so that SDK code can safely
call sdk.close() without affecting other users of the shared client. Actual
cleanup happens at shutdown via close_cached_http_clients().
"""

import asyncio
from functools import cache
from typing import cast

import httpx
from nemo_helix import DefaultAsyncHttpxClient, DefaultHttpxClient


class _SharedAsyncHttpClient(DefaultAsyncHttpxClient):
    """Shared async HTTP client that ignores aclose() calls."""

    async def aclose(self) -> None:
        pass

    async def _real_close(self) -> None:
        await super().aclose()


class _SharedSyncHttpClient(DefaultHttpxClient):
    """Shared sync HTTP client that ignores close() calls."""

    def close(self) -> None:
        pass

    def _real_close(self) -> None:
        super().close()


_shared_async_http_clients: dict[asyncio.AbstractEventLoop, _SharedAsyncHttpClient] = {}


def shared_async_http_client() -> httpx.AsyncClient:
    """Get the shared async HTTP client for SDK requests.

    Returns a cached async HTTP client scoped to the current event loop.
    Use this when creating SDK instances that need to share a connection
    pool and SSL context within a single loop.

    If called when no event loop is currently running, return a fresh async
    client instead of caching one globally. This keeps sync setup code safe
    without binding a shared client to an arbitrary thread-local loop.

    The returned client ignores aclose() calls - cleanup happens at shutdown
    via close_cached_http_clients(). This allows SDKs to safely call close()
    without breaking other users of the shared client.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return DefaultAsyncHttpxClient()
    client = _shared_async_http_clients.get(loop)
    if client is None or client.is_closed:
        client = _SharedAsyncHttpClient()
        _shared_async_http_clients[loop] = client
    return client


@cache
def shared_sync_http_client() -> httpx.Client:
    """Get the shared sync HTTP client for SDK requests.

    Returns a cached sync HTTP client with SDK-compatible defaults.
    Use this when creating SDK instances that need to share a global
    connection pool and SSL context.

    The returned client ignores close() calls - cleanup happens at shutdown
    via close_cached_http_clients(). This allows SDKs to safely call close()
    without breaking other users of the shared client.
    """
    return _SharedSyncHttpClient()


async def close_shared_http_clients() -> None:
    """Close shared HTTP clients during graceful shutdown.

    Called from the platform lifespan shutdown to clean up module-level
    shared clients used by non-service code (tasks, jobs, legacy callers).
    """
    loop = asyncio.get_running_loop()
    async_client = _shared_async_http_clients.pop(loop, None)
    if async_client is not None and not async_client.is_closed:
        await async_client._real_close()

    sync_client = cast(_SharedSyncHttpClient, shared_sync_http_client())
    sync_client._real_close()
    shared_sync_http_client.cache_clear()
