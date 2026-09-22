# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authentication protocols and helpers for NemoClient.

Defines the protocols that any token provider must satisfy, plus simple
concrete implementations.

OIDC-specific machinery lives in :mod:`~.oidc` (token provider, token set,
discovery) and :mod:`~.oidc_factory` (provider caching, config persistence).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, Generator
from typing import Protocol, cast, runtime_checkable

import httpx

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenProvider(Protocol):
    """Sync protocol for objects that can supply an access token."""

    def get_access_token(self) -> str: ...


@runtime_checkable
class AsyncTokenProvider(Protocol):
    """Async protocol for objects that can supply an access token."""

    async def get_access_token(self) -> str: ...


# ---------------------------------------------------------------------------
# StaticToken
# ---------------------------------------------------------------------------


class StaticToken:
    """Wraps a plain token string into a TokenProvider."""

    def __init__(self, token: str) -> None:
        self._token = token

    def get_access_token(self) -> str:
        return self._token

    async def get_access_token_async(self) -> str:
        return self._token


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


async def resolve_token_async(provider: TokenProvider | AsyncTokenProvider) -> str:
    """Get an access token from *provider* without blocking the event loop.

    Three cases, in priority order:

    1. Provider has ``get_access_token_async()`` (e.g. OIDCTokenProvider) — use it.
    2. ``get_access_token()`` is a coroutine function — await it.
    3. ``get_access_token()`` is sync — run in a thread, since it may perform IO
       such as a token refresh.
    """
    get_async = getattr(provider, "get_access_token_async", None)
    if get_async is not None and callable(get_async):
        return await get_async()
    if inspect.iscoroutinefunction(provider.get_access_token):
        return await provider.get_access_token()
    return await asyncio.to_thread(cast(TokenProvider, provider).get_access_token)


# ---------------------------------------------------------------------------
# httpx transport auth
# ---------------------------------------------------------------------------


class TokenProviderAuth(httpx.Auth):
    """Applies a :class:`TokenProvider`'s bearer token at the transport layer.

    ``NemoClient.send()`` sets ``Authorization`` itself, but raw calls made
    through the exposed ``_client`` transport (plugin SDK resources carried over
    from the Stainless SDK) never reach ``send()``. Installing this on the httpx
    client keeps those requests authenticated.

    Requests that already carry an ``Authorization`` header are left alone, so
    ``send()`` and per-call header overrides stay authoritative.
    """

    def __init__(self, provider: TokenProvider | AsyncTokenProvider) -> None:
        self._provider = provider

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if "Authorization" not in request.headers:
            token = self._provider.get_access_token()
            if inspect.isawaitable(token):
                raise TypeError(
                    "Async token provider used on a synchronous transport; "
                    "use AsyncNemoClient with an AsyncTokenProvider."
                )
            request.headers["Authorization"] = f"Bearer {token}"
        yield request

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if "Authorization" not in request.headers:
            request.headers["Authorization"] = f"Bearer {await resolve_token_async(self._provider)}"
        yield request


# ---------------------------------------------------------------------------
# Auth errors
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Authentication-related error."""
