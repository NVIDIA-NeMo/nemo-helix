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
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Protocol

import httpx

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class TokenProvider(Protocol):
    """Sync protocol for objects that can supply an access token."""

    def get_access_token(self) -> str: ...


class AsyncTokenProvider(Protocol):
    """Async protocol for objects that can supply an access token."""

    async def get_access_token_async(self) -> str: ...


class AsyncClientTokenProvider(Protocol):
    """Async client auth provider normalized for per-request token resolution."""

    def set_http_client(self, http_client: httpx.AsyncClient) -> None: ...

    async def get_access_token_async(self) -> str: ...

    async def get_access_token_or_none_async(self) -> str | None: ...


class ServicePrincipalTokenProvider(ABC):
    """Token provider that authenticates the local client as a platform service."""

    @abstractmethod
    def get_access_token(self) -> str:
        """Return a valid access token for the service principal."""

    async def get_access_token_async(self) -> str:
        return await asyncio.to_thread(self.get_access_token)

    def set_http_client(self, http_client: httpx.AsyncClient) -> None:
        _ = http_client
        return None

    async def get_access_token_or_none_async(self) -> str:
        return await self.get_access_token_async()

    @property
    @abstractmethod
    def service_principal_id(self) -> str:
        """Return the platform service principal id this provider authenticates as."""

    @property
    def delegates_principal_identity(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# StaticToken
# ---------------------------------------------------------------------------


class StaticToken:
    """Wraps a plain token string into a TokenProvider."""

    def __init__(self, token: str) -> None:
        self._token = token

    def get_access_token(self) -> str:
        return self._token

    def set_http_client(self, http_client: httpx.AsyncClient) -> None:
        _ = http_client
        return None

    async def get_access_token_async(self) -> str:
        return self._token

    async def get_access_token_or_none_async(self) -> str:
        return self._token


class AsyncFromSyncTokenProvider:
    """Async adapter for sync token providers used by async clients."""

    def __init__(self, provider: TokenProvider) -> None:
        self._provider = provider

    def set_http_client(self, http_client: httpx.AsyncClient) -> None:
        _ = http_client
        return None

    async def get_access_token_async(self) -> str:
        return await asyncio.to_thread(self._provider.get_access_token)

    async def get_access_token_or_none_async(self) -> str:
        return await self.get_access_token_async()


class AsyncClientTokenProviderAdapter:
    """Adapt an async token provider to the full async-client auth contract."""

    def __init__(self, provider: AsyncTokenProvider) -> None:
        self._provider = provider

    def set_http_client(self, http_client: httpx.AsyncClient) -> None:
        _ = http_client
        return None

    async def get_access_token_async(self) -> str:
        return await self._provider.get_access_token_async()

    async def get_access_token_or_none_async(self) -> str:
        return await self.get_access_token_async()


# ---------------------------------------------------------------------------
# httpx transport auth
# ---------------------------------------------------------------------------


class TokenProviderAuth(httpx.Auth):
    """Applies a :class:`TokenProvider`'s bearer token at the transport layer.

    This is for explicit proxy or legacy transports that authenticate every
    request made through that transport. ``NemoClient`` keeps its default
    transport neutral and resolves auth into per-request headers instead.
    """

    def __init__(self, provider: TokenProvider) -> None:
        self._provider = provider

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if "Authorization" not in request.headers:
            request.headers["Authorization"] = f"Bearer {self._provider.get_access_token()}"
        yield request

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if "Authorization" not in request.headers:
            token = await asyncio.to_thread(self._provider.get_access_token)
            request.headers["Authorization"] = f"Bearer {token}"
        yield request


# ---------------------------------------------------------------------------
# Auth errors
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Authentication-related error."""
