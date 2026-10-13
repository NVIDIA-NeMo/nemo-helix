# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter to create a typed client from an existing typed platform client.

Accepts a :class:`NemoClient` / :class:`AsyncNemoClient` and derives the
requested typed service client sharing its transport, auth, headers, retry
policy and timeout.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, overload, runtime_checkable

import httpx
from nemo_helix_plugin.client.auth import TokenProvider
from nemo_helix_plugin.client.client import (
    AsyncNemoClient,
    NemoClient,
    NemoClientRuntime,
)

SyncT = TypeVar("SyncT", bound=NemoClient)
AsyncT = TypeVar("AsyncT", bound=AsyncNemoClient)


@runtime_checkable
class HelixClient(Protocol):
    """Structural shape shared by every platform handle :func:`client_from_platform` accepts.

    Satisfied by :class:`NemoClient` / :class:`AsyncNemoClient`. Use it to
    annotate ``sdk`` / ``async_sdk`` parameters that are only forwarded to
    :func:`client_from_platform`, so the annotating module does not need to
    import a concrete client type. Prefer :class:`SyncHelixClient` or
    :class:`AsyncHelixClient` when the parameter is one or the other.
    """

    @property
    def base_url(self) -> str | httpx.URL: ...

    @property
    def workspace(self) -> str | None: ...

    @property
    def nemo_client_runtime(self) -> NemoClientRuntime: ...

    @property
    def nemo_client_auth(self) -> TokenProvider | None: ...


@runtime_checkable
class SyncHelixClient(HelixClient, Protocol):
    """A sync platform handle (:class:`NemoClient`)."""

    def __enter__(self) -> object: ...


@runtime_checkable
class AsyncHelixClient(HelixClient, Protocol):
    """An async platform handle (:class:`AsyncNemoClient`)."""

    async def __aenter__(self) -> object: ...


@overload
def client_from_platform(
    platform: SyncHelixClient,
    client_cls: type[SyncT],
) -> SyncT: ...
@overload
def client_from_platform(
    platform: AsyncHelixClient,
    client_cls: type[AsyncT],
) -> AsyncT: ...


def client_from_platform(
    platform: NemoClient | AsyncNemoClient,
    client_cls: type[NemoClient] | type[AsyncNemoClient],
) -> NemoClient | AsyncNemoClient:
    """Create a typed client sharing a platform client's transport.

    A :class:`NemoClient` / :class:`AsyncNemoClient` is derived with
    ``client_cls.from_client`` and shares its auth, headers, retry policy and
    transport.

    The overloads pair sync platforms with sync clients and async with async,
    so a mismatch is a type error at the call site.
    """
    if isinstance(platform, NemoClient):
        if not issubclass(client_cls, NemoClient):
            raise TypeError(f"NemoClient cannot back {client_cls.__name__}: sync/async mismatch")
        return platform if isinstance(platform, client_cls) else client_cls.from_client(platform)
    if isinstance(platform, AsyncNemoClient):
        if not issubclass(client_cls, AsyncNemoClient):
            raise TypeError(f"AsyncNemoClient cannot back {client_cls.__name__}: sync/async mismatch")
        return platform if isinstance(platform, client_cls) else client_cls.from_client(platform)
    raise TypeError(f"Unsupported platform client type {type(platform).__name__}; pass a NemoClient or AsyncNemoClient")


def platform_default_headers(platform: NemoClient | AsyncNemoClient) -> dict[str, str]:
    """Return a copy of the default headers *platform* sends on every request.

    Reads ``default_headers`` off a :class:`NemoClient` / :class:`AsyncNemoClient`,
    so callers forwarding identity headers through a non-SDK HTTP client do not
    need to know which platform handle they were given.
    """
    return dict(platform.default_headers)
