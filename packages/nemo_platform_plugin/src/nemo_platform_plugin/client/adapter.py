# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter to create a typed client from an existing platform client.

Accepts either a legacy ``NeMoPlatform`` SDK instance or a :class:`NemoClient`
/ :class:`AsyncNemoClient`, so plugins registered via ``NemoPluginSDKResources``
can use the typed endpoint/client infrastructure regardless of which platform
client the caller holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, cast, overload, runtime_checkable

import httpx
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.types import RetryPolicy

SyncT = TypeVar("SyncT", bound=NemoClient)
AsyncT = TypeVar("AsyncT", bound=AsyncNemoClient)


@runtime_checkable
class PlatformClient(Protocol):
    """Structural shape shared by every platform handle :func:`client_from_platform` accepts.

    Satisfied by :class:`NemoClient` / :class:`AsyncNemoClient` and by the
    generated ``NeMoPlatform`` / ``AsyncNeMoPlatform`` SDK classes. Use it to
    annotate ``sdk`` / ``async_sdk`` parameters that are only forwarded to
    :func:`client_from_platform`, so the annotating module does not need to
    import the generated SDK. Prefer :class:`SyncPlatformClient` or
    :class:`AsyncPlatformClient` when the parameter is one or the other.
    """

    @property
    def base_url(self) -> str | httpx.URL: ...

    @property
    def workspace(self) -> str | None: ...


@runtime_checkable
class SyncPlatformClient(PlatformClient, Protocol):
    """A sync platform handle (``NeMoPlatform`` or :class:`NemoClient`)."""

    def __enter__(self) -> Any: ...


@runtime_checkable
class AsyncPlatformClient(PlatformClient, Protocol):
    """An async platform handle (``AsyncNeMoPlatform`` or :class:`AsyncNemoClient`)."""

    async def __aenter__(self) -> Any: ...


class _PlatformClient(Protocol):
    base_url: str | httpx.URL
    workspace: str | None
    max_retries: int
    timeout: float | httpx.Timeout | None
    _custom_headers: Mapping[str, str]
    _client: httpx.Client | httpx.AsyncClient

    def _prepare_url(self, url: str) -> httpx.URL: ...


def platform_default_headers(platform: PlatformClient) -> dict[str, str]:
    """Return a copy of the default headers *platform* sends on every request.

    Reads ``default_headers`` off a :class:`NemoClient` / :class:`AsyncNemoClient`
    and ``_custom_headers`` off a generated ``NeMoPlatform`` SDK instance, so
    callers forwarding identity headers through a non-SDK HTTP client do not
    need to know which platform handle they were given.
    """
    if isinstance(platform, (NemoClient, AsyncNemoClient)):
        return dict(platform.default_headers)
    return dict(cast(_PlatformClient, platform)._custom_headers)


def _platform_default_headers(platform: _PlatformClient) -> dict[str, str] | None:
    # Prefer _custom_headers (set via with_options/set_default_headers),
    # fall back to the httpx client's actual headers (set at construction,
    # e.g. TestClient(headers={...})), filtering out httpx defaults.
    headers = {key: value for key, value in platform._custom_headers.items() if isinstance(value, str)}
    if not headers:
        skip = {"accept", "accept-encoding", "connection", "user-agent", "host"}
        headers = {key: value for key, value in platform._client.headers.items() if key.lower() not in skip}
    return headers or None


@overload
def client_from_platform(platform: SyncPlatformClient, client_cls: type[SyncT]) -> SyncT: ...
@overload
def client_from_platform(platform: AsyncPlatformClient, client_cls: type[AsyncT]) -> AsyncT: ...


def client_from_platform(
    platform: PlatformClient,
    client_cls: type[NemoClient] | type[AsyncNemoClient],
) -> NemoClient | AsyncNemoClient:
    """Create a typed client sharing a platform client's transport.

    A :class:`NemoClient` / :class:`AsyncNemoClient` is derived with
    ``client_cls.from_client`` and shares its auth, headers, retry policy and
    transport. A generated ``NeMoPlatform`` / ``AsyncNeMoPlatform`` is adapted
    onto its httpx client.

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

    platform_client = cast(_PlatformClient, platform)
    headers = _platform_default_headers(platform_client)
    retry = RetryPolicy(
        max_retries=platform_client.max_retries,
        retryable_status_codes=(408, 409, 429),
        retry_all_server_errors=True,
        respect_retry_decision_headers=True,
        respect_retry_after_headers=True,
    )
    url_resolver = platform_client._prepare_url

    # Carry the platform's timeout across as a per-request override. The shared
    # httpx client keeps whatever timeout it was built with, so a caller's
    # ``platform.with_options(timeout=...)`` would otherwise be silently dropped
    # on the way to the typed client — the httpx client it hands over is the
    # *same* transport instance, with the *original* timeout still on it.
    timeout = platform_client.timeout
    if timeout is None:
        # ``None`` on the platform means "no timeout at all", but the typed
        # client reads None as "defer to the transport". Say the same thing in
        # the form httpx itself uses, so the override survives.
        timeout = httpx.Timeout(None)

    if isinstance(platform_client._client, httpx.AsyncClient):
        if not issubclass(client_cls, AsyncNemoClient):
            raise TypeError("AsyncNeMoPlatform requires an AsyncNemoClient class")
        return client_cls(
            base_url=str(platform_client.base_url).rstrip("/"),
            workspace=platform_client.workspace,
            default_headers=headers,
            timeout=timeout,
            retry=retry,
            http_client=platform_client._client,
            owns_http_client=False,
            url_resolver=url_resolver,
        )

    if not issubclass(client_cls, NemoClient):
        raise TypeError("NeMoPlatform requires a NemoClient class")
    return client_cls(
        base_url=str(platform_client.base_url).rstrip("/"),
        workspace=platform_client.workspace,
        default_headers=headers,
        timeout=timeout,
        retry=retry,
        http_client=platform_client._client,
        owns_http_client=False,
        url_resolver=url_resolver,
    )
