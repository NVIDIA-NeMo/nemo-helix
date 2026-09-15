# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter to create a :class:`NemoClient` from an existing :class:`NeMoPlatform`.

This bridges the generated ``NeMoPlatform`` SDK with the new typed client,
allowing plugins registered via ``NemoPluginSDKResources`` to use the
new endpoint/client infrastructure internally.

Usage::

    from nemo_platform_plugin.client.adapter import client_from_platform

    def make_sync_resource(platform: NeMoPlatform) -> NemoClient:
        return client_from_platform(platform, NemoClient)
"""

from __future__ import annotations

from typing import TypeVar, overload

import httpx
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.types import RetryPolicy

SyncT = TypeVar("SyncT", bound=NemoClient)
AsyncT = TypeVar("AsyncT", bound=AsyncNemoClient)


def _platform_default_headers(platform: NeMoPlatform | AsyncNeMoPlatform) -> dict[str, str] | None:
    # Prefer _custom_headers (set via with_options/set_default_headers),
    # fall back to the httpx client's actual headers (set at construction,
    # e.g. TestClient(headers={...})), filtering out httpx defaults.
    headers = {key: value for key, value in platform._custom_headers.items() if isinstance(value, str)}
    if not headers:
        skip = {"accept", "accept-encoding", "connection", "user-agent", "host"}
        headers = {key: value for key, value in platform._client.headers.items() if key.lower() not in skip}
    return headers or None


@overload
def client_from_platform(platform: NeMoPlatform, client_cls: type[SyncT]) -> SyncT: ...
@overload
def client_from_platform(platform: AsyncNeMoPlatform, client_cls: type[AsyncT]) -> AsyncT: ...


def client_from_platform(
    platform: NeMoPlatform | AsyncNeMoPlatform,
    client_cls: type[NemoClient] | type[AsyncNemoClient],
) -> NemoClient | AsyncNemoClient:
    """Create a typed client sharing a generated platform SDK's transport.

    The overloads preserve the sync/async pairing between platform and client.
    """
    headers = _platform_default_headers(platform)
    retry = RetryPolicy(
        max_retries=platform.max_retries,
        retryable_status_codes=(408, 409, 429),
        retry_all_server_errors=True,
        respect_retry_decision_headers=True,
        respect_retry_after_headers=True,
    )
    url_resolver = platform._prepare_url

    # Carry the platform's timeout across as a per-request override. The shared
    # httpx client keeps whatever timeout it was built with, so a caller's
    # ``platform.with_options(timeout=...)`` would otherwise be silently dropped
    # on the way to the typed client — the httpx client it hands over is the
    # *same* transport instance, with the *original* timeout still on it.
    timeout = platform.timeout
    if timeout is None:
        # ``None`` on the platform means "no timeout at all", but the typed
        # client reads None as "defer to the transport". Say the same thing in
        # the form httpx itself uses, so the override survives.
        timeout = httpx.Timeout(None)

    if isinstance(platform, AsyncNeMoPlatform):
        if not issubclass(client_cls, AsyncNemoClient):
            raise TypeError("AsyncNeMoPlatform requires an AsyncNemoClient class")
        return client_cls(
            base_url=str(platform.base_url).rstrip("/"),
            workspace=platform.workspace,
            default_headers=headers,
            timeout=timeout,
            retry=retry,
            http_client=platform._client,
            owns_http_client=False,
            url_resolver=url_resolver,
        )

    if not issubclass(client_cls, NemoClient):
        raise TypeError("NeMoPlatform requires a NemoClient class")
    return client_cls(
        base_url=str(platform.base_url).rstrip("/"),
        workspace=platform.workspace,
        default_headers=headers,
        timeout=timeout,
        retry=retry,
        http_client=platform._client,
        owns_http_client=False,
        url_resolver=url_resolver,
    )
