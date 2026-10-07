# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constructor arguments for the generated ``nemo_helix`` SDK clients, resolved from the nhx config.

The generated SDK calls :func:`build_client_init_kwargs` and
:func:`build_async_client_init_kwargs` to turn the nhx config file into a
base URL, workspace, default headers and (for OAuth and workload-identity users)
an httpx client whose request hook injects and refreshes the Bearer token. The
resolution itself lives in :mod:`nemo_helix_ext.client.bootstrap`; typed
clients are built there with :func:`~nemo_helix_ext.client.bootstrap.build_nemo_client`.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from nemo_helix_ext.client.bootstrap import (
    _TOKEN_PROVIDER_CACHE,
    _TOKEN_PROVIDER_CACHE_LOCK,
    AccessTokenProvider,
    ResolvedBootstrap,
    _headers_with_seeded_auth,
    _make_async_auth_event_hook,
    _make_auth_event_hook,
    resolve_bootstrap,
    resolve_timeout,
)

__all__ = [
    "_TOKEN_PROVIDER_CACHE",
    "_TOKEN_PROVIDER_CACHE_LOCK",
    "AccessTokenProvider",
    "ClientInitConfig",
    "ResolvedBootstrap",
    "build_async_client_init_kwargs",
    "build_client_init_kwargs",
]

_SyncRequestHook = Callable[[httpx.Request], None]
_AsyncRequestHook = Callable[[httpx.Request], Awaitable[None]]
_SyncHttpClientFactory = Callable[[_SyncRequestHook, str | Literal[True]], httpx.Client]
_AsyncHttpClientFactory = Callable[[_AsyncRequestHook, str | Literal[True]], httpx.AsyncClient]

_CONNECTION_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)


@dataclass(frozen=True)
class ClientInitConfig:
    """Everything a client constructor needs after config resolution.

    For non-OAuth users this just carries base_url/workspace/headers.
    For OAuth users it also includes a custom httpx client with an event
    hook that injects/refreshes the Bearer token on every request.
    """

    base_url: str
    workspace: str | None
    default_headers: Mapping[str, object] | None = None
    http_client: httpx.Client | httpx.AsyncClient | None = None
    client_verify: str | Literal[True] = True


def _split_header_sentinels(
    extra_headers: Mapping[str, object] | None,
) -> tuple[dict[str, str], dict[str, object]]:
    """Separate real header values from non-string sentinels.

    The generated SDK marks one of its own default headers for removal with a
    non-string sentinel. The bootstrap only deals in real values, so sentinels
    are carried around it and merged back into the returned config.
    """
    values: dict[str, str] = {}
    sentinels: dict[str, object] = {}
    for key, value in (extra_headers or {}).items():
        if isinstance(value, str):
            values[key] = value
        else:
            sentinels[key] = value
    return values, sentinels


def _with_sentinels(headers: Mapping[str, str] | None, sentinels: Mapping[str, object]) -> dict[str, object] | None:
    merged: dict[str, object] = {**(headers or {}), **sentinels}
    return merged or None


def build_client_init_kwargs(
    *,
    config_path: Path | None = None,
    base_url: str | httpx.URL | None = None,
    context_name: str | None = None,
    access_token: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    http_client_factory: _SyncHttpClientFactory | None = None,
) -> ClientInitConfig:
    """Build constructor kwargs for a **sync** client.

    For OAuth users, returns a ``ClientInitConfig`` whose ``http_client``
    has a request event hook that transparently injects and refreshes the
    Bearer token before every request.
    """
    header_values, sentinels = _split_header_sentinels(extra_headers)
    bootstrap = resolve_bootstrap(
        config_path=config_path,
        base_url=base_url,
        context_name=context_name,
        access_token=access_token,
        extra_headers=header_values,
    )
    if bootstrap.token_provider is None:
        # Non-OAuth: static headers, no custom http_client needed.
        return ClientInitConfig(
            base_url=bootstrap.base_url,
            workspace=bootstrap.workspace,
            default_headers=_with_sentinels(bootstrap.default_headers, sentinels),
            client_verify=bootstrap.client_verify,
        )

    # Seed the default headers with a current token so that code that inspects
    # headers sees a value. Workload identity only seeds when the request-time
    # provider already has a token. The event hook overwrites it with a fresh
    # token on each request.
    headers = _headers_with_seeded_auth(bootstrap.default_headers, bootstrap.token_provider)
    hook = _make_auth_event_hook(bootstrap.token_provider)
    http_client = (
        http_client_factory(hook, bootstrap.client_verify)
        if http_client_factory is not None
        else httpx.Client(
            event_hooks={"request": [hook], "response": []},
            timeout=resolve_timeout(None),
            limits=_CONNECTION_LIMITS,
            follow_redirects=True,
            verify=bootstrap.client_verify,
        )
    )
    return ClientInitConfig(
        base_url=bootstrap.base_url,
        workspace=bootstrap.workspace,
        default_headers=_with_sentinels(headers, sentinels),
        http_client=http_client,
        client_verify=bootstrap.client_verify,
    )


def build_async_client_init_kwargs(
    *,
    config_path: Path | None = None,
    base_url: str | httpx.URL | None = None,
    context_name: str | None = None,
    access_token: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    http_client_factory: _AsyncHttpClientFactory | None = None,
) -> ClientInitConfig:
    """Build constructor kwargs for an **async** client.

    Same as ``build_client_init_kwargs`` but returns an async httpx client
    whose event hook calls ``provider.get_access_token_async()`` (runs
    the refresh in a worker thread so it doesn't block the event loop).
    """
    header_values, sentinels = _split_header_sentinels(extra_headers)
    bootstrap = resolve_bootstrap(
        config_path=config_path,
        base_url=base_url,
        context_name=context_name,
        access_token=access_token,
        extra_headers=header_values,
    )
    if bootstrap.token_provider is None:
        return ClientInitConfig(
            base_url=bootstrap.base_url,
            workspace=bootstrap.workspace,
            default_headers=_with_sentinels(bootstrap.default_headers, sentinels),
            client_verify=bootstrap.client_verify,
        )

    headers = _headers_with_seeded_auth(bootstrap.default_headers, bootstrap.token_provider)
    hook = _make_async_auth_event_hook(bootstrap.token_provider)
    http_client = (
        http_client_factory(hook, bootstrap.client_verify)
        if http_client_factory is not None
        else httpx.AsyncClient(
            event_hooks={"request": [hook], "response": []},
            timeout=resolve_timeout(None),
            limits=_CONNECTION_LIMITS,
            follow_redirects=True,
            verify=bootstrap.client_verify,
        )
    )
    return ClientInitConfig(
        base_url=bootstrap.base_url,
        workspace=bootstrap.workspace,
        default_headers=_with_sentinels(headers, sentinels),
        http_client=http_client,
        client_verify=bootstrap.client_verify,
    )
