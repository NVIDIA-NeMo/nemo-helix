# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for creating NeMoPlatform SDK clients from nmp config.

Layers the generated ``NeMoPlatform`` constructors on top of the shared,
SDK-independent bootstrap in :mod:`nemo_platform_ext.client.bootstrap`, which
resolves the nmp config file, discovers OIDC settings, and builds the token
provider that keeps every request authenticated.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from nemo_platform import (
    DefaultAsyncHttpxClient,
    DefaultHttpxClient,
    NeMoPlatform,
    NotGiven,
    Omit,
    not_given,
)

from nemo_platform_ext.client.bootstrap import (
    _TOKEN_PROVIDER_CACHE,
    _TOKEN_PROVIDER_CACHE_LOCK,
    AccessTokenProvider,
    ResolvedBootstrap,
    _headers_with_seeded_auth,
    _make_async_auth_event_hook,
    _make_auth_event_hook,
    resolve_bootstrap,
)

__all__ = [
    "_TOKEN_PROVIDER_CACHE",
    "_TOKEN_PROVIDER_CACHE_LOCK",
    "AccessTokenProvider",
    "ClientInitConfig",
    "ResolvedBootstrap",
    "build_async_client_init_kwargs",
    "build_client_init_kwargs",
    "create_client",
]

_SyncRequestHook = Callable[[httpx.Request], None]
_AsyncRequestHook = Callable[[httpx.Request], Awaitable[None]]
_SyncHttpClientFactory = Callable[[_SyncRequestHook, str | Literal[True]], httpx.Client]
_AsyncHttpClientFactory = Callable[[_AsyncRequestHook, str | Literal[True]], httpx.AsyncClient]


@dataclass(frozen=True)
class ClientInitConfig:
    """Everything a generated SDK client constructor needs after config resolution.

    For non-OAuth users this just carries base_url/workspace/headers.
    For OAuth users it also includes a custom httpx client with an event
    hook that injects/refreshes the Bearer token on every request.
    """

    base_url: str
    workspace: str | None
    default_headers: Mapping[str, str | Omit] | None = None
    http_client: httpx.Client | httpx.AsyncClient | None = None
    client_verify: str | Literal[True] = True


def _split_omitted(
    extra_headers: Mapping[str, str | Omit] | None,
) -> tuple[dict[str, str], dict[str, Omit]]:
    """Separate real header values from ``Omit`` sentinels.

    ``Omit`` tells the generated SDK to drop one of its own default headers; the
    SDK-independent bootstrap only deals in real values, so the sentinels are
    carried around it and merged back into the returned config.
    """
    values: dict[str, str] = {}
    omitted: dict[str, Omit] = {}
    for key, value in (extra_headers or {}).items():
        if isinstance(value, Omit):
            omitted[key] = value
        else:
            values[key] = value
    return values, omitted


def _with_omitted(headers: Mapping[str, str] | None, omitted: Mapping[str, Omit]) -> dict[str, str | Omit] | None:
    merged: dict[str, str | Omit] = {**(headers or {}), **omitted}
    return merged or None


# ---------------------------------------------------------------------------
# Public API: build kwargs / create client
# ---------------------------------------------------------------------------


def build_client_init_kwargs(
    *,
    config_path: Path | None = None,
    base_url: str | httpx.URL | None = None,
    context_name: str | None = None,
    access_token: str | None = None,
    extra_headers: Mapping[str, str | Omit] | None = None,
    http_client_factory: _SyncHttpClientFactory | None = None,
) -> ClientInitConfig:
    """Build constructor kwargs for a **sync** NeMoPlatform client.

    For OAuth users, returns a ``ClientInitConfig`` whose ``http_client``
    has a request event hook that transparently injects and refreshes the
    Bearer token before every request.
    """
    header_values, omitted = _split_omitted(extra_headers)
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
            default_headers=_with_omitted(bootstrap.default_headers, omitted),
            client_verify=bootstrap.client_verify,
        )

    # Seed the default headers with a current token so that SDK internals
    # that inspect headers (e.g. auth_headers property) see a value. Workload
    # identity only seeds when the request-time provider already has a token.
    # The event hook will overwrite it with a fresh token on each request.
    headers = _headers_with_seeded_auth(bootstrap.default_headers, bootstrap.token_provider)
    hook = _make_auth_event_hook(bootstrap.token_provider)
    http_client = (
        http_client_factory(hook, bootstrap.client_verify)
        if http_client_factory is not None
        else DefaultHttpxClient(
            event_hooks={"request": [hook], "response": []},
            follow_redirects=True,
            verify=bootstrap.client_verify,
        )
    )
    return ClientInitConfig(
        base_url=bootstrap.base_url,
        workspace=bootstrap.workspace,
        default_headers=_with_omitted(headers, omitted),
        http_client=http_client,
        client_verify=bootstrap.client_verify,
    )


def build_async_client_init_kwargs(
    *,
    config_path: Path | None = None,
    base_url: str | httpx.URL | None = None,
    context_name: str | None = None,
    access_token: str | None = None,
    extra_headers: Mapping[str, str | Omit] | None = None,
    http_client_factory: _AsyncHttpClientFactory | None = None,
) -> ClientInitConfig:
    """Build constructor kwargs for an **async** AsyncNeMoPlatform client.

    Same as ``build_client_init_kwargs`` but returns an async httpx client
    whose event hook calls ``provider.get_access_token_async()`` (runs
    the refresh in a worker thread so it doesn't block the event loop).
    """
    header_values, omitted = _split_omitted(extra_headers)
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
            default_headers=_with_omitted(bootstrap.default_headers, omitted),
            client_verify=bootstrap.client_verify,
        )

    headers = _headers_with_seeded_auth(bootstrap.default_headers, bootstrap.token_provider)
    hook = _make_async_auth_event_hook(bootstrap.token_provider)
    http_client = (
        http_client_factory(hook, bootstrap.client_verify)
        if http_client_factory is not None
        else DefaultAsyncHttpxClient(
            event_hooks={"request": [hook], "response": []},
            follow_redirects=True,
            verify=bootstrap.client_verify,
        )
    )
    return ClientInitConfig(
        base_url=bootstrap.base_url,
        workspace=bootstrap.workspace,
        default_headers=_with_omitted(headers, omitted),
        http_client=http_client,
        client_verify=bootstrap.client_verify,
    )


def create_client(
    *,
    config_path: Path | None = None,
    base_url: str | httpx.URL | None = None,
    context_name: str | None = None,
    access_token: str | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = not_given,
    max_retries: int = 2,
    extra_headers: Mapping[str, str | Omit] | None = None,
) -> NeMoPlatform:
    """Create a NeMoPlatform client from the nmp config.

    This is a convenience wrapper that calls ``build_client_init_kwargs``
    and passes the result to the SDK constructor.
    """
    client_init_kwargs = build_client_init_kwargs(
        config_path=config_path,
        base_url=base_url,
        context_name=context_name,
        access_token=access_token,
        extra_headers=extra_headers,
    )
    http_client = client_init_kwargs.http_client
    if http_client is not None and not isinstance(http_client, httpx.Client):
        raise TypeError("build_client_init_kwargs returned a non-sync HTTP client")
    if http_client is None and client_init_kwargs.client_verify is not True:
        http_client = DefaultHttpxClient(verify=client_init_kwargs.client_verify)

    return NeMoPlatform(
        config_path=config_path,
        context_name=context_name,
        access_token=access_token,
        base_url=client_init_kwargs.base_url,
        workspace=client_init_kwargs.workspace,
        default_headers=client_init_kwargs.default_headers,
        http_client=http_client,
        max_retries=max_retries,
        timeout=timeout,
    )
