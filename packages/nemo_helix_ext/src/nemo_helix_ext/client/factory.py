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

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from nemo_helix_ext.auth.bootstrap import (
    _TOKEN_PROVIDER_CACHE,
    _TOKEN_PROVIDER_CACHE_LOCK,
    AccessTokenProvider,
    AsyncHttpClientFactory,
    AuthBootstrapContext,
    AuthClientConfig,
    SyncHttpClientFactory,
    new_deferred_async_auth_client,
    new_deferred_sync_auth_client,
)
from nemo_helix_ext.client.bootstrap import (
    ResolvedBootstrap,
    bootstrap_requires_discovery_client,
    resolve_bootstrap,
    resolve_bootstrap_context,
    resolve_bootstrap_without_discovery,
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


def _client_init_config(
    *,
    bootstrap: ResolvedBootstrap,
    sentinels: Mapping[str, object],
    client_config: AuthClientConfig,
) -> ClientInitConfig:
    return ClientInitConfig(
        base_url=bootstrap.base_url,
        workspace=bootstrap.workspace,
        default_headers=_with_sentinels(client_config.default_headers, sentinels),
        http_client=client_config.http_client,
        client_verify=bootstrap.client_verify,
    )


# ---------------------------------------------------------------------------
# Public API: build kwargs
# ---------------------------------------------------------------------------


def build_client_init_kwargs(
    *,
    config_path: Path | None = None,
    base_url: str | httpx.URL | None = None,
    context_name: str | None = None,
    access_token: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    http_client_factory: SyncHttpClientFactory | None = None,
) -> ClientInitConfig:
    """Build constructor kwargs for a **sync** client.

    For OAuth users, returns a ``ClientInitConfig`` whose ``http_client``
    has a request event hook that transparently injects and refreshes the
    Bearer token before every request.
    """
    header_values, sentinels = _split_header_sentinels(extra_headers)
    bootstrap_context = resolve_bootstrap_context(
        config_path=config_path,
        base_url=base_url,
        context_name=context_name,
        access_token=access_token,
        extra_headers=header_values,
    )
    if not bootstrap_requires_discovery_client(bootstrap_context):
        bootstrap = resolve_bootstrap_without_discovery(bootstrap_context)
        return _client_init_config(
            bootstrap=bootstrap,
            sentinels=sentinels,
            client_config=AuthClientConfig(default_headers=bootstrap.default_headers),
        )

    auth_client = new_deferred_sync_auth_client(
        http_client_factory=http_client_factory,
        client_verify=bootstrap_context.client_verify,
        timeout=resolve_timeout(None),
    )
    try:
        bootstrap = resolve_bootstrap(bootstrap_context, http_client=auth_client.http_client)
        client_config = auth_client.resolve(
            default_headers=bootstrap.default_headers,
            token_provider=bootstrap.token_provider,
        )
        return _client_init_config(
            bootstrap=bootstrap,
            sentinels=sentinels,
            client_config=client_config,
        )
    except Exception:
        auth_client.close()
        raise


def build_async_client_init_kwargs(
    *,
    config_path: Path | None = None,
    base_url: str | httpx.URL | None = None,
    context_name: str | None = None,
    access_token: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    http_client_factory: AsyncHttpClientFactory | None = None,
) -> ClientInitConfig:
    """Build constructor kwargs for an **async** client.

    Same as ``build_client_init_kwargs`` but returns an async httpx client
    whose event hook calls ``provider.get_access_token_async()`` (runs
    the refresh in a worker thread so it doesn't block the event loop).
    """
    header_values, sentinels = _split_header_sentinels(extra_headers)
    bootstrap_context = resolve_bootstrap_context(
        config_path=config_path,
        base_url=base_url,
        context_name=context_name,
        access_token=access_token,
        extra_headers=header_values,
    )
    if bootstrap_requires_discovery_client(bootstrap_context):
        auth_client = new_deferred_async_auth_client(
            context=AuthBootstrapContext(
                resolved=bootstrap_context.resolved,
                config_exists=bootstrap_context.config_exists,
                config_path=bootstrap_context.config_path,
                base_url=bootstrap_context.base_url,
                certificate_authority=bootstrap_context.certificate_authority,
                default_headers=bootstrap_context.default_headers,
                access_token=bootstrap_context.access_token,
            ),
            http_client_factory=http_client_factory,
            client_verify=bootstrap_context.client_verify,
            timeout=resolve_timeout(None),
        )
        return ClientInitConfig(
            base_url=bootstrap_context.base_url,
            workspace=bootstrap_context.resolved.workspace,
            default_headers=_with_sentinels(bootstrap_context.default_headers, sentinels),
            http_client=auth_client.http_client,
            client_verify=bootstrap_context.client_verify,
        )
    else:
        bootstrap = resolve_bootstrap_without_discovery(bootstrap_context)

    return _client_init_config(
        bootstrap=bootstrap,
        sentinels=sentinels,
        client_config=AuthClientConfig(default_headers=bootstrap.default_headers),
    )
