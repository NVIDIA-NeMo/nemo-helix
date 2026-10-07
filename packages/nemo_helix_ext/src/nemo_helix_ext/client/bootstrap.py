# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config, TLS, and typed-client construction for NeMo Helix clients.

This module bridges the nhx CLI config file (~/.config/nhx/config.yaml) and the
platform HTTP clients. It owns config resolution, TLS verification, timeout
normalization, and concrete httpx client construction. OAuth/workload auth
resolution lives in :mod:`nemo_helix_ext.auth.bootstrap`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import httpx
from nemo_helix_plugin.client.auth import AsyncClientTokenProvider, AsyncClientTokenProviderAdapter
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY, RetryPolicy

from nemo_helix_ext.auth.bootstrap import (
    AccessTokenProvider,
    AuthBootstrapContext,
    ResolvedAuthBootstrap,
    auth_bootstrap_requires_discovery_client,
    new_deferred_async_auth_provider,
    resolve_auth_bootstrap,
    resolve_auth_bootstrap_without_discovery,
)
from nemo_helix_ext.client.tls import client_verify_from_env
from nemo_helix_ext.config.models import Context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedBootstrap:
    """Result of resolving config + OIDC discovery into client parameters."""

    base_url: str
    workspace: str | None
    default_headers: dict[str, str]
    token_provider: AccessTokenProvider | None  # None for non-OAuth users
    client_verify: str | Literal[True]
    certificate_authority: str | None = None


@dataclass(frozen=True)
class BootstrapContext:
    """Config-derived inputs needed before HTTP discovery can run."""

    resolved: Context
    config_exists: bool
    config_path: Path
    base_url: str
    certificate_authority: str | None
    client_verify: str | Literal[True]
    default_headers: dict[str, str]
    access_token: str | None


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _resolve_client_context(
    *,
    config_path: Path | None,
    base_url: str | httpx.URL | None,
    context_name: str | None,
    access_token: str | None,
) -> tuple[Context, bool, Path]:
    """Load the nhx config file, apply overrides, and resolve the active context.

    Returns ``(resolved_context, config_exists, config_path)`` so the caller
    knows whether to enable config-backed features (provider caching,
    token persistence, file locking).
    """
    from nemo_helix_ext.config.config import Config, ConfigParams

    resolved_config_path = config_path or Config.get_default_config_path()
    config_exists = resolved_config_path.exists()
    if config_exists:
        logger.info("Reading nhx config from %s", resolved_config_path)

    # Constructor args override whatever is in the config file.
    overrides: ConfigParams | None = None
    if context_name is not None or access_token is not None or base_url is not None:
        overrides = {}
        if base_url is not None:
            overrides["base_url"] = str(base_url)
        if context_name is not None:
            overrides["current_context"] = context_name
        if access_token is not None:
            overrides["access_token"] = access_token

    config = Config.load(config_path=config_path, overrides=overrides)
    if context_name is not None:
        available_contexts = [ctx.name for ctx in config.get_config_file().contexts]
        if context_name not in available_contexts:
            available = ", ".join(available_contexts) if available_contexts else "(none)"
            raise ValueError(f"Context '{context_name}' not found. Available contexts: {available}")

    return config.resolve(), config_exists, resolved_config_path


# ---------------------------------------------------------------------------
# Bootstrap: config + OIDC discovery → resolved client params
# ---------------------------------------------------------------------------


def resolve_bootstrap_context(
    *,
    config_path: Path | None,
    base_url: str | httpx.URL | None,
    context_name: str | None,
    access_token: str | None,
    extra_headers: Mapping[str, str] | None,
) -> BootstrapContext:
    """Resolve config-derived bootstrap inputs before HTTP discovery."""
    resolved, config_exists, resolved_config_path = _resolve_client_context(
        config_path=config_path,
        base_url=base_url,
        context_name=context_name,
        access_token=access_token,
    )

    base_url = str(resolved.cluster.base_url)
    certificate_authority = resolved.cluster.certificate_authority
    client_verify = client_verify_from_env(certificate_authority)
    headers: dict[str, str] = dict(extra_headers) if extra_headers else {}
    return BootstrapContext(
        resolved=resolved,
        config_exists=config_exists,
        config_path=resolved_config_path,
        base_url=base_url,
        certificate_authority=certificate_authority,
        client_verify=client_verify,
        default_headers=headers,
        access_token=access_token,
    )


def bootstrap_requires_discovery_client(bootstrap_context: BootstrapContext) -> bool:
    """Return True when resolving bootstrap needs a caller-provided discovery client."""
    return auth_bootstrap_requires_discovery_client(_auth_bootstrap_context(bootstrap_context))


def resolve_bootstrap_without_discovery(bootstrap_context: BootstrapContext) -> ResolvedBootstrap:
    """Resolve bootstrap paths that do not need OAuth discovery."""
    auth_bootstrap = resolve_auth_bootstrap_without_discovery(_auth_bootstrap_context(bootstrap_context))
    return _resolved_bootstrap_from_auth(bootstrap_context, auth_bootstrap)


def _auth_bootstrap_context(bootstrap_context: BootstrapContext) -> AuthBootstrapContext:
    return AuthBootstrapContext(
        resolved=bootstrap_context.resolved,
        config_exists=bootstrap_context.config_exists,
        config_path=bootstrap_context.config_path,
        base_url=bootstrap_context.base_url,
        certificate_authority=bootstrap_context.certificate_authority,
        default_headers=bootstrap_context.default_headers,
        access_token=bootstrap_context.access_token,
    )


def _resolved_bootstrap_from_auth(
    bootstrap_context: BootstrapContext,
    auth_bootstrap: ResolvedAuthBootstrap,
) -> ResolvedBootstrap:
    return ResolvedBootstrap(
        base_url=bootstrap_context.base_url,
        workspace=auth_bootstrap.workspace,
        default_headers=auth_bootstrap.default_headers,
        token_provider=auth_bootstrap.token_provider,
        client_verify=bootstrap_context.client_verify,
        certificate_authority=bootstrap_context.certificate_authority,
    )


def resolve_bootstrap(
    bootstrap_context: BootstrapContext,
    *,
    http_client: httpx.Client,
) -> ResolvedBootstrap:
    """Resolve OIDC discovery and token provider state using *http_client*.

    ``http_client`` is owned by the caller. SDK constructors pass either their
    final shared sync transport or a constructor-scoped discovery transport.
    """
    auth_bootstrap = resolve_auth_bootstrap(_auth_bootstrap_context(bootstrap_context), http_client=http_client)
    return _resolved_bootstrap_from_auth(bootstrap_context, auth_bootstrap)


# ---------------------------------------------------------------------------
# Typed client construction
# ---------------------------------------------------------------------------

DEFAULT_RETRY_POLICY = PLATFORM_DEFAULT_RETRY_POLICY


# Connect phase cap for CLI clients. A blackholed endpoint fails in seconds
# rather than waiting out the full read timeout on every attempt.
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_REQUEST_TIMEOUT = 60.0


def resolve_timeout(timeout: float | httpx.Timeout | None) -> httpx.Timeout:
    """Normalize a caller timeout into phase-aware httpx form.

    A bare number (or ``None`` for the 60 s default) is the read/write/pool
    budget; the connect phase is capped at :data:`DEFAULT_CONNECT_TIMEOUT`
    separately. An explicit :class:`httpx.Timeout` is used as given.
    """
    if isinstance(timeout, httpx.Timeout):
        return timeout
    return httpx.Timeout(DEFAULT_REQUEST_TIMEOUT if timeout is None else timeout, connect=DEFAULT_CONNECT_TIMEOUT)


def _client_headers(bootstrap: ResolvedBootstrap) -> dict[str, str]:
    """Return the default headers for a client built from *bootstrap*.

    A static bearer token (API key user) already lives in ``default_headers``.
    Token providers inject ``Authorization`` per request, so nothing is seeded
    here; raw ``_client`` calls that need auth must pass request headers
    resolved by the owning ``NemoClient``.
    """
    return dict(bootstrap.default_headers)


def build_nemo_client(
    *,
    config_path: Path | None = None,
    base_url: str | httpx.URL | None = None,
    context_name: str | None = None,
    access_token: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    workspace: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    retry: RetryPolicy | None = DEFAULT_RETRY_POLICY,
) -> NemoClient:
    """Build a sync :class:`NemoClient` from the nhx config and auth bootstrap.

    For OAuth and workload-identity users the resulting client refreshes and
    injects the Bearer token before every request; API-key users get static
    headers. *workspace* overrides the configured default when given.
    """
    resolved_timeout = resolve_timeout(timeout)
    bootstrap_context = resolve_bootstrap_context(
        config_path=config_path,
        base_url=base_url,
        context_name=context_name,
        access_token=access_token,
        extra_headers=extra_headers,
    )
    http_client = httpx.Client(
        timeout=resolved_timeout,
        follow_redirects=True,
        verify=bootstrap_context.client_verify,
    )
    try:
        bootstrap = resolve_bootstrap(bootstrap_context, http_client=http_client)
        return NemoClient(
            base_url=bootstrap.base_url,
            workspace=workspace if workspace is not None else bootstrap.workspace,
            auth=bootstrap.token_provider,
            default_headers=_client_headers(bootstrap) or None,
            timeout=resolved_timeout,
            retry=retry,
            http_client=http_client,
            owns_http_client=True,
        )
    except Exception:
        http_client.close()
        raise


def build_async_nemo_client(
    *,
    config_path: Path | None = None,
    base_url: str | httpx.URL | None = None,
    context_name: str | None = None,
    access_token: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    workspace: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    retry: RetryPolicy | None = DEFAULT_RETRY_POLICY,
) -> AsyncNemoClient:
    """Async twin of :func:`build_nemo_client`."""
    resolved_timeout = resolve_timeout(timeout)
    bootstrap_context = resolve_bootstrap_context(
        config_path=config_path,
        base_url=base_url,
        context_name=context_name,
        access_token=access_token,
        extra_headers=extra_headers,
    )
    if bootstrap_requires_discovery_client(bootstrap_context):
        http_client = httpx.AsyncClient(
            timeout=resolved_timeout,
            follow_redirects=True,
            verify=bootstrap_context.client_verify,
        )
        token_provider = new_deferred_async_auth_provider(
            context=_auth_bootstrap_context(bootstrap_context),
            http_client=http_client,
        )
        default_headers = dict(bootstrap_context.default_headers)
        resolved_workspace = bootstrap_context.resolved.workspace
    else:
        bootstrap = resolve_bootstrap_without_discovery(bootstrap_context)
        http_client = httpx.AsyncClient(
            timeout=resolved_timeout,
            follow_redirects=True,
            verify=bootstrap.client_verify,
        )
        token_provider: AsyncClientTokenProvider | None = (
            AsyncClientTokenProviderAdapter(bootstrap.token_provider) if bootstrap.token_provider is not None else None
        )
        default_headers = _client_headers(bootstrap)
        resolved_workspace = bootstrap.workspace
    return AsyncNemoClient(
        base_url=bootstrap_context.base_url,
        workspace=workspace if workspace is not None else resolved_workspace,
        auth=token_provider,
        default_headers=default_headers or None,
        timeout=resolved_timeout,
        retry=retry,
        http_client=http_client,
        owns_http_client=True,
    )


def build_direct_nemo_client(
    *,
    base_url: str,
    workspace: str | None = None,
    default_headers: Mapping[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    certificate_authority: str | None = None,
    retry: RetryPolicy | None = DEFAULT_RETRY_POLICY,
) -> NemoClient:
    """Build a sync :class:`NemoClient` without reading the nhx config.

    Direct mode: no config file is consulted and only *default_headers* are
    sent. TLS verification still honours the environment override and any
    saved cluster certificate authority.
    """
    resolved_timeout = resolve_timeout(timeout)
    http_client = httpx.Client(
        timeout=resolved_timeout,
        follow_redirects=True,
        verify=client_verify_from_env(certificate_authority),
    )
    return NemoClient(
        base_url=base_url,
        workspace=workspace,
        default_headers=default_headers,
        timeout=resolved_timeout,
        retry=retry,
        http_client=http_client,
        owns_http_client=True,
    )


def build_direct_async_nemo_client(
    *,
    base_url: str,
    workspace: str | None = None,
    default_headers: Mapping[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    certificate_authority: str | None = None,
    retry: RetryPolicy | None = DEFAULT_RETRY_POLICY,
) -> AsyncNemoClient:
    """Async twin of :func:`build_direct_nemo_client`."""
    resolved_timeout = resolve_timeout(timeout)
    http_client = httpx.AsyncClient(
        timeout=resolved_timeout,
        follow_redirects=True,
        verify=client_verify_from_env(certificate_authority),
    )
    return AsyncNemoClient(
        base_url=base_url,
        workspace=workspace,
        default_headers=default_headers,
        timeout=resolved_timeout,
        retry=retry,
        http_client=http_client,
        owns_http_client=True,
    )
