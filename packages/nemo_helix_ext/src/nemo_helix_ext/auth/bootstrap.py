# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth bootstrap helpers for config-backed NeMo Helix clients."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR

from nemo_helix_ext.auth.helpers import (
    NHXOIDCConfig,
    build_effective_scope,
    discover_nhx_config,
    discover_nhx_config_async,
)
from nemo_helix_ext.auth.token_provider import OIDCTokenProvider, TokenSet
from nemo_helix_ext.auth.workload_exchange import WorkloadTokenExchangeProvider
from nemo_helix_ext.config.models import Context, OAuthUser

logger = logging.getLogger(__name__)

# Refresh the access token when fewer than 60 s remain before expiry.
_TOKEN_REFRESH_MARGIN_SECONDS = 60

# Guards _TOKEN_PROVIDER_CACHE; acquired only during dict lookup/insert (fast).
_TOKEN_PROVIDER_CACHE_LOCK = threading.Lock()


class AccessTokenProvider(Protocol):
    def get_access_token(self) -> str: ...

    async def get_access_token_async(self) -> str: ...


SyncRequestHook = Callable[[httpx.Request], None]
AsyncRequestHook = Callable[[httpx.Request], Awaitable[None]]
SyncHttpClientFactory = Callable[[SyncRequestHook, str | Literal[True]], httpx.Client]
AsyncHttpClientFactory = Callable[[AsyncRequestHook, str | Literal[True]], httpx.AsyncClient]

_AUTH_CLIENT_CONNECTION_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)


@dataclass(frozen=True)
class AuthBootstrapContext:
    """Config-derived auth inputs for resolving request-time credentials."""

    resolved: Context
    config_exists: bool
    config_path: Path
    base_url: str
    certificate_authority: str | None
    default_headers: Mapping[str, str]
    access_token: str | None


@dataclass(frozen=True)
class ResolvedAuthBootstrap:
    """Auth state that client/bootstrap code applies to a concrete client."""

    workspace: str | None
    default_headers: dict[str, str]
    token_provider: AccessTokenProvider | None


@dataclass(frozen=True)
class AuthClientConfig:
    """HTTP client and headers needed by SDK constructors for request-time auth."""

    default_headers: dict[str, str]
    http_client: httpx.Client | httpx.AsyncClient | None = None


class DeferredAuthEventHook:
    """Request hook installed before discovery, then connected to a provider."""

    def __init__(self) -> None:
        self._provider: AccessTokenProvider | None = None

    def set_provider(self, provider: AccessTokenProvider | None) -> None:
        self._provider = provider

    def __call__(self, request: httpx.Request) -> None:
        provider = self._provider
        if provider is not None:
            request.headers["Authorization"] = f"Bearer {provider.get_access_token()}"


class DeferredAsyncAuthEventHook:
    """Async request hook installed before its lazy provider exists."""

    def __init__(self) -> None:
        self._provider: DeferredAsyncAuthProvider | None = None

    def set_provider(self, provider: DeferredAsyncAuthProvider) -> None:
        self._provider = provider

    async def __call__(self, request: httpx.Request) -> None:
        if request.url.path.rstrip("/") == "/apis/auth/discovery":
            return
        provider = self._provider
        if provider is None:
            return
        token = await provider.get_access_token_or_none_async()
        if token is not None:
            request.headers["Authorization"] = f"Bearer {token}"


@dataclass(frozen=True)
class DeferredSyncAuthClient:
    """Sync auth transport created before discovery and connected after bootstrap resolution."""

    http_client: httpx.Client
    hook: DeferredAuthEventHook

    def close(self) -> None:
        self.http_client.close()

    def resolve(
        self,
        *,
        default_headers: Mapping[str, str],
        token_provider: AccessTokenProvider | None,
    ) -> AuthClientConfig:
        if token_provider is None:
            self.close()
            return AuthClientConfig(default_headers=dict(default_headers))

        self.hook.set_provider(token_provider)
        return AuthClientConfig(
            default_headers=_headers_with_seeded_auth(default_headers, token_provider),
            http_client=self.http_client,
        )


class DeferredAsyncAuthProvider:
    """Async token provider that resolves OIDC discovery through its final transport."""

    def __init__(self, *, context: AuthBootstrapContext, http_client: httpx.AsyncClient) -> None:
        self._context = context
        self._http_client = http_client
        self._resolved: ResolvedAuthBootstrap | None = None
        self._lock = asyncio.Lock()

    def set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._http_client = http_client

    async def resolve(self) -> ResolvedAuthBootstrap:
        """Resolve discovery once, sharing concurrent first requests."""
        resolved = self._resolved
        if resolved is not None:
            return resolved

        async with self._lock:
            resolved = self._resolved
            if resolved is None:
                resolved = await resolve_auth_bootstrap_async(self._context, http_client=self._http_client)
                self._resolved = resolved
            return resolved

    async def get_access_token_or_none_async(self) -> str | None:
        resolved = await self.resolve()
        provider = resolved.token_provider
        if provider is None:
            return None
        return await provider.get_access_token_async()

    async def get_access_token_async(self) -> str:
        token = await self.get_access_token_or_none_async()
        if token is None:
            raise RuntimeError("Async auth bootstrap resolved no bearer token")
        return token


@dataclass(frozen=True)
class DeferredAsyncAuthClient:
    """Async auth transport whose first request resolves discovery on the same transport."""

    http_client: httpx.AsyncClient
    provider: DeferredAsyncAuthProvider

    def resolve(self, *, default_headers: Mapping[str, str]) -> AuthClientConfig:
        return AuthClientConfig(default_headers=dict(default_headers), http_client=self.http_client)


@dataclass(frozen=True)
class _ProviderCacheKey:
    """Composite key for the provider cache."""

    config_path: Path
    context_name: str
    token_endpoint: str
    client_id: str
    refresh_scope: str | None
    bearer_token_source: str
    certificate_authority: str | None


# Process-wide cache: (config_path, context, OIDC settings, CA) -> shared OIDCTokenProvider.
_TOKEN_PROVIDER_CACHE: dict[_ProviderCacheKey, OIDCTokenProvider] = {}


_OIDC_DISCOVERY_FALLBACK = NHXOIDCConfig(
    auth_enabled=False,
    client_id="",
    token_endpoint="",
    default_scopes="openid profile email",
    scope_prefix=None,
)


def auth_bootstrap_requires_discovery_client(context: AuthBootstrapContext) -> bool:
    """Return True when auth resolution needs a caller-owned discovery client."""
    return isinstance(context.resolved.user, OAuthUser) or workload_identity_token_file(context) is not None


def workload_identity_token_file(context: AuthBootstrapContext) -> Path | None:
    """Return the configured workload identity token file, if workload bootstrap is active."""
    if context.access_token is not None or os.environ.get("NHX_ACCESS_TOKEN"):
        return None
    token_file = os.environ.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR)
    return Path(token_file) if token_file else None


def resolve_auth_bootstrap_without_discovery(context: AuthBootstrapContext) -> ResolvedAuthBootstrap:
    """Resolve non-discovery auth state from the already-resolved config context."""
    if auth_bootstrap_requires_discovery_client(context):
        raise RuntimeError("Auth bootstrap requires a discovery HTTP client")

    user = context.resolved.user
    headers = dict(context.default_headers)
    user_config = user.get_client_config() if user else {}
    user_headers = user_config.get("default_headers", {})
    if isinstance(user_headers, dict):
        headers.update(
            {key: value for key, value in user_headers.items() if isinstance(key, str) and isinstance(value, str)}
        )
    return ResolvedAuthBootstrap(context.resolved.workspace, headers, None)


def resolve_auth_bootstrap(
    context: AuthBootstrapContext,
    *,
    http_client: httpx.Client,
) -> ResolvedAuthBootstrap:
    """Resolve auth discovery and token provider state using *http_client*."""
    subject_token_file = workload_identity_token_file(context)
    if subject_token_file is not None:
        return _resolve_workload_bootstrap(
            context,
            subject_token_file=subject_token_file,
            http_client=http_client,
        )

    if not auth_bootstrap_requires_discovery_client(context):
        return resolve_auth_bootstrap_without_discovery(context)

    user = context.resolved.user
    if not isinstance(user, OAuthUser):
        raise RuntimeError("OAuth bootstrap requires an OAuth user")

    return _resolve_oauth_bootstrap(context, user=user, http_client=http_client)


async def resolve_auth_bootstrap_async(
    context: AuthBootstrapContext,
    *,
    http_client: httpx.AsyncClient,
) -> ResolvedAuthBootstrap:
    """Resolve auth discovery and token provider state using *http_client*."""
    subject_token_file = workload_identity_token_file(context)
    if subject_token_file is not None:
        return await _resolve_workload_bootstrap_async(
            context,
            subject_token_file=subject_token_file,
            http_client=http_client,
        )

    if not auth_bootstrap_requires_discovery_client(context):
        return resolve_auth_bootstrap_without_discovery(context)

    user = context.resolved.user
    if not isinstance(user, OAuthUser):
        raise RuntimeError("OAuth bootstrap requires an OAuth user")

    return await _resolve_oauth_bootstrap_async(context, user=user, http_client=http_client)


def new_deferred_sync_auth_client(
    *,
    http_client_factory: SyncHttpClientFactory | None,
    client_verify: str | Literal[True],
    timeout: httpx.Timeout,
) -> DeferredSyncAuthClient:
    """Create a sync auth client whose provider can be attached after discovery."""
    hook = DeferredAuthEventHook()
    http_client = (
        http_client_factory(hook, client_verify)
        if http_client_factory is not None
        else httpx.Client(
            event_hooks={"request": [hook], "response": []},
            timeout=timeout,
            limits=_AUTH_CLIENT_CONNECTION_LIMITS,
            follow_redirects=True,
            verify=client_verify,
        )
    )
    return DeferredSyncAuthClient(http_client=http_client, hook=hook)


def new_deferred_async_auth_provider(
    *,
    context: AuthBootstrapContext,
    http_client: httpx.AsyncClient,
) -> DeferredAsyncAuthProvider:
    """Create a lazy async token provider bound to a final async transport."""
    return DeferredAsyncAuthProvider(context=context, http_client=http_client)


def new_deferred_async_auth_client(
    *,
    context: AuthBootstrapContext,
    http_client_factory: AsyncHttpClientFactory | None,
    client_verify: str | Literal[True],
    timeout: httpx.Timeout,
) -> DeferredAsyncAuthClient:
    """Create an async auth client that resolves discovery on first use."""
    hook = DeferredAsyncAuthEventHook()
    http_client = (
        http_client_factory(hook, client_verify)
        if http_client_factory is not None
        else httpx.AsyncClient(
            event_hooks={"request": [hook], "response": []},
            timeout=timeout,
            limits=_AUTH_CLIENT_CONNECTION_LIMITS,
            follow_redirects=True,
            verify=client_verify,
        )
    )
    provider = DeferredAsyncAuthProvider(context=context, http_client=http_client)
    hook.set_provider(provider)
    return DeferredAsyncAuthClient(http_client=http_client, provider=provider)


def _headers_with_seeded_auth(headers: Mapping[str, str], provider: AccessTokenProvider) -> dict[str, str]:
    """Return headers with the provider's current bearer token when one is available."""
    seeded_headers = dict(headers)
    if isinstance(provider, WorkloadTokenExchangeProvider):
        tokens = provider.tokens
        token = (
            tokens.access_token
            if tokens.access_token and not tokens.is_expired(provider.refresh_margin_seconds)
            else None
        )
    else:
        token = provider.get_access_token()
    if token:
        seeded_headers["Authorization"] = f"Bearer {token}"
    return seeded_headers


def _discover_oidc_client_settings_with_client(base_url: str, http_client: httpx.Client) -> NHXOIDCConfig:
    """Fetch OIDC config using an existing transport."""
    return discover_nhx_config(base_url, http_client=http_client)


async def _discover_oidc_client_settings_with_async_client(
    base_url: str,
    http_client: httpx.AsyncClient,
) -> NHXOIDCConfig:
    """Fetch OIDC config using an existing async transport."""
    return await discover_nhx_config_async(base_url, http_client=http_client)


def _resolve_oauth_bootstrap(
    context: AuthBootstrapContext,
    *,
    user: OAuthUser,
    http_client: httpx.Client,
) -> ResolvedAuthBootstrap:
    try:
        oidc_config = _discover_oidc_client_settings_with_client(context.base_url, http_client)
        if not oidc_config.auth_enabled and context.access_token is None:
            return ResolvedAuthBootstrap(context.resolved.workspace, dict(context.default_headers), None)
    except (httpx.HTTPError, json.JSONDecodeError, AttributeError):
        logger.debug("Could not discover OIDC settings from %s", context.base_url, exc_info=True)
        oidc_config = _OIDC_DISCOVERY_FALLBACK

    provider = _build_oauth_token_provider(context, user=user, oidc_config=oidc_config)
    return ResolvedAuthBootstrap(context.resolved.workspace, dict(context.default_headers), provider)


async def _resolve_oauth_bootstrap_async(
    context: AuthBootstrapContext,
    *,
    user: OAuthUser,
    http_client: httpx.AsyncClient,
) -> ResolvedAuthBootstrap:
    try:
        oidc_config = await _discover_oidc_client_settings_with_async_client(context.base_url, http_client)
        if not oidc_config.auth_enabled and context.access_token is None:
            return ResolvedAuthBootstrap(context.resolved.workspace, dict(context.default_headers), None)
    except (httpx.HTTPError, json.JSONDecodeError, AttributeError):
        logger.debug("Could not discover OIDC settings from %s", context.base_url, exc_info=True)
        oidc_config = _OIDC_DISCOVERY_FALLBACK

    provider = _build_oauth_token_provider(context, user=user, oidc_config=oidc_config)
    return ResolvedAuthBootstrap(context.resolved.workspace, dict(context.default_headers), provider)


def _resolve_workload_bootstrap(
    context: AuthBootstrapContext,
    *,
    subject_token_file: Path,
    http_client: httpx.Client,
) -> ResolvedAuthBootstrap:
    try:
        oidc_config = _discover_oidc_client_settings_with_client(context.base_url, http_client)
    except (httpx.HTTPError, json.JSONDecodeError, AttributeError):
        logger.debug("Could not discover OIDC settings from %s", context.base_url, exc_info=True)
        oidc_config = _OIDC_DISCOVERY_FALLBACK
    provider = _create_workload_exchange_provider(
        oidc_config,
        subject_token_file,
        certificate_authority=context.certificate_authority,
    )
    return ResolvedAuthBootstrap(context.resolved.workspace, dict(context.default_headers), provider)


async def _resolve_workload_bootstrap_async(
    context: AuthBootstrapContext,
    *,
    subject_token_file: Path,
    http_client: httpx.AsyncClient,
) -> ResolvedAuthBootstrap:
    try:
        oidc_config = await _discover_oidc_client_settings_with_async_client(context.base_url, http_client)
    except (httpx.HTTPError, json.JSONDecodeError, AttributeError):
        logger.debug("Could not discover OIDC settings from %s", context.base_url, exc_info=True)
        oidc_config = _OIDC_DISCOVERY_FALLBACK
    provider = _create_workload_exchange_provider(
        oidc_config,
        subject_token_file,
        certificate_authority=context.certificate_authority,
    )
    return ResolvedAuthBootstrap(context.resolved.workspace, dict(context.default_headers), provider)


def _create_workload_exchange_provider(
    oidc_config: NHXOIDCConfig,
    subject_token_file: Path,
    *,
    certificate_authority: str | None = None,
) -> WorkloadTokenExchangeProvider:
    """Create a workload identity token exchange provider from NeMo auth discovery metadata."""
    if not oidc_config.workload_token_exchange_enabled:
        raise RuntimeError(
            f"{WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR} is set but workload token exchange is not enabled by auth discovery"
        )

    token_endpoint = oidc_config.workload_token_endpoint or oidc_config.token_endpoint or ""
    client_id = oidc_config.workload_client_id or oidc_config.client_id or ""
    if not token_endpoint:
        raise RuntimeError(
            "Workload token exchange is enabled but auth discovery did not return workload_token_endpoint or token_endpoint"
        )
    if not client_id:
        raise RuntimeError(
            "Workload token exchange is enabled but auth discovery did not return workload_client_id or client_id"
        )

    return WorkloadTokenExchangeProvider(
        token_endpoint=token_endpoint,
        client_id=client_id,
        subject_token_file=subject_token_file,
        audience=oidc_config.workload_audience,
        scope=oidc_config.workload_scope,
        certificate_authority=certificate_authority,
        refresh_margin_seconds=_TOKEN_REFRESH_MARGIN_SECONDS,
    )


def _build_oauth_token_provider(
    context: AuthBootstrapContext,
    *,
    user: OAuthUser,
    oidc_config: NHXOIDCConfig,
) -> OIDCTokenProvider:
    tokens = TokenSet.from_access_token(
        user.token.get_secret_value(),
        user.refresh_token.get_secret_value() if user.refresh_token else None,
        expires_at=user.expires_at,
    )

    token_endpoint = oidc_config.token_endpoint or ""
    client_id = oidc_config.cli_client_id or oidc_config.client_id or ""
    refresh_scope = build_effective_scope(oidc_config.default_scopes, oidc_config.scope_prefix)

    if not (context.config_exists and context.access_token is None):
        return OIDCTokenProvider(
            token_endpoint=token_endpoint,
            client_id=client_id,
            tokens=tokens,
            refresh_margin_seconds=_TOKEN_REFRESH_MARGIN_SECONDS,
            refresh_scope=refresh_scope,
            bearer_token_source=oidc_config.bearer_token_source,
            certificate_authority=context.certificate_authority,
        )

    normalized_config_path = _normalize_config_path(context.config_path)
    provider_key = _ProviderCacheKey(
        config_path=normalized_config_path,
        context_name=context.resolved.context_name,
        token_endpoint=token_endpoint,
        client_id=client_id,
        refresh_scope=refresh_scope,
        bearer_token_source=oidc_config.bearer_token_source,
        certificate_authority=context.certificate_authority,
    )
    on_refreshed = _make_config_persister(context.resolved.context_name, context.config_path)
    load_tokens = _make_config_token_loader(context.resolved.context_name, context.config_path)
    refresh_lock = _make_refresh_lock(context.config_path, context.resolved.context_name)

    return _get_or_create_provider(
        provider_key,
        lambda: OIDCTokenProvider(
            token_endpoint=token_endpoint,
            client_id=client_id,
            tokens=tokens,
            refresh_margin_seconds=_TOKEN_REFRESH_MARGIN_SECONDS,
            refresh_scope=refresh_scope,
            bearer_token_source=oidc_config.bearer_token_source,
            certificate_authority=context.certificate_authority,
            load_tokens=load_tokens,
            refresh_lock=refresh_lock,
            on_tokens_refreshed=on_refreshed,
        ),
    )


def _make_config_persister(context_name: str, config_path: Path | None = None):
    """Create an ``on_tokens_refreshed`` callback that writes new tokens to the nhx config file."""
    from nemo_helix_ext.config.config import Config, ConfigParams

    def persist(tokens: TokenSet) -> None:
        params: ConfigParams = {
            "access_token": tokens.access_token,
            "expires_at": tokens.expires_at,
        }
        if tokens.refresh_token:
            params["refresh_token"] = tokens.refresh_token
        Config.write(params, context_name=context_name, config_path=config_path)
        logger.debug("Persisted refreshed tokens to nhx config (context=%s)", context_name)

    return persist


def _make_config_token_loader(context_name: str, config_path: Path):
    """Create a ``load_tokens`` callback that re-reads tokens from the config file."""
    from nemo_helix_ext.config.config import Config, ConfigParams

    def load_tokens() -> TokenSet | None:
        overrides: ConfigParams = {"current_context": context_name}
        try:
            config = Config.load(config_path=config_path, overrides=overrides)
            resolved = config.resolve()
        except Exception:
            logger.debug("Failed to reload tokens from nhx config (context=%s)", context_name, exc_info=True)
            return None

        if not isinstance(resolved.user, OAuthUser):
            return None

        return TokenSet.from_access_token(
            resolved.user.token.get_secret_value(),
            resolved.user.refresh_token.get_secret_value() if resolved.user.refresh_token else None,
            expires_at=resolved.user.expires_at,
        )

    return load_tokens


def _build_refresh_lock_path(config_path: Path, context_name: str) -> Path:
    safe_context = context_name.replace(os.sep, "_")
    if os.altsep:
        safe_context = safe_context.replace(os.altsep, "_")
    return config_path.with_name(f"{config_path.name}.{safe_context}.oauth-refresh.lock")


def _make_refresh_lock(config_path: Path, context_name: str):
    """Create a cross-process file lock for serializing token refreshes."""
    lock_path = _build_refresh_lock_path(config_path, context_name)

    @contextmanager
    def refresh_lock():
        try:
            import fcntl
        except ImportError:
            yield
            return

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return refresh_lock


def _normalize_config_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _get_or_create_provider(
    key: _ProviderCacheKey,
    create_provider: Callable[[], OIDCTokenProvider],
) -> OIDCTokenProvider:
    """Return the cached provider for *key*, or create and cache a new one."""
    with _TOKEN_PROVIDER_CACHE_LOCK:
        provider = _TOKEN_PROVIDER_CACHE.get(key)
        if provider is None:
            provider = create_provider()
            _TOKEN_PROVIDER_CACHE[key] = provider
            return provider

    provider.reload_tokens()
    return provider
