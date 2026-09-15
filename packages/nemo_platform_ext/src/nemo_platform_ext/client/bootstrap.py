# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config and auth bootstrap shared by NeMo Platform clients.

This module bridges the nmp CLI config file (~/.config/nmp/config.yaml) and the
platform HTTP clients.  When the active user is an OAuthUser, the bootstrap
wires up **transparent token refresh** so that every HTTP request made through a
client automatically carries a valid Bearer token — no manual token management
needed.  It has no dependency on any generated SDK; ``factory.py`` layers the
``NeMoPlatform`` constructors on top, and :func:`build_nemo_client` /
:func:`build_async_nemo_client` build typed ``NemoClient`` instances directly.

High-level flow
===============

    build_nemo_client() / build_client_init_kwargs()
           │
           ├─ resolve_bootstrap()
           │      ├─ _resolve_client_context()   → reads nmp config, resolves context
           │      ├─ _discover_oidc_client_settings()  → GET /apis/auth/discovery
           │      └─ _get_or_create_provider()   → reuses or creates OIDCTokenProvider
           │
           └─ returns a resolved bootstrap with:
                  • base_url, workspace, default_headers
                  • a token provider consulted before every request

Token refresh is **lazy** (on-demand), not proactive.  No background threads are
started.  The token is checked on each HTTP request; if it expires within
_TOKEN_REFRESH_MARGIN_SECONDS (60 s), a synchronous refresh_token grant is
performed inline before the request proceeds.

Concurrency safety
==================

Three layers prevent race conditions when multiple SDK instances or processes
share the same config file:

1. **In-process thread lock** — ``OIDCTokenProvider._lock`` serializes concurrent
   ``get_access_token()`` calls from different threads.
2. **Provider cache** — ``_TOKEN_PROVIDER_CACHE`` ensures multiple clients created
   in the same process with the same (config_path, context) share a single
   ``OIDCTokenProvider`` instance, avoiding redundant refreshes.
3. **Cross-process file lock** — An ``fcntl.flock``-based lock file next to the
   config prevents multiple processes from racing on refresh and clobbering each
   other's tokens.

See also: ``architecture/docs/auth/sdk-cli-oauth.md`` for a full design doc.
"""

import asyncio
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol

import httpx
from nemo_platform_plugin.client.auth import TokenProviderAuth
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_platform_plugin.client.types import RetryPolicy

from nemo_platform_ext.auth.helpers import NMPOIDCConfig, build_effective_scope, discover_nmp_config
from nemo_platform_ext.auth.token_provider import (
    OIDCTokenProvider,
    TokenSet,
)
from nemo_platform_ext.auth.workload_exchange import WorkloadTokenExchangeProvider
from nemo_platform_ext.client.tls import client_verify_from_env

logger = logging.getLogger(__name__)

# Refresh the access token when fewer than 60 s remain before expiry.
# This gives enough headroom for the refresh HTTP round-trip to complete
# before the token actually expires.
_TOKEN_REFRESH_MARGIN_SECONDS = 60

# Guards _TOKEN_PROVIDER_CACHE; acquired only during dict lookup/insert (fast).
_TOKEN_PROVIDER_CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class AccessTokenProvider(Protocol):
    def get_access_token(self) -> str: ...

    async def get_access_token_async(self) -> str: ...


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
class _ProviderCacheKey:
    """Composite key for the provider cache.

    Two clients share the same provider iff they read from the same config
    file, same context, OIDC settings (endpoint, client, scope), and CA match.
    """

    config_path: Path
    context_name: str
    token_endpoint: str
    client_id: str
    refresh_scope: str | None
    certificate_authority: str | None


# Process-wide cache: (config_path, context, OIDC settings, CA) → shared OIDCTokenProvider.
# This avoids redundant refresh-token grants when the user creates multiple
# NeMoPlatform() instances pointing at the same context.
_TOKEN_PROVIDER_CACHE: dict[_ProviderCacheKey, OIDCTokenProvider] = {}


# ---------------------------------------------------------------------------
# OIDC discovery
# ---------------------------------------------------------------------------

# Fallback returned when NMP configdiscovery fails.
_OIDC_DISCOVERY_FALLBACK = NMPOIDCConfig(
    auth_enabled=False,
    client_id="",
    token_endpoint="",
    default_scopes="openid profile email",
    scope_prefix=None,
)


def _discover_oidc_client_settings(base_url: str, certificate_authority: str | None = None) -> NMPOIDCConfig:
    """Fetch OIDC config from the NeMo Platform cluster's discovery endpoint.

    Returns a safe fallback (auth_enabled=False) if the cluster is
    unreachable or doesn't have OIDC configured.  This lets non-OIDC
    clusters work without errors during client construction.
    """
    try:
        return discover_nmp_config(base_url, certificate_authority=certificate_authority)
    except Exception:
        logger.debug("Could not discover OIDC settings from %s", base_url, exc_info=True)
        return _OIDC_DISCOVERY_FALLBACK


def _workload_identity_token_file_from_env() -> Path | None:
    """Return the configured workload identity token file, if workload bootstrap is active."""
    token_file = os.environ.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR)
    return Path(token_file) if token_file else None


def _create_workload_exchange_provider(
    base_url: str,
    subject_token_file: Path,
    *,
    certificate_authority: str | None = None,
) -> WorkloadTokenExchangeProvider:
    """Create a workload identity token exchange provider from NeMo auth discovery metadata."""
    oidc_config = _discover_oidc_client_settings(base_url, certificate_authority=certificate_authority)
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


class _LazyWorkloadTokenExchangeProvider:
    """Create the workload exchange provider on the first token request."""

    def __init__(
        self,
        *,
        base_url: str,
        subject_token_file: Path,
        certificate_authority: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._subject_token_file = subject_token_file
        self._certificate_authority = certificate_authority
        self._provider: WorkloadTokenExchangeProvider | None = None
        self._lock = threading.Lock()

    def _get_provider(self) -> WorkloadTokenExchangeProvider:
        provider = self._provider
        if provider is not None:
            return provider
        with self._lock:
            provider = self._provider
            if provider is None:
                provider = _create_workload_exchange_provider(
                    self._base_url,
                    self._subject_token_file,
                    certificate_authority=self._certificate_authority,
                )
                self._provider = provider
            return provider

    def get_cached_access_token(self) -> str | None:
        provider = self._provider
        if provider is None:
            return None
        tokens = provider.tokens
        if not tokens.access_token or tokens.is_expired(provider.refresh_margin_seconds):
            return None
        return tokens.access_token

    def get_access_token(self) -> str:
        return self._get_provider().get_access_token()

    async def get_access_token_async(self) -> str:
        provider = self._provider
        if provider is None:
            provider = await asyncio.to_thread(self._get_provider)
        return await provider.get_access_token_async()


# ---------------------------------------------------------------------------
# httpx event hooks — the core of transparent token injection
# ---------------------------------------------------------------------------


def _make_auth_event_hook(provider: AccessTokenProvider):
    """Create a **sync** httpx request event hook that injects the Bearer token.

    Called before every SDK HTTP request.  ``provider.get_access_token()``
    returns the cached token if still valid, or performs an inline
    refresh_token grant if the token is expired/about-to-expire.
    """

    def inject_auth(request: httpx.Request) -> None:
        token = provider.get_access_token()
        request.headers["Authorization"] = f"Bearer {token}"

    return inject_auth


def _make_async_auth_event_hook(provider: AccessTokenProvider):
    """Create an **async** httpx request event hook for AsyncNeMoPlatform.

    The actual refresh still runs in a worker thread (via
    ``provider.get_access_token_async``) so it doesn't block the event loop.
    """

    async def inject_auth(request: httpx.Request) -> None:
        token = await provider.get_access_token_async()
        request.headers["Authorization"] = f"Bearer {token}"

    return inject_auth


def _headers_with_seeded_auth(headers: Mapping[str, str], provider: AccessTokenProvider) -> dict[str, str]:
    seeded_headers = dict(headers)
    if isinstance(provider, _LazyWorkloadTokenExchangeProvider):
        token = provider.get_cached_access_token()
    else:
        token = provider.get_access_token()
    if token:
        seeded_headers["Authorization"] = f"Bearer {token}"
    return seeded_headers


# ---------------------------------------------------------------------------
# Callbacks wired into OIDCTokenProvider for config-file integration
# ---------------------------------------------------------------------------


def _make_config_persister(context_name: str, config_path: Path | None = None):
    """Create an ``on_tokens_refreshed`` callback that writes new tokens to the
    nmp config file.

    After a successful refresh, the provider calls this so that the CLI and
    other SDK processes pick up the rotated tokens without re-authenticating.
    """
    from nemo_platform_ext.config.config import Config, ConfigParams

    def persist(tokens: TokenSet) -> None:
        params: ConfigParams = {"access_token": tokens.access_token}
        if tokens.refresh_token:
            params["refresh_token"] = tokens.refresh_token
        Config.write(params, context_name=context_name, config_path=config_path)
        logger.debug("Persisted refreshed tokens to nmp config (context=%s)", context_name)

    return persist


def _make_config_token_loader(context_name: str, config_path: Path):
    """Create a ``load_tokens`` callback that re-reads tokens from the config file.

    This is the recovery mechanism for the ``invalid_grant`` scenario:
    when another process already rotated the refresh token, our local
    copy is stale.  The provider calls this to reload whatever that
    other process wrote, then retries the refresh with the fresh token.
    """
    from nemo_platform_ext.config.config import Config, ConfigParams
    from nemo_platform_ext.config.models import OAuthUser

    def load_tokens() -> TokenSet | None:
        overrides: ConfigParams = {"current_context": context_name}
        try:
            config = Config.load(config_path=config_path, overrides=overrides)
            resolved = config.resolve()
        except Exception:
            logger.debug("Failed to reload tokens from nmp config (context=%s)", context_name, exc_info=True)
            return None

        if not isinstance(resolved.user, OAuthUser):
            return None

        return TokenSet.from_access_token(
            resolved.user.token.get_secret_value(),
            resolved.user.refresh_token.get_secret_value() if resolved.user.refresh_token else None,
        )

    return load_tokens


# ---------------------------------------------------------------------------
# Cross-process file lock for refresh serialization
# ---------------------------------------------------------------------------


def _build_refresh_lock_path(config_path: Path, context_name: str) -> Path:
    """Derive the lock-file path from the config path and context name.

    Example: ``~/.config/nmp/config.yaml.default.oauth-refresh.lock``
    """
    safe_context = context_name.replace(os.sep, "_")
    if os.altsep:
        safe_context = safe_context.replace(os.altsep, "_")
    return config_path.with_name(f"{config_path.name}.{safe_context}.oauth-refresh.lock")


def _make_refresh_lock(config_path: Path, context_name: str):
    """Create a ``refresh_lock`` context-manager factory for OIDCTokenProvider.

    Uses ``fcntl.flock`` (POSIX) to serialize refresh transactions across
    processes.  On platforms without fcntl (Windows), falls back to a no-op
    so the provider still works — just without cross-process serialization.
    """
    lock_path = _build_refresh_lock_path(config_path, context_name)

    @contextmanager
    def refresh_lock():
        try:
            import fcntl
        except ImportError:
            # Windows: no fcntl — skip cross-process locking.
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
    """Canonicalize the config path so that cache keys match regardless of
    how the caller spelled the path (``~/…`` vs absolute)."""
    return path.expanduser().resolve(strict=False)


# ---------------------------------------------------------------------------
# Provider cache
# ---------------------------------------------------------------------------


def _get_or_create_provider(
    key: _ProviderCacheKey,
    create_provider: Callable[[], OIDCTokenProvider],
) -> OIDCTokenProvider:
    """Return the cached provider for *key*, or create and cache a new one.

    If a cached provider already exists, we call ``reload_tokens()`` so it
    picks up any tokens that another process may have written to the config
    file since the provider was last used.
    """
    with _TOKEN_PROVIDER_CACHE_LOCK:
        provider = _TOKEN_PROVIDER_CACHE.get(key)
        if provider is None:
            provider = create_provider()
            _TOKEN_PROVIDER_CACHE[key] = provider
            return provider

    # Provider already existed — reload tokens from disk in case another
    # process refreshed them since we last checked.
    provider.reload_tokens()
    return provider


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _resolve_client_context(
    *,
    config_path: Path | None,
    base_url: str | httpx.URL | None,
    context_name: str | None,
    access_token: str | None,
) -> tuple[Any, bool, Path]:
    """Load the nmp config file, apply overrides, and resolve the active context.

    Returns ``(resolved_context, config_exists, config_path)`` so the caller
    knows whether to enable config-backed features (provider caching,
    token persistence, file locking).
    """
    from nemo_platform_ext.config.config import Config, ConfigParams

    resolved_config_path = config_path or Config.get_default_config_path()
    config_exists = resolved_config_path.exists()
    if config_exists:
        logger.info("Reading nmp config from %s", resolved_config_path)

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


def resolve_bootstrap(
    *,
    config_path: Path | None,
    base_url: str | httpx.URL | None,
    context_name: str | None,
    access_token: str | None,
    extra_headers: Mapping[str, str] | None,
) -> ResolvedBootstrap:
    """Resolve the full client bootstrap: config, OIDC discovery, token provider.

    For **non-OAuth users** (no-auth): returns a bootstrap with
    ``token_provider=None`` and static headers (e.g. ``Authorization: Bearer <api-key>``).

    For **OAuth users**: discovers the cluster's OIDC settings, then either
    reuses a cached OIDCTokenProvider or creates a new one.  The provider is
    wired with:

    - ``load_tokens``: re-reads tokens from the config file (for ``invalid_grant`` recovery)
    - ``refresh_lock``: cross-process fcntl lock to serialize refreshes
    - ``on_tokens_refreshed``: writes rotated tokens back to the config file
    """
    from nemo_platform_ext.config.models import OAuthUser

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

    workload_identity_token_file = _workload_identity_token_file_from_env()
    if workload_identity_token_file is not None and access_token is None and not os.environ.get("NMP_ACCESS_TOKEN"):
        provider = _LazyWorkloadTokenExchangeProvider(
            base_url=base_url,
            subject_token_file=workload_identity_token_file,
            certificate_authority=certificate_authority,
        )
        return ResolvedBootstrap(base_url, resolved.workspace, headers, provider, client_verify, certificate_authority)

    # --- Non-OAuth path (no auth) ---
    if not isinstance(resolved.user, OAuthUser):
        user_config = resolved.user.get_client_config() if resolved.user else {}
        user_headers = user_config.get("default_headers", {})
        if isinstance(user_headers, dict):
            headers.update(user_headers)
        return ResolvedBootstrap(base_url, resolved.workspace, headers, None, client_verify, certificate_authority)

    # --- OAuth path: set up transparent token refresh ---
    try:
        oidc_config = discover_nmp_config(base_url, certificate_authority=certificate_authority)
        if not oidc_config.auth_enabled and access_token is None:
            # Discovery succeeded and confirmed the cluster has no OIDC. The
            # stored OAuthUser token can't be refreshed here (no endpoint), so
            # fall back to no-auth. This guard only fires on a successful
            # discovery response — not on discovery failures, where the stored
            # token may still be valid and should be used as-is.
            return ResolvedBootstrap(base_url, resolved.workspace, headers, None, client_verify, certificate_authority)
    except Exception:
        logger.debug("Could not discover OIDC settings from %s", base_url, exc_info=True)
        oidc_config = _OIDC_DISCOVERY_FALLBACK

    tokens = TokenSet.from_access_token(
        resolved.user.token.get_secret_value(),
        resolved.user.refresh_token.get_secret_value() if resolved.user.refresh_token else None,
    )

    token_endpoint = oidc_config.token_endpoint or ""
    client_id = oidc_config.client_id or ""
    refresh_scope = build_effective_scope(oidc_config.default_scopes, oidc_config.scope_prefix)

    # Only share the provider (and enable persistence/locking) when reading
    # from an actual config file.  If the caller passed an explicit
    # access_token, they own the token lifecycle — don't cache or persist.
    share_provider = config_exists and access_token is None

    if share_provider:
        normalized_config_path = _normalize_config_path(resolved_config_path)
        provider_key = _ProviderCacheKey(
            config_path=normalized_config_path,
            context_name=resolved.context_name,
            token_endpoint=token_endpoint,
            client_id=client_id,
            refresh_scope=refresh_scope,
            certificate_authority=certificate_authority,
        )
        on_refreshed = _make_config_persister(resolved.context_name, resolved_config_path)
        load_tokens = _make_config_token_loader(resolved.context_name, resolved_config_path)
        refresh_lock = _make_refresh_lock(resolved_config_path, resolved.context_name)

        provider = _get_or_create_provider(
            provider_key,
            lambda: OIDCTokenProvider(
                token_endpoint=token_endpoint,
                client_id=client_id,
                tokens=tokens,
                refresh_margin_seconds=_TOKEN_REFRESH_MARGIN_SECONDS,
                refresh_scope=refresh_scope,
                certificate_authority=certificate_authority,
                load_tokens=load_tokens,
                refresh_lock=refresh_lock,
                on_tokens_refreshed=on_refreshed,
            ),
        )
    else:
        # Ephemeral provider: no persistence, no file locking, no caching.
        provider = OIDCTokenProvider(
            token_endpoint=token_endpoint,
            client_id=client_id,
            tokens=tokens,
            refresh_margin_seconds=_TOKEN_REFRESH_MARGIN_SECONDS,
            refresh_scope=refresh_scope,
            certificate_authority=certificate_authority,
        )

    return ResolvedBootstrap(base_url, resolved.workspace, headers, provider, client_verify, certificate_authority)


# ---------------------------------------------------------------------------
# Typed client construction
# ---------------------------------------------------------------------------

# Matches the retry behaviour the generated SDK applied by default, so the CLI
# keeps the same resilience against transient gateway errors.
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    retryable_status_codes=(408, 409, 429),
    retry_all_server_errors=True,
    respect_retry_decision_headers=True,
    respect_retry_after_headers=True,
)


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
    here; the transport auth handles raw ``_client`` calls too.
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
    """Build a sync :class:`NemoClient` from the nmp config and auth bootstrap.

    For OAuth and workload-identity users the resulting client refreshes and
    injects the Bearer token before every request; API-key users get static
    headers. *workspace* overrides the configured default when given.
    """
    bootstrap = resolve_bootstrap(
        config_path=config_path,
        base_url=base_url,
        context_name=context_name,
        access_token=access_token,
        extra_headers=extra_headers,
    )
    resolved_timeout = resolve_timeout(timeout)
    http_client = httpx.Client(
        headers=_client_headers(bootstrap) or None,
        timeout=resolved_timeout,
        follow_redirects=True,
        verify=bootstrap.client_verify,
        auth=TokenProviderAuth(bootstrap.token_provider) if bootstrap.token_provider is not None else None,
    )
    return NemoClient(
        base_url=bootstrap.base_url,
        workspace=workspace if workspace is not None else bootstrap.workspace,
        auth=bootstrap.token_provider,
        default_headers=_client_headers(bootstrap) or None,
        timeout=resolved_timeout,
        retry=retry,
        http_client=http_client,
    )


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
    bootstrap = resolve_bootstrap(
        config_path=config_path,
        base_url=base_url,
        context_name=context_name,
        access_token=access_token,
        extra_headers=extra_headers,
    )
    resolved_timeout = resolve_timeout(timeout)
    http_client = httpx.AsyncClient(
        headers=_client_headers(bootstrap) or None,
        timeout=resolved_timeout,
        follow_redirects=True,
        verify=bootstrap.client_verify,
        auth=TokenProviderAuth(bootstrap.token_provider) if bootstrap.token_provider is not None else None,
    )
    return AsyncNemoClient(
        base_url=bootstrap.base_url,
        workspace=workspace if workspace is not None else bootstrap.workspace,
        auth=bootstrap.token_provider,
        default_headers=_client_headers(bootstrap) or None,
        timeout=resolved_timeout,
        retry=retry,
        http_client=http_client,
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
    """Build a sync :class:`NemoClient` without reading the nmp config.

    Direct mode: no config file is consulted and only *default_headers* are
    sent. TLS verification still honours the environment override and any
    saved cluster certificate authority.
    """
    resolved_timeout = resolve_timeout(timeout)
    http_client = httpx.Client(
        headers=dict(default_headers) if default_headers else None,
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
        headers=dict(default_headers) if default_headers else None,
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
    )
