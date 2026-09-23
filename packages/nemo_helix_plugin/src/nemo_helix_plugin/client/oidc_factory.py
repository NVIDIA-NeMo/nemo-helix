# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OIDC provider factory for NemoClient.from_config().

Manages provider caching, config-file token persistence, and cross-process
locking so that multiple ``NemoClient`` instances (and processes) sharing the
same config file coordinate token refreshes correctly.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

from nemo_helix_plugin.client.auth import AuthError
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.client.oidc import (
    DEFAULT_REFRESH_MARGIN_SECONDS,
    OIDCTokenProvider,
    TokenSet,
    WorkloadTokenExchangeProvider,
    _discover_oidc_client_settings,
    build_effective_scope,
)

logger = logging.getLogger(__name__)

# Guards _TOKEN_PROVIDER_CACHE; acquired only during dict lookup/insert (fast).
_TOKEN_PROVIDER_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _ProviderCacheKey:
    """Composite key for the provider cache."""

    config_path: Path
    context_name: str
    token_endpoint: str
    client_id: str
    refresh_scope: str | None


# Process-wide cache: (config_path, context) → shared OIDCTokenProvider.
_TOKEN_PROVIDER_CACHE: dict[_ProviderCacheKey, OIDCTokenProvider] = {}


def _make_config_persister(context_name: str, config_path: Path | None = None) -> Callable[[TokenSet], None]:
    """Create an ``on_tokens_refreshed`` callback that writes new tokens to the config file."""

    def persist(tokens: TokenSet) -> None:
        from nemo_helix_plugin.client.config.config import Config
        from nemo_helix_plugin.client.config.models import ConfigParams

        params: ConfigParams = {"access_token": tokens.access_token}
        if tokens.refresh_token:
            params["refresh_token"] = tokens.refresh_token
        Config.write(params, context_name=context_name, config_path=config_path)
        logger.debug("Persisted refreshed tokens to nhx config (context=%s)", context_name)

    return persist


def _make_config_token_loader(context_name: str, config_path: Path) -> Callable[[], TokenSet | None]:
    """Create a ``load_tokens`` callback that re-reads tokens from the config file."""

    def load_tokens() -> TokenSet | None:
        from nemo_helix_plugin.client.config.config import Config
        from nemo_helix_plugin.client.config.models import ConfigParams, OAuthUser

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
        )

    return load_tokens


def _build_refresh_lock_path(config_path: Path, context_name: str) -> Path:
    safe_context = context_name.replace(os.sep, "_")
    if os.altsep:
        safe_context = safe_context.replace(os.altsep, "_")
    return config_path.with_name(f"{config_path.name}.{safe_context}.oauth-refresh.lock")


def _make_refresh_lock(config_path: Path, context_name: str) -> Callable[[], AbstractContextManager[None]]:
    """Create a cross-process file lock for serializing token refreshes."""
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

    # Provider already existed — reload tokens from disk in case another
    # process refreshed them since we last checked.
    provider.reload_tokens()
    return provider


def resolve_oidc_provider(
    *,
    base_url: str,
    context_name: str,
    access_token: str,
    refresh_token: str | None,
    config_exists: bool,
    config_path: Path,
    explicit_access_token: bool = False,
) -> OIDCTokenProvider:
    """Build or retrieve a cached OIDCTokenProvider for a resolved config context.

    This is the bridge between ``NemoClient.from_config()`` and the OIDC machinery.
    """
    oidc_config = _discover_oidc_client_settings(base_url)
    tokens = TokenSet.from_access_token(access_token, refresh_token)

    token_endpoint = oidc_config.token_endpoint or ""
    client_id = oidc_config.client_id or ""
    refresh_scope = build_effective_scope(oidc_config.default_scopes, oidc_config.scope_prefix)

    if refresh_token and (not token_endpoint or not client_id):
        raise AuthError(
            "OIDC discovery did not return token_endpoint/client_id; "
            "cannot refresh OAuth tokens. Check cluster auth configuration."
        )

    # Only share the provider (and enable persistence/locking) when reading
    # from an actual config file.  If the caller passed an explicit
    # access_token, they own the token lifecycle.
    share_provider = config_exists and not explicit_access_token

    if share_provider:
        normalized_config_path = _normalize_config_path(config_path)
        provider_key = _ProviderCacheKey(
            config_path=normalized_config_path,
            context_name=context_name,
            token_endpoint=token_endpoint,
            client_id=client_id,
            refresh_scope=refresh_scope,
        )
        on_refreshed = _make_config_persister(context_name, config_path)
        load_tokens_cb = _make_config_token_loader(context_name, config_path)
        refresh_lock = _make_refresh_lock(config_path, context_name)

        return _get_or_create_provider(
            provider_key,
            lambda: OIDCTokenProvider(
                token_endpoint=token_endpoint,
                client_id=client_id,
                tokens=tokens,
                refresh_margin_seconds=DEFAULT_REFRESH_MARGIN_SECONDS,
                refresh_scope=refresh_scope,
                load_tokens=load_tokens_cb,
                refresh_lock=refresh_lock,
                on_tokens_refreshed=on_refreshed,
            ),
        )

    # Ephemeral provider: no persistence, no file locking, no caching.
    return OIDCTokenProvider(
        token_endpoint=token_endpoint,
        client_id=client_id,
        tokens=tokens,
        refresh_margin_seconds=DEFAULT_REFRESH_MARGIN_SECONDS,
        refresh_scope=refresh_scope,
    )


def resolve_workload_exchange_provider(*, base_url: str, subject_token_file: Path) -> WorkloadTokenExchangeProvider:
    """Build a workload identity token exchange provider from auth discovery."""
    oidc_config = _discover_oidc_client_settings(base_url)
    if not oidc_config.workload_token_exchange_enabled:
        raise AuthError(
            f"{WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR} is set but workload token exchange is not enabled by auth discovery"
        )

    token_endpoint = oidc_config.workload_token_endpoint or oidc_config.token_endpoint or ""
    client_id = oidc_config.workload_client_id or oidc_config.client_id or ""
    if not token_endpoint:
        raise AuthError(
            "Workload token exchange is enabled but auth discovery did not return workload_token_endpoint or token_endpoint"
        )
    if not client_id:
        raise AuthError(
            "Workload token exchange is enabled but auth discovery did not return workload_client_id or client_id"
        )

    return WorkloadTokenExchangeProvider(
        token_endpoint=token_endpoint,
        client_id=client_id,
        subject_token_file=subject_token_file,
        audience=oidc_config.workload_audience,
        scope=oidc_config.workload_scope,
        refresh_margin_seconds=DEFAULT_REFRESH_MARGIN_SECONDS,
    )
