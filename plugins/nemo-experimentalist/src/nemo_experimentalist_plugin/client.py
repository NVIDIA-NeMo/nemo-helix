# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared NeMo Platform client construction for Insight consumers.

Auth lives in the active ``nemo auth login`` context in
``~/.config/nmp/config.yaml``. Passing ``base_url`` alone puts the typed client in
"direct mode", which injects no auth headers -- fine for an unauthenticated local
``nemo services run``, but it 401s against a remote deployment. To authenticate
against a remote URL we combine the explicit URL with the context's credentials.
"""

from pathlib import Path
from urllib.parse import urlparse

from nemo_platform_plugin.client.auth import AsyncTokenProvider, TokenProvider
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.config.config import Config
from nemo_platform_plugin.client.config.models import ConfigParams, OAuthUser
from nemo_platform_plugin.client.oidc import discover_nmp_config
from nemo_platform_plugin.client.oidc_factory import resolve_oidc_provider

# Loopback hosts are served by an unauthenticated local platform; attaching
# (and refreshing) OAuth tokens there is both unnecessary and a failure mode
# when the cached token is stale and OIDC discovery against localhost fails.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _is_loopback_url(base_url: str) -> bool:
    return (urlparse(base_url).hostname or "").lower() in LOOPBACK_HOSTS


def _require_secure_authenticated_remote(base_url: str) -> None:
    parsed = urlparse(base_url)
    if _is_loopback_url(base_url):
        return
    if parsed.scheme != "https":
        raise ValueError("Authenticated remote NeMo Platform URLs must use HTTPS")


def _client_from_config_with_base_url(base_url: str, config_path: Path) -> AsyncNemoClient:
    _require_secure_authenticated_remote(base_url)
    overrides: ConfigParams = {"base_url": base_url}
    config = Config.load(config_path=config_path, overrides=overrides)
    actual_config_path = config.get_config_path() or Config.get_default_config_path()
    ctx = config.resolve()
    auth: TokenProvider | AsyncTokenProvider | str | None = None
    if isinstance(ctx.user, OAuthUser):
        auth = resolve_oidc_provider(
            base_url=base_url,
            context_name=ctx.context_name,
            access_token=ctx.user.token.get_secret_value(),
            refresh_token=ctx.user.refresh_token.get_secret_value() if ctx.user.refresh_token else None,
            config_exists=actual_config_path.exists(),
            config_path=actual_config_path,
            explicit_access_token=config.access_token is not None,
        )
    return AsyncNemoClient(base_url=base_url, workspace=ctx.workspace, auth=auth)


def make_client(base_url: str | None) -> AsyncNemoClient:
    """Construct an :class:`AsyncNemoClient` honoring an optional ``base_url``.

    - No ``base_url``: use the active nmp context for both URL and auth.
    - Loopback ``base_url``: direct mode (local platform is unauthenticated).
    - Authenticated remote ``base_url`` with an nmp config present: combine the
      URL with the context's auth so the client injects and refreshes a Bearer token.
    - Unauthenticated remote ``base_url``: direct mode, even when an unrelated
      OAuth context exists locally.
    - Remote ``base_url`` without an nmp config: direct mode (no credentials to
      use; the request will surface a clear auth error).
    """
    if not base_url:
        return AsyncNemoClient.from_config()

    config_path = Config.get_default_config_path()
    if _is_loopback_url(base_url) or not config_path.exists():
        return AsyncNemoClient(base_url=base_url)

    if not discover_nmp_config(base_url).auth_enabled:
        return AsyncNemoClient(base_url=base_url)

    return _client_from_config_with_base_url(base_url, config_path)
