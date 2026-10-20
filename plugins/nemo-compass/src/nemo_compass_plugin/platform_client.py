# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared typed platform client construction for analyst read methods."""

from urllib.parse import urlparse

from nemo_helix_ext.client.bootstrap import build_async_nemo_client, build_direct_async_nemo_client
from nemo_helix_ext.config.config import Config
from nemo_helix_plugin.client.client import AsyncNemoClient

# Loopback hosts are served by an unauthenticated local platform; attaching
# (and refreshing) OAuth tokens there is both unnecessary and a failure mode
# when the cached token is stale and OIDC discovery against localhost fails.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def make_client(base_url: str | None) -> AsyncNemoClient:
    """Construct an :class:`AsyncNemoClient` honoring an optional ``base_url``.

    - No ``base_url``: use the active nhx context for both URL and auth.
    - Loopback ``base_url``: direct mode (local platform is unauthenticated).
    - HTTPS remote ``base_url`` with an nhx config present: combine the URL
      with the context's auth/bootstrap state.
    - Remote ``base_url`` without an nhx config: direct mode (no credentials to
      use; the request will surface a clear auth error).
    """
    if not base_url:
        return build_async_nemo_client()

    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    config_path = Config.get_default_config_path()
    if host in LOOPBACK_HOSTS or not config_path.exists():
        return build_direct_async_nemo_client(base_url=base_url)

    if parsed.scheme.lower() != "https":
        raise ValueError(f"refusing to send credentials to a non-HTTPS remote URL: {base_url}")

    return build_async_nemo_client(base_url=base_url, config_path=config_path)
