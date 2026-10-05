# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared typed platform client construction for analyst read methods.

Auth lives in the active ``nemo auth login`` context in
``~/.config/nhx/config.yaml``. The config bootstrap wires up that context (and
the transparent OIDC token refresh that comes with it). A direct client built
from ``base_url`` alone skips the bootstrap and injects **no** auth headers,
which is fine for an unauthenticated local ``nemo services run`` but 401s
against a remote deployment. To authenticate against a remote URL we run the
bootstrap so the explicit ``base_url`` is combined with the context's
credentials.

The analyst run takes ``base_url`` from its CLI/job context, so this helper is
the one place that branch lives.
"""

from urllib.parse import urlparse

import httpx
from nemo_helix_ext.auth.helpers import discover_nhx_config
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
    - Authenticated remote ``base_url`` with an nhx config present: combine the
      URL with the context's auth so the client injects and refreshes a Bearer token.
    - Unauthenticated remote ``base_url``: direct mode, even when an unrelated
      OAuth context exists locally.
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

    try:
        nhx_config = discover_nhx_config(base_url)
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"could not discover NeMo Helix auth configuration at {base_url}: {exc}") from exc

    if not nhx_config.auth_enabled:
        return build_direct_async_nemo_client(base_url=base_url)

    if parsed.scheme.lower() != "https":
        raise ValueError(f"refusing to send credentials to a non-HTTPS remote URL: {base_url}")

    return build_async_nemo_client(base_url=base_url, config_path=config_path)
