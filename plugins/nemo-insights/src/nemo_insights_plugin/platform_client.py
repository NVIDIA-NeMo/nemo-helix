# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared NeMo Helix SDK client construction for analyst read methods.

Auth lives in the active ``nemo auth login`` context in
``~/.config/nhx/config.yaml``. The SDK only wires up that context (and the
transparent OIDC token refresh that comes with it) when it runs its config
bootstrap. Passing ``base_url`` *alone* puts the SDK in "direct mode", which
skips the bootstrap and injects **no** auth headers — fine for an
unauthenticated local ``nemo services run``, but it 401s against a remote
deployment. To authenticate against a remote URL we must trigger the bootstrap
(by also passing ``config_path``) so the explicit ``base_url`` is combined with
the context's credentials.

The analyst run takes ``base_url`` from its CLI/job context, so this helper is
the one place that branch lives.
"""

from urllib.parse import urlparse

import httpx
from nemo_helix import AsyncNeMoHelix
from nemo_helix_ext.auth.helpers import discover_nhx_config
from nemo_helix_ext.config.config import Config

# Loopback hosts are served by an unauthenticated local platform; attaching
# (and refreshing) OAuth tokens there is both unnecessary and a failure mode
# when the cached token is stale and OIDC discovery against localhost fails.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def make_client(base_url: str | None) -> AsyncNeMoHelix:
    """Construct an :class:`AsyncNeMoHelix` honoring an optional ``base_url``.

    - No ``base_url``: use the active nhx context for both URL and auth.
    - Loopback ``base_url``: direct mode (local platform is unauthenticated).
    - Authenticated remote ``base_url`` with an nhx config present: combine the
      URL with the context's auth so the SDK injects and refreshes a Bearer token.
    - Unauthenticated remote ``base_url``: direct mode, even when an unrelated
      OAuth context exists locally.
    - Remote ``base_url`` without an nhx config: direct mode (no credentials to
      use; the request will surface a clear auth error).
    """
    if not base_url:
        return AsyncNeMoHelix()

    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    config_path = Config.get_default_config_path()
    if host in LOOPBACK_HOSTS or not config_path.exists():
        return AsyncNeMoHelix(base_url=base_url)

    try:
        nhx_config = discover_nhx_config(base_url)
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"could not discover NeMo Helix auth configuration at {base_url}: {exc}") from exc

    if not nhx_config.auth_enabled:
        return AsyncNeMoHelix(base_url=base_url)

    if parsed.scheme.lower() != "https":
        raise ValueError(f"refusing to send credentials to a non-HTTPS remote URL: {base_url}")

    return AsyncNeMoHelix(base_url=base_url, config_path=config_path)
