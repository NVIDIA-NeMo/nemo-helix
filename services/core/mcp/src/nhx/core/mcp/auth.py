# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bearer-token authentication for the MCP streamable-http transport.

The HTTP transport exposes tools that act on the platform with the server's own
credentials, so it must not accept anonymous callers. A single shared bearer
token, read from ``NHX_MCP_AUTH_TOKEN``, gates every request. The stdio
transport is a local pipe to the parent process and is not affected.
"""

from __future__ import annotations

import hmac
import ipaddress
import os

from fastmcp.server.auth import AccessToken, TokenVerifier

AUTH_TOKEN_ENV_VAR = "NHX_MCP_AUTH_TOKEN"
MIN_AUTH_TOKEN_LENGTH = 16


class MCPAuthConfigError(ValueError):
    """The requested transport/auth combination is not allowed."""


class SharedBearerTokenVerifier(TokenVerifier):
    """Accept requests that present one preconfigured bearer token."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token.encode()

    async def verify_token(self, token: str) -> AccessToken | None:
        if not hmac.compare_digest(token.encode(), self._token):
            return None
        return AccessToken(token=token, client_id="nemo-mcp-shared-token", scopes=[])


def is_loopback_host(host: str) -> bool:
    """Return True for ``localhost`` and loopback IP literals; other hostnames are not trusted."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_http_auth(host: str, *, allow_unauthenticated: bool) -> SharedBearerTokenVerifier | None:
    """Pick the auth provider for the HTTP transport, or refuse the combination.

    Returns a verifier when ``NHX_MCP_AUTH_TOKEN`` is set. Returns ``None`` only for
    an explicit ``allow_unauthenticated`` opt-in on a loopback host.
    """
    token = os.environ.get(AUTH_TOKEN_ENV_VAR, "")
    if token:
        if token != token.strip():
            raise MCPAuthConfigError(f"{AUTH_TOKEN_ENV_VAR} must not start or end with whitespace")
        if len(token) < MIN_AUTH_TOKEN_LENGTH:
            raise MCPAuthConfigError(f"{AUTH_TOKEN_ENV_VAR} must be at least {MIN_AUTH_TOKEN_LENGTH} characters")
        return SharedBearerTokenVerifier(token)
    if not allow_unauthenticated:
        raise MCPAuthConfigError(
            f"The HTTP transport requires authentication: set {AUTH_TOKEN_ENV_VAR}, "
            "or pass --allow-unauthenticated for local debugging on a loopback host."
        )
    if not is_loopback_host(host):
        raise MCPAuthConfigError(f"--allow-unauthenticated is only permitted on a loopback host, not {host!r}.")
    return None
