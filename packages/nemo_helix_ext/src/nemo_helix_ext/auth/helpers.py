# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this code except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auth helpers for NeMo Helix CLI (scope normalization, JWT decode, scope validation)."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
from nemo_helix_plugin.authz_format import is_valid_nhx_scope_id
from nemo_helix_plugin.client.oidc import (
    AdvertisedOidcClient as AdvertisedOidcClient,
)
from nemo_helix_plugin.client.oidc import (
    BearerTokenSource as BearerTokenSource,
)
from nemo_helix_plugin.client.oidc import (
    NHXOIDCConfig as NHXOIDCConfig,
)
from nemo_helix_plugin.client.oidc import (
    parse_bearer_token_source as parse_bearer_token_source,
)
from nemo_helix_plugin.client.oidc import (
    parse_nhx_config,
)
from nemo_helix_plugin.client.oidc import (
    refresh_target as refresh_target,
)
from nemo_helix_plugin.client.oidc import (
    select_advertised_client as select_advertised_client,
)


class AuthError(Exception):
    """Authentication-related error (CLI auth commands)."""

    pass


def normalize_scope_prefix(prefix: str | None) -> str:
    """Normalize scope prefix to ensure it ends with a separator.

    Azure AD scope URIs require a '/' between the app ID and scope name.
    This handles misconfigured clusters that omit the trailing slash.

    Args:
        prefix: The scope prefix from cluster configuration (may be None or empty)

    Returns:
        Empty string if prefix is None/empty, otherwise prefix with trailing '/'
    """
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else f"{prefix}/"


def scope_short(scope: str, scope_prefix: str) -> str:
    """Return scope in short form for comparison (strip prefix if present)."""
    if scope_prefix and scope.startswith(scope_prefix):
        return scope[len(scope_prefix) :]
    return scope


def _decode_jwt_segment(token: str, index: int) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        payload = parts[index]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def decode_jwt_header(token: str) -> dict[str, Any]:
    """Decode JWT header without verification."""
    return _decode_jwt_segment(token, 0)


def decode_jwt_claims(token: str) -> dict[str, Any]:
    """Decode JWT claims without verification (for display purposes only)."""
    return _decode_jwt_segment(token, 1)


def is_unsigned_jwt(token: str) -> bool:
    """Return True when JWT uses ``alg=none``."""
    header = decode_jwt_header(token)
    return str(header.get("alg", "")).lower() == "none"


def _base64url_encode_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")


def generate_unsigned_jwt(
    principal_id: str,
    *,
    email: str | None = None,
    groups: list[str] | None = None,
    scopes: list[str] | None = None,
    expires_in_seconds: int | None = 3600,
    issued_at: int | None = None,
    audience: str | None = None,
    issuer: str | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Generate an unsigned JWT (`alg=none`) for local development and testing."""
    now = issued_at if issued_at is not None else int(time.time())
    claims: dict[str, Any] = {
        "sub": principal_id,
        "iat": now,
    }

    if email:
        claims["email"] = email
    if groups:
        claims["groups"] = groups
    if scopes:
        claims["scope"] = " ".join(scopes)
    if expires_in_seconds is not None:
        claims["exp"] = now + expires_in_seconds
    if audience:
        claims["aud"] = audience
    if issuer:
        claims["iss"] = issuer
    if extra_claims:
        claims.update(extra_claims)

    header_segment = _base64url_encode_json({"alg": "none", "typ": "JWT"})
    claims_segment = _base64url_encode_json(claims)
    return f"{header_segment}.{claims_segment}."


def discover_nhx_config(
    base_url: str,
    http_client: httpx.Client,
    timeout: float = 10.0,
) -> NHXOIDCConfig:
    """Fetch OIDC configuration from the NeMo Helix auth discovery endpoint."""
    url = f"{base_url.rstrip('/')}/apis/auth/discovery"
    # TLS verification belongs to the caller-owned httpx.Client; do not override it here.
    # codeql[py/request-without-cert-validation]
    response = http_client.get(url, timeout=timeout)
    response.raise_for_status()
    return _parse_nhx_config(response.json())


async def discover_nhx_config_async(
    base_url: str,
    http_client: httpx.AsyncClient,
    timeout: float = 10.0,
) -> NHXOIDCConfig:
    """Fetch OIDC configuration from the NeMo Helix auth discovery endpoint."""
    url = f"{base_url.rstrip('/')}/apis/auth/discovery"
    # TLS verification belongs to the caller-owned httpx.AsyncClient; do not override it here.
    # codeql[py/request-without-cert-validation]
    response = await http_client.get(url, timeout=timeout)
    response.raise_for_status()
    return _parse_nhx_config(response.json())


def _parse_nhx_config(payload: object) -> NHXOIDCConfig:
    try:
        return parse_nhx_config(payload)
    except ValueError as exc:
        if "bearer_token_source" in str(exc):
            raise
        raise AttributeError("auth discovery response did not match expected shape") from exc


def build_effective_scope(requested_scopes: str, scope_prefix: str | None) -> str:
    """Prepend scope_prefix to custom scopes (those with ':' or ending with '.default')."""
    prefix = normalize_scope_prefix(scope_prefix)
    if not prefix:
        return requested_scopes
    expanded = []
    for s in requested_scopes.split():
        if ":" in s or s.endswith(".default"):
            expanded.append(f"{prefix}{s}")
        else:
            expanded.append(s)
    return " ".join(expanded)


def validate_requested_scopes_granted(
    effective_scope: str,
    granted_scopes: list[str],
    scope_prefix: str,
) -> None:
    """Validate that requested platform scopes appear in granted scopes; raise AuthError if not.

    Compares in short form so IdPs (e.g. Azure AD) that return scp as "platform:read"
    match requested "api://nhx/platform:read".
    """
    requested_platform = {s for s in effective_scope.split() if is_valid_nhx_scope_id(scope_short(s, scope_prefix))}
    requested_short = {scope_short(s, scope_prefix) for s in requested_platform}
    granted_set = set(granted_scopes)
    granted_short = {scope_short(s, scope_prefix) for s in granted_set}
    missing_short = requested_short - granted_short
    if not missing_short:
        return
    full_missing = sorted(s for s in requested_platform if scope_short(s, scope_prefix) in missing_short)
    hint = ""
    if scope_prefix and "api://" in scope_prefix:
        hint = (
            "\nHint: For Azure AD, add the scopes in the app registration (Expose an API) and grant "
            "admin consent. See tools/auth/azure/README.md."
        )
    raise AuthError(
        f"Token is missing requested scopes: {' '.join(full_missing)}.\n"
        "The identity provider did not grant the requested scopes. "
        "Check IdP configuration." + hint
    )
