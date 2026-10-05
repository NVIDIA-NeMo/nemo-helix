# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OIDC token management for NemoClient.

Provides:

- :class:`OIDCTokenProvider` — thread-safe OIDC token refresh.
- :class:`TokenSet` — API bearer + refresh token pair with expiry.
- :class:`NHXOIDCConfig` — OIDC discovery response model.
- JWT decode helpers (no verification — for expiry extraction only).
- OIDC discovery and scope helpers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from nemo_helix_plugin.client.constants import (
    DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE as _DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
)
from nemo_helix_plugin.client.constants import (
    JWT_WORKLOAD_SUBJECT_TOKEN_TYPE,
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    subject_token_type_for_exchange,
)
from nemo_helix_plugin.client.tls import client_verify_from_env

logger = logging.getLogger(__name__)

BearerTokenSource = Literal["access_token", "id_token"]


def _parse_bearer_token_source(value: object) -> BearerTokenSource:
    """Validate a bearer-token response field received from discovery."""
    if value == "access_token":
        return "access_token"
    if value == "id_token":
        return "id_token"
    raise ValueError("OIDC bearer_token_source must be 'access_token' or 'id_token'")


class TokenRefreshError(RuntimeError):
    """Structured error raised for OAuth refresh_token grant failures."""

    def __init__(self, *, error: str, error_description: str) -> None:
        self.error = error
        self.error_description = error_description
        super().__init__(f"Token refresh failed: {error} - {error_description}")


def _select_bearer_token(token_data: dict, source: BearerTokenSource) -> str:
    source = _parse_bearer_token_source(source)
    token = token_data.get(source)
    if not isinstance(token, str) or not token:
        raise RuntimeError(f"OIDC refresh response did not include the configured {source}")
    return token


# ---------------------------------------------------------------------------
# JWT helpers (decode only, no verification)
# ---------------------------------------------------------------------------


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


def decode_jwt_claims(token: str) -> dict[str, Any]:
    """Decode JWT claims without verification (for display/expiry extraction only)."""
    return _decode_jwt_segment(token, 1)


def decode_jwt_header(token: str) -> dict[str, Any]:
    """Decode JWT header without verification."""
    return _decode_jwt_segment(token, 0)


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
    """Generate an unsigned JWT (``alg=none``) for local development and testing."""
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


# ---------------------------------------------------------------------------
# OIDC discovery
# ---------------------------------------------------------------------------

DEFAULT_OAUTH_SCOPES = "openid profile email offline_access"
TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_TOKEN_TYPE = JWT_WORKLOAD_SUBJECT_TOKEN_TYPE
DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE = _DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


@dataclass(frozen=True)
class NHXOIDCConfig:
    """OIDC configuration discovered from the NeMo Helix."""

    auth_enabled: bool
    issuer: str | None = None
    client_id: str | None = None
    token_endpoint: str | None = None
    device_authorization_endpoint: str | None = None
    default_scopes: str = DEFAULT_OAUTH_SCOPES
    scope_prefix: str | None = None
    workload_token_exchange_enabled: bool = False
    workload_client_id: str | None = None
    workload_token_endpoint: str | None = None
    workload_audience: str | None = None
    workload_scope: str | None = None
    cli_client_id: str | None = None
    bearer_token_source: BearerTokenSource = "access_token"
    device_authorization_requires_device_id: bool = False
    device_authorization_display_name: str | None = None
    device_token_request_includes_scope: bool = True


def discover_nhx_config(base_url: str, timeout: float = 10.0) -> NHXOIDCConfig:
    """Fetch OIDC configuration from the NeMo Helix auth discovery endpoint."""
    response = httpx.get(
        f"{base_url.rstrip('/')}/apis/auth/discovery",
        timeout=timeout,
        verify=client_verify_from_env(),
    )
    response.raise_for_status()
    data = response.json()

    oidc = data.get("oidc") or {}
    return NHXOIDCConfig(
        auth_enabled=data.get("auth_enabled", False),
        issuer=oidc.get("issuer"),
        client_id=oidc.get("client_id"),
        cli_client_id=oidc.get("cli_client_id"),
        bearer_token_source=_parse_bearer_token_source(oidc.get("bearer_token_source", "access_token")),
        token_endpoint=oidc.get("token_endpoint"),
        device_authorization_endpoint=oidc.get("device_authorization_endpoint"),
        device_authorization_requires_device_id=oidc.get("device_authorization_requires_device_id", False),
        device_authorization_display_name=oidc.get("device_authorization_display_name"),
        device_token_request_includes_scope=oidc.get("device_token_request_includes_scope", True),
        default_scopes=oidc.get("default_scopes", DEFAULT_OAUTH_SCOPES),
        scope_prefix=oidc.get("scope_prefix"),
        workload_token_exchange_enabled=oidc.get("workload_token_exchange_enabled", False),
        workload_client_id=oidc.get("workload_client_id"),
        workload_token_endpoint=oidc.get("workload_token_endpoint"),
        workload_audience=oidc.get("workload_audience"),
        workload_scope=oidc.get("workload_scope"),
    )


def _discover_oidc_client_settings(base_url: str) -> NHXOIDCConfig:
    """Fetch OIDC config with a safe fallback if unreachable."""
    try:
        return discover_nhx_config(base_url)
    except (httpx.HTTPError, json.JSONDecodeError):
        logger.debug("Could not discover OIDC settings from %s", base_url, exc_info=True)
        return NHXOIDCConfig(
            auth_enabled=False,
            client_id="",
            token_endpoint="",
            default_scopes="openid profile email",
            scope_prefix=None,
        )


def _normalize_scope_prefix(prefix: str | None) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else f"{prefix}/"


def build_effective_scope(requested_scopes: str, scope_prefix: str | None) -> str:
    """Prepend scope_prefix to custom scopes (those with ':' or ending with '.default')."""
    prefix = _normalize_scope_prefix(scope_prefix)
    if not prefix:
        return requested_scopes
    expanded = []
    for s in requested_scopes.split():
        if ":" in s or s.endswith(".default"):
            expanded.append(f"{prefix}{s}")
        else:
            expanded.append(s)
    return " ".join(expanded)


# ---------------------------------------------------------------------------
# OAuth refresh_token grant
# ---------------------------------------------------------------------------

# Refresh proactively when less than this many seconds remain before expiry.
DEFAULT_REFRESH_MARGIN_SECONDS = 60


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname == "localhost":
        return True
    if hostname is None:
        return False
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_token_endpoint(token_endpoint: str) -> None:
    """Reject non-HTTPS token endpoints (except loopback for local dev)."""
    parsed = urlparse(token_endpoint)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
        return
    raise ValueError(
        f"OIDC token endpoint must use HTTPS (got {token_endpoint!r}). "
        "HTTP is only allowed for loopback addresses (localhost, 127.0.0.1, ::1)."
    )


def refresh_token_grant(
    token_endpoint: str,
    client_id: str,
    refresh_token: str,
    *,
    scope: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Execute OAuth refresh_token grant and return token response JSON."""
    _validate_token_endpoint(token_endpoint)
    data: dict[str, str] = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }
    if scope:
        data["scope"] = scope

    response = httpx.post(token_endpoint, data=data, timeout=timeout, verify=client_verify_from_env())

    if response.status_code != 200:
        error_data: dict[str, str] = {}
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                error_data = response.json()
            except (json.JSONDecodeError, ValueError):
                error_data = {}
        error = error_data.get("error", "unknown_error")
        error_description = error_data.get("error_description", response.text)
        raise TokenRefreshError(error=error, error_description=error_description)

    return response.json()


class WorkloadTokenExchangeError(RuntimeError):
    """Raised when workload identity token exchange fails."""


def _workload_exchange_error(error: str, error_description: str) -> WorkloadTokenExchangeError:
    return WorkloadTokenExchangeError(f"Workload token exchange failed: {error} - {error_description}")


def read_subject_token_file(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Unable to read {WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR} at {path}: {exc}") from exc
    if not token:
        raise ValueError(f"{WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR} at {path} is empty")
    return token


def token_exchange_grant(
    *,
    token_endpoint: str,
    client_id: str,
    subject_token: str,
    audience: str | None = None,
    scope: str | None = None,
    allow_http: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Execute an OAuth 2.0 Token Exchange (RFC 8693) request."""
    # Backward-compatible argument: endpoint host controls HTTP allowance.
    _validate_token_endpoint(token_endpoint)
    data: dict[str, str] = {
        "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
        "client_id": client_id,
        "subject_token": subject_token,
        "subject_token_type": subject_token_type_for_exchange(subject_token),
        "requested_token_type": ACCESS_TOKEN_TYPE,
    }
    if audience:
        data["audience"] = audience
    if scope:
        data["scope"] = scope

    response = httpx.post(token_endpoint, data=data, timeout=timeout, verify=client_verify_from_env())
    if response.status_code != 200:
        error_data: dict[str, object] = {}
        if response.headers.get("content-type", "").startswith("application/json"):
            error_data = _response_json_object(
                response,
                error_description="Token endpoint error response was not a JSON object",
            )
        error = _response_string(error_data, "error", "unknown_error")
        error_description = _response_string(error_data, "error_description", response.text)
        raise _workload_exchange_error(error, error_description)

    token_data = _response_json_object(
        response,
        error_description="Token endpoint response was not a JSON object",
    )
    _access_token_from_response(token_data)
    return token_data


def _response_json_object(response: httpx.Response, *, error_description: str) -> dict[str, object]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise _workload_exchange_error("invalid_response", error_description) from exc
    if not isinstance(payload, dict):
        raise _workload_exchange_error("invalid_response", error_description)
    return payload


def _response_string(payload: dict[str, object], key: str, default: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) and value else default


def _access_token_from_response(token_data: dict[str, object]) -> str:
    access_token = token_data.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise _workload_exchange_error(
            "invalid_response",
            "Token endpoint response did not include a non-empty access_token",
        )
    return access_token


def _expires_in_from_response(token_data: dict[str, object]) -> int | float | None:
    expires_in = token_data.get("expires_in")
    if isinstance(expires_in, bool):
        return None
    return expires_in if isinstance(expires_in, int | float) else None


def _validate_expiry(value: object, field_name: str) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


# ---------------------------------------------------------------------------
# TokenSet
# ---------------------------------------------------------------------------


@dataclass
class TokenSet:
    """An API bearer + refresh token pair with expiry metadata.

    ``access_token`` keeps its historical name but may contain the configured
    ID token when that is the bearer accepted by the platform.
    """

    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_at: float | None = None

    @staticmethod
    def from_access_token(
        access_token: str,
        refresh_token: str | None = None,
        *,
        expires_in: int | float | None = None,
        expires_at: float | None = None,
    ) -> TokenSet:
        """Create a TokenSet, preferring JWT expiry over persisted or relative expiry metadata.

        Falls back to a persisted ``expires_at`` value and then ``expires_in``
        (seconds from now) for opaque tokens that don't contain a JWT ``exp`` claim.
        """
        claims = decode_jwt_claims(access_token)

        jwt_expires_at = _validate_expiry(claims.get("exp"), "JWT exp") if claims else None
        persisted_expires_at = _validate_expiry(expires_at, "expires_at")
        validated_expires_in = _validate_expiry(expires_in, "expires_in")

        resolved_expires_at = jwt_expires_at
        if resolved_expires_at is None:
            resolved_expires_at = persisted_expires_at
        if resolved_expires_at is None and validated_expires_in is not None:
            resolved_expires_at = time.time() + validated_expires_in
        return TokenSet(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=resolved_expires_at,
        )

    def is_expired(self, margin_seconds: float = DEFAULT_REFRESH_MARGIN_SECONDS) -> bool:
        """Check if the access token is expired or about to expire."""
        if self.expires_at is None:
            return False
        return not math.isfinite(self.expires_at) or time.time() >= (self.expires_at - margin_seconds)


# ---------------------------------------------------------------------------
# OIDCTokenProvider
# ---------------------------------------------------------------------------


@dataclass
class OIDCTokenProvider:
    """Provides API bearer tokens with automatic refresh via the OAuth2 refresh_token grant.

    This is the core component for SDK-level token management. It:
    - Holds the current access + refresh tokens
    - Proactively refreshes the access token before it expires
    - Is thread-safe (uses a lock for concurrent access)
    - Optionally persists refreshed tokens via a callback
    """

    token_endpoint: str
    client_id: str
    tokens: TokenSet = field(default_factory=lambda: TokenSet(access_token=""), repr=False)
    refresh_margin_seconds: float = DEFAULT_REFRESH_MARGIN_SECONDS
    refresh_scope: str | None = None
    load_tokens: Callable[[], TokenSet | None] | None = None
    refresh_lock: Callable[[], AbstractContextManager[None]] | None = None
    on_tokens_refreshed: Callable[[TokenSet], None] | None = None
    bearer_token_source: BearerTokenSource = "access_token"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_access_token(self) -> str:
        """Return a valid access token, refreshing if necessary."""
        with self._lock:
            if self.tokens.is_expired(self.refresh_margin_seconds):
                self._refresh()
            return self.tokens.access_token

    async def get_access_token_async(self) -> str:
        """Return a valid access token in async contexts.

        Runs refresh logic in a worker thread so token refresh does not block the
        event loop.
        """
        return await asyncio.to_thread(self.get_access_token)

    def reload_tokens(self) -> bool:
        """Reload tokens from a shared store, if configured."""
        with self._lock:
            return self._reload_tokens_from_source()

    def _reload_tokens_from_source(self) -> bool:
        if self.load_tokens is None:
            return False

        try:
            loaded_tokens = self.load_tokens()
        except Exception:
            logger.warning("Failed to reload shared tokens", exc_info=True)
            return False

        if loaded_tokens is None or loaded_tokens == self.tokens:
            return False

        self.tokens = loaded_tokens
        logger.debug("Reloaded shared tokens (expires_at=%s)", self.tokens.expires_at)
        return True

    def _refresh(self, *, force: bool = False) -> None:
        """Refresh the access token using the refresh_token grant."""
        lock_context = self.refresh_lock() if self.refresh_lock is not None else nullcontext()
        with lock_context:
            self._reload_tokens_from_source()
            if not force and not self.tokens.is_expired(self.refresh_margin_seconds):
                return

            if not self.tokens.refresh_token:
                raise RuntimeError(
                    "Access token has expired and no refresh token is available. "
                    "Re-authenticate with `nemo auth login` to obtain new tokens."
                )

            logger.debug("Refreshing access token via %s", self.token_endpoint)

            token_data: dict
            try:
                token_data = refresh_token_grant(
                    token_endpoint=self.token_endpoint,
                    client_id=self.client_id,
                    refresh_token=self.tokens.refresh_token,
                    scope=self.refresh_scope,
                )
            except TokenRefreshError as exc:
                if exc.error != "invalid_grant":
                    raise

                if not self._reload_tokens_from_source():
                    raise

                if not force and not self.tokens.is_expired(self.refresh_margin_seconds):
                    logger.debug("Recovered from invalid_grant with shared tokens")
                    return

                if not self.tokens.refresh_token:
                    raise RuntimeError(
                        "Access token has expired and no refresh token is available. "
                        "Re-authenticate with `nemo auth login` to obtain new tokens."
                    )

                token_data = refresh_token_grant(
                    token_endpoint=self.token_endpoint,
                    client_id=self.client_id,
                    refresh_token=self.tokens.refresh_token,
                    scope=self.refresh_scope,
                )

            new_access_token = _select_bearer_token(token_data, self.bearer_token_source)
            # The IdP may rotate the refresh token.
            old_refresh_token = self.tokens.refresh_token
            new_refresh_token = token_data.get("refresh_token", old_refresh_token)
            refresh_token_rotated = new_refresh_token != old_refresh_token

            self.tokens = TokenSet.from_access_token(
                new_access_token, new_refresh_token, expires_in=token_data.get("expires_in")
            )
            logger.debug("Access token refreshed successfully (expires_at=%s)", self.tokens.expires_at)

            if self.on_tokens_refreshed:
                try:
                    self.on_tokens_refreshed(self.tokens)
                except Exception:
                    if refresh_token_rotated:
                        # The IdP rotated the refresh token but we failed to
                        # persist it. The old refresh token is now invalid, so
                        # swallowing this would silently lose the user's session.
                        raise
                    logger.warning("Failed to persist refreshed tokens", exc_info=True)

    def force_refresh(self) -> str:
        """Force a token refresh regardless of expiry. Returns the new access token."""
        with self._lock:
            self._refresh(force=True)
            return self.tokens.access_token


@dataclass
class WorkloadTokenExchangeProvider:
    """Provides access tokens by exchanging a refreshed workload subject-token file."""

    token_endpoint: str
    client_id: str
    subject_token_file: Path
    audience: str | None = None
    scope: str | None = None
    allow_http: bool = False
    refresh_margin_seconds: float = DEFAULT_REFRESH_MARGIN_SECONDS
    tokens: TokenSet | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_access_token(self) -> str:
        with self._lock:
            if self.tokens is None or self.tokens.is_expired(self.refresh_margin_seconds):
                self._exchange()
            assert self.tokens is not None
            return self.tokens.access_token

    async def get_access_token_async(self) -> str:
        return await asyncio.to_thread(self.get_access_token)

    def _exchange(self) -> None:
        subject_token = read_subject_token_file(self.subject_token_file)
        token_data = token_exchange_grant(
            token_endpoint=self.token_endpoint,
            client_id=self.client_id,
            subject_token=subject_token,
            audience=self.audience,
            scope=self.scope,
            allow_http=self.allow_http,
        )
        access_token = _access_token_from_response(token_data)
        try:
            tokens = TokenSet.from_access_token(
                access_token,
                expires_in=_expires_in_from_response(token_data),
            )
        except (OverflowError, TypeError, ValueError) as exc:
            raise _workload_exchange_error(
                "invalid_response",
                "Token endpoint response did not include a usable access_token lifetime",
            ) from exc
        if tokens.expires_at is None or not math.isfinite(tokens.expires_at):
            raise _workload_exchange_error(
                "invalid_response",
                "Token endpoint response did not include a usable access_token lifetime",
            )
        if tokens.is_expired(0):
            raise _workload_exchange_error(
                "invalid_response",
                "Token endpoint response returned an expired access_token",
            )
        self.tokens = tokens
