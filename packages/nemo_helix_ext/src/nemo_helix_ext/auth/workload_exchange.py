# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workload identity token exchange for SDK authentication."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
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

from nemo_helix_ext.auth.token_provider import DEFAULT_REFRESH_MARGIN_SECONDS, TokenSet
from nemo_helix_ext.client.tls import httpx_tls_config_from_env

logger = logging.getLogger(__name__)

TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_TOKEN_TYPE = JWT_WORKLOAD_SUBJECT_TOKEN_TYPE
DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE = _DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


class WorkloadTokenExchangeError(RuntimeError):
    """Structured error raised for RFC 8693 workload token exchange failures."""

    def __init__(self, *, error: str, error_description: str) -> None:
        self.error = error
        self.error_description = error_description
        super().__init__(f"Workload token exchange failed: {error} - {error_description}")


def read_subject_token_file(path: Path) -> str:
    """Read a subject token from a workload identity token file."""
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Unable to read {WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR} at {path}: {exc}") from exc
    if not token:
        raise ValueError(f"{WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR} at {path} is empty")
    return token


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


def token_exchange_grant(
    *,
    token_endpoint: str,
    client_id: str,
    subject_token: str,
    audience: str | None = None,
    scope: str | None = None,
    certificate_authority: str | None = None,
    timeout: float = 30.0,
) -> dict[str, object]:
    """Execute RFC 8693 token exchange and return token response JSON."""
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

    response = httpx.post(
        token_endpoint,
        data=data,
        timeout=timeout,
        **httpx_tls_config_from_env(certificate_authority),
    )

    if response.status_code != 200:
        error_data: dict[str, object] = {}
        if response.headers.get("content-type", "").startswith("application/json"):
            error_data = _response_json_object(
                response,
                error_description="Token endpoint error response was not a JSON object",
            )
        error = _response_string(error_data, "error", "unknown_error")
        error_description = _response_string(error_data, "error_description", response.text)
        raise WorkloadTokenExchangeError(error=error, error_description=error_description)

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
        raise WorkloadTokenExchangeError(error="invalid_response", error_description=error_description) from exc
    if not isinstance(payload, dict):
        raise WorkloadTokenExchangeError(error="invalid_response", error_description=error_description)
    return payload


def _response_string(payload: dict[str, object], key: str, default: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) and value else default


def _access_token_from_response(token_data: dict[str, object]) -> str:
    access_token = token_data.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise WorkloadTokenExchangeError(
            error="invalid_response",
            error_description="Token endpoint response did not include a non-empty access_token",
        )
    return access_token


def _expires_in_from_response(token_data: dict[str, object]) -> int | float | None:
    expires_in = token_data.get("expires_in")
    if isinstance(expires_in, bool):
        return None
    return expires_in if isinstance(expires_in, int | float) else None


@dataclass
class WorkloadTokenExchangeProvider:
    """Provides access tokens by exchanging a workload identity subject token file."""

    token_endpoint: str
    client_id: str
    subject_token_file: Path
    audience: str | None = None
    scope: str | None = None
    certificate_authority: str | None = None
    refresh_margin_seconds: float = DEFAULT_REFRESH_MARGIN_SECONDS
    tokens: TokenSet = field(default_factory=lambda: TokenSet(access_token=""))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_access_token(self) -> str:
        """Return a valid access token, exchanging the current subject token if needed."""
        with self._lock:
            if not self.tokens.access_token or self.tokens.is_expired(self.refresh_margin_seconds):
                self._exchange()
            return self.tokens.access_token

    async def get_access_token_async(self) -> str:
        """Return a valid access token in async contexts."""
        return await asyncio.to_thread(self.get_access_token)

    def _exchange(self) -> None:
        subject_token = read_subject_token_file(self.subject_token_file)
        logger.debug("Exchanging workload identity token via %s", self.token_endpoint)
        token_data = token_exchange_grant(
            token_endpoint=self.token_endpoint,
            client_id=self.client_id,
            subject_token=subject_token,
            audience=self.audience,
            scope=self.scope,
            certificate_authority=self.certificate_authority,
        )
        access_token = _access_token_from_response(token_data)
        try:
            tokens = TokenSet.from_access_token(
                access_token,
                refresh_token=None,
                expires_in=_expires_in_from_response(token_data),
            )
        except (OverflowError, TypeError, ValueError) as exc:
            raise WorkloadTokenExchangeError(
                error="invalid_response",
                error_description="Token endpoint response did not include a usable access_token lifetime",
            ) from exc
        if tokens.expires_at is None or not math.isfinite(tokens.expires_at):
            raise WorkloadTokenExchangeError(
                error="invalid_response",
                error_description="Token endpoint response did not include a usable access_token lifetime",
            )
        if tokens.is_expired(0):
            raise WorkloadTokenExchangeError(
                error="invalid_response",
                error_description="Token endpoint response returned an expired access_token",
            )
        self.tokens = tokens
