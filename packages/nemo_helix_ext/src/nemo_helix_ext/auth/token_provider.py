# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token provider with automatic refresh for NeMo Helix SDK authentication."""

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

import httpx
from typing_extensions import Self

from nemo_helix_ext.auth.helpers import decode_jwt_claims
from nemo_helix_ext.client.tls import httpx_tls_config_from_env

logger = logging.getLogger(__name__)

# Refresh proactively when less than this many seconds remain before expiry.
DEFAULT_REFRESH_MARGIN_SECONDS = 60


class TokenRefreshError(RuntimeError):
    """Structured error raised for OAuth refresh_token grant failures."""

    def __init__(self, *, error: str, error_description: str) -> None:
        self.error = error
        self.error_description = error_description
        super().__init__(f"Token refresh failed: {error} - {error_description}")


def _validate_expires_in(expires_in: object) -> int | float | None:
    if isinstance(expires_in, bool):
        return None
    return expires_in if isinstance(expires_in, int | float) else None


def refresh_token_grant(
    token_endpoint: str,
    client_id: str,
    refresh_token: str,
    *,
    scope: str | None = None,
    certificate_authority: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Execute OAuth refresh_token grant and return token response JSON."""
    data: dict[str, str] = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }
    if scope:
        data["scope"] = scope

    response = httpx.post(
        token_endpoint,
        data=data,
        timeout=timeout,
        **httpx_tls_config_from_env(certificate_authority),
    )

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


@dataclass
class TokenSet:
    """A pair of access + refresh tokens with expiry metadata."""

    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None

    @staticmethod
    def from_access_token(
        access_token: str,
        refresh_token: str | None = None,
        expires_in: object = None,
    ) -> Self:
        """Create a TokenSet, extracting expiry from the JWT's `exp` claim."""
        expires_at = None
        claims = decode_jwt_claims(access_token)
        if claims:
            expires_at = claims.get("exp")
        validated_expires_in = _validate_expires_in(expires_in)
        if expires_at is None and validated_expires_in is not None:
            expires_at = time.time() + float(validated_expires_in)
        return TokenSet(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=float(expires_at) if expires_at is not None else None,
        )

    def is_expired(self, margin_seconds: float = DEFAULT_REFRESH_MARGIN_SECONDS) -> bool:
        """Check if the access token is expired or about to expire."""
        if self.expires_at is None:
            return False
        return time.time() >= (self.expires_at - margin_seconds)


@dataclass
class OIDCTokenProvider:
    """Provides access tokens with automatic refresh via the OAuth2 refresh_token grant.

    This is the core component for SDK-level token management. It:
    - Holds the current access + refresh tokens
    - Proactively refreshes the access token before it expires
    - Is thread-safe (uses a lock for concurrent access)
    - Optionally persists refreshed tokens via a callback

    Args:
        token_endpoint: The OAuth2 token endpoint URL.
        client_id: The OAuth2 client ID.
        tokens: The current token set.
        refresh_margin_seconds: Seconds before expiry to proactively refresh.
        load_tokens: Optional callback to reload tokens from a shared store (e.g.
            config file) before refresh attempts.
        refresh_lock: Optional context manager factory for serializing refresh
            transactions across processes.
        on_tokens_refreshed: Optional callback invoked with the new ``TokenSet``
            after a successful refresh. Use this to persist tokens (e.g. write
            them back to ``~/.config/nhx/config.yaml``).
    """

    token_endpoint: str
    client_id: str
    tokens: TokenSet = field(default_factory=lambda: TokenSet(access_token=""))
    refresh_margin_seconds: float = DEFAULT_REFRESH_MARGIN_SECONDS
    refresh_scope: str | None = None
    certificate_authority: str | None = None
    load_tokens: Callable[[], TokenSet | None] | None = None
    refresh_lock: Callable[[], AbstractContextManager[None]] | None = None
    on_tokens_refreshed: Callable[[TokenSet], None] | None = None
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
        """Refresh the access token using the refresh_token grant.

        Raises:
            RuntimeError: If no refresh token is available or the refresh request fails.
        """
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
                    certificate_authority=self.certificate_authority,
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
                    certificate_authority=self.certificate_authority,
                )

            new_access_token = token_data["access_token"]
            # The IdP may rotate the refresh token.
            new_refresh_token = token_data.get("refresh_token", self.tokens.refresh_token)

            self.tokens = TokenSet.from_access_token(
                new_access_token,
                new_refresh_token,
                expires_in=token_data.get("expires_in"),
            )
            logger.debug("Access token refreshed successfully (expires_at=%s)", self.tokens.expires_at)

            if self.on_tokens_refreshed:
                try:
                    self.on_tokens_refreshed(self.tokens)
                except Exception:
                    logger.warning("Failed to persist refreshed tokens", exc_info=True)

    def force_refresh(self) -> str:
        """Force a token refresh regardless of expiry. Returns the new access token."""
        with self._lock:
            self._refresh(force=True)
            return self.tokens.access_token
