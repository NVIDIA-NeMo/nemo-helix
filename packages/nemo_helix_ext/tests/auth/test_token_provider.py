# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for token_provider module."""

import json
import time
from base64 import urlsafe_b64encode
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_ext.auth.token_provider import (
    OIDCTokenProvider,
    TokenPersistenceError,
    TokenSet,
    refresh_token_grant,
)
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR


def _make_jwt(claims: dict, header: dict | None = None) -> str:
    """Create a fake JWT token with the given claims (no real signature)."""
    header = header or {"alg": "RS256", "typ": "JWT"}
    h = urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p = urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    s = urlsafe_b64encode(b"fake-signature").rstrip(b"=").decode()
    return f"{h}.{p}.{s}"


class TestTokenSet:
    def test_from_access_token_extracts_exp(self):
        exp = int(time.time()) + 3600
        token = _make_jwt({"sub": "user1", "exp": exp})
        ts = TokenSet.from_access_token(token, refresh_token="refresh_abc")

        assert ts.access_token == token
        assert ts.refresh_token == "refresh_abc"
        assert ts.expires_at == float(exp)

    def test_from_access_token_no_exp_claim(self):
        token = _make_jwt({"sub": "user1"})
        ts = TokenSet.from_access_token(token)

        assert ts.expires_at is None

    @pytest.mark.parametrize("expires_in", [120, 120.5])
    def test_from_access_token_uses_numeric_expires_in_when_no_exp_claim(self, expires_in):
        token = _make_jwt({"sub": "user1"})
        before = time.time()
        ts = TokenSet.from_access_token(token, expires_in=expires_in)

        assert ts.expires_at is not None
        assert before + expires_in <= ts.expires_at <= time.time() + expires_in

    def test_from_access_token_rejects_bool_expires_in(self):
        token = _make_jwt({"sub": "user1"})
        ts = TokenSet.from_access_token(token, expires_in=True)

        assert ts.expires_at is None

    def test_from_access_token_non_jwt(self):
        ts = TokenSet.from_access_token("not-a-jwt", refresh_token="r")

        assert ts.access_token == "not-a-jwt"
        assert ts.refresh_token == "r"
        assert ts.expires_at is None

    def test_is_expired_when_past_expiry(self):
        ts = TokenSet(access_token="t", expires_at=time.time() - 100)
        assert ts.is_expired(margin_seconds=0) is True

    def test_is_expired_within_margin(self):
        ts = TokenSet(access_token="t", expires_at=time.time() + 30)
        assert ts.is_expired(margin_seconds=60) is True

    def test_is_not_expired(self):
        ts = TokenSet(access_token="t", expires_at=time.time() + 3600)
        assert ts.is_expired(margin_seconds=60) is False

    def test_is_not_expired_when_no_expiry(self):
        ts = TokenSet(access_token="t", expires_at=None)
        assert ts.is_expired() is False


class TestOIDCTokenProvider:
    def test_get_access_token_returns_current_when_not_expired(self):
        token = _make_jwt({"exp": int(time.time()) + 3600})
        tokens = TokenSet.from_access_token(token, refresh_token="r")

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
        )
        assert provider.get_access_token() == token

    @pytest.mark.asyncio
    async def test_get_access_token_async_returns_current_when_not_expired(self):
        token = _make_jwt({"exp": int(time.time()) + 3600})
        tokens = TokenSet.from_access_token(token, refresh_token="r")

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
        )

        assert await provider.get_access_token_async() == token

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_token_grant_uses_nemo_scoped_ca_bundle(self, mock_post, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "new_access"}
        mock_post.return_value = mock_response
        monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        result = refresh_token_grant(
            token_endpoint="https://idp/token",
            client_id="client",
            refresh_token="refresh_abc",
        )

        assert result == {"access_token": "new_access"}
        assert mock_post.call_args.kwargs["verify"] == "/tmp/nemo-ca.pem"

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_token_grant_uses_context_certificate_authority(self, mock_post, tmp_path, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "new_access"}
        mock_post.return_value = mock_response
        context_ca = str(tmp_path / "context-ca.pem")
        monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        result = refresh_token_grant(
            token_endpoint="https://idp/token",
            client_id="client",
            refresh_token="refresh_abc",
            certificate_authority=context_ca,
        )

        assert result == {"access_token": "new_access"}
        assert mock_post.call_args.kwargs["verify"] == context_ca

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_get_access_token_refreshes_when_expired(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        new_token = _make_jwt({"exp": int(time.time()) + 3600})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_token,
            "refresh_token": "new_refresh",
        }
        mock_post.return_value = mock_response

        tokens = TokenSet.from_access_token(old_token, refresh_token="old_refresh")
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
            refresh_margin_seconds=0,
        )

        result = provider.get_access_token()
        assert result == new_token
        assert provider.tokens.refresh_token == "new_refresh"

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["data"]["grant_type"] == "refresh_token"
        assert call_kwargs[1]["data"]["client_id"] == "client"
        assert call_kwargs[1]["data"]["refresh_token"] == "old_refresh"

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_selects_configured_id_token(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        new_id_token = _make_jwt({"exp": int(time.time()) + 3600})
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "opaque-access-token",
            "id_token": new_id_token,
            "refresh_token": "rotated-refresh",
        }
        mock_post.return_value = mock_response
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet.from_access_token(old_token, refresh_token="old-refresh"),
            refresh_margin_seconds=0,
            bearer_token_source="id_token",
        )

        assert provider.get_access_token() == new_id_token
        assert provider.tokens.refresh_token == "rotated-refresh"

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_requires_configured_id_token(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "opaque-access-token"}
        mock_post.return_value = mock_response
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet.from_access_token(old_token, refresh_token="old-refresh"),
            refresh_margin_seconds=0,
            bearer_token_source="id_token",
        )

        with pytest.raises(RuntimeError, match="configured id_token"):
            provider.get_access_token()

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_get_access_token_refreshes_opaque_token_with_expires_in(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "opaque_access",
            "expires_in": 120,
        }
        mock_post.return_value = mock_response

        tokens = TokenSet.from_access_token(old_token, refresh_token="old_refresh")
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
            refresh_margin_seconds=0,
        )

        before = time.time()
        result = provider.get_access_token()

        assert result == "opaque_access"
        assert provider.tokens.expires_at is not None
        assert before + 120 <= provider.tokens.expires_at <= time.time() + 120

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_reloads_tokens_before_request(self, mock_post):
        stale_token = _make_jwt({"exp": int(time.time()) - 200})
        shared_token = _make_jwt({"exp": int(time.time()) - 100})
        new_token = _make_jwt({"exp": int(time.time()) + 3600})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": new_token}
        mock_post.return_value = mock_response

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet.from_access_token(stale_token, refresh_token="stale_refresh"),
            refresh_margin_seconds=0,
            load_tokens=lambda: TokenSet.from_access_token(shared_token, refresh_token="shared_refresh"),
        )

        provider.get_access_token()

        assert mock_post.call_args[1]["data"]["refresh_token"] == "shared_refresh"

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_invalid_grant_recovers_with_reloaded_tokens(self, mock_post):
        stale_token = _make_jwt({"exp": int(time.time()) - 200})
        shared_token = _make_jwt({"exp": int(time.time()) + 3600})

        invalid_grant = MagicMock()
        invalid_grant.status_code = 400
        invalid_grant.text = "invalid refresh token"
        invalid_grant.headers = {"content-type": "application/json"}
        invalid_grant.json.return_value = {
            "error": "invalid_grant",
            "error_description": "invalid refresh token",
        }
        mock_post.return_value = invalid_grant

        load_states = iter(
            [
                TokenSet.from_access_token(stale_token, refresh_token="stale_refresh"),
                TokenSet.from_access_token(shared_token, refresh_token="rotated_refresh"),
            ]
        )

        def load_tokens() -> TokenSet:
            try:
                return next(load_states)
            except StopIteration:
                return TokenSet.from_access_token(shared_token, refresh_token="rotated_refresh")

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet.from_access_token(stale_token, refresh_token="stale_refresh"),
            refresh_margin_seconds=0,
            load_tokens=load_tokens,
        )

        result = provider.get_access_token()

        assert result == shared_token
        assert mock_post.call_count == 1

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_invalid_grant_retries_with_reloaded_refresh_token(self, mock_post):
        stale_token = _make_jwt({"exp": int(time.time()) - 200})
        rotated_access_token = _make_jwt({"exp": int(time.time()) - 100})
        new_token = _make_jwt({"exp": int(time.time()) + 3600})

        invalid_grant = MagicMock()
        invalid_grant.status_code = 400
        invalid_grant.text = "invalid refresh token"
        invalid_grant.headers = {"content-type": "application/json"}
        invalid_grant.json.return_value = {
            "error": "invalid_grant",
            "error_description": "invalid refresh token",
        }

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {"access_token": new_token, "refresh_token": "new_refresh"}

        mock_post.side_effect = [invalid_grant, success]

        load_states = iter(
            [
                TokenSet.from_access_token(stale_token, refresh_token="stale_refresh"),
                TokenSet.from_access_token(rotated_access_token, refresh_token="rotated_refresh"),
            ]
        )

        def load_tokens() -> TokenSet:
            try:
                return next(load_states)
            except StopIteration:
                return TokenSet.from_access_token(rotated_access_token, refresh_token="rotated_refresh")

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet.from_access_token(stale_token, refresh_token="stale_refresh"),
            refresh_margin_seconds=0,
            load_tokens=load_tokens,
        )

        result = provider.get_access_token()

        assert result == new_token
        assert mock_post.call_count == 2
        assert mock_post.call_args_list[0][1]["data"]["refresh_token"] == "stale_refresh"
        assert mock_post.call_args_list[1][1]["data"]["refresh_token"] == "rotated_refresh"

    @pytest.mark.asyncio
    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    async def test_get_access_token_async_refreshes_when_expired(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        new_token = _make_jwt({"exp": int(time.time()) + 3600})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_token,
            "refresh_token": "new_refresh",
        }
        mock_post.return_value = mock_response

        tokens = TokenSet.from_access_token(old_token, refresh_token="old_refresh")
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
            refresh_margin_seconds=0,
        )

        result = await provider.get_access_token_async()
        assert result == new_token
        assert provider.tokens.refresh_token == "new_refresh"

        mock_post.assert_called_once()

    def test_refresh_raises_when_no_refresh_token(self):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        tokens = TokenSet.from_access_token(old_token)  # no refresh token

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
            refresh_margin_seconds=0,
        )

        with pytest.raises(RuntimeError, match="no refresh token"):
            provider.get_access_token()

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_raises_on_http_error(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "bad request"
        mock_response.json.return_value = {"error": "invalid_grant"}
        mock_post.return_value = mock_response

        tokens = TokenSet.from_access_token(old_token, refresh_token="r")
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
            refresh_margin_seconds=0,
        )

        with pytest.raises(RuntimeError, match="Token refresh failed"):
            provider.get_access_token()

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_on_tokens_refreshed_callback_called(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        new_token = _make_jwt({"exp": int(time.time()) + 3600})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": new_token}
        mock_post.return_value = mock_response

        callback = MagicMock()
        tokens = TokenSet.from_access_token(old_token, refresh_token="r")
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
            refresh_margin_seconds=0,
            on_tokens_refreshed=callback,
        )

        provider.get_access_token()

        callback.assert_called_once()
        persisted_tokens = callback.call_args[0][0]
        assert persisted_tokens.access_token == new_token

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_on_tokens_refreshed_callback_error_does_not_propagate(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        new_token = _make_jwt({"exp": int(time.time()) + 3600})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": new_token}
        mock_post.return_value = mock_response

        callback = MagicMock(side_effect=IOError("disk full"))
        tokens = TokenSet.from_access_token(old_token, refresh_token="r")
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
            refresh_margin_seconds=0,
            on_tokens_refreshed=callback,
        )

        # Should not raise despite callback failure
        result = provider.get_access_token()
        assert result == new_token

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_rotated_refresh_token_persistence_error_propagates(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        new_token = _make_jwt({"exp": int(time.time()) + 3600})
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_token,
            "refresh_token": "rotated-refresh",
        }
        mock_post.return_value = mock_response
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet.from_access_token(old_token, refresh_token="old-refresh"),
            refresh_margin_seconds=0,
            on_tokens_refreshed=MagicMock(side_effect=OSError("disk full")),
        )

        with pytest.raises(TokenPersistenceError, match="rotated the refresh token") as exc_info:
            provider.get_access_token()

        assert isinstance(exc_info.value.__cause__, OSError)

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_keeps_old_refresh_token_if_not_rotated(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        new_token = _make_jwt({"exp": int(time.time()) + 3600})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_token,
            # no refresh_token in response
        }
        mock_post.return_value = mock_response

        tokens = TokenSet.from_access_token(old_token, refresh_token="original_refresh")
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
            refresh_margin_seconds=0,
        )

        provider.get_access_token()
        assert provider.tokens.refresh_token == "original_refresh"

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_force_refresh(self, mock_post):
        valid_token = _make_jwt({"exp": int(time.time()) + 3600})
        new_token = _make_jwt({"exp": int(time.time()) + 7200})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": new_token}
        mock_post.return_value = mock_response

        tokens = TokenSet.from_access_token(valid_token, refresh_token="r")
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=tokens,
        )

        result = provider.force_refresh()
        assert result == new_token
        mock_post.assert_called_once()

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_uses_refresh_lock(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        new_token = _make_jwt({"exp": int(time.time()) + 7200})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": new_token}
        mock_post.return_value = mock_response

        lock_events: list[str] = []

        @contextmanager
        def refresh_lock():
            lock_events.append("enter")
            try:
                yield
            finally:
                lock_events.append("exit")

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet.from_access_token(old_token, refresh_token="r"),
            refresh_margin_seconds=0,
            refresh_lock=refresh_lock,
        )

        provider.get_access_token()

        assert lock_events == ["enter", "exit"]

    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_includes_scope_when_configured(self, mock_post):
        old_token = _make_jwt({"exp": int(time.time()) - 100})
        new_token = _make_jwt({"exp": int(time.time()) + 3600})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": new_token}
        mock_post.return_value = mock_response

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet.from_access_token(old_token, refresh_token="r"),
            refresh_scope="openid profile",
            refresh_margin_seconds=0,
        )

        provider.get_access_token()

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["data"]["scope"] == "openid profile"
