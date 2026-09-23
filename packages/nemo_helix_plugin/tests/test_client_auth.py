# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NemoClient first-class auth support (AIRCORE-828)."""

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import nemo_helix_plugin.client.oidc as oidc_module
import pytest
import respx
import yaml
from nemo_helix_plugin.client.auth import (
    StaticToken,
    TokenProvider,
)
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.config.config import Config
from nemo_helix_plugin.client.config.models import (
    NoAuthUser,
    OAuthUser,
)
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.client.oidc import (
    ACCESS_TOKEN_TYPE,
    DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
    JWT_TOKEN_TYPE,
    TOKEN_EXCHANGE_GRANT_TYPE,
    NHXOIDCConfig,
    OIDCTokenProvider,
    TokenSet,
    WorkloadTokenExchangeError,
    WorkloadTokenExchangeProvider,
    discover_nhx_config,
    generate_unsigned_jwt,
    refresh_token_grant,
    token_exchange_grant,
)
from nemo_helix_plugin.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR
from nemo_helix_plugin.client_provider import (
    DefaultNemoClientProvider,
    get_async_nemo_client,
    get_nemo_client,
    set_nemo_client_provider,
)


def _assert_no_nhx_common_import(module_file: str | None) -> None:
    assert module_file is not None
    tree = ast.parse(Path(module_file).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            continue
        assert not any(name == "nhx.common" or name.startswith("nhx.common.") for name in imported)


def test_oidc_module_has_no_nhx_common_dependency():
    _assert_no_nhx_common_import(oidc_module.__file__)


# ---------------------------------------------------------------------------
# StaticToken
# ---------------------------------------------------------------------------


class TestStaticToken:
    def test_get_access_token(self):
        provider = StaticToken("my-token")
        assert provider.get_access_token() == "my-token"

    def test_satisfies_protocol(self):
        provider = StaticToken("t")
        assert isinstance(provider, TokenProvider)

    def test_async_get_access_token(self):
        provider = StaticToken("my-token")
        result = asyncio.run(provider.get_access_token_async())
        assert result == "my-token"


# ---------------------------------------------------------------------------
# NemoClient auth parameter
# ---------------------------------------------------------------------------


class TestNemoClientAuth:
    @respx.mock
    def test_string_auth_sets_bearer_header(self):
        """NemoClient(auth='token') sets Authorization header on requests."""
        route = respx.get("http://localhost:8080/test").mock(return_value=httpx.Response(200, json={"ok": True}))
        client = NemoClient(base_url="http://localhost:8080", auth="my-secret-token")

        from nemo_helix_plugin.client.types import PreparedRequest

        req = PreparedRequest(
            method="GET", path_template="/test", path_params={}, content=None, content_type=None, response_type=None
        )
        client.send(req)

        assert route.called
        assert route.calls[0].request.headers["Authorization"] == "Bearer my-secret-token"

    @respx.mock
    def test_custom_provider_called_per_request(self):
        """NemoClient(auth=CustomProvider()) calls get_access_token() per request."""
        route = respx.get("http://localhost:8080/test").mock(return_value=httpx.Response(200, json={"ok": True}))

        call_count = 0

        class CountingProvider:
            def get_access_token(self) -> str:
                nonlocal call_count
                call_count += 1
                return f"token-{call_count}"

        client = NemoClient(base_url="http://localhost:8080", auth=CountingProvider())

        from nemo_helix_plugin.client.types import PreparedRequest

        req = PreparedRequest(
            method="GET", path_template="/test", path_params={}, content=None, content_type=None, response_type=None
        )
        client.send(req)
        client.send(req)

        assert route.calls[0].request.headers["Authorization"] == "Bearer token-1"
        assert route.calls[1].request.headers["Authorization"] == "Bearer token-2"
        assert call_count == 2

    @respx.mock
    def test_no_auth_no_header(self):
        """NemoClient without auth does not add Authorization header."""
        route = respx.get("http://localhost:8080/test").mock(return_value=httpx.Response(200, json={"ok": True}))
        client = NemoClient(base_url="http://localhost:8080")

        from nemo_helix_plugin.client.types import PreparedRequest

        req = PreparedRequest(
            method="GET", path_template="/test", path_params={}, content=None, content_type=None, response_type=None
        )
        client.send(req)

        assert route.called
        assert "Authorization" not in route.calls[0].request.headers

    def test_constructor_uses_workload_exchange_provider_from_env(self, monkeypatch, tmp_path):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token\n", encoding="utf-8")
        provider = object()
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

        with patch(
            "nemo_helix_plugin.client.oidc_factory.resolve_workload_exchange_provider",
            return_value=provider,
        ) as resolve_provider:
            client = NemoClient(base_url="https://nemo.example.com")

        assert client._auth is provider
        resolve_provider.assert_called_once_with(
            base_url="https://nemo.example.com",
            subject_token_file=subject_token_file,
        )

    def test_constructor_workload_exchange_does_not_override_authorization_header(self, monkeypatch, tmp_path):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token\n", encoding="utf-8")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

        with patch("nemo_helix_plugin.client.oidc_factory.resolve_workload_exchange_provider") as resolve_provider:
            client = NemoClient(
                base_url="https://nemo.example.com",
                default_headers={"Authorization": "Bearer explicit-token"},
            )

        assert client._auth is None
        resolve_provider.assert_not_called()

    def test_constructor_workload_exchange_does_not_override_principal_header(self, monkeypatch, tmp_path):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token\n", encoding="utf-8")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

        with patch("nemo_helix_plugin.client.oidc_factory.resolve_workload_exchange_provider") as resolve_provider:
            client = NemoClient(
                base_url="https://nemo.example.com",
                default_headers={"X-NHX-Principal-Id": "service:jobs"},
            )

        assert client._auth is None
        resolve_provider.assert_not_called()


# ---------------------------------------------------------------------------
# AsyncNemoClient auth parameter
# ---------------------------------------------------------------------------


class TestAsyncNemoClientAuth:
    @respx.mock
    def test_async_provider_called(self):
        """AsyncNemoClient(auth=AsyncProvider()) calls async get_access_token()."""
        route = respx.get("http://localhost:8080/test").mock(return_value=httpx.Response(200, json={"ok": True}))

        class AsyncProvider:
            async def get_access_token(self) -> str:
                return "async-token"

        client = AsyncNemoClient(base_url="http://localhost:8080", auth=AsyncProvider())

        from nemo_helix_plugin.client.types import PreparedRequest

        req = PreparedRequest(
            method="GET", path_template="/test", path_params={}, content=None, content_type=None, response_type=None
        )

        asyncio.run(client.send(req))

        assert route.called
        assert route.calls[0].request.headers["Authorization"] == "Bearer async-token"

    @respx.mock
    def test_sync_provider_works_in_async_client(self):
        """AsyncNemoClient accepts a sync TokenProvider too."""
        route = respx.get("http://localhost:8080/test").mock(return_value=httpx.Response(200, json={"ok": True}))
        client = AsyncNemoClient(base_url="http://localhost:8080", auth="sync-token")

        from nemo_helix_plugin.client.types import PreparedRequest

        req = PreparedRequest(
            method="GET", path_template="/test", path_params={}, content=None, content_type=None, response_type=None
        )

        asyncio.run(client.send(req))

        assert route.called
        assert route.calls[0].request.headers["Authorization"] == "Bearer sync-token"

    def test_constructor_uses_workload_exchange_provider_from_env(self, monkeypatch, tmp_path):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token\n", encoding="utf-8")
        provider = object()
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

        with patch(
            "nemo_helix_plugin.client.oidc_factory.resolve_workload_exchange_provider",
            return_value=provider,
        ) as resolve_provider:
            client = AsyncNemoClient(base_url="https://nemo.example.com")

        assert client._auth is provider
        resolve_provider.assert_called_once_with(
            base_url="https://nemo.example.com",
            subject_token_file=subject_token_file,
        )


# ---------------------------------------------------------------------------
# OIDCTokenProvider
# ---------------------------------------------------------------------------


def _make_jwt(exp: float | None = None, sub: str = "user") -> str:
    """Create a minimal unsigned JWT for testing."""
    return generate_unsigned_jwt(
        principal_id=sub,
        expires_in_seconds=int(exp - time.time()) if exp else 3600,
    )


class TestOIDCTokenProvider:
    def test_discover_nhx_config_uses_nemo_scoped_ca_bundle(self, monkeypatch):
        monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        with patch("nemo_helix_plugin.client.oidc.httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(
                200,
                json={"auth_enabled": False},
                request=httpx.Request("GET", "https://nemo.example.com/apis/auth/discovery"),
            )

            result = discover_nhx_config("https://nemo.example.com")

        assert result.auth_enabled is False
        mock_get.assert_called_once_with(
            "https://nemo.example.com/apis/auth/discovery",
            timeout=10.0,
            verify="/tmp/nemo-ca.pem",
        )

    def test_discover_nhx_config_rejects_unknown_bearer_token_source(self):
        with patch("nemo_helix_plugin.client.oidc.httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(
                200,
                json={"auth_enabled": True, "oidc": {"bearer_token_source": "refresh_token"}},
                request=httpx.Request("GET", "https://nemo.example.com/apis/auth/discovery"),
            )

            with pytest.raises(ValueError, match="bearer_token_source"):
                discover_nhx_config("https://nemo.example.com")

    def test_returns_token_when_not_expired(self):
        token = _make_jwt(exp=time.time() + 3600)
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet.from_access_token(token),
        )
        assert provider.get_access_token() == token

    def test_refreshes_when_expired(self):
        expired_token = _make_jwt(exp=time.time() - 100)
        new_token = _make_jwt(exp=time.time() + 3600)

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet(access_token=expired_token, refresh_token="refresh-me", expires_at=time.time() - 100),
        )

        with patch("nemo_helix_plugin.client.oidc.refresh_token_grant") as mock_grant:
            mock_grant.return_value = {"access_token": new_token}
            result = provider.get_access_token()

        assert result == new_token
        mock_grant.assert_called_once()

    def test_refresh_selects_configured_id_token(self):
        expired_token = _make_jwt(exp=time.time() - 100)
        new_id_token = _make_jwt(exp=time.time() + 3600)
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet(access_token=expired_token, refresh_token="refresh-me", expires_at=time.time() - 100),
            bearer_token_source="id_token",
        )

        with patch("nemo_helix_plugin.client.oidc.refresh_token_grant") as mock_grant:
            mock_grant.return_value = {
                "access_token": "opaque-access-token",
                "id_token": new_id_token,
                "refresh_token": "rotated-refresh",
            }
            result = provider.get_access_token()

        assert result == new_id_token
        assert provider.tokens.refresh_token == "rotated-refresh"

    def test_refresh_requires_configured_id_token(self):
        expired_token = _make_jwt(exp=time.time() - 100)
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet(access_token=expired_token, refresh_token="refresh-me", expires_at=time.time() - 100),
            bearer_token_source="id_token",
        )

        with (
            patch("nemo_helix_plugin.client.oidc.refresh_token_grant") as mock_grant,
            pytest.raises(RuntimeError, match="configured id_token"),
        ):
            mock_grant.return_value = {"access_token": "opaque-access-token"}
            provider.get_access_token()

    def test_persists_rotated_tokens(self):
        expired_token = _make_jwt(exp=time.time() - 100)
        new_token = _make_jwt(exp=time.time() + 3600)
        persisted = []

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet(access_token=expired_token, refresh_token="old-refresh", expires_at=time.time() - 100),
            on_tokens_refreshed=lambda ts: persisted.append(ts),
        )

        with patch("nemo_helix_plugin.client.oidc.refresh_token_grant") as mock_grant:
            mock_grant.return_value = {"access_token": new_token, "refresh_token": "new-refresh"}
            provider.get_access_token()

        assert len(persisted) == 1
        assert persisted[0].access_token == new_token
        assert persisted[0].refresh_token == "new-refresh"

    def test_raises_when_no_refresh_token(self):
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet(access_token="expired", expires_at=time.time() - 100),
        )
        with pytest.raises(RuntimeError, match="no refresh token"):
            provider.get_access_token()

    def test_invalid_grant_recovery_with_shared_tokens(self):
        """On invalid_grant, reload tokens from shared store and retry."""
        expired_token = _make_jwt(exp=time.time() - 100)
        fresh_token = _make_jwt(exp=time.time() + 3600)

        from nemo_helix_plugin.client.oidc import TokenRefreshError

        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet(access_token=expired_token, refresh_token="stale-refresh", expires_at=time.time() - 100),
            load_tokens=lambda: TokenSet(
                access_token=fresh_token, refresh_token="fresh-refresh", expires_at=time.time() + 3600
            ),
        )

        with patch("nemo_helix_plugin.client.oidc.refresh_token_grant") as mock_grant:
            mock_grant.side_effect = TokenRefreshError(error="invalid_grant", error_description="token revoked")
            result = provider.get_access_token()

        # Should have recovered by reloading fresh tokens from the shared store
        assert result == fresh_token

    def test_refresh_token_grant_uses_nemo_scoped_ca_bundle(self, monkeypatch):
        monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        with patch("nemo_helix_plugin.client.oidc.httpx.post") as mock_post:
            mock_post.return_value = httpx.Response(200, json={"access_token": "new-access"})

            result = refresh_token_grant(
                token_endpoint="https://idp.example.com/token",
                client_id="client",
                refresh_token="refresh-token",
            )

        assert result == {"access_token": "new-access"}
        assert mock_post.call_args.kwargs["verify"] == "/tmp/nemo-ca.pem"


class TestWorkloadTokenExchangeProvider:
    def test_token_exchange_grant_sends_rfc8693_request(self, monkeypatch):
        monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        with patch("nemo_helix_plugin.client.oidc.httpx.post") as mock_post:
            mock_post.return_value = httpx.Response(200, json={"access_token": "exchanged-token", "expires_in": 300})

            token_data = token_exchange_grant(
                token_endpoint="https://idp.example.com/token",
                client_id="nemo-helix-workload",
                subject_token="subject-token",
                audience="nemo-helix",
                scope="openid email groups",
                timeout=5.0,
            )

        assert token_data["access_token"] == "exchanged-token"
        mock_post.assert_called_once_with(
            "https://idp.example.com/token",
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
                "client_id": "nemo-helix-workload",
                "subject_token": "subject-token",
                "subject_token_type": JWT_TOKEN_TYPE,
                "requested_token_type": ACCESS_TOKEN_TYPE,
                "audience": "nemo-helix",
                "scope": "openid email groups",
            },
            timeout=5.0,
            verify=True,
        )

    def test_token_exchange_grant_uses_docker_opaque_subject_token_type(self, monkeypatch):
        monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        with patch("nemo_helix_plugin.client.oidc.httpx.post") as mock_post:
            mock_post.return_value = httpx.Response(200, json={"access_token": "exchanged-token", "expires_in": 300})

            token_exchange_grant(
                token_endpoint="https://idp.example.com/token",
                client_id="nemo-helix-workload",
                subject_token="nhx_obo_v1.delegation.secret",
            )

        assert mock_post.call_args.kwargs["data"]["subject_token_type"] == DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE

    def test_token_exchange_grant_uses_nemo_scoped_ca_bundle(self, monkeypatch):
        monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        with patch("nemo_helix_plugin.client.oidc.httpx.post") as mock_post:
            mock_post.return_value = httpx.Response(200, json={"access_token": "exchanged-token", "expires_in": 300})

            token_data = token_exchange_grant(
                token_endpoint="https://idp.example.com/token",
                client_id="nemo-helix-workload",
                subject_token="subject-token",
            )

        assert token_data["access_token"] == "exchanged-token"
        assert mock_post.call_args.kwargs["verify"] == "/tmp/nemo-ca.pem"

    def test_token_exchange_grant_allows_http_loopback_endpoint(self):
        with patch("nemo_helix_plugin.client.oidc.httpx.post") as mock_post:
            mock_post.return_value = httpx.Response(200, json={"access_token": "exchanged-token", "expires_in": 300})

            token_data = token_exchange_grant(
                token_endpoint="http://localhost:8080/apis/auth/token",
                client_id="nemo-helix-workload",
                subject_token="subject-token",
            )

        assert token_data["access_token"] == "exchanged-token"

    def test_token_exchange_grant_rejects_http_non_loopback_endpoint_when_allow_http_is_set(self):
        with patch("nemo_helix_plugin.client.oidc.httpx.post") as mock_post:
            with pytest.raises(ValueError, match="HTTPS"):
                token_exchange_grant(
                    token_endpoint="http://nemo-gateway:8080/apis/auth/token",
                    client_id="nemo-helix-workload",
                    subject_token="subject-token",
                    allow_http=True,
                )

        mock_post.assert_not_called()

    def test_token_exchange_grant_rejects_non_object_error_payload(self):
        with patch("nemo_helix_plugin.client.oidc.httpx.post") as mock_post:
            mock_post.return_value = httpx.Response(
                400,
                json=[],
                headers={"content-type": "application/json"},
            )

            with pytest.raises(WorkloadTokenExchangeError, match="invalid_response - Token endpoint error response"):
                token_exchange_grant(
                    token_endpoint="https://idp.example.com/token",
                    client_id="nemo-helix-workload",
                    subject_token="bad-subject-token",
                )

    @pytest.mark.parametrize("payload", [[], {}, {"access_token": ""}, {"access_token": None}])
    def test_token_exchange_grant_rejects_success_response_without_non_empty_access_token(self, payload):
        with patch("nemo_helix_plugin.client.oidc.httpx.post") as mock_post:
            mock_post.return_value = httpx.Response(200, json=payload)

            with pytest.raises(WorkloadTokenExchangeError, match="invalid_response"):
                token_exchange_grant(
                    token_endpoint="https://idp.example.com/token",
                    client_id="nemo-helix-workload",
                    subject_token="subject-token",
                )

    def test_token_exchange_grant_rejects_non_json_success_response(self):
        with patch("nemo_helix_plugin.client.oidc.httpx.post") as mock_post:
            mock_post.return_value = httpx.Response(200, content=b"not-json")

            with pytest.raises(WorkloadTokenExchangeError, match="Token endpoint response was not a JSON object"):
                token_exchange_grant(
                    token_endpoint="https://idp.example.com/token",
                    client_id="nemo-helix-workload",
                    subject_token="subject-token",
                )

    def test_provider_rejects_exchange_response_without_access_token(self, tmp_path):
        subject_token_file = tmp_path / "token"
        subject_token_file.write_text("subject-token", encoding="utf-8")
        provider = WorkloadTokenExchangeProvider(
            token_endpoint="https://idp.example.com/token",
            client_id="nemo-helix-workload",
            subject_token_file=subject_token_file,
        )

        with (
            patch("nemo_helix_plugin.client.oidc.token_exchange_grant", return_value={}),
            pytest.raises(WorkloadTokenExchangeError, match="non-empty access_token"),
        ):
            provider.get_access_token()

    @pytest.mark.parametrize(
        "expires_in",
        [None, "300", True, float("nan"), float("inf"), 10**400],
    )
    def test_provider_rejects_exchange_response_without_usable_lifetime(self, tmp_path, expires_in):
        subject_token_file = tmp_path / "token"
        subject_token_file.write_text("subject-token", encoding="utf-8")
        token_data = {"access_token": "opaque-access-token"}
        if expires_in is not None:
            token_data["expires_in"] = expires_in
        provider = WorkloadTokenExchangeProvider(
            token_endpoint="https://idp.example.com/token",
            client_id="nemo-helix-workload",
            subject_token_file=subject_token_file,
        )

        with (
            patch("nemo_helix_plugin.client.oidc.token_exchange_grant", return_value=token_data),
            pytest.raises(WorkloadTokenExchangeError, match="usable access_token lifetime"),
        ):
            provider.get_access_token()

        assert provider.tokens is None

    def test_provider_rejects_expired_exchange_response_and_retries_with_current_subject_token(self, tmp_path):
        subject_token_file = tmp_path / "token"
        subject_token_file.write_text("subject-token-one", encoding="utf-8")
        expired_access_token = _make_jwt(exp=time.time() - 10)
        fresh_access_token = _make_jwt(exp=time.time() + 3600)
        provider = WorkloadTokenExchangeProvider(
            token_endpoint="https://idp.example.com/token",
            client_id="nemo-helix-workload",
            subject_token_file=subject_token_file,
            audience="nemo-helix",
            scope="openid email groups",
            refresh_margin_seconds=0,
        )

        with patch(
            "nemo_helix_plugin.client.oidc.token_exchange_grant",
            side_effect=[
                {"access_token": expired_access_token},
                {"access_token": fresh_access_token},
            ],
        ) as mock_exchange:
            with pytest.raises(WorkloadTokenExchangeError, match="expired access_token"):
                provider.get_access_token()

            assert provider.tokens is None
            subject_token_file.write_text("subject-token-two", encoding="utf-8")

            assert provider.get_access_token() == fresh_access_token
            assert mock_exchange.call_count == 2
            assert mock_exchange.call_args_list[0].kwargs["subject_token"] == "subject-token-one"
            assert mock_exchange.call_args_list[1].kwargs["subject_token"] == "subject-token-two"
            assert mock_exchange.call_args_list[1].kwargs["audience"] == "nemo-helix"
            assert mock_exchange.call_args_list[1].kwargs["scope"] == "openid email groups"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_load_and_resolve(self, tmp_path):
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [{"name": "test-user", "type": "oauth", "token": "my-token"}],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user", "workspace": "ws1"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))

        config = Config.load(config_path=config_file)
        ctx = config.resolve()

        assert ctx.context_name == "test"
        assert str(ctx.cluster.base_url).rstrip("/") == "http://localhost:9090"
        assert ctx.workspace == "ws1"
        assert isinstance(ctx.user, OAuthUser)
        assert ctx.user.token.get_secret_value() == "my-token"

    def test_load_nonexistent_explicit_path_raises(self, tmp_path):
        missing = tmp_path / "does-not-exist.yaml"
        with pytest.raises(FileNotFoundError):
            Config.load(config_path=missing)

    def test_write_then_read_round_trip(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        Config.write(
            {"base_url": "http://localhost:9999", "access_token": "round-trip-token"},
            context_name="rt",
            config_path=config_file,
            set_current_on_create=True,
        )

        config = Config.load(config_path=config_file)
        ctx = config.resolve()
        assert ctx.context_name == "rt"
        assert str(ctx.cluster.base_url).rstrip("/") == "http://localhost:9999"
        assert isinstance(ctx.user, OAuthUser)
        assert ctx.user.token.get_secret_value() == "round-trip-token"

    def test_migrate_legacy_api_key_to_oauth(self, tmp_path):
        """Legacy api-key users are migrated to oauth on load."""
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [{"name": "test-user", "type": "api-key", "api_key": "my-api-key"}],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))

        config = Config.load(config_path=config_file)
        ctx = config.resolve()
        assert isinstance(ctx.user, OAuthUser)
        assert ctx.user.token.get_secret_value() == "my-api-key"

    def test_migrate_legacy_api_key_empty_becomes_no_auth(self, tmp_path):
        """Legacy api-key users with empty keys become no-auth."""
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [{"name": "test-user", "type": "api-key", "api_key": ""}],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))

        config = Config.load(config_path=config_file)
        ctx = config.resolve()
        assert isinstance(ctx.user, NoAuthUser)

    def test_resolve_no_auth_user(self, tmp_path):
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [{"name": "test-user", "type": "no-auth"}],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))

        config = Config.load(config_path=config_file)
        ctx = config.resolve()

        assert isinstance(ctx.user, NoAuthUser)


# ---------------------------------------------------------------------------
# NemoClient.from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_with_oauth(self, tmp_path):
        token = _make_jwt()
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [{"name": "test-user", "type": "oauth", "token": token}],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))

        with patch("nemo_helix_plugin.client.oidc_factory._discover_oidc_client_settings") as mock_discover:
            from nemo_helix_plugin.client.oidc import NHXOIDCConfig

            mock_discover.return_value = NHXOIDCConfig(
                auth_enabled=True,
                client_id="web-client",
                cli_client_id="cli-client",
                bearer_token_source="id_token",
                token_endpoint="https://idp/token",
            )
            client = NemoClient.from_config(config_path=config_file)

        assert client.base_url == "http://localhost:9090"
        assert isinstance(client._auth, OIDCTokenProvider)
        assert client._auth.client_id == "cli-client"
        assert client._auth.bearer_token_source == "id_token"

    def test_from_config_restores_opaque_token_expiry(self, tmp_path):
        expires_at = time.time() + 3600
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [
                {
                    "name": "test-user",
                    "type": "oauth",
                    "token": "opaque-access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": expires_at,
                }
            ],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))

        with patch("nemo_helix_plugin.client.oidc_factory._discover_oidc_client_settings") as mock_discover:
            mock_discover.return_value = NHXOIDCConfig(
                auth_enabled=True,
                client_id="cli-client",
                token_endpoint="https://idp/token",
            )
            client = NemoClient.from_config(config_path=config_file)

        assert isinstance(client._auth, OIDCTokenProvider)
        assert client._auth.tokens.expires_at == expires_at

    def test_from_config_does_not_downgrade_discovery_validation_failure(self, tmp_path):
        token = _make_jwt()
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [{"name": "test-user", "type": "oauth", "token": token}],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))

        with patch(
            "nemo_helix_plugin.client.oidc.discover_nhx_config",
            side_effect=ValueError("OIDC bearer_token_source must be 'access_token' or 'id_token'"),
        ):
            with pytest.raises(ValueError, match="bearer_token_source"):
                NemoClient.from_config(config_path=config_file)

    def test_from_config_with_no_auth(self, tmp_path):
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [{"name": "test-user", "type": "no-auth"}],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))

        client = NemoClient.from_config(config_path=config_file)
        assert client.base_url == "http://localhost:9090"
        assert client._auth is None

    @respx.mock
    def test_from_config_with_workload_identity_token_file(self, tmp_path, monkeypatch):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token\n", encoding="utf-8")
        exchanged_token = _make_jwt()
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [{"name": "test-user", "type": "no-auth"}],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))
        route = respx.get("http://localhost:9090/test").mock(return_value=httpx.Response(200, json={"ok": True}))
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
        monkeypatch.delenv("NHX_ACCESS_TOKEN", raising=False)

        with (
            patch("nemo_helix_plugin.client.oidc_factory._discover_oidc_client_settings") as mock_discover,
            patch("nemo_helix_plugin.client.oidc.token_exchange_grant") as mock_exchange,
        ):
            mock_discover.return_value = NHXOIDCConfig(
                auth_enabled=True,
                client_id="nemo-helix-cli",
                token_endpoint="https://idp.example.com/token",
                workload_token_exchange_enabled=True,
                workload_client_id="nemo-helix-workload",
                workload_token_endpoint="https://workload-idp.example.com/token",
                workload_audience="nemo-helix",
                workload_scope="openid email groups",
            )
            mock_exchange.return_value = {"access_token": exchanged_token, "expires_in": 300}

            client = NemoClient.from_config(config_path=config_file)

            from nemo_helix_plugin.client.types import PreparedRequest

            req = PreparedRequest(
                method="GET",
                path_template="/test",
                path_params={},
                content=None,
                content_type=None,
                response_type=None,
            )
            client.send(req)

        assert route.calls[0].request.headers["Authorization"] == f"Bearer {exchanged_token}"
        assert mock_exchange.call_args.kwargs["token_endpoint"] == "https://workload-idp.example.com/token"
        assert mock_exchange.call_args.kwargs["client_id"] == "nemo-helix-workload"
        assert mock_exchange.call_args.kwargs["subject_token"] == "subject-token"
        assert mock_exchange.call_args.kwargs["audience"] == "nemo-helix"
        assert mock_exchange.call_args.kwargs["scope"] == "openid email groups"
        assert mock_exchange.call_args.kwargs["allow_http"] is False

    def test_from_config_rejects_discovered_http_non_loopback_token_endpoint(self, tmp_path, monkeypatch):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token\n", encoding="utf-8")
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [{"name": "test-user", "type": "no-auth"}],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
        monkeypatch.delenv("NHX_ACCESS_TOKEN", raising=False)

        with (
            patch("nemo_helix_plugin.client.oidc_factory._discover_oidc_client_settings") as mock_discover,
            patch("nemo_helix_plugin.client.oidc.httpx.post") as mock_post,
        ):
            mock_discover.return_value = NHXOIDCConfig(
                auth_enabled=True,
                client_id="nemo-helix-cli",
                token_endpoint="https://idp.example.com/token",
                workload_token_exchange_enabled=True,
                workload_client_id="nemo-helix-workload",
                workload_token_endpoint="http://idp.example.com/token",
            )
            client = NemoClient.from_config(config_path=config_file)

            from nemo_helix_plugin.client.types import PreparedRequest

            req = PreparedRequest(
                method="GET",
                path_template="/test",
                path_params={},
                content=None,
                content_type=None,
                response_type=None,
            )
            with pytest.raises(ValueError, match="HTTPS"):
                client.send(req)

        mock_post.assert_not_called()

    def test_from_config_selects_context(self, tmp_path):
        """from_config(context='staging') uses the staging context, not the default."""
        config_data = {
            "current_context": "prod",
            "clusters": [
                {"name": "prod-cluster", "base_url": "http://prod:8080"},
                {"name": "staging-cluster", "base_url": "http://staging:9090"},
            ],
            "users": [
                {"name": "prod-user", "type": "no-auth"},
                {"name": "staging-user", "type": "no-auth"},
            ],
            "contexts": [
                {"name": "prod", "cluster": "prod-cluster", "user": "prod-user"},
                {"name": "staging", "cluster": "staging-cluster", "user": "staging-user"},
            ],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))

        client = NemoClient.from_config(context="staging", config_path=config_file)
        assert client.base_url == "http://staging:9090"

    def test_from_config_nonexistent_path_raises(self, tmp_path):
        missing = tmp_path / "nope.yaml"
        with pytest.raises(FileNotFoundError):
            NemoClient.from_config(config_path=missing)

    def test_async_from_config(self, tmp_path):
        config_data = {
            "current_context": "test",
            "clusters": [{"name": "test-cluster", "base_url": "http://localhost:9090"}],
            "users": [{"name": "test-user", "type": "no-auth"}],
            "contexts": [{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump(config_data))

        client = AsyncNemoClient.from_config(config_path=config_file)
        assert isinstance(client, AsyncNemoClient)
        assert client.base_url == "http://localhost:9090"


# ---------------------------------------------------------------------------
# get_nemo_client / get_async_nemo_client
# ---------------------------------------------------------------------------


class TestGetNemoClient:
    @pytest.fixture(autouse=True)
    def _use_default_provider(self):
        set_nemo_client_provider(DefaultNemoClientProvider())
        yield
        set_nemo_client_provider(None)

    def test_returns_sync_client(self):
        client = get_nemo_client(as_service="test-svc", internal=True)
        assert isinstance(client, NemoClient)

    def test_returns_async_client(self):
        client = get_async_nemo_client(as_service="test-svc")
        assert isinstance(client, AsyncNemoClient)

    def test_sync_client_uses_workload_exchange_provider_from_env(self, monkeypatch, tmp_path):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token\n", encoding="utf-8")
        provider = object()
        monkeypatch.setenv("NHX_BASE_URL", "https://nemo.example.com")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        with patch(
            "nemo_helix_plugin.client.oidc_factory.resolve_workload_exchange_provider",
            return_value=provider,
        ) as resolve_provider:
            client = get_nemo_client()

        assert client._auth is provider
        resolve_provider.assert_called_once_with(
            base_url="https://nemo.example.com",
            subject_token_file=subject_token_file,
        )

    def test_async_client_uses_workload_exchange_provider_from_env(self, monkeypatch, tmp_path):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token\n", encoding="utf-8")
        provider = object()
        monkeypatch.setenv("NHX_BASE_URL", "https://nemo.example.com")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        with patch(
            "nemo_helix_plugin.client.oidc_factory.resolve_workload_exchange_provider",
            return_value=provider,
        ) as resolve_provider:
            client = get_async_nemo_client()

        assert client._auth is provider
        resolve_provider.assert_called_once_with(
            base_url="https://nemo.example.com",
            subject_token_file=subject_token_file,
        )

    def test_workload_exchange_provider_does_not_override_explicit_service(self, monkeypatch, tmp_path):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token\n", encoding="utf-8")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        with patch("nemo_helix_plugin.client.oidc_factory.resolve_workload_exchange_provider") as resolve_provider:
            client = get_nemo_client(as_service="jobs")

        assert client._auth is None
        resolve_provider.assert_not_called()


# ---------------------------------------------------------------------------
# Security: token repr and endpoint validation
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_token_set_repr_hides_tokens(self):
        ts = TokenSet(access_token="secret-access", refresh_token="secret-refresh", expires_at=123.0)
        r = repr(ts)
        assert "secret-access" not in r
        assert "secret-refresh" not in r
        assert "123.0" in r  # expires_at is still visible

    def test_oidc_provider_repr_hides_tokens(self):
        provider = OIDCTokenProvider(
            token_endpoint="https://idp/token",
            client_id="client",
            tokens=TokenSet(access_token="secret", expires_at=999.0),
        )
        r = repr(provider)
        assert "secret" not in r
        assert "idp" in r  # non-sensitive fields are visible

    def test_refresh_token_grant_rejects_http_endpoint(self):
        from nemo_helix_plugin.client.oidc import _validate_token_endpoint

        # HTTPS is fine
        _validate_token_endpoint("https://idp.example.com/token")
        # HTTP loopback is fine
        _validate_token_endpoint("http://localhost:8080/token")
        _validate_token_endpoint("http://127.0.0.1:8080/token")
        # HTTP non-loopback is rejected
        with pytest.raises(ValueError, match="HTTPS"):
            _validate_token_endpoint("http://evil.example.com/token")
