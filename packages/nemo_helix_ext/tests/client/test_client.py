# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for building typed clients from the nhx config: auth wiring, TLS, overrides and failures."""

import json
import time
from base64 import urlsafe_b64encode
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml
from nemo_helix_ext.auth.helpers import NHXOIDCConfig, decode_jwt_claims
from nemo_helix_ext.client.bootstrap import build_async_nemo_client, build_nemo_client
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR
from nemo_helix_plugin.client.auth import TokenProviderAuth
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.client.endpoint import get
from pydantic import BaseModel


def _make_jwt(claims: dict) -> str:
    """Create a fake JWT token with the given claims."""
    header = {"alg": "RS256", "typ": "JWT"}
    h = urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p = urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    s = urlsafe_b64encode(b"fake-signature").rstrip(b"=").decode()
    return f"{h}.{p}.{s}"


def _write_config(
    tmp_path,
    *,
    user_type="oauth",
    token=None,
    refresh_token=None,
    api_key=None,
    certificate_authority=None,
):
    """Write a minimal nhx config file and return its path."""
    if user_type == "oauth":
        user = {
            "name": "default",
            "type": "oauth",
            "token": token,
            "refresh_token": refresh_token,
        }
    elif user_type == "api-key":
        user = {"name": "default", "type": "api-key", "api_key": api_key}
    else:
        user = {"name": "default", "type": "no-auth"}

    cluster = {"name": "default", "base_url": "http://localhost:8080"}
    if certificate_authority:
        cluster["certificate_authority"] = certificate_authority

    config = {
        "current_context": "default",
        "clusters": [cluster],
        "users": [user],
        "contexts": [
            {
                "name": "default",
                "cluster": "default",
                "user": "default",
                "workspace": "test-workspace",
            }
        ],
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return config_path


_MOCK_NHX_CONFIG = NHXOIDCConfig(
    auth_enabled=True,
    client_id="nhx-client-id",
    token_endpoint="https://idp/token",
)

_MOCK_WORKLOAD_NHX_CONFIG = NHXOIDCConfig(
    auth_enabled=True,
    client_id="nhx-client-id",
    token_endpoint="https://idp/token",
    workload_token_exchange_enabled=True,
    workload_client_id="nhx-workload-client-id",
    workload_token_endpoint="https://workload-idp/token",
    workload_audience="nemo-helix",
    workload_scope="openid email groups",
)


class Probe(BaseModel):
    ok: bool


@get("/apis/test/v2/probe")
def probe() -> Probe:
    raise NotImplementedError


def _wire(client: NemoClient) -> list[httpx.Request]:
    """Swap the transport for a recorder that answers every request, keeping the builder's auth hook."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    client._http = httpx.Client(
        transport=httpx.MockTransport(handler), auth=client._http.auth, headers=client._http.headers
    )
    return seen


def _async_wire(client: AsyncNemoClient) -> list[httpx.Request]:
    """Async twin of :func:`_wire`."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), auth=client._http.auth, headers=client._http.headers
    )
    return seen


def _sent_authorization(client: NemoClient) -> str | None:
    """Send a probe request and return the ``Authorization`` header that reached the wire."""
    seen = _wire(client)
    client.send(probe())
    return seen[0].headers.get("Authorization")


def _multi_context_config(tmp_path):
    config = {
        "current_context": "default",
        "clusters": [
            {"name": "cluster-one", "base_url": "http://localhost:8080"},
            {"name": "cluster-two", "base_url": "http://localhost:9090"},
        ],
        "users": [
            {"name": "user-one", "type": "api-key", "api_key": "nvapi-one"},
            {"name": "user-two", "type": "api-key", "api_key": "nvapi-two"},
        ],
        "contexts": [
            {"name": "default", "cluster": "cluster-one", "user": "user-one", "workspace": "workspace-one"},
            {"name": "target-context", "cluster": "cluster-two", "user": "user-two", "workspace": "workspace-two"},
        ],
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return config_path


class TestBuildNemoClientOAuth:
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_builds_client_from_stored_oauth_tokens(self, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc")

        client = build_nemo_client(config_path=config_path)

        assert isinstance(client, NemoClient)
        assert client.base_url.rstrip("/") == "http://localhost:8080"
        assert client.workspace == "test-workspace"

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_sends_the_stored_token_on_each_request(self, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc")

        client = build_nemo_client(config_path=config_path)

        assert isinstance(client._http.auth, TokenProviderAuth)
        assert _sent_authorization(client) == f"Bearer {token}"

    @patch("nemo_helix_ext.client.bootstrap.httpx.Client")
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_uses_nemo_scoped_ca_bundle(self, _mock_discover, mock_httpx_client, tmp_path, monkeypatch):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc")
        monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        build_nemo_client(config_path=config_path)

        assert mock_httpx_client.call_args.kwargs["verify"] == "/tmp/nemo-ca.pem"

    @patch("nemo_helix_ext.client.bootstrap.httpx.Client")
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_uses_context_certificate_authority(self, _mock_discover, mock_httpx_client, tmp_path, monkeypatch):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        context_ca = str(tmp_path / "context-ca.pem")
        config_path = _write_config(
            tmp_path, token=token, refresh_token="refresh_abc", certificate_authority=context_ca
        )
        monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        build_nemo_client(config_path=config_path)

        assert mock_httpx_client.call_args.kwargs["verify"] == context_ca
        assert _mock_discover.call_args.kwargs["certificate_authority"] == context_ca

    @patch("nemo_helix_ext.client.bootstrap.httpx.Client")
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_env_ca_bundle_overrides_context_certificate_authority(
        self, _mock_discover, mock_httpx_client, tmp_path, monkeypatch
    ):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(
            tmp_path, token=token, refresh_token="refresh_abc", certificate_authority="/tmp/context-ca.pem"
        )
        monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/env-ca.pem")

        build_nemo_client(config_path=config_path)

        assert mock_httpx_client.call_args.kwargs["verify"] == "/tmp/env-ca.pem"
        assert _mock_discover.call_args.kwargs["certificate_authority"] == "/tmp/context-ca.pem"

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_persist_refreshed_tokens_writes_to_config(self, mock_post, _mock_discover, tmp_path):
        expired_token = _make_jwt({"exp": int(time.time()) - 100, "sub": "user1"})
        new_token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=expired_token, refresh_token="refresh_abc")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": new_token, "refresh_token": "new_refresh"}
        mock_post.return_value = mock_response

        client = build_nemo_client(config_path=config_path)

        assert _sent_authorization(client) == f"Bearer {new_token}"
        mock_post.assert_called_once()
        with open(config_path) as f:
            saved_user = yaml.safe_load(f)["users"][0]
        assert saved_user["token"] == new_token
        assert saved_user["refresh_token"] == "new_refresh"

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_explicit_access_token_overrides_config_auth(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        client = build_nemo_client(config_path=config_path, access_token="explicit-access-token-123")

        assert _sent_authorization(client) == "Bearer explicit-access-token-123"


class TestBuildNemoClientAuthDisabledCluster:
    """Regression tests: OAuthUser context pointed at a cluster with no auth.

    This happens when a config context was previously authenticated against a
    real OIDC cluster and is later pointed at a local/no-auth instance. Once
    the stored access token expires, the client must not attempt a
    refresh_token grant against an empty token endpoint.
    """

    @patch(
        "nemo_helix_ext.client.bootstrap.discover_nhx_config",
        return_value=NHXOIDCConfig(auth_enabled=False, client_id="", token_endpoint=""),
    )
    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_expired_token_on_auth_disabled_cluster_does_not_attempt_refresh(self, mock_post, _mock_discover, tmp_path):
        expired_token = _make_jwt({"exp": int(time.time()) - 100, "sub": "user1"})
        config_path = _write_config(tmp_path, token=expired_token, refresh_token="refresh_abc")

        client = build_nemo_client(config_path=config_path)

        assert client.base_url.rstrip("/") == "http://localhost:8080"
        assert _sent_authorization(client) is None
        mock_post.assert_not_called()

    @patch(
        "nemo_helix_ext.client.bootstrap.discover_nhx_config",
        return_value=NHXOIDCConfig(auth_enabled=False, client_id="", token_endpoint=""),
    )
    def test_valid_token_on_auth_disabled_cluster_skips_token_provider(self, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc")

        client = build_nemo_client(config_path=config_path)

        assert client._auth is None
        assert client._http.auth is None or not isinstance(client._http.auth, TokenProviderAuth)
        assert _sent_authorization(client) is None

    @patch(
        "nemo_helix_ext.client.bootstrap.discover_nhx_config",
        side_effect=httpx.ConnectError("network error"),
    )
    def test_discovery_failure_preserves_stored_token(self, _mock_discover, tmp_path):
        # A discovery failure must not strip auth — the stored token may still
        # be valid and should be used as-is without attempting a refresh.
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc")

        client = build_nemo_client(config_path=config_path)

        assert _sent_authorization(client) == f"Bearer {token}"

    @patch(
        "nemo_helix_ext.client.bootstrap.discover_nhx_config",
        side_effect=json.JSONDecodeError("Expecting value", "<html>", 0),
    )
    def test_non_json_discovery_preserves_stored_token(self, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc")

        client = build_nemo_client(config_path=config_path)

        assert _sent_authorization(client) == f"Bearer {token}"

    @patch(
        "nemo_helix_ext.client.bootstrap.discover_nhx_config",
        side_effect=ValueError("OIDC bearer_token_source must be 'access_token' or 'id_token'"),
    )
    def test_discovery_validation_failure_is_not_downgraded_to_fallback(self, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc")

        with pytest.raises(ValueError, match="bearer_token_source"):
            build_nemo_client(config_path=config_path)


class TestBuildNemoClientWorkloadIdentity:
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_WORKLOAD_NHX_CONFIG)
    @patch("nemo_helix_ext.auth.workload_exchange.token_exchange_grant")
    def test_exchanges_workload_identity_token_file(self, mock_exchange, _mock_discover, tmp_path, monkeypatch):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token-one\n", encoding="utf-8")
        access_token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "workload-user"})
        mock_exchange.return_value = {"access_token": access_token, "expires_in": 300}
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

        client = build_nemo_client()

        try:
            assert client.base_url.rstrip("/") == "https://api.example.com"
            assert "Authorization" not in client.default_headers
            _mock_discover.assert_not_called()
            mock_exchange.assert_not_called()

            assert _sent_authorization(client) == f"Bearer {access_token}"
        finally:
            client.close()

        mock_exchange.assert_called_once()
        assert mock_exchange.call_args.kwargs["token_endpoint"] == "https://workload-idp/token"
        assert mock_exchange.call_args.kwargs["client_id"] == "nhx-workload-client-id"
        assert mock_exchange.call_args.kwargs["subject_token"] == "subject-token-one"
        assert mock_exchange.call_args.kwargs["audience"] == "nemo-helix"
        assert mock_exchange.call_args.kwargs["scope"] == "openid email groups"

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_WORKLOAD_NHX_CONFIG)
    @patch("nemo_helix_ext.auth.workload_exchange.token_exchange_grant")
    def test_workload_identity_discovery_uses_context_certificate_authority(
        self, mock_exchange, _mock_discover, tmp_path, monkeypatch
    ):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token-one\n", encoding="utf-8")
        context_ca = str(tmp_path / "context-ca.pem")
        access_token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "workload-user"})
        mock_exchange.return_value = {"access_token": access_token, "expires_in": 300}
        config_path = _write_config(tmp_path, user_type="no-auth", certificate_authority=context_ca)
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
        monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        real_httpx_client = httpx.Client

        def default_httpx_client(*args, **kwargs):
            kwargs["verify"] = True
            return real_httpx_client(*args, **kwargs)

        with patch("nemo_helix_ext.client.bootstrap.httpx.Client", side_effect=default_httpx_client) as client_cls:
            client = build_nemo_client(config_path=config_path)
        try:
            assert _sent_authorization(client) == f"Bearer {access_token}"
        finally:
            client.close()

        assert _mock_discover.call_args.kwargs["certificate_authority"] == context_ca
        assert mock_exchange.call_args.kwargs["certificate_authority"] == context_ca
        assert client_cls.call_args.kwargs["verify"] == context_ca

    @pytest.mark.asyncio
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_WORKLOAD_NHX_CONFIG)
    @patch("nemo_helix_ext.auth.workload_exchange.token_exchange_grant")
    async def test_async_exchanges_workload_identity_token_file_at_request_time(
        self, mock_exchange, _mock_discover, tmp_path, monkeypatch
    ):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token-one\n", encoding="utf-8")
        access_token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "workload-user"})
        mock_exchange.return_value = {"access_token": access_token, "expires_in": 300}
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

        client = build_async_nemo_client()
        seen = _async_wire(client)

        try:
            assert isinstance(client, AsyncNemoClient)
            assert client.base_url.rstrip("/") == "https://api.example.com"
            assert "Authorization" not in client.default_headers
            _mock_discover.assert_not_called()
            mock_exchange.assert_not_called()

            await client.send(probe())
            assert seen[0].headers["Authorization"] == f"Bearer {access_token}"
        finally:
            await client.close()

        mock_exchange.assert_called_once()
        assert mock_exchange.call_args.kwargs["token_endpoint"] == "https://workload-idp/token"
        assert mock_exchange.call_args.kwargs["client_id"] == "nhx-workload-client-id"
        assert mock_exchange.call_args.kwargs["subject_token"] == "subject-token-one"
        assert mock_exchange.call_args.kwargs["audience"] == "nemo-helix"
        assert mock_exchange.call_args.kwargs["scope"] == "openid email groups"

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_WORKLOAD_NHX_CONFIG)
    def test_env_access_token_takes_precedence_over_workload_identity_file(self, _mock_discover, tmp_path, monkeypatch):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token-one\n", encoding="utf-8")
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("NHX_ACCESS_TOKEN", "env-access-token-123")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

        client = build_nemo_client()

        try:
            assert _sent_authorization(client) == "Bearer env-access-token-123"
        finally:
            client.close()


class TestBuildNemoClientApiKey:
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_builds_client_with_api_key(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        client = build_nemo_client(config_path=config_path)

        assert client.base_url.rstrip("/") == "http://localhost:8080"
        assert client.workspace == "test-workspace"
        assert _sent_authorization(client) == "Bearer nvapi-test-key-123"

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_builds_client_with_email_api_key(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="admin@example.com")

        client = build_nemo_client(config_path=config_path)

        token = (_sent_authorization(client) or "").split(" ", 1)[1]
        claims = decode_jwt_claims(token)
        assert claims["sub"] == "admin@example.com"
        assert claims["email"] == "admin@example.com"

    @pytest.mark.asyncio
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    async def test_async_client_uses_config_for_api_key(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        client = build_async_nemo_client(config_path=config_path)
        try:
            assert client.base_url.rstrip("/") == "http://localhost:8080"
            assert client.workspace == "test-workspace"
            assert client._auth is not None
        finally:
            await client.close()


class TestBuildNemoClientNoAuth:
    def test_builds_client_without_auth(self, tmp_path):
        config_path = _write_config(tmp_path, user_type="no-auth")

        client = build_nemo_client(config_path=config_path)

        assert client.base_url.rstrip("/") == "http://localhost:8080"
        assert client.workspace == "test-workspace"
        assert "Authorization" not in client.default_headers
        assert _sent_authorization(client) is None

    @patch("nemo_helix_ext.client.bootstrap.httpx.Client")
    def test_non_oauth_context_uses_context_certificate_authority(self, mock_httpx_client, tmp_path, monkeypatch):
        context_ca = str(tmp_path / "context-ca.pem")
        config_path = _write_config(tmp_path, user_type="no-auth", certificate_authority=context_ca)
        monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        client = build_nemo_client(config_path=config_path)

        assert client.base_url.rstrip("/") == "http://localhost:8080"
        assert mock_httpx_client.call_args.kwargs["verify"] == context_ca


class TestBuildNemoClientProviderReuse:
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    @patch("nemo_helix_ext.client.bootstrap.OIDCTokenProvider")
    def test_reuses_oauth_provider_for_same_context(self, mock_provider_cls, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc")

        provider = MagicMock()
        provider.get_access_token.return_value = token
        provider.reload_tokens.return_value = False
        mock_provider_cls.return_value = provider

        build_nemo_client(config_path=config_path)
        build_nemo_client(config_path=config_path)

        assert mock_provider_cls.call_count == 1
        provider_kwargs = mock_provider_cls.call_args.kwargs
        assert callable(provider_kwargs["load_tokens"])
        assert callable(provider_kwargs["refresh_lock"])

    @patch("nemo_helix_ext.client.bootstrap.httpx.Client")
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    @patch("nemo_helix_ext.client.bootstrap.OIDCTokenProvider")
    def test_context_certificate_authority_participates_in_provider_cache_key(
        self, mock_provider_cls, _mock_discover, _mock_httpx_client, tmp_path, monkeypatch
    ):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        first_ca = str(tmp_path / "first-ca.pem")
        second_ca = str(tmp_path / "second-ca.pem")
        first_provider = MagicMock()
        first_provider.get_access_token.return_value = token
        first_provider.reload_tokens.return_value = False
        second_provider = MagicMock()
        second_provider.get_access_token.return_value = token
        second_provider.reload_tokens.return_value = False
        mock_provider_cls.side_effect = [first_provider, second_provider]
        monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc", certificate_authority=first_ca)
        first_client = build_nemo_client(config_path=config_path)
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc", certificate_authority=second_ca)
        second_client = build_nemo_client(config_path=config_path)
        try:
            assert first_client is not None
            assert second_client is not None
        finally:
            second_client.close()
            first_client.close()

        assert mock_provider_cls.call_count == 2
        certificate_authorities = [call.kwargs["certificate_authority"] for call in mock_provider_cls.call_args_list]
        assert certificate_authorities == [first_ca, second_ca]


class TestBuildNemoClientOverrides:
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_base_url_override_uses_explicit_url_with_context_auth(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        client = build_nemo_client(config_path=config_path, base_url="http://localhost:9090")

        assert client.base_url.rstrip("/") == "http://localhost:9090"
        assert client.workspace == "test-workspace"
        assert _sent_authorization(client) == "Bearer nvapi-test-key-123"

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_context_override_uses_selected_context(self, _mock_discover, tmp_path):
        config_path = _multi_context_config(tmp_path)

        client = build_nemo_client(config_path=config_path, context_name="target-context")

        assert client.base_url.rstrip("/") == "http://localhost:9090"
        assert client.workspace == "workspace-two"
        assert _sent_authorization(client) == "Bearer nvapi-two"

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_async_context_override_uses_selected_context(self, _mock_discover, tmp_path):
        config_path = _multi_context_config(tmp_path)

        client = build_async_nemo_client(config_path=config_path, context_name="target-context")

        assert client.base_url.rstrip("/") == "http://localhost:9090"
        assert client.workspace == "workspace-two"

    def test_context_override_fails_for_missing_context(self, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        with pytest.raises(ValueError, match="Context 'missing-context' not found"):
            build_nemo_client(config_path=config_path, context_name="missing-context")

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_access_token_override_uses_bearer_token(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "override-user"})

        client = build_nemo_client(config_path=config_path, access_token=token)

        assert _sent_authorization(client) == f"Bearer {token}"


class TestBuildNemoClientBootstrapFailures:
    def test_explicit_missing_config_file_fails_fast(self, tmp_path):
        missing_config_path = tmp_path / "missing-config.yaml"

        with pytest.raises(FileNotFoundError, match=f"Config file not found at {missing_config_path}"):
            build_nemo_client(config_path=missing_config_path)

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    def test_expired_oauth_token_without_refresh_token_fails(self, _mock_discover, tmp_path):
        expired_token = _make_jwt({"exp": int(time.time()) - 100, "sub": "user1"})
        config_path = _write_config(tmp_path, token=expired_token, refresh_token=None)

        client = build_nemo_client(config_path=config_path)

        with pytest.raises(RuntimeError, match="no refresh token is available"):
            _sent_authorization(client)

    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    @patch("nemo_helix_ext.auth.token_provider.httpx.post")
    def test_refresh_grant_failure_surfaces_clear_error(self, mock_post, _mock_discover, tmp_path):
        expired_token = _make_jwt({"exp": int(time.time()) - 100, "sub": "user1"})
        config_path = _write_config(tmp_path, token=expired_token, refresh_token="refresh_abc")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "invalid refresh token"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "invalid refresh token",
        }
        mock_post.return_value = mock_response

        client = build_nemo_client(config_path=config_path)

        with pytest.raises(RuntimeError, match=r"Token refresh failed: invalid_grant - invalid refresh token"):
            _sent_authorization(client)


class TestBuildAsyncNemoClientOAuth:
    @pytest.mark.asyncio
    @patch("nemo_helix_ext.client.bootstrap.discover_nhx_config", return_value=_MOCK_NHX_CONFIG)
    async def test_sends_the_stored_token_on_each_request(self, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc")

        client = build_async_nemo_client(config_path=config_path)
        seen = _async_wire(client)
        try:
            assert isinstance(client._http.auth, TokenProviderAuth)
            await client.send(probe())
            assert seen[0].headers["Authorization"] == f"Bearer {token}"
        finally:
            await client.close()
