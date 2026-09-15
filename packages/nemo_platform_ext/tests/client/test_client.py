# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for client_factory module."""

import json
import sys
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml
from nemo_platform import AsyncNeMoPlatform, DefaultHttpxClient, NeMoPlatform, not_given
from nemo_platform_ext.auth.helpers import NMPOIDCConfig, decode_jwt_claims
from nemo_platform_ext.client.factory import create_client
from nemo_platform_ext.client.tls import NMP_CLIENT_SSL_CERT_FILE_ENVVAR
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR


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
    """Write a minimal nmp config file and return its path."""
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


_MOCK_NMP_CONFIG = NMPOIDCConfig(
    auth_enabled=True,
    client_id="nmp-client-id",
    token_endpoint="https://idp/token",
)

_MOCK_WORKLOAD_NMP_CONFIG = NMPOIDCConfig(
    auth_enabled=True,
    client_id="nmp-client-id",
    token_endpoint="https://idp/token",
    workload_token_exchange_enabled=True,
    workload_client_id="nmp-workload-client-id",
    workload_token_endpoint="https://workload-idp/token",
    workload_audience="nemo-platform",
    workload_scope="openid email groups",
)


class TestCreateClientOAuth:
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_creates_client_from_stored_oauth_tokens(self, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
        )

        client = create_client(config_path=config_path)

        assert str(client.base_url).rstrip("/") == "http://localhost:8080"
        assert client.workspace == "test-workspace"

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_event_hook_injects_fresh_token(self, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
        )

        client = create_client(config_path=config_path)

        httpx_client = client._client
        assert len(httpx_client._event_hooks["request"]) == 1

        request = httpx_client.build_request("GET", "http://localhost:8080/test")
        httpx_client._event_hooks["request"][0](request)
        assert request.headers["Authorization"] == f"Bearer {token}"

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_oauth_uses_sdk_default_httpx_client(self, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
        )

        client = create_client(config_path=config_path)
        try:
            assert isinstance(client._client, DefaultHttpxClient)
        finally:
            client.close()

    @patch("nemo_platform_ext.client.factory.DefaultHttpxClient")
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_oauth_uses_nemo_scoped_ca_bundle(self, _mock_discover, mock_default_httpx_client, tmp_path, monkeypatch):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
        )
        http_client = httpx.Client()
        mock_default_httpx_client.return_value = http_client
        monkeypatch.setenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        client = create_client(config_path=config_path)
        try:
            assert client is not None
        finally:
            client.close()

        assert mock_default_httpx_client.call_args.kwargs["verify"] == "/tmp/nemo-ca.pem"

    @patch("nemo_platform_ext.client.factory.DefaultHttpxClient")
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_oauth_uses_context_certificate_authority(
        self, _mock_discover, mock_default_httpx_client, tmp_path, monkeypatch
    ):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        context_ca = str(tmp_path / "context-ca.pem")
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
            certificate_authority=context_ca,
        )
        http_client = httpx.Client()
        mock_default_httpx_client.return_value = http_client
        monkeypatch.delenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        client = create_client(config_path=config_path)
        try:
            assert client is not None
        finally:
            client.close()

        assert mock_default_httpx_client.call_args.kwargs["verify"] == context_ca
        assert _mock_discover.call_args.kwargs["certificate_authority"] == context_ca

    @patch("nemo_platform_ext.client.factory.DefaultHttpxClient")
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_env_ca_bundle_overrides_context_certificate_authority(
        self, _mock_discover, mock_default_httpx_client, tmp_path, monkeypatch
    ):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
            certificate_authority="/tmp/context-ca.pem",
        )
        http_client = httpx.Client()
        mock_default_httpx_client.return_value = http_client
        monkeypatch.setenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/env-ca.pem")

        client = create_client(config_path=config_path)
        try:
            assert client is not None
        finally:
            client.close()

        assert mock_default_httpx_client.call_args.kwargs["verify"] == "/tmp/env-ca.pem"
        assert _mock_discover.call_args.kwargs["certificate_authority"] == "/tmp/context-ca.pem"

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    @patch("nemo_platform_ext.auth.token_provider.httpx.post")
    def test_persist_refreshed_tokens_writes_to_config(self, mock_post, _mock_discover, tmp_path):
        expired_token = _make_jwt({"exp": int(time.time()) - 100, "sub": "user1"})
        new_token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})

        config_path = _write_config(
            tmp_path,
            token=expired_token,
            refresh_token="refresh_abc",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": new_token,
            "refresh_token": "new_refresh",
        }
        mock_post.return_value = mock_response

        client = create_client(config_path=config_path)

        assert client is not None
        mock_post.assert_called_once()

        with open(config_path) as f:
            saved_config = yaml.safe_load(f)
        saved_user = saved_config["users"][0]
        assert saved_user["token"] == new_token
        assert saved_user["refresh_token"] == "new_refresh"

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_explicit_access_token_overrides_config_auth(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        client = create_client(config_path=config_path, access_token="explicit-access-token-123")

        request = client._client.build_request("GET", "http://localhost:8080/test")
        client._client._event_hooks["request"][0](request)
        assert request.headers["Authorization"] == "Bearer explicit-access-token-123"


class TestCreateClientOAuthUserAuthDisabledCluster:
    """Regression tests: OAuthUser context pointed at a cluster with no auth.

    This happens when a config context was previously authenticated against a
    real OIDC cluster and is later pointed at a local/no-auth instance. Once
    the stored access token expires, the client must not attempt a
    refresh_token grant against an empty token endpoint.
    """

    @patch(
        "nemo_platform.client.bootstrap.discover_nmp_config",
        return_value=NMPOIDCConfig(auth_enabled=False, client_id="", token_endpoint=""),
    )
    @patch(
        "nemo_platform_ext.client.bootstrap.discover_nmp_config",
        return_value=NMPOIDCConfig(auth_enabled=False, client_id="", token_endpoint=""),
    )
    @patch("nemo_platform_ext.auth.token_provider.httpx.post")
    def test_expired_token_on_auth_disabled_cluster_does_not_attempt_refresh(
        self, mock_post, _mock_ext_discover, _mock_sdk_discover, tmp_path
    ):
        expired_token = _make_jwt({"exp": int(time.time()) - 100, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=expired_token,
            refresh_token="refresh_abc",
        )

        client = create_client(config_path=config_path)

        assert str(client.base_url).rstrip("/") == "http://localhost:8080"
        mock_post.assert_not_called()
        assert "Authorization" not in client._custom_headers

    @patch(
        "nemo_platform.client.bootstrap.discover_nmp_config",
        return_value=NMPOIDCConfig(auth_enabled=False, client_id="", token_endpoint=""),
    )
    @patch(
        "nemo_platform_ext.client.bootstrap.discover_nmp_config",
        return_value=NMPOIDCConfig(auth_enabled=False, client_id="", token_endpoint=""),
    )
    def test_valid_token_on_auth_disabled_cluster_skips_token_provider(
        self, _mock_ext_discover, _mock_sdk_discover, tmp_path
    ):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
        )

        client = create_client(config_path=config_path)

        assert "Authorization" not in client._custom_headers
        assert client._client._event_hooks["request"] == []

    @patch(
        "nemo_platform_ext.client.bootstrap.discover_nmp_config",
        side_effect=Exception("network error"),
    )
    def test_discovery_failure_preserves_stored_token(self, _mock_discover, tmp_path):
        # A discovery failure must not strip auth — the stored token may still
        # be valid and should be used as-is without attempting a refresh.
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(tmp_path, token=token, refresh_token="refresh_abc")

        client = create_client(config_path=config_path)

        request = client._client.build_request("GET", "http://localhost:8080/test")
        client._client._event_hooks["request"][0](request)
        assert request.headers["Authorization"] == f"Bearer {token}"


class TestCreateClientWorkloadIdentity:
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_WORKLOAD_NMP_CONFIG)
    @patch("nemo_platform_ext.auth.workload_exchange.token_exchange_grant")
    def test_exchanges_workload_identity_token_file(self, mock_exchange, _mock_discover, tmp_path, monkeypatch):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token-one\n", encoding="utf-8")
        access_token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "workload-user"})
        mock_exchange.return_value = {"access_token": access_token, "expires_in": 300}
        monkeypatch.setenv("NMP_BASE_URL", "https://api.example.com")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

        client = create_client()

        try:
            assert str(client.base_url).rstrip("/") == "https://api.example.com"
            assert "Authorization" not in client._custom_headers
            _mock_discover.assert_not_called()
            mock_exchange.assert_not_called()

            request = client._client.build_request("GET", "https://api.example.com/test")
            client._client._event_hooks["request"][0](request)
            assert request.headers["Authorization"] == f"Bearer {access_token}"
        finally:
            client.close()

        mock_exchange.assert_called_once()
        assert mock_exchange.call_args.kwargs["token_endpoint"] == "https://workload-idp/token"
        assert mock_exchange.call_args.kwargs["client_id"] == "nmp-workload-client-id"
        assert mock_exchange.call_args.kwargs["subject_token"] == "subject-token-one"
        assert mock_exchange.call_args.kwargs["audience"] == "nemo-platform"
        assert mock_exchange.call_args.kwargs["scope"] == "openid email groups"

    @patch("nemo_platform_ext.client.factory.DefaultHttpxClient")
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_WORKLOAD_NMP_CONFIG)
    @patch("nemo_platform_ext.auth.workload_exchange.token_exchange_grant")
    def test_workload_identity_discovery_uses_context_certificate_authority(
        self, mock_exchange, _mock_discover, mock_default_httpx_client, tmp_path, monkeypatch
    ):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token-one\n", encoding="utf-8")
        context_ca = str(tmp_path / "context-ca.pem")
        access_token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "workload-user"})
        mock_exchange.return_value = {"access_token": access_token, "expires_in": 300}
        config_path = _write_config(tmp_path, user_type="no-auth", certificate_authority=context_ca)
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
        monkeypatch.delenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        def default_httpx_client(*args, **kwargs):
            kwargs["verify"] = True
            return httpx.Client(*args, **kwargs)

        mock_default_httpx_client.side_effect = default_httpx_client

        client = create_client(config_path=config_path)
        try:
            request = client._client.build_request("GET", "http://localhost:8080/test")
            client._client._event_hooks["request"][0](request)
            assert request.headers["Authorization"] == f"Bearer {access_token}"
        finally:
            client.close()

        assert _mock_discover.call_args.kwargs["certificate_authority"] == context_ca
        assert mock_exchange.call_args.kwargs["certificate_authority"] == context_ca
        assert mock_default_httpx_client.call_args.kwargs["verify"] == context_ca

    @pytest.mark.asyncio
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_WORKLOAD_NMP_CONFIG)
    @patch("nemo_platform_ext.auth.workload_exchange.token_exchange_grant")
    async def test_async_exchanges_workload_identity_token_file_at_request_time(
        self, mock_exchange, _mock_discover, tmp_path, monkeypatch
    ):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token-one\n", encoding="utf-8")
        access_token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "workload-user"})
        mock_exchange.return_value = {"access_token": access_token, "expires_in": 300}
        monkeypatch.setenv("NMP_BASE_URL", "https://api.example.com")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

        client = AsyncNeMoPlatform()

        try:
            assert str(client.base_url).rstrip("/") == "https://api.example.com"
            assert "Authorization" not in client._custom_headers
            _mock_discover.assert_not_called()
            mock_exchange.assert_not_called()

            request = client._client.build_request("GET", "https://api.example.com/test")
            await client._client._event_hooks["request"][0](request)
            assert request.headers["Authorization"] == f"Bearer {access_token}"
        finally:
            await client.close()

        mock_exchange.assert_called_once()
        assert mock_exchange.call_args.kwargs["token_endpoint"] == "https://workload-idp/token"
        assert mock_exchange.call_args.kwargs["client_id"] == "nmp-workload-client-id"
        assert mock_exchange.call_args.kwargs["subject_token"] == "subject-token-one"
        assert mock_exchange.call_args.kwargs["audience"] == "nemo-platform"
        assert mock_exchange.call_args.kwargs["scope"] == "openid email groups"

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_WORKLOAD_NMP_CONFIG)
    def test_env_access_token_takes_precedence_over_workload_identity_file(self, _mock_discover, tmp_path, monkeypatch):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token-one\n", encoding="utf-8")
        monkeypatch.setenv("NMP_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("NMP_ACCESS_TOKEN", "env-access-token-123")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

        client = create_client()

        try:
            request = client._client.build_request("GET", "https://api.example.com/test")
            client._client._event_hooks["request"][0](request)
            assert request.headers["Authorization"] == "Bearer env-access-token-123"
        finally:
            client.close()


class TestCreateClientApiKey:
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_creates_client_with_api_key(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        client = create_client(config_path=config_path)

        assert str(client.base_url).rstrip("/") == "http://localhost:8080"
        assert client.workspace == "test-workspace"
        assert "Authorization" in client._custom_headers
        assert client._custom_headers["Authorization"] == "Bearer nvapi-test-key-123"

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_creates_client_with_email_api_key(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="admin@example.com")

        client = create_client(config_path=config_path)

        assert "Authorization" in client._custom_headers
        token = client._custom_headers["Authorization"].split(" ", 1)[1]
        claims = decode_jwt_claims(token)
        assert claims["sub"] == "admin@example.com"
        assert claims["email"] == "admin@example.com"


class TestCreateClientNoAuth:
    def test_creates_client_without_auth(self, tmp_path):
        config_path = _write_config(tmp_path, user_type="no-auth")

        client = create_client(config_path=config_path)

        assert str(client.base_url).rstrip("/") == "http://localhost:8080"
        assert client.workspace == "test-workspace"
        # No auth headers should be set
        assert "Authorization" not in client._custom_headers

    @patch("nemo_platform_ext.client.factory.DefaultHttpxClient")
    def test_non_oauth_context_uses_context_certificate_authority(
        self, mock_default_httpx_client, tmp_path, monkeypatch
    ):
        context_ca = str(tmp_path / "context-ca.pem")
        config_path = _write_config(
            tmp_path,
            user_type="no-auth",
            certificate_authority=context_ca,
        )
        http_client = httpx.Client()
        mock_default_httpx_client.return_value = http_client
        monkeypatch.delenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        client = create_client(config_path=config_path)
        try:
            assert str(client.base_url).rstrip("/") == "http://localhost:8080"
        finally:
            client.close()

        mock_default_httpx_client.assert_called_once_with(verify=context_ca)


class TestCreateClientTimeout:
    @patch("nemo_platform_ext.client.factory.NeMoPlatform")
    def test_default_timeout_preserves_sdk_constructor_default(self, mock_client_ctor, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        create_client(config_path=config_path)

        assert mock_client_ctor.call_args.kwargs["timeout"] is not_given

    @patch("nemo_platform_ext.client.factory.NeMoPlatform")
    def test_explicit_timeout_is_forwarded(self, mock_client_ctor, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        create_client(config_path=config_path, timeout=42.0)

        assert mock_client_ctor.call_args.kwargs["timeout"] == 42.0


class TestCreateClientProviderReuse:
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    @patch("nemo_platform_ext.client.bootstrap.OIDCTokenProvider")
    def test_reuses_oauth_provider_for_same_context(self, mock_provider_cls, _mock_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
        )

        provider = MagicMock()
        provider.get_access_token.return_value = token
        provider.reload_tokens.return_value = False
        mock_provider_cls.return_value = provider

        create_client(config_path=config_path)
        create_client(config_path=config_path)

        assert mock_provider_cls.call_count == 1
        provider_kwargs = mock_provider_cls.call_args.kwargs
        assert callable(provider_kwargs["load_tokens"])
        assert callable(provider_kwargs["refresh_lock"])

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    @patch("nemo_platform_ext.client.factory.DefaultHttpxClient")
    @patch("nemo_platform_ext.client.bootstrap.OIDCTokenProvider")
    def test_context_certificate_authority_participates_in_provider_cache_key(
        self, mock_provider_cls, mock_default_httpx_client, _mock_discover, tmp_path, monkeypatch
    ):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        first_ca = str(tmp_path / "first-ca.pem")
        second_ca = str(tmp_path / "second-ca.pem")
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
            certificate_authority=first_ca,
        )
        first_provider = MagicMock()
        first_provider.get_access_token.return_value = token
        first_provider.reload_tokens.return_value = False
        second_provider = MagicMock()
        second_provider.get_access_token.return_value = token
        second_provider.reload_tokens.return_value = False
        mock_provider_cls.side_effect = [first_provider, second_provider]
        mock_default_httpx_client.side_effect = [httpx.Client(), httpx.Client()]
        monkeypatch.delenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        first_client = create_client(config_path=config_path)
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
            certificate_authority=second_ca,
        )
        second_client = create_client(config_path=config_path)
        try:
            assert first_client is not None
            assert second_client is not None
        finally:
            second_client.close()
            first_client.close()

        assert mock_provider_cls.call_count == 2
        certificate_authorities = [call.kwargs["certificate_authority"] for call in mock_provider_cls.call_args_list]
        assert certificate_authorities == [first_ca, second_ca]


class TestCreateClientOverrides:
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_base_url_override_uses_explicit_url_with_context_auth(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        client = create_client(config_path=config_path, base_url="http://localhost:9090")

        assert str(client.base_url).rstrip("/") == "http://localhost:9090"
        assert client.workspace == "test-workspace"
        assert client._custom_headers["Authorization"] == "Bearer nvapi-test-key-123"

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_context_override_uses_selected_context(self, _mock_discover, tmp_path):
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
                {
                    "name": "default",
                    "cluster": "cluster-one",
                    "user": "user-one",
                    "workspace": "workspace-one",
                },
                {
                    "name": "target-context",
                    "cluster": "cluster-two",
                    "user": "user-two",
                    "workspace": "workspace-two",
                },
            ],
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        client = create_client(config_path=config_path, context_name="target-context")

        assert str(client.base_url).rstrip("/") == "http://localhost:9090"
        assert client.workspace == "workspace-two"
        assert client._custom_headers["Authorization"] == "Bearer nvapi-two"

    def test_context_override_fails_for_missing_context(self, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        with pytest.raises(ValueError, match="Context 'missing-context' not found"):
            create_client(config_path=config_path, context_name="missing-context")

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_access_token_override_uses_bearer_token(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "override-user"})

        client = create_client(config_path=config_path, access_token=token)

        request = client._client.build_request("GET", "http://localhost:8080/test")
        client._client._event_hooks["request"][0](request)
        assert request.headers["Authorization"] == f"Bearer {token}"


class TestCreateClientBootstrapFailures:
    def test_explicit_missing_config_file_fails_fast(self, tmp_path):
        missing_config_path = tmp_path / "missing-config.yaml"

        with pytest.raises(FileNotFoundError, match=f"Config file not found at {missing_config_path}"):
            create_client(config_path=missing_config_path)

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    def test_expired_oauth_token_without_refresh_token_fails(self, _mock_discover, tmp_path):
        expired_token = _make_jwt({"exp": int(time.time()) - 100, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=expired_token,
            refresh_token=None,
        )

        with pytest.raises(RuntimeError, match="no refresh token is available"):
            create_client(config_path=config_path)

    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    @patch("nemo_platform_ext.auth.token_provider.httpx.post")
    def test_refresh_grant_failure_surfaces_clear_error(self, mock_post, _mock_discover, tmp_path):
        expired_token = _make_jwt({"exp": int(time.time()) - 100, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=expired_token,
            refresh_token="refresh_abc",
        )

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "invalid refresh token"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "invalid refresh token",
        }
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match=r"Token refresh failed: invalid_grant - invalid refresh token"):
            create_client(config_path=config_path)


class TestClientConstructorBootstrapBypass:
    @patch("nemo_platform_ext.client.factory.build_client_init_kwargs")
    def test_sync_constructor_with_base_url_skips_config_bootstrap(self, mock_build_client_kwargs):
        mock_build_client_kwargs.side_effect = AssertionError("bootstrap should not be called")

        client = NeMoPlatform(base_url="http://override-host:8081", workspace="test-workspace")
        try:
            assert str(client.base_url).rstrip("/") == "http://override-host:8081"
            assert client.workspace == "test-workspace"
        finally:
            client.close()

        mock_build_client_kwargs.assert_not_called()

    @patch("nemo_platform_ext.client.factory.build_client_init_kwargs")
    def test_sync_constructor_env_base_url_still_bootstraps_when_base_url_omitted(
        self, mock_build_client_kwargs, monkeypatch
    ):
        monkeypatch.setenv("NEMO_PLATFORM_BASE_URL", "http://env-host:8081")
        mock_build_client_kwargs.return_value = MagicMock(
            base_url="http://env-host:8081",
            workspace="test-workspace",
            default_headers=None,
            http_client=None,
            client_verify=True,
        )

        client = NeMoPlatform()
        try:
            assert str(client.base_url).rstrip("/") == "http://env-host:8081"
            assert client.workspace == "test-workspace"
        finally:
            client.close()

        assert mock_build_client_kwargs.call_args.kwargs["base_url"] == "http://env-host:8081"

    @patch("nemo_platform._client.DefaultHttpxClient")
    @patch("nemo_platform_ext.client.factory.build_client_init_kwargs")
    def test_sync_constructor_direct_mode_uses_nemo_scoped_ca_bundle(
        self, mock_build_client_kwargs, mock_default_httpx_client, monkeypatch
    ):
        mock_build_client_kwargs.side_effect = AssertionError("bootstrap should not be called")
        mock_default_httpx_client.return_value = httpx.Client()
        monkeypatch.setenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        client = NeMoPlatform(base_url="https://override-host:8081", workspace="test-workspace")
        try:
            assert str(client.base_url).rstrip("/") == "https://override-host:8081"
        finally:
            client.close()

        mock_build_client_kwargs.assert_not_called()
        mock_default_httpx_client.assert_called_once_with(verify="/tmp/nemo-ca.pem")

    @patch("nemo_platform._client.DefaultHttpxClient")
    @patch("nemo_platform_ext.client.factory.build_client_init_kwargs")
    def test_sync_constructor_uses_context_certificate_authority(
        self, mock_build_client_kwargs, mock_default_httpx_client, monkeypatch
    ):
        mock_build_client_kwargs.return_value = MagicMock(
            base_url="https://config-host:8443",
            workspace="test-workspace",
            default_headers=None,
            http_client=None,
            client_verify="/ctx/ca.pem",
        )
        mock_default_httpx_client.return_value = httpx.Client()
        monkeypatch.delenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        client = NeMoPlatform(config_path=Path("/tmp/config.yaml"))
        try:
            assert str(client.base_url).rstrip("/") == "https://config-host:8443"
            assert client.workspace == "test-workspace"
        finally:
            client.close()

        mock_default_httpx_client.assert_called_once_with(verify="/ctx/ca.pem")

    @patch("nemo_platform_ext.client.factory.build_client_init_kwargs")
    def test_sync_copy_with_access_token_bootstraps_instead_of_reusing_http_client(self, mock_build_client_kwargs):
        mock_build_client_kwargs.return_value = MagicMock(
            base_url="http://override-host:8081",
            workspace="test-workspace",
            default_headers={"Authorization": "Bearer replacement-token"},
            http_client=None,
            client_verify=True,
        )

        client = NeMoPlatform(base_url="http://original-host:8081", workspace="original-workspace")
        original_http_client = client._client
        copied = client.copy(access_token="replacement-token")
        try:
            assert copied._client is not original_http_client
        finally:
            copied.close()
            client.close()

        assert mock_build_client_kwargs.call_args.kwargs["access_token"] == "replacement-token"

    @patch("nemo_platform_ext.client.factory.build_client_init_kwargs")
    def test_sync_constructor_with_workload_file_and_base_url_bootstraps(self, mock_build_client_kwargs, monkeypatch):
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, "/var/run/secrets/nemo-platform/workload/token")
        mock_build_client_kwargs.return_value = MagicMock(
            base_url="http://override-host:8081",
            workspace="test-workspace",
            default_headers=None,
            http_client=None,
            client_verify=True,
        )

        client = NeMoPlatform(base_url="http://override-host:8081", workspace="test-workspace")
        try:
            assert str(client.base_url).rstrip("/") == "http://override-host:8081"
        finally:
            client.close()

        assert mock_build_client_kwargs.call_args.kwargs["base_url"] == "http://override-host:8081"

    @patch("nemo_platform_ext.client.factory.build_client_init_kwargs")
    def test_sync_constructor_passes_context_name_to_bootstrap(self, mock_build_client_kwargs):
        mock_build_client_kwargs.return_value = MagicMock(
            base_url="http://override-host:8081",
            workspace="test-workspace",
            default_headers=None,
            http_client=None,
            client_verify=True,
        )

        client = NeMoPlatform(config_path=Path("/tmp/config.yaml"), context_name="target-context")
        try:
            assert str(client.base_url).rstrip("/") == "http://override-host:8081"
            assert client.workspace == "test-workspace"
        finally:
            client.close()

        assert mock_build_client_kwargs.call_args.kwargs["context_name"] == "target-context"

    def test_sync_constructor_rejects_legacy_context_argument(self):
        with pytest.raises(TypeError, match="unexpected keyword argument 'context'"):
            NeMoPlatform(context="ctx-b")  # ty: ignore[unknown-argument]

    @patch("nemo_platform_ext.client.factory.build_client_init_kwargs")
    def test_sync_constructor_with_http_client_skips_config_bootstrap(self, mock_build_client_kwargs):
        mock_build_client_kwargs.side_effect = AssertionError("bootstrap should not be called")

        http_client = httpx.Client(base_url="http://override-host:8081")
        client = NeMoPlatform(
            base_url="http://override-host:8081",
            workspace="test-workspace",
            http_client=http_client,
        )
        try:
            assert str(client.base_url).rstrip("/") == "http://override-host:8081"
            assert client.workspace == "test-workspace"
        finally:
            client.close()

        mock_build_client_kwargs.assert_not_called()

    @pytest.mark.asyncio
    @patch("nemo_platform_ext.client.factory.build_async_client_init_kwargs")
    async def test_async_constructor_with_base_url_skips_config_bootstrap(self, mock_build_client_kwargs):
        mock_build_client_kwargs.side_effect = AssertionError("bootstrap should not be called")

        client = AsyncNeMoPlatform(base_url="http://override-host:8081", workspace="test-workspace")
        try:
            assert str(client.base_url).rstrip("/") == "http://override-host:8081"
            assert client.workspace == "test-workspace"
        finally:
            await client.close()

        mock_build_client_kwargs.assert_not_called()

    @pytest.mark.asyncio
    @patch("nemo_platform_ext.client.factory.build_async_client_init_kwargs")
    async def test_async_constructor_env_base_url_still_bootstraps_when_base_url_omitted(
        self, mock_build_client_kwargs, monkeypatch
    ):
        monkeypatch.setenv("NEMO_PLATFORM_BASE_URL", "http://env-host:8081")
        mock_build_client_kwargs.return_value = MagicMock(
            base_url="http://env-host:8081",
            workspace="test-workspace",
            default_headers=None,
            http_client=None,
            client_verify=True,
        )

        client = AsyncNeMoPlatform()
        try:
            assert str(client.base_url).rstrip("/") == "http://env-host:8081"
            assert client.workspace == "test-workspace"
        finally:
            await client.close()

        assert mock_build_client_kwargs.call_args.kwargs["base_url"] == "http://env-host:8081"

    @pytest.mark.asyncio
    @patch("nemo_platform._client.DefaultAsyncHttpxClient")
    @patch("nemo_platform_ext.client.factory.build_async_client_init_kwargs")
    async def test_async_constructor_direct_mode_uses_nemo_scoped_ca_bundle(
        self, mock_build_client_kwargs, mock_default_httpx_client, monkeypatch
    ):
        mock_build_client_kwargs.side_effect = AssertionError("bootstrap should not be called")
        mock_default_httpx_client.return_value = httpx.AsyncClient()
        monkeypatch.setenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        client = AsyncNeMoPlatform(base_url="https://override-host:8081", workspace="test-workspace")
        try:
            assert str(client.base_url).rstrip("/") == "https://override-host:8081"
        finally:
            await client.close()

        mock_build_client_kwargs.assert_not_called()
        mock_default_httpx_client.assert_called_once_with(verify="/tmp/nemo-ca.pem")

    @pytest.mark.asyncio
    @patch("nemo_platform._client.DefaultAsyncHttpxClient")
    @patch("nemo_platform_ext.client.factory.build_async_client_init_kwargs")
    async def test_async_constructor_uses_context_certificate_authority(
        self, mock_build_client_kwargs, mock_default_httpx_client, monkeypatch
    ):
        mock_build_client_kwargs.return_value = MagicMock(
            base_url="https://config-host:8443",
            workspace="test-workspace",
            default_headers=None,
            http_client=None,
            client_verify="/ctx/ca.pem",
        )
        mock_default_httpx_client.return_value = httpx.AsyncClient()
        monkeypatch.delenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        client = AsyncNeMoPlatform(config_path=Path("/tmp/config.yaml"))
        try:
            assert str(client.base_url).rstrip("/") == "https://config-host:8443"
            assert client.workspace == "test-workspace"
        finally:
            await client.close()

        mock_default_httpx_client.assert_called_once_with(verify="/ctx/ca.pem")

    @pytest.mark.asyncio
    @patch("nemo_platform_ext.client.factory.build_async_client_init_kwargs")
    async def test_async_copy_with_access_token_bootstraps_instead_of_reusing_http_client(
        self, mock_build_client_kwargs
    ):
        mock_build_client_kwargs.return_value = MagicMock(
            base_url="http://override-host:8081",
            workspace="test-workspace",
            default_headers={"Authorization": "Bearer replacement-token"},
            http_client=None,
            client_verify=True,
        )

        client = AsyncNeMoPlatform(base_url="http://original-host:8081", workspace="original-workspace")
        original_http_client = client._client
        copied = client.copy(access_token="replacement-token")
        try:
            assert copied._client is not original_http_client
        finally:
            await copied.close()
            await client.close()

        assert mock_build_client_kwargs.call_args.kwargs["access_token"] == "replacement-token"

    @pytest.mark.asyncio
    @patch("nemo_platform_ext.client.factory.build_async_client_init_kwargs")
    async def test_async_constructor_with_workload_file_and_base_url_bootstraps(
        self, mock_build_client_kwargs, monkeypatch
    ):
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, "/var/run/secrets/nemo-platform/workload/token")
        mock_build_client_kwargs.return_value = MagicMock(
            base_url="http://override-host:8081",
            workspace="test-workspace",
            default_headers=None,
            http_client=None,
            client_verify=True,
        )

        client = AsyncNeMoPlatform(base_url="http://override-host:8081", workspace="test-workspace")
        try:
            assert str(client.base_url).rstrip("/") == "http://override-host:8081"
        finally:
            await client.close()

        assert mock_build_client_kwargs.call_args.kwargs["base_url"] == "http://override-host:8081"

    @pytest.mark.asyncio
    @patch("nemo_platform_ext.client.factory.build_async_client_init_kwargs")
    async def test_async_constructor_with_http_client_skips_config_bootstrap(self, mock_build_client_kwargs):
        mock_build_client_kwargs.side_effect = AssertionError("bootstrap should not be called")

        http_client = httpx.AsyncClient(base_url="http://override-host:8081")
        client = AsyncNeMoPlatform(
            base_url="http://override-host:8081",
            workspace="test-workspace",
            http_client=http_client,
        )
        try:
            assert str(client.base_url).rstrip("/") == "http://override-host:8081"
            assert client.workspace == "test-workspace"
        finally:
            await client.close()

        mock_build_client_kwargs.assert_not_called()

    @pytest.mark.asyncio
    @patch("nemo_platform_ext.client.factory.build_async_client_init_kwargs")
    async def test_async_constructor_passes_context_name_to_bootstrap(self, mock_build_client_kwargs):
        mock_build_client_kwargs.return_value = MagicMock(
            base_url="http://override-host:8081",
            workspace="test-workspace",
            default_headers=None,
            http_client=None,
            client_verify=True,
        )

        client = AsyncNeMoPlatform(config_path=Path("/tmp/config.yaml"), context_name="target-context")
        try:
            assert str(client.base_url).rstrip("/") == "http://override-host:8081"
            assert client.workspace == "test-workspace"
        finally:
            await client.close()

        assert mock_build_client_kwargs.call_args.kwargs["context_name"] == "target-context"


class TestAsyncNeMoPlatformInit:
    @pytest.mark.asyncio
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    async def test_async_client_uses_config_for_api_key(self, _mock_discover, tmp_path):
        config_path = _write_config(tmp_path, user_type="api-key", api_key="nvapi-test-key-123")

        client = AsyncNeMoPlatform(config_path=config_path)
        try:
            assert str(client.base_url).rstrip("/") == "http://localhost:8080"
            assert client.workspace == "test-workspace"
            assert client._custom_headers["Authorization"] == "Bearer nvapi-test-key-123"
        finally:
            await client.close()


class TestPluginSDKMounting:
    def test_sync_client_mounts_plugin_resource(self):
        plugin_resource = MagicMock(name="plugin-resource")
        plugin_container = MagicMock()
        plugin_container.sync_resource.return_value = plugin_resource
        plugin_discovery = SimpleNamespace(discover_sdk=MagicMock(return_value={"example": plugin_container}))

        with patch.dict(sys.modules, {"nemo_platform_plugin.discovery": plugin_discovery}):
            client = NeMoPlatform(base_url="http://localhost:8080", workspace="test-workspace")

            assert client.example is plugin_resource
            assert client.example is plugin_resource
            client.close()

        plugin_container.sync_resource.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_client_mounts_plugin_resource(self):
        plugin_resource = MagicMock(name="async-plugin-resource")
        plugin_container = MagicMock()
        plugin_container.async_resource.return_value = plugin_resource
        plugin_discovery = SimpleNamespace(discover_sdk=MagicMock(return_value={"example": plugin_container}))

        with patch.dict(sys.modules, {"nemo_platform_plugin.discovery": plugin_discovery}):
            client = AsyncNeMoPlatform(base_url="http://localhost:8080", workspace="test-workspace")

            assert client.example is plugin_resource
            assert client.example is plugin_resource
            await client.close()

        plugin_container.async_resource.assert_called_once()

    def test_sync_client_raises_attribute_error_for_async_only_plugin(self):
        plugin_container = SimpleNamespace(sync_resource=None, async_resource=MagicMock())
        plugin_discovery = SimpleNamespace(discover_sdk=MagicMock(return_value={"example": plugin_container}))

        with patch.dict(sys.modules, {"nemo_platform_plugin.discovery": plugin_discovery}):
            client = NeMoPlatform(base_url="http://localhost:8080", workspace="test-workspace")

            with pytest.raises(AttributeError, match="example"):
                _ = client.example

            client.close()

    @pytest.mark.asyncio
    async def test_async_client_raises_attribute_error_for_sync_only_plugin(self):
        plugin_container = SimpleNamespace(sync_resource=MagicMock(), async_resource=None)
        plugin_discovery = SimpleNamespace(discover_sdk=MagicMock(return_value={"example": plugin_container}))

        with patch.dict(sys.modules, {"nemo_platform_plugin.discovery": plugin_discovery}):
            client = AsyncNeMoPlatform(base_url="http://localhost:8080", workspace="test-workspace")

            with pytest.raises(AttributeError, match="example"):
                _ = client.example

            await client.close()

    @pytest.mark.asyncio
    @patch("nemo_platform.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    @patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_MOCK_NMP_CONFIG)
    async def test_async_client_oauth_hook_injects_fresh_token(self, _mock_ext_discover, _mock_sdk_discover, tmp_path):
        token = _make_jwt({"exp": int(time.time()) + 3600, "sub": "user1"})
        config_path = _write_config(
            tmp_path,
            token=token,
            refresh_token="refresh_abc",
        )

        client = AsyncNeMoPlatform(config_path=config_path)
        try:
            httpx_client = client._client
            assert len(httpx_client._event_hooks["request"]) == 1

            request = httpx_client.build_request("GET", "http://localhost:8080/test")
            await httpx_client._event_hooks["request"][0](request)
            assert request.headers["Authorization"] == f"Bearer {token}"
        finally:
            await client.close()
