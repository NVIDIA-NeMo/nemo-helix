# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from nemo_helix_ext.auth.helpers import (
    AuthError,
    NHXOIDCConfig,
    build_effective_scope,
    decode_jwt_claims,
    decode_jwt_header,
    discover_nhx_config,
    generate_unsigned_jwt,
    is_unsigned_jwt,
    normalize_scope_prefix,
    validate_requested_scopes_granted,
)
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR
from pytest_httpserver import HTTPServer


def _make_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    signature = base64.urlsafe_b64encode(b"fake-signature").rstrip(b"=").decode()
    return f"{header}.{body}.{signature}"


def _make_jwt_with_alg(payload: dict, alg: str) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": alg}).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    signature = base64.urlsafe_b64encode(b"fake-signature").rstrip(b"=").decode()
    return f"{header}.{body}.{signature}"


class TestDecodeJwtClaims:
    def test_decodes_valid_jwt(self):
        token = _make_jwt({"sub": "user-1", "exp": 9999999999})
        claims = decode_jwt_claims(token)
        assert claims["sub"] == "user-1"
        assert claims["exp"] == 9999999999

    def test_returns_empty_for_non_jwt(self):
        assert decode_jwt_claims("not-a-jwt") == {}

    def test_returns_empty_for_two_parts(self):
        assert decode_jwt_claims("header.payload") == {}

    def test_returns_empty_for_empty_string(self):
        assert decode_jwt_claims("") == {}

    def test_returns_empty_for_invalid_base64(self):
        assert decode_jwt_claims("a.!!!invalid!!!.c") == {}

    def test_handles_missing_padding(self):
        payload = {"email": "test@example.com"}
        token = _make_jwt(payload)
        claims = decode_jwt_claims(token)
        assert claims["email"] == "test@example.com"


class TestDecodeJwtHeader:
    def test_decodes_valid_header(self):
        token = _make_jwt({"sub": "user-1"})
        header = decode_jwt_header(token)
        assert header["alg"] == "RS256"

    def test_returns_empty_for_non_jwt(self):
        assert decode_jwt_header("not-a-jwt") == {}


class TestIsUnsignedJwt:
    def test_true_for_alg_none(self):
        token = _make_jwt_with_alg({"sub": "user-1"}, "none")
        assert is_unsigned_jwt(token) is True

    def test_false_for_signed_alg(self):
        token = _make_jwt_with_alg({"sub": "user-1"}, "RS256")
        assert is_unsigned_jwt(token) is False


class TestGenerateUnsignedJwt:
    def test_generates_unsigned_token_with_expected_claims(self):
        token = generate_unsigned_jwt(
            principal_id="user-123",
            email="user@example.com",
            groups=["dev", "admin"],
            scopes=["platform:read", "platform:write"],
            expires_in_seconds=3600,
            issued_at=1700000000,
        )

        claims = decode_jwt_claims(token)
        assert claims["sub"] == "user-123"
        assert claims["email"] == "user@example.com"
        assert claims["groups"] == ["dev", "admin"]
        assert claims["scope"] == "platform:read platform:write"
        assert claims["iat"] == 1700000000
        assert claims["exp"] == 1700003600

    def test_generates_token_without_exp_when_requested(self):
        token = generate_unsigned_jwt(
            principal_id="user-123",
            expires_in_seconds=None,
        )

        claims = decode_jwt_claims(token)
        assert claims["sub"] == "user-123"
        assert "exp" not in claims


class TestDiscoverNhxConfig:
    def test_parses_full_response(self, httpserver: HTTPServer):
        config_response = {
            "auth_enabled": True,
            "oidc": {
                "issuer": "https://idp.example.com",
                "client_id": "nhx-app",
                "cli_client_id": "nhx-cli",
                "bearer_token_source": "id_token",
                "token_endpoint": "https://idp.example.com/api/login/oauth/access_token",
                "device_authorization_endpoint": "https://idp.example.com/api/login/oauth/device/code",
                "device_authorization_requires_device_id": True,
                "device_authorization_display_name": "NeMo Helix CLI",
                "device_token_request_includes_scope": False,
                "default_scopes": "openid profile",
                "scope_prefix": "api://nhx/",
            },
        }
        httpserver.expect_request("/apis/auth/discovery").respond_with_json(config_response)
        result = discover_nhx_config(httpserver.url_for(""))
        assert result == NHXOIDCConfig(
            auth_enabled=True,
            issuer="https://idp.example.com",
            client_id="nhx-app",
            cli_client_id="nhx-cli",
            bearer_token_source="id_token",
            token_endpoint="https://idp.example.com/api/login/oauth/access_token",
            device_authorization_endpoint="https://idp.example.com/api/login/oauth/device/code",
            device_authorization_requires_device_id=True,
            device_authorization_display_name="NeMo Helix CLI",
            device_token_request_includes_scope=False,
            default_scopes="openid profile",
            scope_prefix="api://nhx/",
        )

    def test_handles_auth_disabled(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_json({"auth_enabled": False})
        result = discover_nhx_config(httpserver.url_for(""))
        assert result.auth_enabled is False
        assert result.client_id is None
        assert result.token_endpoint is None

    def test_rejects_unknown_bearer_token_source(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_json(
            {"auth_enabled": True, "oidc": {"bearer_token_source": "refresh_token"}}
        )

        with pytest.raises(ValueError, match="bearer_token_source"):
            discover_nhx_config(httpserver.url_for(""))

    def test_handles_missing_oidc_key(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_json({"auth_enabled": True})
        result = discover_nhx_config(httpserver.url_for(""))
        assert result.auth_enabled is True
        assert result.issuer is None
        assert result.default_scopes == "openid profile email offline_access"

    def test_raises_on_http_error(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_data("Not Found", status=404)
        with pytest.raises(httpx.HTTPStatusError):
            discover_nhx_config(httpserver.url_for(""))

    def test_strips_trailing_slash(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_json({"auth_enabled": False})
        result = discover_nhx_config(httpserver.url_for("") + "/")
        assert result.auth_enabled is False

    @patch("nemo_helix_ext.auth.helpers.httpx.get")
    def test_uses_nemo_scoped_ca_bundle(self, mock_get, monkeypatch):
        response = MagicMock()
        response.json.return_value = {"auth_enabled": False}
        mock_get.return_value = response
        monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        result = discover_nhx_config("https://nemo.example.com")

        assert result.auth_enabled is False
        mock_get.assert_called_once_with(
            "https://nemo.example.com/apis/auth/discovery",
            timeout=10.0,
            verify="/tmp/nemo-ca.pem",
        )

    @patch("nemo_helix_ext.auth.helpers.httpx.get")
    def test_uses_context_certificate_authority(self, mock_get, tmp_path, monkeypatch):
        response = MagicMock()
        response.json.return_value = {"auth_enabled": False}
        mock_get.return_value = response
        context_ca = str(tmp_path / "context-ca.pem")
        monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        result = discover_nhx_config("https://nemo.example.com", certificate_authority=context_ca)

        assert result.auth_enabled is False
        mock_get.assert_called_once_with(
            "https://nemo.example.com/apis/auth/discovery",
            timeout=10.0,
            verify=context_ca,
        )


class TestBuildEffectiveScope:
    def test_no_prefix_returns_unchanged(self):
        assert build_effective_scope("openid profile email", None) == "openid profile email"

    def test_empty_prefix_returns_unchanged(self):
        assert build_effective_scope("openid profile email", "") == "openid profile email"

    def test_standard_scopes_not_prefixed(self):
        result = build_effective_scope("openid profile email", "api://nhx/")
        assert result == "openid profile email"

    def test_custom_scope_with_colon_gets_prefixed(self):
        result = build_effective_scope("openid nhx:admin", "api://nhx/")
        assert result == "openid api://nhx/nhx:admin"

    def test_default_scope_gets_prefixed(self):
        result = build_effective_scope("openid api.default", "api://nhx/")
        assert result == "openid api://nhx/api.default"

    def test_mixed_scopes(self):
        result = build_effective_scope("openid profile nhx:read nhx:write", "prefix/")
        assert result == "openid profile prefix/nhx:read prefix/nhx:write"

    def test_single_standard_scope(self):
        assert build_effective_scope("openid", "api://") == "openid"

    def test_single_custom_scope(self):
        assert build_effective_scope("nhx:all", "api://") == "api://nhx:all"

    def test_prefix_without_trailing_slash_is_normalized(self):
        assert build_effective_scope("platform:read", "api://nhx") == "api://nhx/platform:read"


class TestNormalizeScopePrefix:
    def test_none_returns_empty(self):
        assert normalize_scope_prefix(None) == ""

    def test_empty_returns_empty(self):
        assert normalize_scope_prefix("") == ""

    def test_preserves_trailing_slash(self):
        assert normalize_scope_prefix("api://nhx/") == "api://nhx/"

    def test_appends_trailing_slash(self):
        assert normalize_scope_prefix("api://nhx") == "api://nhx/"


class TestValidateRequestedScopesGranted:
    def test_all_requested_platform_scopes_granted(self):
        validate_requested_scopes_granted(
            effective_scope="openid api://nhx/platform:read api://nhx/platform:write",
            granted_scopes=["openid", "platform:read", "platform:write"],
            scope_prefix="api://nhx/",
        )

    def test_missing_requested_scope_raises_auth_error(self):
        with pytest.raises(AuthError, match="Token is missing requested scopes"):
            validate_requested_scopes_granted(
                effective_scope="openid api://nhx/platform:read api://nhx/platform:write",
                granted_scopes=["openid", "platform:read"],
                scope_prefix="api://nhx/",
            )
