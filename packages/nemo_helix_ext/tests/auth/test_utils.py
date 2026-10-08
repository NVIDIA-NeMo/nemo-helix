# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json

import httpx
import pytest
from nemo_helix_ext.auth.helpers import (
    AdvertisedOidcClient,
    AuthError,
    NHXOIDCConfig,
    build_effective_scope,
    decode_jwt_claims,
    decode_jwt_header,
    discover_nhx_config,
    discover_nhx_config_async,
    generate_unsigned_jwt,
    is_unsigned_jwt,
    normalize_scope_prefix,
    parse_nhx_config,
    refresh_target,
    select_advertised_client,
    validate_requested_scopes_granted,
)
from nemo_helix_plugin.client.oidc import OidcClientName
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


def _discover_with_client(base_url: str) -> NHXOIDCConfig:
    with httpx.Client() as http_client:
        return discover_nhx_config(base_url, http_client=http_client)


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
                "clients": [
                    {
                        "name": "confidential",
                        "client_id": "nhx-app",
                        "client_authentication": "client_secret_basic",
                        "default": False,
                        "authorization_start_endpoint": "https://nemo.example.com/apis/auth/v2/authorize",
                        "broker_token_endpoint": "https://nemo.example.com/apis/auth/v2/token",
                        "default_scopes": "openid profile",
                        "bearer_token_source": "id_token",
                        "scope_prefix": "api://nhx/",
                    },
                    {
                        "name": "public",
                        "client_id": "nhx-public",
                        "client_authentication": "public",
                        "default": True,
                        "token_endpoint": "https://idp.example.com/api/login/oauth/access_token",
                        "device_authorization_endpoint": "https://idp.example.com/api/login/oauth/device/code",
                        "device_authorization_requires_device_id": True,
                        "device_authorization_display_name": "NeMo Helix CLI",
                        "device_token_request_includes_scope": False,
                        "default_scopes": "openid profile",
                        "bearer_token_source": "id_token",
                        "scope_prefix": "api://nhx/",
                    },
                ],
            },
        }
        httpserver.expect_request("/apis/auth/discovery").respond_with_json(config_response)
        result = _discover_with_client(httpserver.url_for(""))
        assert result == NHXOIDCConfig(
            auth_enabled=True,
            issuer="https://idp.example.com",
            clients=(
                AdvertisedOidcClient(
                    name="confidential",
                    client_id="nhx-app",
                    client_authentication="client_secret_basic",
                    default=False,
                    server_side_sessions=True,
                    authorization_start_endpoint="https://nemo.example.com/apis/auth/v2/authorize",
                    broker_token_endpoint="https://nemo.example.com/apis/auth/v2/token",
                    default_scopes="openid profile",
                    bearer_token_source="id_token",
                    scope_prefix="api://nhx/",
                ),
                AdvertisedOidcClient(
                    name="public",
                    client_id="nhx-public",
                    client_authentication="public",
                    default=True,
                    token_endpoint="https://idp.example.com/api/login/oauth/access_token",
                    device_authorization_endpoint="https://idp.example.com/api/login/oauth/device/code",
                    device_authorization_requires_device_id=True,
                    device_authorization_display_name="NeMo Helix CLI",
                    device_token_request_includes_scope=False,
                    default_scopes="openid profile",
                    bearer_token_source="id_token",
                    scope_prefix="api://nhx/",
                ),
            ),
        )

    def test_handles_auth_disabled(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_json({"auth_enabled": False})
        result = _discover_with_client(httpserver.url_for(""))
        assert result.auth_enabled is False
        assert result.clients == ()

    def test_rejects_wrong_scalar_types_as_malformed_discovery(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_json(
            {
                "auth_enabled": "true",
                "oidc": {
                    "issuer": 123,
                    "client_id": ["nhx-app"],
                    "device_authorization_requires_device_id": "true",
                    "device_token_request_includes_scope": "false",
                    "default_scopes": 42,
                },
            }
        )

        with pytest.raises(AttributeError, match="expected shape"):
            _discover_with_client(httpserver.url_for(""))

    def test_rejects_unknown_bearer_token_source(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_json(
            {
                "auth_enabled": True,
                "oidc": {
                    "issuer": "https://idp.example.com",
                    "clients": [
                        {
                            "name": "public",
                            "client_id": "public",
                            "client_authentication": "public",
                            "default": True,
                            "default_scopes": "openid",
                            "bearer_token_source": "refresh_token",
                            "device_authorization_requires_device_id": False,
                            "device_token_request_includes_scope": True,
                        }
                    ],
                },
            }
        )

        with pytest.raises(ValueError, match="bearer_token_source"):
            _discover_with_client(httpserver.url_for(""))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("clients", [{"name": "other", "client_id": "client", "client_authentication": "public"}]),
            (
                "clients",
                [{"name": "public", "client_id": "client", "client_authentication": "private_key_jwt"}],
            ),
        ],
    )
    def test_rejects_unsupported_advertised_client_values(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match="OIDC"):
            parse_nhx_config({"auth_enabled": True, "oidc": {"issuer": "https://idp.example.com", field: value}})

    @pytest.mark.parametrize("value", [[], "invalid", 1])
    def test_rejects_non_object_discovery_response(self, value: object) -> None:
        with pytest.raises(ValueError, match="discovery response must be an object"):
            parse_nhx_config(value)

    def test_rejects_non_object_client_entry(self) -> None:
        with pytest.raises(ValueError, match="client entries must be objects"):
            parse_nhx_config(
                {
                    "auth_enabled": True,
                    "oidc": {"issuer": "https://idp.example.com", "clients": ["public"]},
                }
            )

    def test_rejects_duplicate_advertised_client_names(self) -> None:
        client = {
            "name": "public",
            "client_id": "public",
            "client_authentication": "public",
            "default": True,
            "default_scopes": "openid",
            "bearer_token_source": "access_token",
            "device_authorization_requires_device_id": False,
            "device_token_request_includes_scope": True,
        }
        with pytest.raises(ValueError, match="duplicate 'public' clients"):
            parse_nhx_config(
                {
                    "auth_enabled": True,
                    "oidc": {"issuer": "https://idp.example.com", "clients": [client, client]},
                }
            )

    def test_parses_brokered_public_client(self) -> None:
        result = parse_nhx_config(
            {
                "auth_enabled": True,
                "oidc": {
                    "issuer": "https://idp.example.com",
                    "clients": [
                        {
                            "name": "public",
                            "client_id": "public-client",
                            "client_authentication": "public",
                            "default": True,
                            "server_side_sessions": True,
                            "authorization_start_endpoint": "https://nemo.example.com/apis/auth/v2/authorize?client=public",
                            "broker_token_endpoint": "https://nemo.example.com/apis/auth/v2/token?client=public",
                            "default_scopes": "openid profile",
                            "bearer_token_source": "access_token",
                            "device_authorization_requires_device_id": False,
                            "device_token_request_includes_scope": True,
                        }
                    ],
                },
            }
        )

        assert result.clients[0].server_side_sessions is True
        assert result.clients[0].token_endpoint is None

    def test_rejects_brokered_public_client_without_broker_endpoints(self) -> None:
        with pytest.raises(ValueError, match="public server-side client"):
            parse_nhx_config(
                {
                    "auth_enabled": True,
                    "oidc": {
                        "issuer": "https://idp.example.com",
                        "clients": [
                            {
                                "name": "public",
                                "client_id": "public-client",
                                "client_authentication": "public",
                                "default": True,
                                "server_side_sessions": True,
                                "default_scopes": "openid profile",
                                "bearer_token_source": "access_token",
                                "device_authorization_requires_device_id": False,
                                "device_token_request_includes_scope": True,
                            }
                        ],
                    },
                }
            )

    def test_handles_missing_oidc_key(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_json({"auth_enabled": True})
        result = _discover_with_client(httpserver.url_for(""))
        assert result.auth_enabled is True
        assert result.issuer is None
        assert result.clients == ()

    def test_raises_on_http_error(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_data("Not Found", status=404)
        with pytest.raises(httpx.HTTPStatusError):
            _discover_with_client(httpserver.url_for(""))

    def test_strips_trailing_slash(self, httpserver: HTTPServer):
        httpserver.expect_request("/apis/auth/discovery").respond_with_json({"auth_enabled": False})
        result = _discover_with_client(httpserver.url_for("") + "/")
        assert result.auth_enabled is False

    def test_uses_injected_http_client(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"auth_enabled": False})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            result = discover_nhx_config("https://nemo.example.com", http_client=http_client)

        assert result.auth_enabled is False
        assert [request.url.path for request in requests] == ["/apis/auth/discovery"]

    @pytest.mark.asyncio
    async def test_async_uses_injected_http_client(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"auth_enabled": False})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            result = await discover_nhx_config_async("https://nemo.example.com", http_client=http_client)

        assert result.auth_enabled is False
        assert [request.url.path for request in requests] == ["/apis/auth/discovery"]


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

    def test_ignores_external_colon_delimited_scope(self):
        validate_requested_scopes_granted(
            effective_scope="openid urn:zitadel:iam:org:project:id:123:aud",
            granted_scopes=["openid"],
            scope_prefix="",
        )


def test_refresh_target_uses_broker_for_confidential_login() -> None:
    config = NHXOIDCConfig(
        auth_enabled=True,
        clients=(
            AdvertisedOidcClient(
                name="confidential",
                client_id="confidential-client",
                client_authentication="client_secret_basic",
                default=True,
                authorization_start_endpoint="https://nemo.example.com/apis/auth/v2/authorize",
                broker_token_endpoint="https://nemo.example.com/apis/auth/v2/token",
            ),
        ),
    )

    endpoint, client_id = refresh_target(config, "https://nemo.example.com/apis/auth/v2/token")

    assert endpoint == "https://nemo.example.com/apis/auth/v2/token"
    assert client_id == "confidential-client"


def test_refresh_target_uses_public_client_for_device_flow() -> None:
    config = NHXOIDCConfig(
        auth_enabled=True,
        clients=(
            AdvertisedOidcClient(
                name="public",
                client_id="public-client",
                client_authentication="public",
                default=True,
                token_endpoint="https://idp.example.com/token",
            ),
        ),
    )

    endpoint, client_id = refresh_target(config, None)

    assert endpoint == "https://idp.example.com/token"
    assert client_id == "public-client"


def test_refresh_target_uses_broker_for_public_server_side_session() -> None:
    broker_url = "https://nemo.example.com/apis/auth/v2/token?client=public"
    config = NHXOIDCConfig(
        auth_enabled=True,
        clients=(
            AdvertisedOidcClient(
                name="public",
                client_id="public-client",
                client_authentication="public",
                default=True,
                server_side_sessions=True,
                authorization_start_endpoint="https://nemo.example.com/apis/auth/v2/authorize?client=public",
                broker_token_endpoint=broker_url,
            ),
        ),
    )

    endpoint, client_id = refresh_target(config, broker_url)

    assert endpoint == broker_url
    assert client_id == "public-client"


def test_refresh_target_uses_public_profile_client_id() -> None:
    config = NHXOIDCConfig(
        auth_enabled=True,
        clients=(
            AdvertisedOidcClient(
                name="public",
                client_id="user-login-client",
                client_authentication="public",
                default=True,
                token_endpoint="https://idp.example.com/token",
            ),
        ),
    )

    endpoint, client_id = refresh_target(config, None)

    assert endpoint == "https://idp.example.com/token"
    assert client_id == "user-login-client"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, "confidential-client"), ("confidential", "confidential-client"), ("public", "public-client")],
)
def test_select_advertised_client(requested: OidcClientName | None, expected: str) -> None:
    config = NHXOIDCConfig(
        auth_enabled=True,
        clients=(
            AdvertisedOidcClient(
                name="confidential",
                client_id="confidential-client",
                client_authentication="client_secret_basic",
                default=True,
                authorization_start_endpoint="https://nemo.example.com/apis/auth/v2/authorize",
                broker_token_endpoint="https://nemo.example.com/apis/auth/v2/token",
            ),
            AdvertisedOidcClient(
                name="public",
                client_id="public-client",
                client_authentication="public",
                default=False,
            ),
        ),
    )

    assert select_advertised_client(config, requested).client_id == expected


def test_select_advertised_client_rejects_unknown_client() -> None:
    config = NHXOIDCConfig(auth_enabled=True)

    with pytest.raises(ValueError, match="OIDC client 'public' is not configured"):
        select_advertised_client(config, "public")
