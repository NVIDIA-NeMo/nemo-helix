# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from nemo_helix_ext.auth.confidential_login import (
    _authorization_url_from_response,
    _json_object_from_response,
    _validated_callback_code,
)
from nemo_helix_ext.auth.device_flow import TokenResponse
from nemo_helix_ext.auth.helpers import AdvertisedOidcClient, AuthError, NHXOIDCConfig, generate_unsigned_jwt
from nemo_helix_ext.auth.login import authenticate_with_oidc


def _confidential_config() -> NHXOIDCConfig:
    return NHXOIDCConfig(
        auth_enabled=True,
        issuer="https://idp.example.com",
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


def _public_config() -> NHXOIDCConfig:
    return NHXOIDCConfig(
        auth_enabled=True,
        issuer="https://idp.example.com",
        clients=(
            AdvertisedOidcClient(
                name="public",
                client_id="public-client",
                client_authentication="public",
                default=True,
                token_endpoint="https://idp.example.com/token",
                device_authorization_endpoint="https://idp.example.com/device",
                scope_prefix="api://nhx",
            ),
        ),
    )


def _brokered_public_config() -> NHXOIDCConfig:
    return NHXOIDCConfig(
        auth_enabled=True,
        issuer="https://idp.example.com",
        clients=(
            AdvertisedOidcClient(
                name="public",
                client_id="public-client",
                client_authentication="public",
                default=True,
                server_side_sessions=True,
                authorization_start_endpoint="https://nemo.example.com/apis/auth/v2/authorize?client=public",
                broker_token_endpoint="https://nemo.example.com/apis/auth/v2/token?client=public",
            ),
        ),
    )


def _confidential_with_public_client_config() -> NHXOIDCConfig:
    return NHXOIDCConfig(
        auth_enabled=True,
        issuer="https://idp.example.com",
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
                token_endpoint="https://idp.example.com/token",
                device_authorization_endpoint="https://idp.example.com/device",
                scope_prefix="api://nhx",
            ),
        ),
    )


def test_confidential_login_uses_broker_refresh_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    token = generate_unsigned_jwt("alice", email="alice@example.com")
    calls: list[dict[str, object]] = []

    def broker(**kwargs: object) -> dict[str, str | int]:
        calls.append(kwargs)
        return {"access_token": token, "refresh_token": "refresh-token", "expires_in": 3600}

    monkeypatch.setattr("nemo_helix_ext.auth.login.login_with_oidc_broker", broker)

    result = authenticate_with_oidc(
        oidc_config=_confidential_config(),
        no_browser=True,
        certificate_authority="/tmp/ca.pem",
    )

    assert calls == [
        {
            "authorization_start_endpoint": "https://nemo.example.com/apis/auth/v2/authorize",
            "broker_token_endpoint": "https://nemo.example.com/apis/auth/v2/token",
            "open_browser": False,
            "certificate_authority": "/tmp/ca.pem",
        }
    ]
    assert result.flow == "confidential"
    assert result.access_token == token
    assert result.refresh_token == "refresh-token"
    assert result.expires_at is not None
    assert result.token_broker_url == "https://nemo.example.com/apis/auth/v2/token"


def test_brokered_public_login_uses_broker_refresh_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    token = generate_unsigned_jwt("alice", email="alice@example.com")

    monkeypatch.setattr(
        "nemo_helix_ext.auth.login.login_with_oidc_broker",
        lambda **_kwargs: {"access_token": token, "refresh_token": "refresh-token", "expires_in": 3600},
    )

    result = authenticate_with_oidc(oidc_config=_brokered_public_config(), no_browser=True)

    assert result.flow == "public"
    assert result.refresh_token == "refresh-token"
    assert result.token_broker_url == "https://nemo.example.com/apis/auth/v2/token?client=public"


def test_brokered_login_rejects_missing_callback_state() -> None:
    with pytest.raises(AuthError, match="did not include state"):
        _validated_callback_code({"code": "one-time-code"}, expected_state="expected-state")


def test_brokered_login_rejects_mismatched_callback_state() -> None:
    with pytest.raises(AuthError, match="did not match"):
        _validated_callback_code(
            {"code": "one-time-code", "state": "different-state"},
            expected_state="expected-state",
        )


def test_brokered_login_reports_callback_error_after_state_validation() -> None:
    with pytest.raises(AuthError, match="access_denied: User cancelled"):
        _validated_callback_code(
            {"error": "access_denied", "error_description": "User cancelled", "state": "expected-state"},
            expected_state="expected-state",
        )


def test_brokered_login_rejects_invalid_authorization_response_json() -> None:
    response = httpx.Response(200, content=b"not-json")

    with pytest.raises(AuthError, match="authorization response was invalid"):
        _authorization_url_from_response(response)


def test_brokered_login_rejects_missing_authorization_url() -> None:
    response = httpx.Response(200, json={"transaction_id": "txn"})

    with pytest.raises(AuthError, match="did not return an authorization URL"):
        _authorization_url_from_response(response)


def test_brokered_login_rejects_non_object_token_response() -> None:
    response = httpx.Response(200, json=["not", "an", "object"])

    with pytest.raises(AuthError, match="token response was invalid"):
        _json_object_from_response(response, "token")


def test_confidential_login_rejects_password_credentials() -> None:
    with pytest.raises(AuthError, match="Password grant is not available"):
        authenticate_with_oidc(
            oidc_config=_confidential_config(),
            username="alice",
        )


def test_public_login_uses_password_grant_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    token = generate_unsigned_jwt(
        "alice",
        email="alice@example.com",
        scopes=["openid", "profile", "email", "offline_access", "platform:read"],
    )
    calls: list[dict[str, object]] = []

    def password_grant(**kwargs: object) -> TokenResponse:
        calls.append(kwargs)
        return TokenResponse(
            access_token=token,
            id_token=None,
            refresh_token="refresh-token",
            token_type="Bearer",
            expires_in=3600,
            scope=None,
        )

    monkeypatch.setenv("NHX_OIDC_USERNAME", "env-user")
    monkeypatch.setenv("NHX_OIDC_PASSWORD", "env-password")
    monkeypatch.setattr("nemo_helix_ext.auth.device_flow.authenticate_with_password_grant", password_grant)

    result = authenticate_with_oidc(
        oidc_config=_public_config(),
        scope="platform:read",
    )

    assert calls == [
        {
            "token_endpoint": "https://idp.example.com/token",
            "client_id": "public-client",
            "username": "env-user",
            "password": "env-password",
            "scope": "openid profile email offline_access api://nhx/platform:read",
            "bearer_token_source": "access_token",
            "certificate_authority": None,
        }
    ]
    assert result.flow == "public"
    assert result.user_email == "alice@example.com"
    assert result.display_granted_scopes == ("openid", "profile", "email", "offline_access", "platform:read")
    assert result.refresh_token == "refresh-token"


def test_explicit_public_client_uses_device_flow_when_platform_client_is_confidential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = generate_unsigned_jwt("alice", scopes=["openid", "profile", "email", "offline_access", "platform:read"])
    calls: list[dict[str, object]] = []

    async def device_login(**kwargs: object) -> TokenResponse:
        calls.append(kwargs)
        return TokenResponse(
            access_token=token,
            id_token=None,
            refresh_token="refresh-token",
            token_type="Bearer",
            expires_in=3600,
            scope="openid profile email offline_access api://nhx/platform:read",
        )

    monkeypatch.setattr("nemo_helix_ext.auth.device_flow.authenticate_with_device_flow", device_login)

    result = authenticate_with_oidc(
        oidc_config=_confidential_with_public_client_config(),
        no_browser=True,
        scope="platform:read",
        oidc_client="public",
    )

    assert calls == [
        {
            "device_authorization_endpoint": "https://idp.example.com/device",
            "token_endpoint": "https://idp.example.com/token",
            "client_id": "public-client",
            "scope": "openid profile email offline_access api://nhx/platform:read",
            "open_browser": False,
            "bearer_token_source": "access_token",
            "include_device_id": False,
            "device_display_name": None,
            "include_scope_in_token_request": True,
            "certificate_authority": None,
        }
    ]
    assert result.flow == "public"
    assert result.requested_scopes[-1].name == "platform:read"
    assert result.requested_scopes[-1].effective_name == "api://nhx/platform:read"
