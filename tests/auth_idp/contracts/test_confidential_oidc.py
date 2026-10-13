# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from nemo_helix_ext.auth.helpers import select_advertised_client
from nhx.core.auth.oidc_broker.crypto import pkce_challenge
from nhx.testing import grant_workspace_role

from tests.auth_idp.common import discover_runtime_nhx_config, require_capability, runtime_tls_config

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
    pytest.mark.xdist_group("idp-live"),
]

_CLI_CODE_VERIFIER = "v" * 48
_CLI_STATE = "cli-state-1234567890"


def test_provider_discovery_exposes_confidential_client_without_secrets(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "confidential_oidc")

    tls_config = runtime_tls_config(auth_idp_runtime)
    response = httpx.get(
        f"{auth_idp_runtime.gateway_base_url.rstrip('/')}/apis/auth/discovery",
        timeout=10.0,
        **tls_config,
    )
    response.raise_for_status()
    body = response.json()
    serialized = response.text

    oidc = body["oidc"]
    assert [client["name"] for client in oidc["clients"]] == ["public", "confidential"]
    assert [client["client_authentication"] for client in oidc["clients"]] == [
        "public",
        "client_secret_basic",
    ]
    confidential = oidc["clients"][1]
    assert confidential["authorization_start_endpoint"]
    assert confidential["broker_token_endpoint"]
    assert "client_secret_env_var" not in serialized
    assert "session_encryption_key_env_var" not in serialized
    assert "NHX_OIDC_CLIENT_SECRET" not in serialized
    assert "NHX_AUTH_SESSION_ENCRYPTION_KEY" not in serialized


def test_provider_confidential_cli_redirect_uses_pkce_without_exposing_secret(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "confidential_oidc")

    oidc = discover_runtime_nhx_config(auth_idp_runtime)
    confidential = select_advertised_client(oidc, "confidential")
    assert confidential.authorization_start_endpoint is not None
    tls_config = runtime_tls_config(auth_idp_runtime)
    start = httpx.post(
        confidential.authorization_start_endpoint,
        json={
            "redirect_uri": "http://127.0.0.1:54321/callback",
            "code_challenge": pkce_challenge(_CLI_CODE_VERIFIER),
            "state": _CLI_STATE,
        },
        timeout=30.0,
        **tls_config,
    )
    start.raise_for_status()
    provider_authorization_url = start.json()["authorization_url"]
    response = httpx.get(provider_authorization_url, follow_redirects=False, timeout=30.0, **tls_config)

    assert response.status_code in {302, 303, 307, 308}
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["client_id"] == [confidential.client_id]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"]
    assert "client_secret" not in query


def test_provider_confidential_cli_login_exchanges_authenticates_and_refreshes(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "confidential_oidc")

    oidc = discover_runtime_nhx_config(auth_idp_runtime)
    confidential = select_advertised_client(oidc, "confidential")
    assert confidential.authorization_start_endpoint is not None
    assert confidential.broker_token_endpoint is not None
    tls_config = runtime_tls_config(auth_idp_runtime)
    loopback_redirect_uri = "http://127.0.0.1:54321/callback"
    start = httpx.post(
        confidential.authorization_start_endpoint,
        json={
            "redirect_uri": loopback_redirect_uri,
            "code_challenge": pkce_challenge(_CLI_CODE_VERIFIER),
            "state": _CLI_STATE,
        },
        timeout=30.0,
        **tls_config,
    )
    start.raise_for_status()
    provider_redirect = httpx.get(
        start.json()["authorization_url"],
        follow_redirects=False,
        timeout=30.0,
        **tls_config,
    )
    assert provider_redirect.status_code in {302, 303, 307, 308}

    provider_callback_url = auth_idp_runtime.complete_confidential_authorization(
        authorization_url=provider_redirect.headers["location"],
        username=auth_idp_case.provider.interactive_user_username,
        password=auth_idp_case.provider.interactive_user_password,
        tls_config=tls_config,
    )
    callback = httpx.get(
        provider_callback_url,
        follow_redirects=False,
        timeout=30.0,
        **tls_config,
    )
    assert callback.status_code in {302, 303, 307, 308}, callback.text
    loopback_location = callback.headers["location"]
    parsed_loopback = urlparse(loopback_location)
    assert f"{parsed_loopback.scheme}://{parsed_loopback.netloc}{parsed_loopback.path}" == loopback_redirect_uri
    one_time_code = parse_qs(parsed_loopback.query)["code"][0]
    assert parse_qs(parsed_loopback.query)["state"] == [_CLI_STATE]

    broker_token_url = confidential.broker_token_endpoint
    exchange = httpx.post(
        broker_token_url,
        json={
            "grant_type": "authorization_code",
            "code": one_time_code,
            "code_verifier": _CLI_CODE_VERIFIER,
        },
        timeout=30.0,
        **tls_config,
    )
    exchange.raise_for_status()
    tokens = exchange.json()
    assert tokens["token_type"].lower() == "bearer"
    assert isinstance(tokens["access_token"], str)
    assert tokens["access_token"]
    assert isinstance(tokens["refresh_token"], str)
    assert tokens["refresh_token"]
    _assert_oidc_access_token_authenticates(auth_idp_runtime, tokens["access_token"], tls_config)

    replay = httpx.post(
        broker_token_url,
        json={
            "grant_type": "authorization_code",
            "code": one_time_code,
            "code_verifier": _CLI_CODE_VERIFIER,
        },
        timeout=30.0,
        **tls_config,
    )
    assert replay.status_code == 400, replay.text

    refresh = httpx.post(
        broker_token_url,
        json={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]},
        timeout=30.0,
        **tls_config,
    )
    refresh.raise_for_status()
    refreshed = refresh.json()
    assert refreshed["token_type"].lower() == "bearer"
    assert refreshed["refresh_token"] != tokens["refresh_token"]
    assert isinstance(refreshed["access_token"], str)
    assert refreshed["access_token"]
    _assert_oidc_access_token_authenticates(auth_idp_runtime, refreshed["access_token"], tls_config)

    replay_refresh = httpx.post(
        broker_token_url,
        json={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]},
        timeout=30.0,
        **tls_config,
    )
    assert replay_refresh.status_code == 400, replay_refresh.text


def test_provider_confidential_browser_session_authenticates_and_revokes(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "confidential_oidc")
    require_capability(auth_idp_case, "workspace_rbac")

    expected_email = auth_idp_case.provider.interactive_user_expected_email
    gateway_base_url = auth_idp_runtime.gateway_base_url.rstrip("/")
    tls_config = runtime_tls_config(auth_idp_runtime)
    with httpx.Client(follow_redirects=False, timeout=30.0, **tls_config) as browser:
        login = browser.get(
            f"{gateway_base_url}/apis/auth/v2/login",
            params={"client": "confidential", "return_to": "/studio/workspaces"},
        )
        assert login.status_code in {302, 303, 307, 308}, login.text

        provider_callback_url = auth_idp_runtime.complete_confidential_authorization(
            authorization_url=login.headers["location"],
            username=auth_idp_case.provider.interactive_user_username,
            password=auth_idp_case.provider.interactive_user_password,
            tls_config=tls_config,
        )
        callback = browser.get(provider_callback_url)
        assert callback.status_code in {302, 303, 307, 308}, callback.text
        assert callback.headers["location"] == "/studio/workspaces"

        set_cookie = callback.headers["set-cookie"].lower()
        assert "nhx_session=" in set_cookie
        assert "httponly" in set_cookie
        assert "secure" in set_cookie
        assert "samesite=lax" in set_cookie

        session = browser.get(f"{gateway_base_url}/apis/auth/v2/session")
        session.raise_for_status()
        session_body = session.json()
        assert session_body["email"] == expected_email
        assert session_body["id"]
        assert session_body["account_id"]
        assert expected_email in session_body["authz_aliases"]
        assert set(session_body) == {"id", "email", "groups", "account_id", "authz_aliases"}

        public_ext_authz = browser.get(
            f"{gateway_base_url}/apis/auth/ext-authz/apis/entities/v2/workspaces/{auth_idp_workspace}"
        )
        assert public_ext_authz.status_code == 404, public_ext_authz.text

        grant_workspace_role(
            auth_idp_runtime.e2e_setup_client(),
            workspace=auth_idp_workspace,
            principal=session_body["id"],
            roles=["Viewer"],
        )
        workspace = browser.get(f"{gateway_base_url}/apis/entities/v2/workspaces/{auth_idp_workspace}")
        assert workspace.status_code == 200, workspace.text
        assert workspace.json()["name"] == auth_idp_workspace

        missing_csrf = browser.post(f"{gateway_base_url}/apis/auth/v2/logout")
        assert missing_csrf.status_code == 403, missing_csrf.text

        logout = browser.post(
            f"{gateway_base_url}/apis/auth/v2/logout",
            headers={"X-Source": "NeMo Studio"},
        )
        assert logout.status_code == 204, logout.text

        revoked = browser.get(f"{gateway_base_url}/apis/auth/v2/session")
        assert revoked.status_code == 401, revoked.text


def _assert_oidc_access_token_authenticates(auth_idp_runtime, access_token: str, tls_config) -> None:
    response = httpx.get(
        f"{auth_idp_runtime.gateway_base_url.rstrip('/')}/apis/auth/authenticate",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30.0,
        **tls_config,
    )
    response.raise_for_status()
    assert response.json()["token_kind"] == "oidc_access_token"
