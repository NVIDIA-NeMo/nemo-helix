# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Studio confidential-client OIDC routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.studio.confidential_oidc import _SESSIONS, VERIFIER_COOKIE
from nhx.studio.config import StudioConfig
from nhx.studio.service import StudioService


class FakeTokenResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return {
            "access_token": "access-token",
            "id_token": "id-token",
            "refresh_token": "refresh-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }


class FakeTokenClient:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    async def post(self, url: str, **kwargs: object) -> FakeTokenResponse:
        self.calls.append({"url": url, **kwargs})
        return FakeTokenResponse()


def _client(monkeypatch, config: StudioConfig, fake_token_client: FakeTokenClient | None = None) -> TestClient:
    app = FastAPI()
    monkeypatch.setattr(
        "nhx.studio.confidential_oidc.shared_async_http_client",
        lambda: fake_token_client or FakeTokenClient(),
    )
    StudioService().with_config(config).configure_app(app)
    return TestClient(app, base_url="https://studio.example.com")


def _config() -> StudioConfig:
    return StudioConfig(
        confidential_oidc={
            "enabled": True,
            "client_id": "studio-client",
            "client_secret_env_var": "TEST_STUDIO_CLIENT_SECRET",
            "authorization_endpoint": "https://idp.example.com/oauth2/authorize",
            "token_endpoint": "https://idp.example.com/oauth2/token",
            "scope": "openid profile email",
        }
    )


def test_login_redirects_to_authorization_endpoint_and_sets_pkce_cookies(monkeypatch):
    monkeypatch.setenv("TEST_STUDIO_CLIENT_SECRET", "secret")
    client = _client(monkeypatch, _config())

    response = client.get("/studio/auth/confidential/login", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"].startswith("https://idp.example.com/oauth2/authorize?")
    assert "response_type=code" in response.headers["location"]
    assert "client_id=studio-client" in response.headers["location"]
    assert "code_challenge_method=S256" in response.headers["location"]
    assert response.cookies.get(VERIFIER_COOKIE)


def test_callback_exchanges_code_with_client_secret_basic_and_token_endpoint_returns_bearer(monkeypatch):
    monkeypatch.setenv("TEST_STUDIO_CLIENT_SECRET", "secret")
    fake_token_client = FakeTokenClient()
    client = _client(monkeypatch, _config(), fake_token_client=fake_token_client)
    login_response = client.get("/studio/auth/confidential/login", follow_redirects=False)
    state = login_response.cookies["nhx_studio_oidc_state"]

    callback_response = client.get(
        f"/studio/auth/confidential/callback?code=auth-code&state={state}",
        follow_redirects=False,
    )

    assert callback_response.status_code == 307
    assert callback_response.headers["location"] == "/studio/"
    assert len(fake_token_client.calls) == 1
    call = fake_token_client.calls[0]
    assert call["url"] == "https://idp.example.com/oauth2/token"
    assert call["data"] == {
        "grant_type": "authorization_code",
        "code": "auth-code",
        "redirect_uri": "https://studio.example.com/studio/auth/confidential/callback",
        "code_verifier": login_response.cookies[VERIFIER_COOKIE],
    }
    assert call["headers"]["Authorization"].startswith("Basic ")  # type: ignore[index]

    token_response = client.get("/studio/auth/confidential/token")

    assert token_response.status_code == 200
    assert token_response.json()["access_token"] == "access-token"
    assert _SESSIONS
