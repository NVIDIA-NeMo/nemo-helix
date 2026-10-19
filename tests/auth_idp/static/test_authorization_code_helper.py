# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest

from tests.auth_idp.oidc_test_driver import OidcTestDriver
from tests.auth_idp.provider_automation import AuthentikUserAutomation, ZitadelUserAutomation

pytestmark = [pytest.mark.auth_idp]


def test_authentik_authorization_code_helper_completes_login_flow() -> None:
    submitted_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/application/o/authorize/":
            return httpx.Response(
                302,
                headers={"location": "/flows/-/default/authentication/?next=%2Fapplication%2Fo%2Fauthorize%2F"},
            )
        if request.url.path == "/api/v3/flows/executor/default-authentication-flow/":
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={"component": "ak-stage-identification", "password_fields": True},
                )
            submitted_payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "component": "xak-flow-redirect",
                    "to": "/if/flow/default-provider-authorization-implicit-consent/?code=flow-code",
                },
            )
        if request.url.path == "/api/v3/flows/executor/default-provider-authorization-implicit-consent/":
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={"component": "ak-stage-consent", "token": "consent-token"},
                    headers={"set-cookie": "authentik_csrf=csrf-token; Path=/; Secure"},
                )
            assert request.headers["referer"] == str(request.url)
            assert request.headers["x-authentik-csrf"] == "csrf-token"
            submitted_payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "component": "xak-flow-redirect",
                    "to": "https://advertised.example/apis/auth/v2/login/callback?code=idp-code&state=txn",
                },
            )
        raise AssertionError(f"Unexpected Authentik request: {request.method} {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        driver = OidcTestDriver("https://gateway.example", AuthentikUserAutomation())
        callback_url = driver.complete_authorization(
            client,
            authorization_url="https://advertised.example/application/o/authorize/?client_id=nemo-helix-user",
            username="nemo-user",
            password="nemo-password",
        )

    assert callback_url == "https://advertised.example/apis/auth/v2/login/callback?code=idp-code&state=txn"
    assert submitted_payloads == [
        {
            "component": "ak-stage-identification",
            "uid_field": "nemo-user",
            "password": "nemo-password",
        },
        {"component": "ak-stage-consent", "token": "consent-token"},
    ]


def test_zitadel_authorization_code_helper_links_session_to_auth_request() -> None:
    submitted_payloads: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/v2/authorize":
            return httpx.Response(302, headers={"location": "/ui/login/login?authRequest=V2_request"})
        if request.url.path == "/v2/sessions":
            assert request.headers["authorization"] == "Bearer login-client-pat"
            submitted_payloads["session"] = json.loads(request.content)
            return httpx.Response(200, json={"sessionId": "session-id", "sessionToken": "session-token"})
        if request.url.path == "/v2/oidc/auth_requests/V2_request":
            assert request.headers["authorization"] == "Bearer login-client-pat"
            submitted_payloads["auth_request"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"callbackUrl": "https://advertised.example/apis/auth/v2/login/callback?code=idp-code&state=txn"},
            )
        raise AssertionError(f"Unexpected Zitadel request: {request.method} {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        driver = OidcTestDriver(
            "https://gateway.example",
            ZitadelUserAutomation("login-client-pat"),
        )
        callback_url = driver.complete_authorization(
            client,
            authorization_url="https://advertised.example/oauth/v2/authorize?client_id=nemo-helix-user",
            username="nemo-user",
            password="nemo-password",
        )

    assert callback_url == "https://advertised.example/apis/auth/v2/login/callback?code=idp-code&state=txn"
    assert submitted_payloads == {
        "session": {
            "checks": {
                "user": {"loginName": "nemo-user"},
                "password": {"password": "nemo-password"},
            }
        },
        "auth_request": {
            "session": {
                "sessionId": "session-id",
                "sessionToken": "session-token",
            }
        },
    }
