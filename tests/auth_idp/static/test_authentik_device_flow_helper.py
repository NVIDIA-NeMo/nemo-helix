# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest

from tests.auth_idp.cli_flow import _broker_callback_location
from tests.auth_idp.device_flow import url_origin
from tests.auth_idp.provider_automation import AuthentikUserAutomation, authentik_flow_executor_url

pytestmark = [pytest.mark.auth_idp]


def test_authentik_flow_executor_url_preserves_default_login_next_query() -> None:
    assert authentik_flow_executor_url(
        "https://127.0.0.1:38080",
        "/flows/-/default/authentication/?next=/device%3Fcode%3D123456789",
    ) == (
        "https://127.0.0.1:38080/api/v3/flows/executor/default-authentication-flow/"
        "?query=next%3D%2Fdevice%253Fcode%253D123456789"
    )


def test_authentik_flow_executor_url_preserves_if_flow_query() -> None:
    assert authentik_flow_executor_url(
        "https://127.0.0.1:38080",
        "/if/flow/default-provider-authorization-implicit-consent/?code=abc&state=xyz",
    ) == (
        "https://127.0.0.1:38080/api/v3/flows/executor/default-provider-authorization-implicit-consent/"
        "?query=code%3Dabc%26state%3Dxyz"
    )


def test_authentik_flow_executor_url_ignores_non_flow_browser_urls() -> None:
    assert authentik_flow_executor_url("https://127.0.0.1:38080", "/device?code=123456789") is None


def test_url_origin_keeps_runtime_port_forward_separate_from_advertised_device_origin() -> None:
    assert url_origin("https://127.0.0.1:18080/application/o/device/") == "https://127.0.0.1:18080"


def test_authentik_device_flow_submissions_include_csrf_headers() -> None:
    submitted_components: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"component": "ak-stage-identification", "password_fields": True},
                headers={"set-cookie": "authentik_csrf=csrf-token; Path=/; Secure"},
            )
        assert request.headers["referer"] == str(request.url)
        assert request.headers["x-authentik-csrf"] == "csrf-token"
        component = json.loads(request.content)["component"]
        submitted_components.append(component)
        next_components = {
            "ak-stage-identification": "ak-provider-oauth2-device-code",
            "ak-provider-oauth2-device-code": "ak-provider-oauth2-device-code-finish",
            "ak-provider-oauth2-device-code-finish": "xak-flow-redirect",
        }
        return httpx.Response(200, json={"component": next_components[component]})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        AuthentikUserAutomation().approve_device_authorization(
            client,
            gateway_base_url="https://gateway.example",
            verification_uri_complete="https://gateway.example/device?code=123456789",
            user_code="123456789",
            username="nemo-user",
            password="nemo-password",
        )

    assert submitted_components == [
        "ak-stage-identification",
        "ak-provider-oauth2-device-code",
        "ak-provider-oauth2-device-code-finish",
    ]


def test_confidential_cli_accepts_broker_callback_redirect() -> None:
    response = httpx.Response(
        302,
        headers={"location": "http://127.0.0.1:54321/callback?code=one-time-code"},
    )

    assert _broker_callback_location(response) == "http://127.0.0.1:54321/callback?code=one-time-code"


def test_confidential_cli_rejects_non_redirecting_broker_callback() -> None:
    response = httpx.Response(200, request=httpx.Request("GET", "https://gateway.example/apis/auth/v2/login/callback"))

    with pytest.raises(AssertionError, match="did not redirect"):
        _broker_callback_location(response)
