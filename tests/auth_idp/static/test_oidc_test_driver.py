# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import parse_qs

import httpx
import pytest

from tests.auth_idp.oidc_test_driver import OidcTestDriver, create_oidc_test_driver
from tests.auth_idp.provider_automation import AuthentikUserAutomation, ZitadelUserAutomation

pytestmark = [pytest.mark.auth_idp]


class RecordingUserAutomation:
    def __init__(self) -> None:
        self.device_approvals: list[tuple[str, str, str, str]] = []

    def approve_device_authorization(
        self,
        client: httpx.Client,
        *,
        gateway_base_url: str,
        verification_uri_complete: str,
        user_code: str,
        username: str,
        password: str,
    ) -> None:
        del client
        self.device_approvals.append((gateway_base_url, verification_uri_complete, user_code, username))
        assert password == "nemo-password"

    def complete_authorization(
        self,
        client: httpx.Client,
        *,
        gateway_base_url: str,
        authorization_url: str,
        username: str,
        password: str,
    ) -> str:
        raise AssertionError("authorization-code automation is not used by this test")


def test_oidc_driver_factory_selects_authentik_automation() -> None:
    driver = create_oidc_test_driver(
        gateway_base_url="https://gateway.example",
        provider_name="authentik",
    )

    assert isinstance(driver.user_automation, AuthentikUserAutomation)


def test_oidc_driver_factory_selects_zitadel_automation() -> None:
    driver = create_oidc_test_driver(
        gateway_base_url="https://gateway.example",
        provider_name="zitadel",
        zitadel_login_client_pat="login-client-pat",
    )

    assert isinstance(driver.user_automation, ZitadelUserAutomation)
    assert driver.user_automation.admin_token == "login-client-pat"


def test_oidc_driver_factory_requires_zitadel_login_client_pat() -> None:
    with pytest.raises(AssertionError, match="seeded login client PAT"):
        create_oidc_test_driver(
            gateway_base_url="https://gateway.example",
            provider_name="zitadel",
        )


def test_oidc_driver_owns_device_authorization_and_token_polling() -> None:
    requests: list[tuple[str, dict[str, list[str]]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode())
        requests.append((request.url.path, form))
        if request.url.path == "/oauth/device/code":
            return httpx.Response(
                200,
                json={
                    "device_code": "device-code",
                    "user_code": "ABCD-EFGH",
                    "verification_uri_complete": "https://advertised.example/device?user_code=ABCD-EFGH",
                },
            )
        if request.url.path == "/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token", "token_type": "Bearer"})
        raise AssertionError(f"Unexpected OIDC request: {request.method} {request.url}")

    automation = RecordingUserAutomation()
    driver = OidcTestDriver("https://gateway.example", automation)
    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        tokens = driver.authenticate_device_flow(
            client,
            device_authorization_endpoint="https://gateway.example/oauth/device/code",
            token_endpoint="https://gateway.example/oauth/token",
            client_id="public-client",
            scope="openid offline_access",
            username="nemo-user",
            password="nemo-password",
        )

    assert tokens == {"access_token": "access-token", "token_type": "Bearer"}
    assert automation.device_approvals == [
        (
            "https://gateway.example",
            "https://advertised.example/device?user_code=ABCD-EFGH",
            "ABCD-EFGH",
            "nemo-user",
        )
    ]
    assert requests == [
        (
            "/oauth/device/code",
            {"client_id": ["public-client"], "scope": ["openid offline_access"]},
        ),
        (
            "/oauth/token",
            {
                "grant_type": ["urn:ietf:params:oauth:grant-type:device_code"],
                "client_id": ["public-client"],
                "device_code": ["device-code"],
                "scope": ["openid offline_access"],
            },
        ),
    ]
