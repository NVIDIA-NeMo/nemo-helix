# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from tests.auth_idp.device_flow import poll_device_token
from tests.auth_idp.provider_automation import AuthentikUserAutomation, IdpUserAutomation, ZitadelUserAutomation
from tests.auth_idp.providers import ProviderName
from tests.auth_idp.runtime_contract import JsonObject


@dataclass(frozen=True)
class OidcTestDriver:
    """Drive standard OIDC operations while delegating provider-owned user interaction."""

    gateway_base_url: str
    user_automation: IdpUserAutomation

    def authenticate_device_flow(
        self,
        client: httpx.Client,
        *,
        device_authorization_endpoint: str,
        token_endpoint: str,
        client_id: str,
        scope: str,
        username: str,
        password: str,
    ) -> JsonObject:
        device_response = client.post(
            device_authorization_endpoint,
            data={
                "client_id": client_id,
                "scope": scope,
            },
            timeout=30.0,
        )
        device_response.raise_for_status()
        device_body = device_response.json()

        self.user_automation.approve_device_authorization(
            client,
            gateway_base_url=self.gateway_base_url,
            verification_uri_complete=urljoin(str(device_response.url), device_body["verification_uri_complete"]),
            user_code=device_body["user_code"],
            username=username,
            password=password,
        )

        return poll_device_token(
            client,
            token_endpoint=token_endpoint,
            client_id=client_id,
            device_code=device_body["device_code"],
            scope=scope,
        )

    def complete_authorization(
        self,
        client: httpx.Client,
        *,
        authorization_url: str,
        username: str,
        password: str,
    ) -> str:
        return self.user_automation.complete_authorization(
            client,
            gateway_base_url=self.gateway_base_url,
            authorization_url=authorization_url,
            username=username,
            password=password,
        )


def create_oidc_test_driver(
    *,
    gateway_base_url: str,
    provider_name: ProviderName,
    zitadel_login_client_pat: str | None = None,
) -> OidcTestDriver:
    if provider_name == "authentik":
        return OidcTestDriver(gateway_base_url, AuthentikUserAutomation())
    if not zitadel_login_client_pat:
        raise AssertionError("ZITADEL login automation requires the seeded login client PAT")
    return OidcTestDriver(gateway_base_url, ZitadelUserAutomation(zitadel_login_client_pat))
