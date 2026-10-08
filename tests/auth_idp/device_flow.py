# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from urllib.parse import urlparse

import httpx
from nemo_helix_ext.auth.helpers import NHXOIDCConfig, select_advertised_client

from tests.auth_idp.runtime_contract import JsonObject

DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
DEVICE_TOKEN_POLL_ATTEMPTS = 3
DEVICE_TOKEN_POLL_INTERVAL_SECONDS = 1.0


def url_origin(url: str) -> str:
    parsed = urlparse(url)
    assert parsed.scheme
    assert parsed.netloc
    return f"{parsed.scheme}://{parsed.netloc}"


def poll_device_token(
    client: httpx.Client,
    *,
    token_endpoint: str,
    client_id: str,
    device_code: str,
    scope: str,
) -> JsonObject:
    last_response: httpx.Response | None = None
    for _ in range(DEVICE_TOKEN_POLL_ATTEMPTS):
        response = client.post(
            token_endpoint,
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "client_id": client_id,
                "device_code": device_code,
                "scope": scope,
            },
            timeout=30.0,
        )
        if response.status_code == 200:
            token_response = response.json()
            assert isinstance(token_response, dict)
            return token_response

        last_response = response
        error = response.json().get("error")
        if error != "authorization_pending":
            response.raise_for_status()

        time.sleep(DEVICE_TOKEN_POLL_INTERVAL_SECONDS)

    raise AssertionError(
        "Device token endpoint did not return tokens after browser-side authorization completed: "
        f"{last_response.text if last_response is not None else 'no response'}"
    )


def advertised_device_client_id(oidc: NHXOIDCConfig) -> str:
    """Return the public device-flow client when one is advertised."""
    return select_advertised_client(oidc, "public").client_id
