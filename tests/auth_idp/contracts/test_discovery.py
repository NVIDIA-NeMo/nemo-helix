# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from nemo_platform_ext.auth.helpers import discover_nmp_config

from tests.auth_idp.common import require_capability, runtime_tls_config
from tests.auth_idp.device_flow import (
    url_origin,
    with_url_origin,
)

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
    pytest.mark.xdist_group("idp-live"),
]


def test_provider_gateway_serves_oidc_discovery(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "gateway_discovery")

    tls_config = runtime_tls_config(auth_idp_runtime)
    response = httpx.get(auth_idp_runtime.discovery_url, timeout=10.0, **tls_config)

    response.raise_for_status()
    discovery = response.json()
    expected_issuer_path = urlparse(auth_idp_case.provider.issuer_url).path.rstrip("/")
    actual_issuer = urlparse(discovery["issuer"])
    if expected_issuer_path:
        assert actual_issuer.path.rstrip("/") == expected_issuer_path
    else:
        assert discovery["issuer"].rstrip("/") == auth_idp_runtime.gateway_base_url.rstrip("/")
    assert discovery["jwks_uri"]


def test_provider_discovery_exposes_device_flow_when_supported(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "device_flow")

    oidc = discover_nmp_config(auth_idp_runtime.gateway_base_url)

    assert oidc.auth_enabled is True
    assert oidc.client_id
    assert oidc.token_endpoint
    assert oidc.device_authorization_endpoint
    assert oidc.default_scopes


def test_provider_device_authorization_endpoint_issues_user_code(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "device_flow")

    oidc = discover_nmp_config(auth_idp_runtime.gateway_base_url)
    tls_config = runtime_tls_config(auth_idp_runtime)
    assert oidc.device_authorization_endpoint is not None
    device_authorization_endpoint = with_url_origin(
        oidc.device_authorization_endpoint,
        auth_idp_runtime.gateway_base_url,
    )
    response = httpx.post(
        device_authorization_endpoint,
        data={
            "client_id": oidc.client_id,
            "scope": oidc.default_scopes,
        },
        timeout=30.0,
        **tls_config,
    )

    response.raise_for_status()
    body = response.json()
    assert body["device_code"]
    assert body["user_code"]
    assert body["verification_uri"].startswith(url_origin(device_authorization_endpoint))

    verification_complete = urlparse(body["verification_uri_complete"])
    verification_uri = urlparse(body["verification_uri"])
    assert verification_complete.scheme == verification_uri.scheme
    assert verification_complete.netloc == verification_uri.netloc
    assert verification_complete.path == verification_uri.path
    verification_query = parse_qs(verification_complete.query)
    assert verification_query.get("code", verification_query.get("user_code")) == [body["user_code"]]


def test_provider_device_flow_returns_refresh_token(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "device_flow")

    oidc = discover_nmp_config(auth_idp_runtime.gateway_base_url)
    assert oidc.client_id is not None
    assert oidc.token_endpoint
    assert oidc.device_authorization_endpoint
    assert "offline_access" in oidc.default_scopes.split()

    tls_config = runtime_tls_config(auth_idp_runtime)
    device_authorization_endpoint = with_url_origin(
        oidc.device_authorization_endpoint,
        auth_idp_runtime.gateway_base_url,
    )
    token_endpoint = with_url_origin(oidc.token_endpoint, auth_idp_runtime.gateway_base_url)
    token_response = auth_idp_runtime.authenticate_device_flow(
        device_authorization_endpoint=device_authorization_endpoint,
        token_endpoint=token_endpoint,
        client_id=oidc.client_id,
        scope=oidc.default_scopes,
        username=auth_idp_case.provider.interactive_user_username,
        password=auth_idp_case.provider.interactive_user_password,
        tls_config=tls_config,
    )

    refresh_token = token_response.get("refresh_token")
    assert token_response.get("access_token")
    assert isinstance(refresh_token, str)
    assert refresh_token

    refresh_response = httpx.post(
        token_endpoint,
        data={
            "grant_type": "refresh_token",
            "client_id": oidc.client_id,
            "refresh_token": refresh_token,
            "scope": oidc.default_scopes,
        },
        timeout=30.0,
        **tls_config,
    )
    refresh_response.raise_for_status()
    refreshed = refresh_response.json()
    assert refreshed["access_token"]
