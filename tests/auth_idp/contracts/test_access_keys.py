# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import uuid

import httpx
import pytest
from nemo_helix_ext.client.tls import HttpxTLSConfig
from nhx.testing import grant_workspace_role

from tests.auth_idp.common import jwt_claims, require_capability, runtime_tls_config
from tests.auth_idp.runtime_contract import AuthIdpRuntime, JsonObject

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
    pytest.mark.xdist_group("idp-live"),
]

REQUEST_TIMEOUT_SECONDS = 10.0
ROLE_GRANT_RETRY_TIMEOUT_SECONDS = 10.0
ROLE_GRANT_RETRY_SLEEP_SECONDS = 0.5


def _create_access_key_with_body(auth_idp_runtime: AuthIdpRuntime, bearer_token: str, body: JsonObject) -> JsonObject:
    tls_config = runtime_tls_config(auth_idp_runtime)
    response = httpx.post(
        f"{auth_idp_runtime.gateway_base_url}/apis/auth/v2/access-keys",
        json=body,
        headers={"Authorization": f"Bearer {bearer_token}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
        **tls_config,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "Bearer"
    assert body["token"]
    return body


def _create_access_key(auth_idp_runtime: AuthIdpRuntime, bearer_token: str) -> JsonObject:
    return _create_access_key_with_body(
        auth_idp_runtime,
        bearer_token,
        {"name": f"auth-idp-contract-{uuid.uuid4().hex[:8]}", "expires_in_seconds": 600},
    )


def _tamper_jwt(token: str) -> str:
    """Return a JWT with its signature bytes changed so validation must fail."""
    parts = token.split(".")
    assert len(parts) == 3
    signature = parts[2]
    replacement = "A" if signature[0] != "A" else "B"
    parts[2] = f"{replacement}{signature[1:]}"
    return ".".join(parts)


def _claim_values(value: object) -> set[str]:
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return set()


def _get_until_workspace_role_grant_applies(
    url: str,
    *,
    headers: dict[str, str],
    tls_config: HttpxTLSConfig,
) -> httpx.Response:
    deadline = time.monotonic() + ROLE_GRANT_RETRY_TIMEOUT_SECONDS
    while True:
        response = httpx.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **tls_config,
        )
        if response.status_code == 200:
            return response

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return response
        time.sleep(min(ROLE_GRANT_RETRY_SLEEP_SECONDS, remaining))


def test_provider_platform_access_key_authenticates_and_uses_workspace_rbac(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "platform_access_keys")
    require_capability(auth_idp_case, "workload_provider_token")
    require_capability(auth_idp_case, "workspace_rbac")

    workload_token = auth_idp_runtime.workload_provider_token()
    created = _create_access_key(auth_idp_runtime, workload_token.access_token)
    access_key = str(created["token"])
    access_key_claims = jwt_claims(access_key)
    tls_config = runtime_tls_config(auth_idp_runtime)

    assert created["principal"] == workload_token.claims["sub"]
    assert access_key_claims["nhx_token_type"] == "access_key"
    assert access_key_claims["sub"] == workload_token.claims["sub"]
    assert access_key_claims["aud"] == "nemo-helix-access-key"
    access_key_headers = {"Authorization": f"Bearer {access_key}"}

    authenticate_response = httpx.get(
        f"{auth_idp_runtime.gateway_base_url}/apis/auth/authenticate",
        headers=access_key_headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
        **tls_config,
    )
    authenticate_response.raise_for_status()
    authenticated = authenticate_response.json()
    assert authenticated["jti"] == created["jti"]
    assert authenticated["principal"] == created["principal"]
    assert authenticated["token_kind"] == "access_key"

    workspace_url = f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces/{auth_idp_workspace}"
    denied_response = httpx.get(
        workspace_url,
        headers=access_key_headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
        **tls_config,
    )
    assert denied_response.status_code == 403, denied_response.text

    e2e_setup_sdk = auth_idp_runtime.e2e_setup_sdk()
    role_principals = sorted(_claim_values(access_key_claims.get("groups"))) or [str(created["principal"])]
    for principal in role_principals:
        grant_workspace_role(e2e_setup_sdk, workspace=auth_idp_workspace, principal=principal, roles=["Viewer"])

    allowed_response = _get_until_workspace_role_grant_applies(
        workspace_url,
        headers=access_key_headers,
        tls_config=tls_config,
    )
    assert allowed_response.status_code == 200, allowed_response.text
    assert allowed_response.headers.get("x-envoy-upstream-service-time") is not None
    assert allowed_response.json()["name"] == auth_idp_workspace


def test_provider_platform_access_key_defaults_expiry_when_omitted(auth_idp_runtime, auth_idp_case):
    require_capability(auth_idp_case, "platform_access_keys")
    require_capability(auth_idp_case, "workload_provider_token")

    workload_token = auth_idp_runtime.workload_provider_token()

    created = _create_access_key_with_body(
        auth_idp_runtime,
        workload_token.access_token,
        {"name": f"default-expiry-{uuid.uuid4().hex[:8]}"},
    )
    access_key_claims = jwt_claims(str(created["token"]))

    assert "exp" in access_key_claims
    assert created["expires_at"] is not None


def test_provider_platform_access_key_rejects_invalid_key(
    auth_idp_case,
    auth_idp_runtime,
):
    require_capability(auth_idp_case, "platform_access_keys")
    require_capability(auth_idp_case, "workload_provider_token")

    workload_token = auth_idp_runtime.workload_provider_token()
    created = _create_access_key(auth_idp_runtime, workload_token.access_token)
    invalid_access_key = _tamper_jwt(str(created["token"]))
    invalid_access_key_headers = {"Authorization": f"Bearer {invalid_access_key}"}
    tls_config = runtime_tls_config(auth_idp_runtime)

    authenticate_response = httpx.get(
        f"{auth_idp_runtime.gateway_base_url}/apis/auth/authenticate",
        headers=invalid_access_key_headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
        **tls_config,
    )
    assert authenticate_response.status_code == 401, authenticate_response.text

    protected_response = httpx.get(
        f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces",
        headers=invalid_access_key_headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
        **tls_config,
    )
    assert protected_response.status_code == 401, protected_response.text


def test_provider_platform_access_key_ignores_spoofed_principal_headers(
    auth_idp_case,
    auth_idp_runtime,
):
    require_capability(auth_idp_case, "platform_access_keys")
    require_capability(auth_idp_case, "workload_provider_token")
    require_capability(auth_idp_case, "spoofed_header_rejection")

    workload_token = auth_idp_runtime.workload_provider_token()
    created = _create_access_key(auth_idp_runtime, workload_token.access_token)
    access_key_headers = {
        "Authorization": f"Bearer {created['token']}",
        "X-NHX-Principal-Id": "service:bootstrap",
        "X-NHX-Principal-Email": "attacker@example.com",
    }
    workspace_name = f"access-key-spoof-{uuid.uuid4().hex[:8]}"
    tls_config = runtime_tls_config(auth_idp_runtime)

    try:
        create_response = httpx.post(
            f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Access-key spoofed header check"},
            headers=access_key_headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **tls_config,
        )
        create_response.raise_for_status()

        created_by = create_response.json()["created_by"]
        assert created_by == created["principal"]
        assert created_by not in {"service:bootstrap", "attacker@example.com"}
    finally:
        httpx.delete(
            f"{auth_idp_runtime.gateway_base_url}/apis/entities/v2/workspaces/{workspace_name}",
            headers=access_key_headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **tls_config,
        )
