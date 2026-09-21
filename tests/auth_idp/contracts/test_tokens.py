# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from tests.auth_idp.common import require_capability, runtime_tls_config

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
]


def _claim_values(value: object) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return set()


def _scope_values(value: object) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, str):
        return {item.strip() for item in value.replace(",", " ").split() if item.strip()}
    return set()


def _authenticate_token(auth_idp_runtime, token: str) -> dict:
    response = httpx.get(
        f"{auth_idp_runtime.gateway_base_url}/apis/auth/authenticate",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
        **runtime_tls_config(auth_idp_runtime),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "principal" in body
    assert "token_kind" in body
    return body


def test_provider_e2e_setup_token_is_real(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "gateway_authn")

    token = auth_idp_runtime.e2e_setup_token()
    grant = auth_idp_case.provider.e2e_setup_password_grant
    assert grant is not None

    assert token.access_token
    assert token.claims
    expected_subject = grant.get("expected_subject") or grant.get("username")
    if expected_subject is not None:
        assert token.claims["sub"] == expected_subject


def test_provider_e2e_setup_token_authenticates_as_oidc_access_token(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "gateway_authn")

    token = auth_idp_runtime.e2e_setup_token()
    authenticated = _authenticate_token(auth_idp_runtime, token.access_token)

    assert authenticated["principal"] == token.claims["sub"]
    assert authenticated["email"] == token.claims["email"]
    assert set(authenticated["groups"]) == _claim_values(token.claims.get("groups"))
    assert authenticated["token_kind"] == "oidc_access_token"
    assert set(authenticated["scopes"]) == _scope_values(token.claims.get("scope"))
    assert authenticated["on_behalf_of"] is None


def test_provider_workload_provider_token_is_real(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "workload_provider_token")

    token = auth_idp_runtime.workload_provider_token()

    assert token.access_token
    assert token.claims


def test_provider_workload_subject_token_authenticates_with_expected_identity(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "workload_provider_token")

    token = auth_idp_runtime.workload_provider_token()
    authenticated = _authenticate_token(auth_idp_runtime, token.access_token)

    assert authenticated["principal"] == auth_idp_case.provider.workload_principal_id
    assert authenticated["principal"] == token.claims["sub"]
    assert set(auth_idp_case.provider.workload_expected_groups).issubset(set(authenticated["groups"]))
    assert authenticated["token_kind"] == "workload_subject_token"
    assert authenticated["on_behalf_of"] is None


def test_provider_workload_provider_token_claims_match_manifest(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "workload_provider_token")

    token = auth_idp_runtime.workload_provider_token()
    provider = auth_idp_runtime.provider
    grant = provider.workload_provider_password_grant
    assert grant is not None

    assert token.claims["sub"] == provider.workload_principal_id
    assert set(provider.workload_expected_groups).issubset(
        _claim_values(token.claims.get(provider.workload_groups_claim))
    )
    assert grant.get("expected_audience", grant["client_id"]) in _claim_values(token.claims.get("aud"))


def test_provider_workload_subject_token_exchanges_for_access_token(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "workload_subject_token")
    require_capability(auth_idp_case, "workload_token_exchange")

    subject_token = auth_idp_runtime.workload_subject_token()
    exchanged = auth_idp_runtime.exchange_workload_token(subject_token)

    assert exchanged.access_token
    assert exchanged.claims


def test_provider_exchanged_workload_token_authenticates_with_expected_identity(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "workload_subject_token")
    require_capability(auth_idp_case, "workload_token_exchange")

    exchanged = auth_idp_runtime.exchange_workload_token(auth_idp_runtime.workload_subject_token())
    authenticated = _authenticate_token(auth_idp_runtime, exchanged.access_token)
    expected_groups = _claim_values(exchanged.claims.get(auth_idp_case.provider.workload_groups_claim))

    assert authenticated["principal"] == exchanged.claims["sub"]
    assert expected_groups
    assert expected_groups.issubset(set(authenticated["groups"]))
    assert authenticated["token_kind"] == "workload_access_token"
    assert {"openid", "groups"}.issubset(set(authenticated["scopes"]))
    assert authenticated["on_behalf_of"] is None
