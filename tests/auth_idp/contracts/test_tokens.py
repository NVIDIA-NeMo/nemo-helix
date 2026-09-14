# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.auth_idp.common import require_capability

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


def test_provider_workload_provider_token_is_real(auth_idp_case, auth_idp_runtime):
    require_capability(auth_idp_case, "workload_provider_token")

    token = auth_idp_runtime.workload_provider_token()

    assert token.access_token
    assert token.claims


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
