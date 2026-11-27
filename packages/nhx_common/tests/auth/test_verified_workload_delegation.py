# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from nemo_helix_plugin.auth import AuthContext
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import AuthContext as CapturedAuthContext
from nhx.common.auth.models import Principal
from nhx.common.auth.token_claims import SUBJECT_ALIASES_CLAIM, ActorClaims, TokenClaims
from nhx.common.auth.token_resolver import ResolvedBearerToken, ResolvedTokenKind
from nhx.common.config import AuthConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("credential", ["workload_access_token", "oidc_access_token", "access_key", "none"])
@pytest.mark.parametrize("substitute", ["none", "actor", "subject"])
async def test_delegation_fact_is_bound_to_verified_credential(credential: str, substitute: str) -> None:
    claims = TokenClaims(
        subject="creator",
        email="creator@example.test",
        groups=["readers"],
        scopes=[],
        raw_claims={SUBJECT_ALIASES_CLAIM: ["creator-alias", 7, "creator"]},
        actor=ActorClaims(subject="docker-runtime", groups=[]),
    )
    workload = ResolvedBearerToken(claims=claims, token_kind="workload_access_token")
    principal = workload.principal
    assert principal.verified_workload_delegation
    assert (principal.id, principal.authz_aliases) == ("docker-runtime", ["docker-runtime"])
    assert principal.on_behalf_of == "creator"
    assert principal.on_behalf_of_email == "creator@example.test"
    assert principal.on_behalf_of_groups == ["readers"]
    assert principal.on_behalf_of_authz_aliases == ["creator", "creator@example.test", "creator-alias"]
    if substitute == "actor":
        principal = principal.model_copy(update={"id": "different-actor"})
    elif substitute == "subject":
        principal = principal.model_copy(update={"on_behalf_of": "different-subject"})
    # Even a model-supplied true value needs the actual matching resolved credential.
    principal.verified_workload_delegation = True
    resolved = None
    if credential == "workload_access_token":
        resolved = workload
    elif credential == "oidc_access_token":
        resolved = ResolvedBearerToken(claims=claims, token_kind="oidc_access_token")
    elif credential == "access_key":
        resolved = ResolvedBearerToken(claims=claims, token_kind="access_key")

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"result": {"allowed": False}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        auth = AuthClient(
            principal=principal,
            resolved_bearer_token=resolved,
            http_client=http,
            service_name="secrets",
            config=AuthConfig(enabled=True, policy_decision_point_base_url="http://localhost"),
        )
        await auth.authorize_request("GET", "/apis/secrets/v2/workspaces/ws/secrets/key/access")
    assert json.loads(requests[0].content)["input"].get("verified_workload_delegation", False) is (
        credential == "workload_access_token" and substitute == "none"
    )


@pytest.mark.parametrize("token_kind", ["oidc_access_token", "access_key"])
def test_a_direct_credential_ignores_subject_alias_claims(token_kind: ResolvedTokenKind) -> None:
    """Only a verified workload token's delegation carries the creator's alias claim; a direct credential can't add one."""
    claims = TokenClaims(
        subject="creator",
        email="creator@example.test",
        groups=["readers"],
        scopes=[],
        raw_claims={SUBJECT_ALIASES_CLAIM: ["creator-alias"]},
    )
    principal = ResolvedBearerToken(claims=claims, token_kind=token_kind).principal
    assert (principal.id, principal.authz_aliases) == ("creator", ["creator", "creator@example.test"])
    assert not principal.verified_workload_delegation
    assert principal.on_behalf_of_authz_aliases == []


def test_verified_fact_does_not_cross_identity_serialization() -> None:
    principal = Principal(id="docker-runtime", on_behalf_of="creator", verified_workload_delegation=True)
    assert principal.caller_kind == "principal"
    assert not principal.is_privileged()
    assert "verified_workload_delegation" not in principal.model_dump()
    assert "verified_workload_delegation" not in principal.get_env_var()["NHX_PRINCIPAL"]
    restored = Principal.from_headers(principal.get_headers())
    assert restored is not None
    assert not restored.verified_workload_delegation
    assert not CapturedAuthContext.from_principal(principal).to_principal().verified_workload_delegation
    assert "verified_workload_delegation" not in AuthContext.from_principal(principal).model_dump_json()
