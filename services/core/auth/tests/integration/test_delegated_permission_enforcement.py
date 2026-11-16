# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegated IAM denials through real middleware, policy and isolated persistence."""

import json
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nhx.common.auth.access_keys import public_jwk_from_private_key_pem, validate_access_key_token
from nhx.common.auth.token_claims import TokenClaims
from nhx.common.config import AuthConfig, HelixConfig
from nhx.common.config.base import AccessKeyConfig, TokenSigningConfig
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.entities.service import EntitiesService
from nhx.testing import ClientContext, create_test_client, short_unique_name

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("delegated_permission_enforcement")]
INFRASTRUCTURE = {"X-NHX-Principal-Id": "service:auth"}
WORKSPACES = "/apis/entities/v2/workspaces"
IAM = "/apis/auth/v2/iam/role-bindings"


@pytest.fixture(scope="module")
def platform(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[ClientContext, AuthConfig]]:
    private_key_file = tmp_path_factory.mktemp("delegated-permissions") / "signing.pem"
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key_file.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    shared = AuthConfig(
        enabled=True,
        policy_decision_point_provider="embedded",
        policy_decision_point_base_url="http://testserver",
        propagation_poll_interval_seconds=0.01,
        token_signing=TokenSigningConfig(
            issuer="http://testserver/apis/auth",
            key_id="delegation-test",
            private_key_file=str(private_key_file),
        ),
        access_keys=AccessKeyConfig(enabled=True),
    )
    # Every PDP request reloads bindings. Keep background refresh outside this
    # sequential proof so its bootstrap cannot replace an in-flight snapshot.
    service = AuthServiceConfig(**shared.model_dump(), bundle_cache_seconds=0, policy_data_refresh_interval=3600)
    with create_test_client(
        EntitiesService,
        client_type=ClientContext,
        auth_enabled=True,
        workspaces=[],
        projects=[],
        service_configs={
            HelixConfig: HelixConfig(base_url="http://127.0.0.1"),
            AuthConfig: shared,
            AuthServiceConfig: service,
        },
    ) as context:
        yield context, shared


@pytest.mark.parametrize("actor_role", [None, "Admin", "HelixAdmin"])
def test_denied_delegated_grant_cannot_elevate_viewer_access_key(
    platform: tuple[ClientContext, AuthConfig], actor_role: str | None
) -> None:
    context, shared = platform
    client = context.test_client
    workspace = short_unique_name("delegated-iam")
    viewer = short_unique_name("viewer")
    actor_name = "service:intake"
    created = client.post(WORKSPACES, json={"name": workspace}, headers={"X-NHX-Principal-Id": "service:files"})
    assert created.status_code == 201, created.text
    response = client.post(
        IAM, json={"principal": viewer, "role": "Viewer", "workspace": workspace}, headers=INFRASTRUCTURE
    )
    assert response.status_code in {200, 201}, response.text
    if actor_role:
        response = client.post(
            IAM,
            json={
                "principal": actor_name,
                "role": actor_role,
                "workspace": "system" if actor_role == "HelixAdmin" else workspace,
            },
            headers=INFRASTRUCTURE,
        )
        assert response.status_code in {200, 201}, response.text
    direct = {"X-NHX-Principal-Id": viewer}
    delegated = {"X-NHX-Principal-Id": actor_name, "X-NHX-Principal-On-Behalf-Of": viewer}
    response = client.post("/apis/auth/v2/access-keys", json={"name": "viewer-key"}, headers=direct)
    assert response.status_code == 200, response.text
    token = response.json()["token"]
    jwks = {"keys": [public_jwk_from_private_key_pem(shared)]}

    async def validate_locally(config: AuthConfig, token: str) -> TokenClaims | None:
        return await validate_access_key_token(config, token, jwks_override=jwks)

    with patch("nhx.common.auth.access_keys.validate_access_key_token", validate_locally):
        key_headers = {"Authorization": f"Bearer {token}"}
        for headers in (direct, key_headers, delegated):
            response = client.get(f"{WORKSPACES}/{workspace}/members", headers=headers)
            assert response.status_code == 403, response.text
            response = client.post(
                f"{WORKSPACES}/{workspace}/members",
                json={"principal": viewer, "roles": ["Admin"]},
                headers=headers,
            )
            assert response.status_code == 403, response.text
            response = client.get(IAM, headers=headers)
            assert response.status_code == 403, response.text
        # A denied delegated grant must not change the user's own key authority.
        response = client.get(f"{WORKSPACES}/{workspace}", headers=key_headers)
        assert response.status_code == 200, response.text
        response = client.get(f"{WORKSPACES}/{workspace}/members", headers=key_headers)
        assert response.status_code == 403, response.text
    response = client.get(f"{WORKSPACES}/{workspace}/members", headers=INFRASTRUCTURE)
    assert response.status_code == 200, response.text
    assert next(member["roles"] for member in response.json()["data"] if member["principal"] == viewer) == ["Viewer"]


def test_generic_control_entity_paths_require_subject_iam_and_preserve_internal_storage(
    platform: tuple[ClientContext, AuthConfig],
) -> None:
    context, _ = platform
    client = context.test_client
    workspace = short_unique_name("delegated-control")
    viewer = short_unique_name("viewer")
    creator = {"X-NHX-Principal-Id": "service:files"}
    response = client.post(WORKSPACES, json={"name": workspace}, headers=creator)
    assert response.status_code == 201, response.text
    response = client.post(
        IAM, json={"principal": viewer, "role": "Viewer", "workspace": workspace}, headers=INFRASTRUCTURE
    )
    assert response.status_code in {200, 201}, response.text
    delegated = {"X-NHX-Principal-Id": "service:intake", "X-NHX-Principal-On-Behalf-Of": viewer}
    base = f"{WORKSPACES}/{workspace}/entities"
    # Ordinary internal storage remains usable without entities.* user grants.
    response = client.post(f"{base}/operational_record", json={"name": "record", "data": {}}, headers=delegated)
    assert response.status_code == 201, response.text
    entity_id = response.json()["id"]
    response = client.get(f"/apis/entities/v2/entities/{entity_id}", headers=delegated)
    assert response.status_code == 200, response.text
    for entity_type in ("role_binding", "access_key"):
        response = client.post(f"{base}/{entity_type}", json={"name": "forged", "data": {}}, headers=delegated)
        assert response.status_code == 403, response.text
        response = client.get(f"{base}/{entity_type}", headers=delegated)
        assert response.status_code == 403, response.text
        response = client.get(f"{base}/{entity_type}/forged", headers=INFRASTRUCTURE)
        assert response.status_code == 404, response.text
    non_member = {**delegated, "X-NHX-Principal-On-Behalf-Of": short_unique_name("non-member")}
    response = client.get(f"{WORKSPACES}/{workspace}", headers=non_member)
    assert response.status_code == 403, response.text
    response = client.get(f"{WORKSPACES}/{workspace}", headers=delegated)
    assert response.status_code == 200, response.text
    # A subject with IAM grants can use the same internal storage path.
    admin = short_unique_name("administrator")
    response = client.post(
        IAM, json={"principal": admin, "role": "Admin", "workspace": workspace}, headers=INFRASTRUCTURE
    )
    assert response.status_code in {200, 201}, response.text
    response = client.get(f"{base}/role_binding", headers={**delegated, "X-NHX-Principal-On-Behalf-Of": admin})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("subject_has_workspace", [True, False])
def test_delegated_cross_workspace_list_filters_by_subject(
    platform: tuple[ClientContext, AuthConfig], subject_has_workspace: bool
) -> None:
    context, _ = platform
    client = context.test_client
    subject = short_unique_name("list-subject")
    actor = "service:models" if subject_has_workspace else "service:files"
    entity_type = short_unique_name("filtered-record")
    workspaces = [short_unique_name("subject-workspace"), short_unique_name("actor-workspace")]
    for workspace in workspaces:
        response = client.post(WORKSPACES, json={"name": workspace}, headers={"X-NHX-Principal-Id": "service:files"})
        assert response.status_code == 201, response.text
        response = client.post(
            f"{WORKSPACES}/{workspace}/entities/{entity_type}",
            json={"name": "record", "data": {}},
            headers={"X-NHX-Principal-Id": "service:intake"},
        )
        assert response.status_code == 201, response.text
    if subject_has_workspace:
        response = client.post(
            IAM,
            json={"principal": subject, "role": "Viewer", "workspace": workspaces[0]},
            headers=INFRASTRUCTURE,
        )
        assert response.status_code in {200, 201}, response.text
    response = client.post(
        IAM,
        json={"principal": actor, "role": "HelixAdmin", "workspace": "system"},
        headers=INFRASTRUCTURE,
    )
    assert response.status_code in {200, 201}, response.text
    response = client.get(
        f"{WORKSPACES}/-/entities/{entity_type}",
        headers={"X-NHX-Principal-Id": actor, "X-NHX-Principal-On-Behalf-Of": subject},
    )
    assert response.status_code == 200, response.text
    rows = response.json()["data"]
    assert [row["workspace"] for row in rows] == ([workspaces[0]] if subject_has_workspace else [])


def test_system_admin_iam_operations_filter_control_rows_and_counts(
    platform: tuple[ClientContext, AuthConfig],
) -> None:
    """System Admin can administer system without gaining IAM access in other memberships."""
    context, _ = platform
    client = context.test_client
    subject = short_unique_name("system-administrator")
    workspace = short_unique_name("viewer-workspace")
    response = client.post(WORKSPACES, json={"name": workspace}, headers={"X-NHX-Principal-Id": "service:files"})
    assert response.status_code == 201, response.text
    for target, role in [("system", "Admin"), (workspace, "Viewer")]:
        response = client.post(
            IAM, json={"principal": subject, "role": role, "workspace": target}, headers=INFRASTRUCTURE
        )
        assert response.status_code in {200, 201}, response.text
    forbidden_name = response.json()["name"]
    delegated = {**INFRASTRUCTURE, "X-NHX-Principal-On-Behalf-Of": subject}

    listed = client.get(IAM, headers=delegated, params={"page_size": 1000})
    assert listed.status_code == 200, listed.text
    bindings = listed.json()["data"]
    assert bindings and {binding["workspace"] for binding in bindings} == {"system"}
    assert listed.json()["pagination"]["total_results"] == len(bindings)

    counted = client.get(
        f"{WORKSPACES}/-/entities/role_binding",
        headers=delegated,
        params={"page_size": 1000, "count_by": "data.principal", "filter": json.dumps({"data.principal": subject})},
    )
    assert counted.status_code == 200, counted.text
    assert [binding["workspace"] for binding in counted.json()["data"]] == ["system"]
    assert counted.json()["pagination"]["total_results"] == 1
    assert counted.json()["group_counts"] == {subject: 1}

    created = client.post(
        IAM,
        headers=delegated,
        json={"principal": short_unique_name("recipient"), "role": "Viewer", "workspace": "system"},
    )
    assert created.status_code in {200, 201}, created.text
    resolved = client.get(f"{IAM}/{created.json()['name']}", headers=delegated)
    assert resolved.status_code == 200, resolved.text
    hidden = client.get(f"{IAM}/{forbidden_name}", headers=delegated)
    assert hidden.status_code == 404, hidden.text
    denied = client.post(
        IAM,
        headers=delegated,
        json={"principal": short_unique_name("denied"), "role": "Admin", "workspace": workspace},
    )
    assert denied.status_code == 403, denied.text
