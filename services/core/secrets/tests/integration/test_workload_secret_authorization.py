# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signed workload exchange through the merged platform and persisted encrypted Secrets."""

from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, CreateWorkspaceRequest
from nhx.common.auth.models import Principal
from nhx.common.auth.workload_delegations import (
    DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
    WorkloadDelegationEntity,
    WorkloadDelegationScope,
    create_opaque_docker_proof_token,
    docker_workload_delegation_name,
)
from nhx.common.config import AuthConfig
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig
from nhx.common.platform_client_context import service_principal_auth_headers
from nhx.core.auth.api.v2.workload_token_exchange import TOKEN_EXCHANGE_GRANT_TYPE
from nhx.core.secrets.config import SecretsServiceConfig
from nhx.core.secrets.service import SecretsService
from nhx.testing import TEST_ADMIN_EMAIL, ClientContext, create_test_client, short_unique_name
from pydantic import SecretStr

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("kind", ["job", "deployment"])
@pytest.mark.parametrize("outcome", ["allowed", "missing-grant", "cross-workspace", "scope-denied", "tampered"])
def test_signed_workload_reads_persisted_secret(
    kind: str,
    outcome: str,
    tmp_path: Path,
    service_config: SecretsServiceConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_file = tmp_path / "workload.pem"
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    config = AuthConfig(
        enabled=True,
        policy_decision_point_base_url="http://127.0.0.1",
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://idp.invalid/",
            workload=OIDCWorkloadConfig(
                client_id="nemo-helix-workload",
                audience="nemo-helix",
                scope="platform:read platform:write secrets:read jobs:read",
                token_private_key_file=str(key_file),
            ),
        ),
    )
    # There is no external IdP in this in-process integration.
    monkeypatch.setattr("nhx.common.auth.middleware.resolve_workload_subject_token", AsyncMock(return_value=None))
    monkeypatch.setattr("nhx.common.auth.jwt.JWTValidator.validate_token", AsyncMock(return_value=None))

    with create_test_client(
        SecretsService,
        auth_enabled=True,
        client_type=ClientContext,
        service_configs={SecretsService: service_config, AuthConfig: config},
    ) as ctx:
        admin = ctx.client.with_headers(
            service_principal_auth_headers(
                "integration-test",
                on_behalf_of=Principal(id=TEST_ADMIN_EMAIL, email=TEST_ADMIN_EMAIL),
            )
        )
        workspaces = WorkspacesClient.from_client(admin)
        workspace = short_unique_name("workload-secret")
        other = short_unique_name("other-secret")
        for name in (workspace, other):
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
            SecretsClient.from_client(admin).create_secret(
                workspace=name,
                body=HelixSecretCreateRequest(name="credential", value=SecretStr("persisted-value")),
            ).data()
        creator = Principal(id="creator@example.com", email="creator@example.com", groups=["runtime-readers"])
        if outcome != "missing-grant":
            workspaces.create_workspace_member(
                workspace=workspace,
                body=CreateWorkspaceMemberRequest(principal="runtime-readers", roles=["Viewer"]),
                query_params={"wait_role_propagation": True},
            ).data()
        scope = WorkloadDelegationScope(
            workload_workspace=workspace,
            workload_kind=kind,
            workload_instance_id="runtime-instance",
        )
        name = docker_workload_delegation_name(scope=scope, workload_generation="generation-1")
        proof, proof_hash = create_opaque_docker_proof_token(name)
        delegation = WorkloadDelegationEntity(
            name=name,
            workspace="system",
            workload_subject=name,
            workload_audience="nemo-helix",
            workload_workspace=workspace,
            workload_kind=kind,
            workload_id="runtime-instance",
            workload_generation="generation-1",
            auth_context=AuthContext.from_principal(creator),
            opaque_subject_token_hash=proof_hash,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )
        assert ctx.test_client.portal is not None
        ctx.test_client.portal.call(partial(ctx.entity_client.as_service("auth", internal=True).create, delegation))
        exchanged = ctx.test_client.post(
            "/apis/auth/token",
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
                "client_id": "nemo-helix-workload",
                "subject_token": proof,
                "subject_token_type": DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
                "scope": "jobs:read" if outcome == "scope-denied" else "secrets:read",
            },
        )
        assert exchanged.status_code == 200, exchanged.text
        token = exchanged.json()["access_token"]
        if outcome == "tampered":
            parts = token.split(".")
            parts[2] = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
            token = ".".join(parts)
        target = other if outcome == "cross-workspace" else workspace
        response = ctx.test_client.get(
            f"/apis/secrets/v2/workspaces/{target}/secrets/credential/access",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == (200 if outcome == "allowed" else 401 if outcome == "tampered" else 403), (
            response.text
        )
        if outcome == "allowed":
            assert response.json()["value"] == "persisted-value"
