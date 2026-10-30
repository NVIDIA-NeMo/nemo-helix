# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Job proof exchange, gateway authentication, and Secret authorization.

Policy/token-shape edge cases live in the focused Rego and middleware unit tests.
These tests protect the signed delegation and the target-workspace creator check.
"""

from collections.abc import Generator

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import WorkloadDelegationScope
from nemo_helix_plugin.auth.workload_identity import build_docker_opaque_workload_delegation
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.common.auth import Principal
from nhx.common.auth.workload_tokens import issue_service_workload_access_token
from nhx.common.config import AuthConfig
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig, TokenSigningConfig
from nhx.common.entities.client import EntityClient
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.auth.service import AuthService
from nhx.core.secrets.service import SecretsService
from nhx.testing import ClientContext, create_test_client, grant_workspace_role, short_unique_name
from pydantic import SecretStr

WORKSPACE = "job-secrets"
pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("workload_secret_authorization")]


@pytest.fixture(scope="module")
def workload_secret_platform(tmp_path_factory: pytest.TempPathFactory) -> Generator[tuple[ClientContext, AuthConfig]]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_file = tmp_path_factory.mktemp("workload-secret") / "workload.pem"
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    config = AuthConfig(
        enabled=True,
        policy_decision_point_provider="embedded",
        policy_decision_point_base_url="http://127.0.0.1",
        token_signing=TokenSigningConfig(
            issuer="http://127.0.0.1/apis/auth", key_id="workload-secret-test", private_key_file=str(key_file)
        ),
        oidc=OIDCConfig(
            enabled=True, workload=OIDCWorkloadConfig(client_id="nemo-helix-workload", audience="nemo-helix")
        ),
    )
    service_config = AuthServiceConfig(**config.model_dump(), bundle_cache_seconds=0, policy_data_refresh_interval=60)
    # Use signed service tokens for setup, not create_test_client's optional
    # trusted-header seeding. Both actual auth configurations are enabled.
    with create_test_client(
        AuthService,
        SecretsService,
        service_configs={AuthConfig: config, AuthServiceConfig: service_config},
        workspaces=[],
        projects=[],
        client_type=ClientContext,
    ) as ctx:
        token = issue_service_workload_access_token(config, service_name="auth")
        service_client = ctx.client.with_headers({"Authorization": f"Bearer {token}"})
        for workspace in (WORKSPACE, "shared-secrets", "other-secrets"):
            WorkspacesClient.from_client(service_client).create_workspace(body=CreateWorkspaceRequest(name=workspace))
            SecretsClient.from_client(service_client).create_secret(
                workspace=workspace,
                body=HelixSecretCreateRequest(name="signing-key", value=SecretStr("test-secret-value")),
            )
            if workspace != "other-secrets":
                grant_workspace_role(
                    service_client, workspace=workspace, principal="creator@example.com", roles=["Viewer"]
                )
        yield ctx, config


def _job_token(ctx: ClientContext, config: AuthConfig, creator: str) -> str:
    attempt = short_unique_name("attempt")
    delegation, proof = build_docker_opaque_workload_delegation(
        scope=WorkloadDelegationScope(workload_workspace=WORKSPACE, workload_kind="job", workload_instance_id="job-1"),
        workload_audience="nemo-helix",
        workload_generation=attempt,
        auth_context=AuthContext.from_principal(Principal(id=creator)),
        ttl_seconds_active=600,
        job_id="job-1",
        attempt_id=attempt,
        step_id="step-1",
    )
    service_token = issue_service_workload_access_token(config, service_name="auth")
    service_headers = {"Authorization": f"Bearer {service_token}"}
    entity_client = EntityClient(AsyncEntitiesClient.from_client(ctx.async_client.with_headers(service_headers)))
    assert ctx.test_client.portal is not None
    ctx.test_client.portal.call(entity_client.create, delegation)
    if creator == "outsider@example.com":
        # The actor's own roles must not replace the creator's missing access.
        grant_workspace_role(
            ctx.client.with_headers(service_headers),
            workspace=WORKSPACE,
            principal=delegation.workload_subject,
            roles=["Viewer", "JobRunner"],
        )
    response = ctx.test_client.post(
        "/apis/auth/token",
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "client_id": "nemo-helix-workload",
            "subject_token": proof,
            "subject_token_type": "urn:nvidia:nemo:params:oauth:token-type:docker-opaque-workload-proof",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest.mark.parametrize(
    "creator,workspace,status",
    [
        pytest.param("creator@example.com", WORKSPACE, 200, id="authorized-creator"),
        pytest.param("outsider@example.com", WORKSPACE, 403, id="actor-roles-do-not-replace-creator-access"),
        pytest.param("creator@example.com", "shared-secrets", 200, id="authorized-cross-workspace-reference"),
        pytest.param("creator@example.com", "other-secrets", 403, id="target-workspace-permission-required"),
    ],
)
def test_signed_job_secret_access(
    workload_secret_platform: tuple[ClientContext, AuthConfig], creator: str, workspace: str, status: int
) -> None:
    ctx, config = workload_secret_platform
    token = _job_token(ctx, config, creator)
    headers = {"Authorization": f"Bearer {token}"}
    path = f"/apis/secrets/v2/workspaces/{workspace}/secrets/signing-key/access"
    gateway = ctx.test_client.get("/apis/auth/ext-authz" + path, headers=headers)
    assert gateway.status_code == 200, gateway.text
    response = ctx.test_client.get(path, headers=headers)
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["value"] == "test-secret-value"
