# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signed workload exchange through the merged platform and persisted encrypted Secrets."""

import json
import os
import shutil
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import (
    CreateWorkspaceMemberRequest,
    CreateWorkspaceRequest,
)
from nhx.common.auth.models import Principal
from nhx.common.auth.workload_delegations import (
    DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
    WorkloadDelegationEntity,
    WorkloadDelegationScope,
    create_opaque_docker_proof_token,
    docker_workload_delegation_name,
    kubernetes_pod_uid_delegation_name,
)
from nhx.common.auth.workload_tokens import (
    issue_service_workload_access_token,
    resolve_workload_access_token,
)
from nhx.common.config import AuthConfig
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig
from nhx.common.entities import EntityClient
from nhx.core.auth.api.v2.workload_token_exchange import (
    KUBERNETES_POD_UID_REFERENCE_NAME,
    TOKEN_EXCHANGE_GRANT_TYPE,
    DecodedSubjectToken,
    VerifiedWorkloadReference,
)
from nhx.core.auth.app.embedded_pdp.data import apply_embedded_policy_document
from nhx.core.auth.app.seeding import run_seeding
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.auth.service import AuthService
from nhx.core.secrets.service import SecretsService
from nhx.testing import (
    TEST_ADMIN_EMAIL,
    ClientContext,
    create_test_client,
    short_unique_name,
)
from pydantic import SecretStr

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def launcher_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    go = shutil.which("go")
    if go is None:
        pytest.skip("Go toolchain is required for the launcher/server integration")
    root = Path(__file__).resolve().parents[4]
    binary = tmp_path_factory.mktemp("secret-launcher") / "jobs-launcher"
    subprocess.run([go, "build", "-o", str(binary), "."], cwd=root / "services/core/jobs/jobs-launcher", check=True)
    return binary


@pytest.mark.parametrize(
    "proof_kind,kind",
    [("docker", "job"), ("docker", "deployment"), ("kubernetes", "job"), ("kubernetes", "deployment")],
)
@pytest.mark.parametrize(
    "outcome",
    [
        "allowed",
        "missing-grant",
        "cross-workspace",
        "cross-workspace-allowed",
        "alias-grant",
        "scope-denied",
        "tampered",
    ],
)
def test_signed_workload_reads_persisted_secret(
    kind: str,
    proof_kind: str,
    outcome: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
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
        policy_decision_point_provider="embedded",
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://idp.invalid/",
            audience="nemo-helix",
            workload=OIDCWorkloadConfig(
                client_id="nemo-helix-workload",
                audience="nemo-helix",
                scope="platform:read platform:write secrets:read jobs:read",
                token_private_key_file=str(key_file),
                token_key_id="local-repro-key",
                kubernetes_token_review_enabled=proof_kind == "kubernetes",
            ),
        ),
    )
    # There is no external IdP in this in-process integration.
    monkeypatch.setattr(
        "nhx.common.auth.middleware.resolve_workload_subject_token",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr("nhx.common.auth.jwt.JWTValidator.validate_token", AsyncMock(return_value=None))

    with create_test_client(
        SecretsService,
        AuthService,
        auth_enabled=False,
        workspaces=[],
        projects=[],
        client_type=ClientContext,
        service_configs={
            AuthConfig: config,
            AuthService: AuthServiceConfig(
                enabled=True,
                policy_decision_point_provider="embedded",
                policy_decision_point_base_url="http://127.0.0.1",
                admin_email=TEST_ADMIN_EMAIL,
                allowed_service_principals=["integration-test"],
                policy_data_refresh_interval=0.1,
                bundle_cache_seconds=0.1,
            ),
        },
    ) as ctx:
        auth_headers = {"Authorization": "Bearer " + issue_service_workload_access_token(config, service_name="auth")}
        auth_entities = EntityClient(AsyncEntitiesClient.from_client(ctx.async_client.with_headers(auth_headers)))
        assert ctx.test_client.portal is not None
        assert ctx.test_client.portal.call(partial(run_seeding, auth_entities))
        ctx.test_client.portal.call(
            partial(
                apply_embedded_policy_document,
                auth_entities,
                skip_static_bootstrap=True,
            )
        )
        admin = ctx.client.with_headers(auth_headers)
        workspaces = WorkspacesClient.from_client(admin)
        workspace = short_unique_name("workload-secret")
        other = short_unique_name("other-secret")
        secret_values = {workspace: "persisted-value", other: "other-workspace-value"}
        for name, value in secret_values.items():
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
            SecretsClient.from_client(admin).create_secret(
                workspace=name,
                body=HelixSecretCreateRequest(name="credential", value=SecretStr(value)),
            ).data()
        creator = Principal(
            id="creator-id",
            email="creator@example.com",
            groups=["runtime-readers"],
            authz_aliases=["creator-alias"],
        )
        if outcome != "missing-grant":
            workspaces.create_workspace_member(
                workspace=workspace,
                body=CreateWorkspaceMemberRequest(
                    principal="creator-alias" if outcome == "alias-grant" else "runtime-readers", roles=["Viewer"]
                ),
                query_params={"wait_role_propagation": True},
            ).data()
        if outcome == "cross-workspace-allowed":
            workspaces.create_workspace_member(
                workspace=other,
                body=CreateWorkspaceMemberRequest(principal="runtime-readers", roles=["Viewer"]),
                query_params={"wait_role_propagation": True},
            ).data()
        scope = WorkloadDelegationScope(
            workload_workspace=workspace,
            workload_kind=kind,
            workload_instance_id="runtime-instance",
        )
        name = docker_workload_delegation_name(scope=scope, workload_generation="generation-1")
        bound_reference_name = None
        bound_reference_value = None
        workload_subject = name
        proof, proof_hash = create_opaque_docker_proof_token(name)
        if proof_kind == "kubernetes":
            workload_subject = "system:serviceaccount:nemo-runs:job-runner"
            bound_reference_name = KUBERNETES_POD_UID_REFERENCE_NAME
            bound_reference_value = "pod-uid-123"
            name = kubernetes_pod_uid_delegation_name(
                workload_audience="nemo-helix", workload_subject=workload_subject, pod_uid=bound_reference_value
            )
            proof = "kubernetes-subject-proof"
            proof_hash = None

            async def review_subject(config: AuthConfig, subject_token: str, audience: str) -> DecodedSubjectToken:
                # TokenReview is the external Kubernetes trust boundary; local
                # delegation lookup, minting and access-token verification stay real.
                if subject_token != proof or audience != "nemo-helix-workload":
                    raise HTTPException(status_code=401, detail="Invalid workload proof")
                return DecodedSubjectToken(
                    claims={"sub": workload_subject, "groups": ["system:serviceaccounts"]},
                    bound_reference=VerifiedWorkloadReference(
                        name=KUBERNETES_POD_UID_REFERENCE_NAME, value="pod-uid-123"
                    ),
                )

            monkeypatch.setattr(
                "nhx.core.auth.api.v2.workload_token_exchange._decode_kubernetes_subject_token", review_subject
            )
        delegation = WorkloadDelegationEntity(
            name=name,
            workspace="system",
            workload_subject=workload_subject,
            workload_audience="nemo-helix",
            workload_workspace=workspace,
            workload_kind=kind,
            workload_id="runtime-instance",
            workload_generation="generation-1",
            auth_context=AuthContext.from_principal(creator),
            opaque_subject_token_hash=proof_hash,
            bound_reference_name=bound_reference_name,
            bound_reference_value=bound_reference_value,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )
        assert ctx.test_client.portal is not None
        ctx.test_client.portal.call(partial(auth_entities.create, delegation))
        exchanged = ctx.test_client.post(
            "/apis/auth/token",
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
                "client_id": "nemo-helix-workload",
                "subject_token": proof,
                "subject_token_type": DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE
                if proof_kind == "docker"
                else "urn:ietf:params:oauth:token-type:jwt",
                "scope": "jobs:read" if outcome == "scope-denied" else "secrets:read",
            },
        )
        print("exchange", outcome, exchanged.status_code, flush=True)
        assert exchanged.status_code == 200, exchanged.text
        token = exchanged.json()["access_token"]
        if outcome == "tampered":
            parts = token.split(".")
            parts[2] = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
            token = ".".join(parts)
        if outcome == "allowed":
            service_token = issue_service_workload_access_token(config, service_name="jobs", on_behalf_of=creator)
            control = ctx.test_client.get(
                f"/apis/secrets/v2/workspaces/{workspace}/secrets/credential/access",
                headers={"Authorization": f"Bearer {service_token}"},
            )
            print(
                "signed service:jobs delegation control",
                control.status_code,
                flush=True,
            )
            assert control.status_code == 200, control.text
            assert ctx.test_client.portal is not None
            resolved = ctx.test_client.portal.call(partial(resolve_workload_access_token, config, token))
            assert resolved is not None
            print(
                "verified exchanged identity",
                {
                    "actor": resolved.principal.id,
                    "caller_kind": resolved.principal.caller_kind,
                    "subject_groups": resolved.principal.on_behalf_of_groups,
                },
                flush=True,
            )
            assert resolved.principal.on_behalf_of == "creator-id"
            assert resolved.principal.on_behalf_of_email == "creator@example.com"
            assert resolved.principal.on_behalf_of_groups == ["runtime-readers"]
            assert resolved.principal.on_behalf_of_authz_aliases == [
                "creator-id",
                "creator@example.com",
                "creator-alias",
            ]
        target = other if outcome.startswith("cross-workspace") else workspace
        response = ctx.test_client.get(
            f"/apis/secrets/v2/workspaces/{target}/secrets/credential/access",
            headers={"Authorization": f"Bearer {token}"},
        )
        print(
            "signed workload Secret access",
            outcome,
            response.status_code,
            response.text if response.status_code != 200 else "value retrieved",
            flush=True,
        )
        allowed = outcome in {"allowed", "cross-workspace-allowed", "alias-grant"}
        assert response.status_code == (200 if allowed else 401 if outcome == "tampered" else 403), response.text
        if allowed:
            assert response.json()["value"] == secret_values[target]
        if outcome == "allowed":
            launcher_binary = request.getfixturevalue("launcher_binary")
            _run_launcher_against_service(ctx, launcher_binary, proof, workspace, tmp_path, creator)


def _run_launcher_against_service(
    ctx: ClientContext, binary: Path, proof: str | None, workspace: str, tmp_path: Path, creator: Principal
) -> None:
    # Bridge real Go HTTP to the in-process services; only discovery's advertised
    # address changes to the temporary listener. Exchange and authorization are real.
    class Bridge(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.forward()

        def do_POST(self) -> None:
            self.forward()

        def log_message(self, format: str, *args: object) -> None:
            pass

        def forward(self) -> None:
            response = ctx.test_client.request(
                self.command,
                self.path,
                content=self.rfile.read(int(self.headers.get("Content-Length", "0"))),
                headers={key: value for key, value in self.headers.items() if key.lower() != "host"},
            )
            content = response.content
            if self.path == "/apis/auth/discovery":
                data = response.json()
                data["oidc"]["workload_token_endpoint"] = f"http://127.0.0.1:{server.server_port}/apis/auth/token"
                content = json.dumps(data).encode()
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Bridge)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        proof_file = tmp_path / "proof"
        proof_file.write_text(proof or "")
        env = {key: value for key, value in os.environ.items() if not key.startswith(("NHX_", "NEMO_", "OTEL_"))}
        env.update(
            {
                "NHX_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                "NHX_WORKLOAD_IDENTITY_TOKEN_FILE": str(proof_file),
                "NHX_ACCESS_TOKEN_FILE": "/missing/ignored-when-workload-selected",
                "NEMO_JOB_SECRETS": f"TEST_SECRET={workspace}/credential",
            }
        )
        if proof is None:
            env.pop("NHX_WORKLOAD_IDENTITY_TOKEN_FILE")
            env.pop("NHX_ACCESS_TOKEN_FILE")
            env["NHX_PRINCIPAL"] = creator.model_dump_json(exclude_none=True)
        child_marker = tmp_path / "child-started"
        check_script = tmp_path / "check-secret.sh"
        check_script.write_text(f'touch "{child_marker}" && test "$TEST_SECRET" = "persisted-value"')
        child = [str(binary), "run", "--", "/bin/sh", str(check_script)]
        result = subprocess.run(child, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert child_marker.exists()
        assert "persisted-value" not in result.stdout + result.stderr
        child_marker.unlink()
        # A selected credential failure must stop before child execution.
        proof_file.write_text("invalid-proof")
        if proof is None:
            env.pop("NHX_PRINCIPAL")
        result = subprocess.run(child, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode != 0
        assert not child_marker.exists()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("grant", ["group", "email", "alias", "account", "delegated"])
def test_trusted_launcher_reads_granted_secret(launcher_binary: Path, tmp_path: Path, grant: str) -> None:
    with create_test_client(SecretsService, auth_enabled=True, client_type=ClientContext) as ctx:
        admin = ctx.client.with_headers({"X-NHX-Principal-Id": TEST_ADMIN_EMAIL})
        workspace = short_unique_name("trusted-secret")
        workspaces = WorkspacesClient.from_client(admin)
        creator = Principal(
            id="creator-id",
            email="creator@example.com",
            groups=["runtime-readers"],
            account_id="account-creator",
            authz_aliases=["creator-alias"],
        )
        granted_principal = {
            "group": "runtime-readers",
            "email": "creator@example.com",
            "alias": "creator-alias",
            "account": "account-creator",
            "delegated": "runtime-readers",
        }[grant]
        if grant == "delegated":
            creator = Principal(
                id="service:garak",
                groups=["actor-group"],
                on_behalf_of="creator-id",
                on_behalf_of_email=creator.email,
                on_behalf_of_groups=creator.groups,
                on_behalf_of_account_id=creator.account_id,
                on_behalf_of_authz_aliases=creator.authz_aliases,
            )
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        workspaces.create_workspace_member(
            workspace=workspace,
            body=CreateWorkspaceMemberRequest(principal=granted_principal, roles=["Viewer"]),
            query_params={"wait_role_propagation": True},
        ).data()
        SecretsClient.from_client(admin).create_secret(
            workspace=workspace,
            body=HelixSecretCreateRequest(name="credential", value=SecretStr("persisted-value")),
        ).data()
        _run_launcher_against_service(ctx, launcher_binary, None, workspace, tmp_path, creator)
