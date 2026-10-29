# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signed workload -> gateway ext-authz -> Secret service/PDP integration tests.

No bearer resolver, policy decision, delegation store, or Secret handler is mocked.
Job actors remain non-service principals and require the creator's read access.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import jwt
import pytest
import yaml
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
from nhx.common.config import AuthConfig, Configuration, HelixConfig
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig, TokenSigningConfig
from nhx.common.entities.client import EntityClient
from nhx.core.auth.api.v2.workload_token_exchange import WorkloadTokenExchangeService
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.auth.service import AuthService
from nhx.core.secrets.service import SecretsService
from nhx.testing import ClientContext, create_test_client, grant_workspace_role, short_unique_name
from pydantic import SecretStr

WORKSPACE = "job-secrets"
SECRET_PATH = f"/apis/secrets/v2/workspaces/{WORKSPACE}/secrets/signing-key/access"

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("workload_secret_authorization")]


@pytest.fixture(scope="module")
def workload_secret_platform(tmp_path_factory: pytest.TempPathFactory) -> Generator[tuple[ClientContext, AuthConfig]]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_file = tmp_path_factory.mktemp("workload-secret") / "workload.pem"
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    config = AuthConfig(
        enabled=True,
        policy_decision_point_provider="embedded",
        policy_decision_point_base_url="http://127.0.0.1",
        token_signing=TokenSigningConfig(
            issuer="http://127.0.0.1/apis/auth",
            key_id="workload-secret-test",
            private_key_file=str(key_file),
        ),
        oidc=OIDCConfig(
            enabled=True,
            workload=OIDCWorkloadConfig(client_id="nemo-helix-workload", audience="nemo-helix"),
        ),
    )
    service_config = AuthServiceConfig(
        **config.model_dump(),
        bundle_cache_seconds=0,
        policy_data_refresh_interval=60,
    )
    # auth_enabled=False prevents the test helper from seeding trusted-header
    # identities. Both actual auth configurations above are enabled; setup uses
    # signed service tokens just as runtime service-to-service requests do.
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
                    service_client,
                    workspace=workspace,
                    principal="creator@example.com",
                    roles=["Viewer"],
                )
        yield ctx, config


def _job_credentials(ctx: ClientContext, config: AuthConfig, creator: str) -> tuple[str, str]:
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
    token = response.json()["access_token"]
    public_key = WorkloadTokenExchangeService().workload_signing_key(config).public_key
    claims = jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        audience="nemo-helix",
        issuer="http://127.0.0.1/apis/auth",
    )
    assert claims["sub"] == creator
    assert claims["act"]["sub"] == delegation.workload_subject
    assert claims["nhx_workload"] == {"kind": "job", "workspace": WORKSPACE, "job_id": "job-1"}
    if creator == "outsider@example.com":
        # An actor's own read/logging roles cannot replace the creator's access.
        grant_workspace_role(
            ctx.client.with_headers(service_headers),
            workspace=WORKSPACE,
            principal=delegation.workload_subject,
            roles=["Viewer", "JobRunner"],
        )
    return token, proof


@pytest.mark.parametrize(
    ("actor", "creator", "status"),
    [
        pytest.param("job", "creator@example.com", 200, id="job-authorized-creator"),
        pytest.param("job", "outsider@example.com", 403, id="job-unauthorized-creator"),
        pytest.param("service", "creator@example.com", 200, id="service-authorized-creator"),
        pytest.param("service", "outsider@example.com", 403, id="service-unauthorized-creator"),
    ],
)
def test_signed_workload_secret_authorization(
    workload_secret_platform: tuple[ClientContext, AuthConfig], actor: str, creator: str, status: int
) -> None:
    ctx, config = workload_secret_platform
    if actor == "job":
        token, _ = _job_credentials(ctx, config, creator)
    else:
        token = issue_service_workload_access_token(config, service_name="jobs", on_behalf_of=creator)
    headers = {"Authorization": f"Bearer {token}"}
    gateway = ctx.test_client.get("/apis/auth/ext-authz" + SECRET_PATH, headers=headers)
    assert gateway.status_code == 200, gateway.text
    assert not any(key.lower().startswith("x-nhx-principal") for key in gateway.headers)
    response = ctx.test_client.get(SECRET_PATH, headers=headers)
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["value"] == "test-secret-value"


@pytest.mark.parametrize("workspace,status", [("shared-secrets", 200), ("other-secrets", 403)])
def test_job_secret_reference_uses_creator_access_in_target_workspace(
    workload_secret_platform: tuple[ClientContext, AuthConfig], workspace: str, status: int
) -> None:
    ctx, config = workload_secret_platform
    token, _ = _job_credentials(ctx, config, "creator@example.com")
    response = ctx.test_client.get(
        SECRET_PATH.replace(WORKSPACE, workspace), headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == status, response.text


@pytest.mark.parametrize("change", ["no_marker", "no_actor", "deployment", "empty_job_id", "wrong_scope"])
def test_secret_capability_requires_signed_job_delegation(
    workload_secret_platform: tuple[ClientContext, AuthConfig], change: str
) -> None:
    ctx, config = workload_secret_platform
    token, _ = _job_credentials(ctx, config, "creator@example.com")
    signing_key = WorkloadTokenExchangeService().workload_signing_key(config)
    claims = jwt.decode(token, signing_key.public_key, algorithms=["RS256"], audience="nemo-helix")
    if change == "no_marker":
        del claims["nhx_workload"]
    elif change == "no_actor":
        del claims["act"]
    elif change == "deployment":
        claims["nhx_workload"]["kind"] = "deployment"
    elif change == "empty_job_id":
        claims["nhx_workload"]["job_id"] = ""
    else:
        claims["scope"] = "files:read"
    token = jwt.encode(claims, signing_key.private_key, algorithm="RS256", headers={"kid": signing_key.kid})
    headers = {"Authorization": f"Bearer {token}"}
    gateway = ctx.test_client.get("/apis/auth/ext-authz" + SECRET_PATH, headers=headers)
    assert gateway.status_code == 200, gateway.text
    response = ctx.test_client.get(SECRET_PATH, headers=headers)
    assert response.status_code == 403, response.text


def test_job_secret_capability_does_not_grant_service_privileges(
    workload_secret_platform: tuple[ClientContext, AuthConfig],
) -> None:
    ctx, config = workload_secret_platform
    token, _ = _job_credentials(ctx, config, "creator@example.com")
    # Viewer creator cannot write Secrets. The Job marker does not change that.
    response = ctx.test_client.post(
        f"/apis/secrets/v2/workspaces/{WORKSPACE}/secrets",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "not-allowed", "value": "test-value"},
    )
    assert response.status_code == 403, response.text


def test_gateway_rejects_forged_workload_token(workload_secret_platform: tuple[ClientContext, AuthConfig]) -> None:
    ctx, config = workload_secret_platform
    token = issue_service_workload_access_token(config, service_name="jobs", on_behalf_of="creator@example.com")
    header, payload, _ = token.split(".")
    response = ctx.test_client.get(
        "/apis/auth/ext-authz" + SECRET_PATH,
        headers={"Authorization": f"Bearer {header}.{payload}.invalid-signature"},
    )
    assert response.status_code == 401, response.text


@pytest.mark.skipif(
    os.environ.get("NHX_TEST_WORKLOAD_GATEWAY") != "1" or sys.platform != "linux",
    reason="opt-in Linux test requiring Docker, cached Envoy v1.37.0, and Go",
)
def test_launcher_through_real_envoy(
    workload_secret_platform: tuple[ClientContext, AuthConfig], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the reference gateway, real launcher, signed tokens, policy, and Secrets.

    A loopback HTTP bridge exposes the in-process ASGI services to Envoy. Only
    the listener TLS and backend addresses differ from the reference config.
    A local IdP discovery/JWKS stub keeps this test independent of external SSO;
    workload proof verification and access-token signing/verification are real.
    """
    ctx, config = workload_secret_platform
    root = Path(__file__).resolve().parents[5]
    calls: list[tuple[str, int]] = []

    class Bridge(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            pass

        def forward(self) -> None:
            if self.path.startswith("/idp/"):
                document = (
                    {"keys": []}
                    if self.path.endswith("/jwks")
                    else {"issuer": config.oidc.issuer, "jwks_uri": config.oidc.issuer + "/jwks"}
                )
                content = json.dumps(document).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            response = ctx.test_client.request(self.command, self.path, content=body, headers=dict(self.headers))
            calls.append((self.path, response.status_code))
            self.send_response(response.status_code)
            self.send_header("content-type", response.headers.get("content-type", "application/json"))
            self.send_header("content-length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

        do_GET = forward
        do_POST = forward

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Bridge)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(config.oidc, "issuer", f"http://127.0.0.1:{backend.server_port}/idp")
    container = "nhx-launcher-auth-test-" + uuid.uuid4().hex[:12]
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        gateway_port = reservation.getsockname()[1]
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    gateway_config = yaml.safe_load((root / "contrib/auth/authentik/gateway/envoy.yaml").read_text())
    listener = gateway_config["static_resources"]["listeners"][0]
    listener["address"]["socket_address"] = {"address": "127.0.0.1", "port_value": gateway_port}
    del listener["filter_chains"][0]["transport_socket"]
    for cluster in gateway_config["static_resources"]["clusters"]:
        endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
        endpoint["address"]["socket_address"] = {"address": "127.0.0.1", "port_value": backend.server_port}
    gateway_file = tmp_path / "envoy.yaml"
    gateway_file.write_text(yaml.safe_dump(gateway_config))
    monkeypatch.setattr(Configuration.get_service_config(HelixConfig), "advertised_base_url", gateway_url)
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--pull=never",
                "--rm",
                "-d",
                "--name",
                container,
                "--network",
                "host",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{gateway_file}:/etc/envoy/envoy.yaml:ro",
                "--entrypoint",
                "envoy",
                "envoyproxy/envoy:v1.37.0",
                "-c",
                "/etc/envoy/envoy.yaml",
                "--concurrency",
                "1",
                "--log-level",
                "warning",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        with httpx.Client(base_url=gateway_url, timeout=30) as gateway:
            for _ in range(100):
                try:
                    if gateway.get("/health/ready").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True, timeout=10)
                pytest.fail("Envoy failed readiness: " + logs.stdout + logs.stderr)
            response = gateway.get(
                SECRET_PATH,
                headers={
                    "Authorization": "Bearer invalid.token.signature",
                    "X-NHX-Principal-Id": "service:jobs",
                    "X-NHX-Principal-On-Behalf-Of": "creator@example.com",
                },
            )
            assert response.status_code == 401, response.text

        binary = tmp_path / "jobs-launcher"
        subprocess.run(
            ["go", "build", "-o", str(binary), "."],
            cwd=root / "services/core/jobs/jobs-launcher",
            check=True,
            timeout=120,
        )
        proof_file = tmp_path / "proof"
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("NHX_JOB_LAUNCHER_", "NEMO_JOB_", "OTEL_"))
        }
        env.update(
            {
                "NHX_BASE_URL": gateway_url,
                "NHX_SECRETS_URL": gateway_url,
                "NHX_WORKLOAD_IDENTITY_TOKEN_FILE": str(proof_file),
                "NHX_PRINCIPAL": "",
                "NEMO_JOB_SECRETS": f"SIGNING_KEY={WORKSPACE}/signing-key",
            }
        )
        for creator, expected in [("creator@example.com", 200), ("outsider@example.com", 403)]:
            _, proof = _job_credentials(ctx, config, creator)
            proof_file.write_text(proof)
            start = len(calls)
            result = subprocess.run(
                [str(binary), "run", "--", "sh", "-c", 'test "$SIGNING_KEY" = test-secret-value && echo CHILD_STARTED'],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert ("/apis/auth/token", 200) in calls[start:]
            assert ("/apis/auth/ext-authz" + SECRET_PATH, 200) in calls[start:]
            assert (SECRET_PATH, expected) in calls[start:]
            if expected == 200:
                assert result.returncode == 0, result.stdout + result.stderr
                assert "CHILD_STARTED" in result.stdout
            else:
                assert result.returncode != 0
                assert "status code 403" in result.stdout + result.stderr
                assert "CHILD_STARTED" not in result.stdout
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=30)
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)
