# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task clients must enforce creator grants and still publish platform artifacts."""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from nemo_anonymizer_plugin.app.task_config import AnonymizerStepConfig
from nemo_anonymizer_plugin.tasks.anonymizer import run as anonymizer
from nemo_data_designer_plugin.jobs import bridge
from nemo_data_designer_plugin.jobs.spec import DataDesignerStepConfig
from nemo_helix_plugin.client.auth import StaticToken
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.client_provider import DefaultNemoClientProvider, set_nemo_client_provider
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.iam.client import IAMClient
from nemo_helix_plugin.iam.types import RoleBindingInput
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nhx.common.auth.models import Principal
from nhx.common.auth.workload_tokens import issue_service_workload_access_token
from nhx.common.client_factory import get_task_nemo_client
from nhx.common.config import Configuration, get_auth_config, get_service_config
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig, TokenSigningConfig
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.testing import ClientContext, create_test_client, subprocess_job_executor_patch

pytestmark = pytest.mark.integration


class InProcessProvider(DefaultNemoClientProvider):
    def __init__(self, http_client: httpx.Client) -> None:
        self.http_client = http_client

    def get_task_nemo_client(
        self, service_name: str, *, workspace: str | None = None, http_client: httpx.Client | None = None
    ) -> NemoClient:
        return get_task_nemo_client(service_name, workspace=workspace, http_client=http_client or self.http_client)


@pytest.mark.parametrize("mode", ["no-auth", "trusted-headers", "token-exchange"])
@pytest.mark.parametrize("entrypoint", ["data-designer", "anonymizer"])
def test_task_input_permissions_and_artifact_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    entrypoint: str,
) -> None:
    auth_enabled = mode != "no-auth"
    for name in ("NHX_ACCESS_TOKEN", "NHX_ACCESS_TOKEN_FILE", "NHX_WORKLOAD_IDENTITY_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    with (
        subprocess_job_executor_patch(),
        create_test_client(
            FilesService,
            JobsService,
            client_type=ClientContext,
            workspaces=["allowed", "blocked"],
            projects=[],
            workspace="allowed",
            auth_enabled=auth_enabled,
            auth_bundle_cache_seconds=0,
            auth_policy_data_refresh_interval=60,
        ) as platform,
    ):
        admin = platform.client.with_headers({"X-NHX-Principal-Id": "service:files"})
        files = FilesClient.from_client(admin)
        for workspace in ("allowed", "blocked"):
            files.create_fileset(workspace=workspace, body=CreateFilesetRequest(name="input"))
            files.upload_file(workspace=workspace, name="input", path="data.txt", content=workspace.encode())
        if auth_enabled:
            IAMClient.from_client(admin).create_role_binding(
                body=RoleBindingInput(principal="task-readers", role="Editor", workspace="allowed"),
                query_params={"wait_role_propagation": True},
            )
        job = (
            JobsClient.from_client(admin)
            .create_job(
                workspace="allowed",
                body=CreateHelixJobRequest(
                    name="task-auth",
                    source=entrypoint,
                    spec={},
                    platform_spec={
                        "steps": [
                            {
                                "name": "step",
                                "executor": {
                                    "provider": "cpu",
                                    "profile": "default",
                                    "container": {"image": "ubuntu:latest", "command": ["true"]},
                                },
                            }
                        ]
                    },
                ),
            )
            .data()
        )
        assert job.attempt_id is not None
        env = {
            "NHX_BASE_URL": str(platform.client.base_url),
            "NHX_PRINCIPAL": json.dumps({"id": "creator", "email": "creator@example.test", "groups": ["task-readers"]}),
            "NEMO_JOB_STEP_CONFIG_FILE_PATH": str(tmp_path / "config.json"),
            "NEMO_JOB_ID": job.id,
            "NEMO_JOB_ATTEMPT_ID": job.attempt_id,
            "NEMO_JOB_WORKSPACE": "allowed",
            "NEMO_JOB_FILESET": job.fileset,
            "NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH": str(tmp_path),
            "NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH": str(tmp_path),
        }
        if mode == "token-exchange":
            key_path = tmp_path / "workload.pem"
            key_path.write_bytes(
                rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
            signing = TokenSigningConfig(key_id="task-test", private_key_file=str(key_path))
            oidc = OIDCConfig(
                enabled=True,
                audience="nemo-helix",
                issuer="https://idp.test",
                workload=OIDCWorkloadConfig(
                    client_id="tasks", token_issuer="https://auth.test", scope="platform:read platform:write"
                ),
            )
            exchange_auth = get_auth_config().model_copy(update={"oidc": oidc, "token_signing": signing})
            exchange_service = get_service_config(AuthServiceConfig).model_copy(
                update={
                    "oidc": oidc,
                    "token_signing": signing,
                    "bundle_cache_seconds": 0,
                }
            )
            Configuration.set_override(exchange_auth)
            Configuration.set_override(exchange_service)
            assert isinstance(platform.test_client.app, FastAPI)
            platform.test_client.app.middleware_stack = None
            token_path = tmp_path / "task-token"
            token_path.write_text(
                issue_service_workload_access_token(
                    exchange_auth,
                    service_name=entrypoint,
                    on_behalf_of=Principal(id="creator", email="creator@example.test", groups=["task-readers"]),
                )
            )
            env["NHX_ACCESS_TOKEN_FILE"] = str(token_path)
            admin = NemoClient(
                base_url=str(platform.client.base_url),
                http_client=platform.client.http_client,
                auth=StaticToken(issue_service_workload_access_token(exchange_auth, service_name="files")),
            )
            files = FilesClient.from_client(admin)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        set_nemo_client_provider(InProcessProvider(platform.client.http_client))
        published: list[str] = []

        def probe(ctx: JobContext, client: NemoClient) -> None:
            task_files = FilesClient.from_client(client)
            assert task_files.download_file(workspace="allowed", name="input", path="data.txt").read() == b"allowed"
            if auth_enabled:
                with pytest.raises(PermissionDeniedError):
                    task_files.download_file(workspace="blocked", name="input", path="data.txt").read()
            else:
                assert task_files.download_file(workspace="blocked", name="input", path="data.txt").read() == b"blocked"
            artifact = tmp_path / "artifact.txt"
            artifact.write_text("published")
            ctx.results.save("artifact", artifact)
            if entrypoint == "anonymizer":
                ctx.usage.report_totals(input_tokens=12, output_tokens=3)
            published.append("artifact")

        def run_designer(*, step_config: DataDesignerStepConfig, ctx: JobContext, client: NemoClient) -> dict[str, int]:
            probe(ctx, client)
            return {"exit_code": 0}

        def run_anonymizer(step_config: AnonymizerStepConfig, *, ctx: JobContext, sdk: NemoClient | None = None) -> int:
            assert sdk is not None
            probe(ctx, sdk)
            return 0

        try:
            config = tmp_path / "config.json"
            if entrypoint == "data-designer":
                config.write_text(
                    json.dumps(
                        {
                            "job_config": {"num_records": 1, "config": {"columns": []}},
                            "model_providers": [],
                            "model_configs": [],
                        }
                    )
                )
                with patch.object(bridge, "run_step_config", side_effect=run_designer):
                    bridge.run()
            else:
                config.write_text(
                    json.dumps(
                        {
                            "request": {
                                "config": {"replace": {"kind": "redact"}},
                                "data": {"source": "https://example.test/input.csv", "text_column": "text"},
                            },
                            "model_configs_yaml": "model_configs: []",
                            "dd_model_providers": [],
                        }
                    )
                )
                with patch.object(anonymizer, "run_step_config", side_effect=run_anonymizer):
                    assert anonymizer.run() == 0
            assert published == ["artifact"]
            results = JobsClient.from_client(admin).list_job_results(workspace="allowed", name=job.id).data()
            assert any(result.name == "artifact" for result in results.data)
            if entrypoint == "anonymizer":
                attempt = (
                    JobsClient.from_client(admin)
                    .get_job_attempt(workspace="allowed", job=job.id, attempt=job.attempt_id)
                    .data()
                )
                assert attempt.status_details["input_tokens"] == 12
                assert attempt.status_details["output_tokens"] == 3
            assert (
                files.download_file(
                    workspace="allowed",
                    name=job.fileset,
                    path=f"results/{job.attempt_id}/artifact",
                ).read()
                == b"published"
            )
        finally:
            set_nemo_client_provider(None)
