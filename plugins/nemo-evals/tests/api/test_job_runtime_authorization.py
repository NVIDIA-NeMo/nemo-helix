# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-runtime image-build routes under the real platform authentication and authorization."""

import asyncio
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nemo_evals.api.schemas import TaskInput
from nemo_evals.api.service.task_service import TaskService
from nemo_evals.api.task_definitions.harbor import HarborArchiveSource, HarborTaskDefinition, HarborTaskHash
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.image_builds.models import ImageBuildRequest, TaskImageBuildEntity, build_entity_name
from nemo_evals.image_builds.runner import TaskImageBuildSpec
from nemo_evals.image_builds.runtime import RuntimeBuildRecords
from nemo_evals.image_builds.service import BuildJob, ImageBuildService
from nemo_evals.revisions import get_revision
from nemo_evals.service import EvaluatorPluginService
from nemo_helix_plugin.auth.workload_delegations import JOB_WORKLOAD_CLAIM, JobWorkloadBinding
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nhx.common.auth import Principal
from nhx.common.auth.workload_tokens import (
    _encode_access_token,
    granted_workload_scope,
    issue_service_workload_access_token,
    workload_signing_key,
)
from nhx.common.config import AuthConfig, get_auth_config
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig, TokenSigningConfig
from nhx.core.entities.service import EntitiesService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client

CREATOR = Principal(id="creator@example.com", email="creator@example.com")
EVALS_HEADERS = {"X-NHX-Principal-Id": "service:evals", "X-NHX-Internal": "true"}


class TestEvalsService(NemoServiceAdapter):
    __test__ = False

    def __init__(self):
        super().__init__(EvaluatorPluginService())


class _Jobs:
    async def create(self, request: CreateHelixJobRequest) -> None:
        pass

    async def get(self, name: str) -> BuildJob | None:
        return None


async def _allow(workspace: str) -> None:
    pass


@dataclass
class Platform:
    ctx: ClientContext
    build: TaskImageBuildEntity
    path: str
    key_file: Path


async def _store_build(ctx: ClientContext, monkeypatch: pytest.MonkeyPatch) -> TaskImageBuildEntity:
    from nhx_evals_sdk.agent_eval.runtimes.harbor.archive import NativeTask

    async def verify(spec, files_client):
        return NativeTask("sort-task", spec.instruction, spec.config)

    monkeypatch.setattr("nemo_evals.api.service.task_service.verify_definition", verify)
    entities = EntityClient(AsyncEntitiesClient.from_client(ctx.async_client.with_headers(EVALS_HEADERS)))
    tasks = TaskService(entities, Mock(), files_client=Mock(), authorize_metric_read=AsyncMock())
    spec = HarborTaskDefinition(
        kind="harbor",
        native_task_id="sort-task",
        harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
        source=HarborArchiveSource(fileset_ref="default/archives#sort-task/task_archive", files_hash="a" * 64),
        config={},
    )
    await tasks.create_task("sort-task", TaskInput(spec=spec), workspace="default")
    head = await entities.get(TaskEntity, name="sort-task", workspace="default")
    digest = (await get_revision(entities, TaskRevisionEntity, head)).content_hash
    await ImageBuildService(entities, _Jobs(), _allow).request_build(
        "default", "sort-task", digest, ImageBuildRequest()
    )
    name = build_entity_name(digest, 1)
    return await entities.get(TaskImageBuildEntity, name=name, workspace="default", parent=head.id)


@pytest.fixture(scope="module")
def platform(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Platform]:
    tmp = tmp_path_factory.mktemp("job-runtime")
    key_file = tmp / "workload-key.pem"
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        create_test_client(
            TestEvalsService,
            EntitiesService,
            client_type=ClientContext,
            auth_enabled=True,
            tmp_dir=tmp,
            workspaces=["default", "other"],
            projects=[],
            auth_bundle_cache_seconds=3600,
            auth_policy_data_refresh_interval=3600,
        ) as ctx,
    ):
        build = asyncio.run(_store_build(ctx, monkeypatch))
        path = (
            f"/apis/evals/v2/workspaces/default/job-runtime/{build.job}/tasks/sort-task"
            f"/revisions/{build.revision_digest}/image-builds/{build.attempt}"
        )
        yield Platform(ctx, build, path, key_file)


@contextmanager
def _workload_exchange(platform: Platform) -> Generator[AuthConfig]:
    """Run the platform in workload token-exchange mode, where trusted identity headers are refused."""
    auth = get_auth_config()
    with (
        patch.object(auth, "token_signing", TokenSigningConfig(key_id="test", private_key_file=str(platform.key_file))),
        patch.object(
            auth,
            "oidc",
            OIDCConfig(
                enabled=True,
                issuer="https://idp.example.test",
                audience="test-audience",
                workload=OIDCWorkloadConfig(client_id="jobs-test"),
            ),
        ),
    ):
        yield auth


def _pod_token(auth: AuthConfig, job: JobWorkloadBinding | None) -> str:
    """The shape of a token exchanged for a Kubernetes or Docker Job's delegation."""
    claims: dict = {"sub": CREATOR.id, "email": CREATOR.email, "act": {"sub": "system:serviceaccount:jobs:runner"}}
    if job is not None:
        claims[JOB_WORKLOAD_CLAIM] = job.model_dump()
    return _encode_access_token(
        auth, workload_signing_key(auth), subject_claims=claims, scope=granted_workload_scope(auth, None)
    )


def _subprocess_token(auth: AuthConfig, job: JobWorkloadBinding | None) -> str:
    return issue_service_workload_access_token(auth, service_name="jobs", on_behalf_of=CREATOR, job=job)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("mint", [_pod_token, _subprocess_token], ids=["pod", "subprocess"])
def test_the_build_job_reads_and_records_its_build(platform: Platform, mint) -> None:
    build = platform.build
    spec = TaskImageBuildSpec(
        workspace="default", task="sort-task", revision_digest=build.revision_digest, attempt=build.attempt
    )

    async def load_and_fail() -> tuple[TaskImageBuildEntity, TaskRevisionEntity, TaskImageBuildEntity]:
        records = RuntimeBuildRecords(platform.ctx.async_client.with_headers(headers), job=build.job, spec=spec)
        loaded, revision = await records.load()
        saved = await records.save(loaded.model_copy(update={"status": "failed", "error": "test"}))
        return loaded, revision, saved

    with _workload_exchange(platform) as auth:
        headers = _bearer(mint(auth, JobWorkloadBinding(workspace="default", job=build.job)))
        loaded, revision, saved = asyncio.run(load_and_fail())

    assert loaded.job == build.job
    assert revision.content_hash == build.revision_digest
    assert (saved.status, saved.error) == ("failed", "test")


@pytest.mark.parametrize(
    "bound_to",
    [("default", "another-job"), ("other", None), None],
    ids=["another-job", "another-workspace", "unbound"],
)
@pytest.mark.parametrize("mint", [_pod_token, _subprocess_token], ids=["pod", "subprocess"])
def test_any_other_workload_is_refused(platform: Platform, mint, bound_to: tuple[str, str | None] | None) -> None:
    job = None
    if bound_to is not None:
        workspace, name = bound_to
        job = JobWorkloadBinding(workspace=workspace, job=name or platform.build.job)
    with _workload_exchange(platform) as auth:
        headers = _bearer(mint(auth, job))
        assert platform.ctx.test_client.get(platform.path, headers=headers).status_code == 403
        refused = platform.ctx.test_client.put(platform.path, headers=headers, json={"status": "ready", "images": []})
        assert refused.status_code == 403


def test_a_job_cannot_use_a_build_another_job_runs(platform: Platform) -> None:
    foreign = platform.path.replace(f"/job-runtime/{platform.build.job}/", "/job-runtime/another-job/")
    with _workload_exchange(platform) as auth:
        headers = _bearer(_pod_token(auth, JobWorkloadBinding(workspace="default", job="another-job")))
        assert platform.ctx.test_client.get(foreign, headers=headers).status_code == 403
        refused = platform.ctx.test_client.put(foreign, headers=headers, json={"status": "ready", "images": []})
        assert refused.status_code == 403


def test_the_build_job_still_cannot_read_entities_directly(platform: Platform) -> None:
    """The Job reaches its record only through these routes; its credential gains no Entity Store access."""
    with _workload_exchange(platform) as auth:
        headers = _bearer(_pod_token(auth, JobWorkloadBinding(workspace="default", job=platform.build.job)))
        raw = platform.ctx.test_client.get(
            "/apis/entities/v2/workspaces/default/entities/task/sort-task", headers=headers
        )
    assert raw.status_code == 403, raw.text


def test_a_user_cannot_use_the_routes(platform: Platform) -> None:
    owner = {"X-NHX-Principal-Id": "owner@example.com"}
    grant = platform.ctx.test_client.post(
        "/apis/auth/v2/iam/role-bindings?wait_role_propagation=false",
        headers=EVALS_HEADERS | {"X-NHX-Principal-Id": "service:integration-test"},
        json={"principal": owner["X-NHX-Principal-Id"], "workspace": "default", "role": "Owner"},
    )
    assert grant.status_code == 200, grant.text
    assert platform.ctx.test_client.get(platform.path, headers=owner).status_code == 403
    refused = platform.ctx.test_client.put(platform.path, headers=owner, json={"status": "ready", "images": []})
    assert refused.status_code == 403


def test_the_evals_service_identity_is_admitted_without_token_exchange(platform: Platform) -> None:
    """Without token exchange a Job authenticates as the Evals service acting for its creator."""
    headers = EVALS_HEADERS | {"X-NHX-Principal-On-Behalf-Of": CREATOR.id}
    assert platform.ctx.test_client.get(platform.path, headers=headers).status_code == 200


def test_a_job_cannot_rewrite_what_its_build_is(platform: Platform) -> None:
    with _workload_exchange(platform) as auth:
        headers = _bearer(_pod_token(auth, JobWorkloadBinding(workspace="default", job=platform.build.job)))
        rewritten = platform.ctx.test_client.put(
            platform.path,
            headers=headers,
            json={"status": "ready", "images": [], "job": "another-job", "allowed_registries": ["evil.example"]},
        )
        assert rewritten.status_code == 422
