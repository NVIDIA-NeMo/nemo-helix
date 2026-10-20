# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MVP preparer over the real ``ImageBuildService``: check first, then one call that is ready or failed."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_evals.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash, ResolvedHarborTaskDefinition
from nemo_evals.api.task_definitions.provenance import TaskProvenance
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.harbor.remote import preparation
from nemo_evals.harbor.remote.preparation import (
    ImagePreparationUnavailableError,
    PreparationClients,
    PreparationErrorInfo,
    TaskPreparation,
    prepare_task_images,
)
from nemo_evals.image_builds.models import BuildImage, BuildImageSource, TaskImageBuildEntity, build_job_name
from nemo_evals.image_builds.planning import TaskArchive
from nemo_evals.image_builds.runner import TaskImageBuildSpec
from nemo_evals.image_builds.service import BuildJob, ImageBuildForbiddenError, ImageBuildService
from nemo_evals.jobs.agent_spec import ResolvedTask
from nemo_evals.jobs.task_image_build import task_image_build_job_request
from nemo_evals.revisions import get_revision
from nemo_helix_plugin.client.errors import ConflictError, InternalServerError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

SEPARATE = {"environment": {}, "verifier": {"environment_mode": "separate"}}


def digest_ref(char: str) -> str:
    return f"registry.invalid/{char}@sha256:{char * 64}"


class _Jobs:
    """Jobs that, like the real service, refuse a second Job under a taken name."""

    def __init__(self) -> None:
        self.created: dict[str, CreateHelixJobRequest] = {}
        self.statuses: dict[str, str] = {}
        self.refuse: Exception | None = None

    async def create(self, request: CreateHelixJobRequest) -> None:
        if self.refuse is not None:
            raise self.refuse
        assert request.name is not None
        if request.name in self.created:
            raise ConflictError(httpx.Response(409, json={"detail": "job exists"}))
        self.created[request.name] = request

    async def get(self, name: str) -> BuildJob | None:
        if name not in self.created:
            return None
        return BuildJob(status=self.statuses.get(name, "active"), custom_fields=self.created[name].custom_fields)


class _Authorizer:
    def __init__(self) -> None:
        self.allowed = True

    async def __call__(self, workspace: str) -> None:
        if not self.allowed:
            raise ImageBuildForbiddenError(f"You may not submit container image builds in workspace {workspace}")


@pytest.fixture
def jobs() -> _Jobs:
    return _Jobs()


@pytest.fixture
def authorizer() -> _Authorizer:
    return _Authorizer()


@pytest.fixture
def clients(entity_store, jobs, authorizer) -> PreparationClients:
    service = ImageBuildService(entity_store, jobs, authorizer)
    return PreparationClients(client=MagicMock(), files=MagicMock(), image_builds=lambda workspace: service)


@pytest.fixture
def archive(monkeypatch) -> dict[str, TaskArchive]:
    holder = {"archive": TaskArchive(root="t", files=frozenset({"t/environment/Dockerfile"}), compose=None)}

    async def read(files, source):
        return holder["archive"]

    monkeypatch.setattr(preparation, "read_task_archive", read)
    return holder


def task(digest: str, *, name: str = "sort-task", config: dict | None = None) -> ResolvedTask:
    return ResolvedTask(
        id=name,
        spec=ResolvedHarborTaskDefinition(
            kind="harbor",
            native_task_id=name,
            instruction="Solve the task.",
            source=HarborArchiveSource(fileset_ref="default/harbor-tasks#t/task_archive", files_hash="a" * 64),
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            config=config if config is not None else SEPARATE,
            provenance=TaskProvenance(entity_name=f"default/{name}", revision_digest=digest),
        ),
    )


async def prepare(resolved: ResolvedTask, clients, policy, recorded: TaskPreparation | None = None):
    return await prepare_task_images(
        resolved, recorded, policy=policy, caller="user:alice", clients=clients, deadline=datetime.now(timezone.utc)
    )


async def settle(entity_store, digest: str, attempt: int, status: str, *, images=(), error=None) -> None:
    """Record the outcome the task-builder Job would write on build ``attempt``."""
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    revision = await get_revision(entity_store, TaskRevisionEntity, head, digest)
    build = await entity_store.get(
        TaskImageBuildEntity, name=f"build.{attempt}", workspace="default", parent=revision.id
    )
    build.status, build.images, build.error = status, list(images), error
    await entity_store.update(build)


def image(role: str, ref: str, *, source: BuildImageSource = "dockerfile") -> BuildImage:
    return BuildImage(
        role=role, source=source, status="ready", image_ref=ref, builder_image=f"harbor-env-{role}", build_job="b"
    )


def failure(result: TaskPreparation) -> tuple[str, str]:
    assert result.status == "failed" and result.error is not None
    return result.error.code, result.error.detail


async def test_a_task_without_a_build_starts_its_first_attempt(clients, store_task, sandbox_policy, jobs):
    digest = await store_task(config=SEPARATE)
    code, detail = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "image_unavailable" and "image build 1 of default/sort-task" in detail and "pending" in detail
    assert list(jobs.created) == [build_job_name("default/sort-task", digest, 1)]


async def test_a_ready_build_gives_agent_and_verifier(clients, store_task, sandbox_policy, entity_store):
    digest = await store_task(config=SEPARATE)
    await prepare(task(digest), clients, sandbox_policy)
    images = [image("main", digest_ref("d")), image("verifier", digest_ref("e"))]
    await settle(entity_store, digest, 1, "ready", images=images)
    result = await prepare(task(digest), clients, sandbox_policy)
    assert result.status == "ready" and result.environment is not None
    agent, verifier = result.environment.agent, result.environment.verifier
    assert agent.image_ref == digest_ref("d") and agent.preparation_key == "sha256:" + "d" * 64
    assert agent.builder_image == "harbor-env-main" and agent.source_archive_digest == "a" * 64
    assert verifier is not None and verifier.image_ref == digest_ref("e")
    assert result.images["agent"].source == "built"


async def test_a_prebuilt_copy_is_marked_prebuilt(clients, store_task, sandbox_policy, entity_store):
    digest = await store_task(config={"environment": {"docker_image": "docker.io/library/ubuntu:24.04"}})
    await prepare(task(digest), clients, sandbox_policy)
    await settle(entity_store, digest, 1, "ready", images=[image("main", digest_ref("c"), source="prebuilt")])
    result = await prepare(task(digest), clients, sandbox_policy)
    assert result.environment is not None and result.environment.verifier is None
    assert result.images["agent"].source == "prebuilt" and result.environment.agent.source_archive_digest is None


async def test_a_failed_build_fails_the_task_without_a_new_attempt(
    clients, store_task, sandbox_policy, entity_store, jobs
):
    digest = await store_task(config=SEPARATE)
    await prepare(task(digest), clients, sandbox_policy)
    await settle(entity_store, digest, 1, "failed", error="Dockerfile step 3 failed")
    code, detail = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "build_failed" and detail.endswith("Dockerfile step 3 failed")
    assert len(jobs.created) == 1  # a broken task doesn't start attempt after attempt


async def test_a_running_build_is_reported_and_left_alone(clients, store_task, sandbox_policy, jobs):
    digest = await store_task(config=SEPARATE)
    await prepare(task(digest), clients, sandbox_policy)
    code, detail = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "image_unavailable" and "resubmit when it's ready" in detail
    assert len(jobs.created) == 1


async def test_a_caller_who_may_not_build_still_learns_a_build_is_pending(
    clients, store_task, sandbox_policy, authorizer
):
    digest = await store_task(config=SEPARATE)
    await prepare(task(digest), clients, sandbox_policy)  # someone else started it
    authorizer.allowed = False
    code, detail = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "image_unavailable" and "is pending" in detail and "may not" not in detail


async def test_a_caller_who_may_not_build_cannot_start_one(clients, store_task, sandbox_policy, authorizer, jobs):
    digest = await store_task(config=SEPARATE)
    authorizer.allowed = False
    code, detail = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "image_unavailable" and "may not submit container image builds" in detail
    assert not jobs.created


async def test_a_lost_build_job_is_created_again(clients, store_task, sandbox_policy, jobs):
    digest = await store_task(config=SEPARATE)
    jobs.refuse = InternalServerError(httpx.Response(500, json={"detail": "jobs is down"}))
    with pytest.raises(ImagePreparationUnavailableError, match="jobs is down"):
        await prepare(task(digest), clients, sandbox_policy)
    jobs.refuse = None
    code, _ = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "image_unavailable" and list(jobs.created) == [build_job_name("default/sort-task", digest, 1)]


async def test_a_build_whose_job_ended_without_a_result_is_replaced(clients, store_task, sandbox_policy, jobs):
    digest = await store_task(config=SEPARATE)
    await prepare(task(digest), clients, sandbox_policy)
    jobs.statuses[build_job_name("default/sort-task", digest, 1)] = "error"
    code, detail = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "image_unavailable" and "image build 2" in detail
    assert len(jobs.created) == 2


async def test_a_job_name_held_by_another_job_is_reported(clients, store_task, sandbox_policy, jobs):
    digest = await store_task(config=SEPARATE)
    name = build_job_name("default/sort-task", digest, 1)
    other = TaskImageBuildSpec(workspace="default", task="other-task", revision_digest=digest, attempt=1)
    jobs.created[name] = task_image_build_job_request(other, name=name)
    code, _ = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "image_unavailable"


async def test_a_compose_service_image_is_not_supported(clients, store_task, sandbox_policy, entity_store):
    digest = await store_task(config=SEPARATE)
    await prepare(task(digest), clients, sandbox_policy)
    images = [image("main", digest_ref("d")), image("db", digest_ref("f"))]
    await settle(entity_store, digest, 1, "ready", images=images)
    code, _ = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "multi_container_not_yet_supported"


@pytest.mark.parametrize("missing", ["task", "revision"])
async def test_a_task_or_revision_that_is_gone_is_invalid(clients, store_task, sandbox_policy, missing):
    digest = await store_task(config=SEPARATE)
    gone = task(digest, name="no-such-task") if missing == "task" else task("f" * 64)
    code, _ = failure(await prepare(gone, clients, sandbox_policy))
    assert code == "invalid_task_config"


async def test_a_non_harbor_task_is_invalid(clients, store_task, sandbox_policy):
    digest = await store_task(harbor=False)
    code, _ = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "invalid_task_config"


async def test_settled_preparation_is_returned_unchanged(clients, sandbox_policy):
    failed = TaskPreparation(status="failed", error=PreparationErrorInfo(code="build_failed", detail="x"))
    assert await prepare(task("f" * 64), clients, sandbox_policy, failed) is failed


@pytest.mark.parametrize(
    ("config", "compose", "code"),
    [
        ({"environment": {"gpus": 1}}, None, "gpu_unsupported"),
        # Its build has only a main image, so only the archive shows it's a compose task.
        ({"environment": {}}, "services: {main: {}}", "multi_container_not_yet_supported"),
        ({"environment": {"os": "windows"}}, None, "windows_unsupported"),
    ],
)
async def test_check_rejects_what_kubernetes_runs_cannot_take(clients, archive, sandbox_policy, config, compose, code):
    archive["archive"] = TaskArchive(root="t", files=frozenset({"t/environment/Dockerfile"}), compose=compose)
    rejected = await prepare_task_images.check(task("f" * 64, config=config), policy=sandbox_policy, clients=clients)
    assert rejected is not None and rejected.code == code


async def test_check_passes_a_dockerfile_task_without_starting_a_build(clients, archive, sandbox_policy, jobs):
    assert await prepare_task_images.check(task("f" * 64), policy=sandbox_policy, clients=clients) is None
    assert not jobs.created
