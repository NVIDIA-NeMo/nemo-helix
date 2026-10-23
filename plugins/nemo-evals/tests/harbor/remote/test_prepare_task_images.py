# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MVP preparer over the real ``ImageBuildService``: check first, then one read that is ready or failed."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_evals.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evals.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash, ResolvedHarborTaskDefinition
from nemo_evals.api.task_definitions.provenance import TaskProvenance
from nemo_evals.entities import TaskEntity
from nemo_evals.harbor.remote import preparation
from nemo_evals.harbor.remote.preparation import (
    ImagePreparationUnavailableError,
    PreparationClients,
    PreparationErrorInfo,
    TaskPreparation,
    prepare_task_images,
)
from nemo_evals.image_builds.models import (
    BuildImage,
    BuildImageSource,
    ImageBuildRequest,
    TaskImageBuildEntity,
    build_entity_name,
)
from nemo_evals.image_builds.planning import TaskArchive
from nemo_evals.image_builds.service import BuildJob, ImageBuildForbiddenError, ImageBuildService
from nemo_evals.jobs.agent_spec import ResolvedTask
from nemo_helix_plugin.client.errors import ConflictError, InternalServerError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

SEPARATE = {"environment": {}, "verifier": {"environment_mode": "separate"}}


def digest_ref(char: str) -> str:
    return f"registry.invalid/{char}@sha256:{char * 64}"


class _Jobs:
    """Jobs that, like the real service, refuse a second Job under a taken name."""

    def __init__(self) -> None:
        self.created: dict[str, CreateHelixJobRequest] = {}

    async def create(self, request: CreateHelixJobRequest) -> None:
        assert request.name is not None
        if request.name in self.created:
            raise ConflictError(httpx.Response(409, json={"detail": "job exists"}))
        self.created[request.name] = request

    async def get(self, name: str) -> BuildJob | None:
        if name not in self.created:
            return None
        return BuildJob(status="active", custom_fields=self.created[name].custom_fields)


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
def builds(entity_store, jobs, authorizer) -> ImageBuildService:
    """Starts builds for a test, as ``nemo evals runs submit`` does before it submits."""
    return ImageBuildService(entity_store, jobs, authorizer)


@pytest.fixture
def clients(entity_store, jobs, authorizer, monkeypatch) -> PreparationClients:
    """The preparer's clients: it reads the same builds through a service that must never start one."""
    service = ImageBuildService(entity_store, jobs, authorizer)

    async def request_build(*args, **kwargs):
        pytest.fail("the preparer must never request a build")

    monkeypatch.setattr(service, "request_build", request_build)
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


async def build(builds: ImageBuildService, digest: str, *, platform: str | None = None, rebuild: bool = False) -> None:
    """Start the next build attempt of ``sort-task`` (or return the one running)."""
    await builds.request_build("default", "sort-task", digest, ImageBuildRequest(platform=platform, rebuild=rebuild))


async def settle(entity_store, digest: str, attempt: int, status: str, *, images=(), error=None) -> None:
    """Record the outcome the task-builder Job would write on build ``attempt``."""
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    build = await entity_store.get(
        TaskImageBuildEntity, name=build_entity_name(digest, attempt), workspace="default", parent=head.id
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


READY = [image("main", digest_ref("d")), image("verifier", digest_ref("e"))]


async def test_a_task_without_a_build_is_refused_and_no_build_starts(clients, store_task, sandbox_policy, jobs):
    digest = await store_task(config=SEPARATE)
    code, detail = failure(await prepare(task(digest), clients, sandbox_policy))
    assert (code, detail) == (
        "image_unavailable",
        f"default/sort-task#{digest} has no image build; build it first (`nemo evals runs submit` builds and waits)",
    )
    assert not jobs.created


@pytest.mark.parametrize("status", ["pending", "running"])
async def test_a_build_in_flight_is_refused_and_left_alone(
    clients, builds, store_task, sandbox_policy, entity_store, jobs, status
):
    digest = await store_task(config=SEPARATE)
    await build(builds, digest)
    await settle(entity_store, digest, 1, status)
    code, detail = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "image_unavailable" and f"image build 1 of default/sort-task#{digest} is {status}" in detail
    assert len(jobs.created) == 1


async def test_a_rebuild_in_flight_hides_the_older_ready_build(
    clients, builds, store_task, sandbox_policy, entity_store
):
    """The newest attempt decides: the run waits for the rebuild instead of taking the images it replaces."""
    digest = await store_task(config=SEPARATE)
    await build(builds, digest)
    await settle(entity_store, digest, 1, "ready", images=READY)
    await build(builds, digest, rebuild=True)
    code, detail = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "image_unavailable" and "image build 2" in detail and "is pending" in detail


async def test_a_ready_build_gives_agent_and_verifier(clients, builds, store_task, sandbox_policy, entity_store):
    digest = await store_task(config=SEPARATE)
    await build(builds, digest)
    await settle(entity_store, digest, 1, "ready", images=READY)
    result = await prepare(task(digest), clients, sandbox_policy)
    assert result.status == "ready" and result.environment is not None
    agent, verifier = result.environment.agent, result.environment.verifier
    assert agent.image_ref == digest_ref("d") and agent.preparation_key == "sha256:" + "d" * 64
    assert agent.builder_image == "harbor-env-main" and agent.source_archive_digest == "a" * 64
    assert verifier is not None and verifier.image_ref == digest_ref("e")
    assert result.images["agent"].source == "built"


async def test_a_caller_who_may_not_build_still_gets_a_ready_build(
    clients, builds, store_task, sandbox_policy, entity_store, authorizer
):
    digest = await store_task(config=SEPARATE)
    await build(builds, digest)
    await settle(entity_store, digest, 1, "ready", images=READY)
    authorizer.allowed = False
    assert (await prepare(task(digest), clients, sandbox_policy)).status == "ready"


async def test_a_prebuilt_copy_is_marked_prebuilt(clients, builds, store_task, sandbox_policy, entity_store):
    digest = await store_task(config={"environment": {"docker_image": "docker.io/library/ubuntu:24.04"}})
    await build(builds, digest)
    await settle(entity_store, digest, 1, "ready", images=[image("main", digest_ref("c"), source="prebuilt")])
    result = await prepare(task(digest), clients, sandbox_policy)
    assert result.environment is not None and result.environment.verifier is None
    assert result.images["agent"].source == "prebuilt" and result.environment.agent.source_archive_digest is None


async def test_a_failed_build_fails_the_task_without_a_new_attempt(
    clients, builds, store_task, sandbox_policy, entity_store, jobs
):
    digest = await store_task(config=SEPARATE)
    await build(builds, digest)
    await settle(entity_store, digest, 1, "failed", error="Dockerfile step 3 failed")
    code, detail = failure(await prepare(task(digest), clients, sandbox_policy))
    assert code == "build_failed" and detail.endswith("Dockerfile step 3 failed")
    assert len(jobs.created) == 1


@pytest.mark.parametrize("other", ["pending", "ready", "failed"])
async def test_a_newer_build_for_another_platform_does_not_hide_the_ready_default_one(
    clients, builds, store_task, sandbox_policy, entity_store, other
):
    """Runs use the deployment's default platform, so another platform's newer build is no answer."""
    digest = await store_task(config=SEPARATE)
    await build(builds, digest)
    await settle(entity_store, digest, 1, "ready", images=READY)
    await build(builds, digest, platform="linux/arm64")
    await settle(entity_store, digest, 2, other, images=[image("main", digest_ref("f"))] if other == "ready" else ())

    result = await prepare(task(digest), clients, sandbox_policy)

    assert result.status == "ready" and result.environment is not None
    assert result.environment.agent.image_ref == digest_ref("d")


async def test_a_default_build_older_than_a_page_of_other_platforms_builds_is_found(
    clients, builds, store_task, sandbox_policy, entity_store, monkeypatch
):
    """A page holds 2 builds here: the default build is on the second page, behind other platforms' newer builds."""
    monkeypatch.setattr("nemo_evals.image_builds.service._LOOKUP_PAGE_SIZE", 2)
    digest = await store_task(config=SEPARATE)
    await build(builds, digest)
    await settle(entity_store, digest, 1, "ready", images=READY)
    for attempt, platform in enumerate(("linux/arm64", "linux/amd64", "linux/riscv64"), start=2):
        await build(builds, digest, platform=platform)
        await settle(entity_store, digest, attempt, "failed", error="no base image for the platform")

    result = await prepare(task(digest), clients, sandbox_policy)

    assert result.status == "ready" and result.environment is not None
    assert result.environment.agent.image_ref == digest_ref("d")


async def test_a_build_read_outage_is_retryable(clients, store_task, sandbox_policy, entity_store, monkeypatch):
    digest = await store_task(config=SEPARATE)

    async def unavailable(*args, **kwargs):
        raise InternalServerError(httpx.Response(503, json={"detail": "entity store is down"}))

    monkeypatch.setattr(entity_store, "list", unavailable)
    with pytest.raises(ImagePreparationUnavailableError, match="entity store is down"):
        await prepare(task(digest), clients, sandbox_policy)


async def test_a_compose_service_image_is_not_supported(clients, builds, store_task, sandbox_policy, entity_store):
    digest = await store_task(config=SEPARATE)
    await build(builds, digest)
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


async def test_a_non_harbor_task_is_invalid(clients, sandbox_policy):
    evaluator = ResolvedTask(id="answer", spec=ResolvedEvaluatorTaskDefinition(kind="evaluator", intent="Answer"))
    code, _ = failure(await prepare(evaluator, clients, sandbox_policy))
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
        # A CPU agent with a GPU verifier would still ask OpenSandbox for a GPU.
        (
            {"environment": {"gpus": 0}, "verifier": {"environment_mode": "separate", "environment": {"gpus": 1}}},
            None,
            "gpu_unsupported",
        ),
    ],
)
async def test_check_rejects_what_kubernetes_runs_cannot_take(clients, archive, sandbox_policy, config, compose, code):
    archive["archive"] = TaskArchive(root="t", files=frozenset({"t/environment/Dockerfile"}), compose=compose)
    rejected = await prepare_task_images.check(task("f" * 64, config=config), policy=sandbox_policy, clients=clients)
    assert rejected is not None and rejected.code == code


async def test_check_passes_a_dockerfile_task_without_starting_a_build(clients, archive, sandbox_policy, jobs):
    assert await prepare_task_images.check(task("f" * 64), policy=sandbox_policy, clients=clients) is None
    assert not jobs.created


async def test_check_reports_a_files_outage_as_retryable(clients, sandbox_policy, monkeypatch):
    async def read(files, source):
        raise InternalServerError(httpx.Response(503, json={"detail": "files is down"}))

    monkeypatch.setattr(preparation, "read_task_archive", read)
    with pytest.raises(ImagePreparationUnavailableError, match="files is down"):
        await prepare_task_images.check(task("f" * 64), policy=sandbox_policy, clients=clients)
