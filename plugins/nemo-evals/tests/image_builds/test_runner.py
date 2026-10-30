# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task-builder Job's work: plan, submit and follow every image of one build."""

from __future__ import annotations

import hashlib
import io
import tarfile
from typing import Literal
from unittest.mock import Mock

import httpx
import pytest
from nemo_builder_plugin.entities import ContainerImage, ImageRuntimeConfig, Provenance
from nemo_builder_plugin.schema import FileSetSource, ImageSource, SubmitBuildResponse
from nemo_evals.api.task_definitions.harbor import HarborArchiveSource, HarborTaskDefinition
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.image_builds.builder import builder_name, copy_builder_name
from nemo_evals.image_builds.models import (
    BuildImage,
    ImageBuildRequest,
    TaskImageBuildEntity,
    build_entity_name,
    build_job_name,
)
from nemo_evals.image_builds.runner import TaskImageBuildSpec, read_task_archive, run_task_image_build
from nemo_evals.image_builds.service import BuildJob, ImageBuildService
from nemo_evals.jobs.task_image_build import is_job_for, task_image_build_job_request
from nemo_evals.revisions import apply_tag, get_revision, publish_revision
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

Outcome = Literal["ready", "failed"]
_SEPARATE_VERIFIER = {"verifier": {"environment_mode": "separate"}}


# How every ready image starts, as Builder records it from the image's config.
RUNTIME_CONFIG = ImageRuntimeConfig(entrypoint=["/bin/bash", "-l", "-c"], working_dir="/app")


def _container_image(
    name: str, status: Literal["pending", "ready", "failed"] = "pending", *, detail: str | None = None
) -> ContainerImage:
    return ContainerImage(
        name=name,
        workspace="default",
        status=status,
        status_detail=detail,
        registry="registry.test",
        repository=f"default/{name}",
        digest="sha256:" + "c" * 64 if status == "ready" else None,
        runtime_config=RUNTIME_CONFIG if status == "ready" else None,
        provenance=Provenance(
            build_set=name.rsplit("-", 1)[0],
            revision=int(name.rsplit("-", 1)[1]),
            job=f"default/builder-{name}",
            request_digest="sha256:" + "a" * 64,
        ),
    )


def _submitted(name: str, revision: int) -> SubmitBuildResponse:
    return SubmitBuildResponse(job=f"builder-{name}-{revision}", images=[_container_image(f"{name}-{revision}")])


class _Jobs:
    async def create(self, request: CreateHelixJobRequest) -> None:
        pass

    async def get(self, name: str) -> BuildJob | None:
        return None


class _Builder:
    """Builder whose images finish with ``outcomes[context_path]`` on the first read after submit."""

    def __init__(
        self,
        outcomes: dict[str, Outcome],
        *,
        refuse: bool = False,
        job: str = "active",
        published: dict[str, list[str]] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.refuse = refuse
        self.job = job
        #: The platforms each prebuilt image is published for, by fully qualified reference.
        self.published = published or {}
        self.images: dict[str, str] = {}
        self.submits: list[FileSetSource | ImageSource] = []
        #: The platform each submit named, by Builder name.
        self.platforms: dict[str, str | None] = {}

    async def get_image(self, name: str) -> ContainerImage | None:
        context = self.images.get(name)
        if context is None:
            return None
        outcome = self.outcomes[context]
        return _container_image(name, outcome, detail="kaniko exited 1" if outcome == "failed" else None)

    async def submit(
        self,
        *,
        name: str,
        revision: int,
        source: FileSetSource | ImageSource,
        dockerfile: str | None,
        platform: str | None,
    ) -> SubmitBuildResponse:
        self.platforms[name] = platform
        if self.refuse:
            raise PermissionDeniedError(httpx.Response(403, json={"detail": "not allowed to build"}))
        what = source.image if isinstance(source, ImageSource) else source.context_path
        assert what is not None
        self.submits.append(source)
        self.images[f"{name}-{revision}"] = what
        return _submitted(name, revision)

    async def job_status(self, job: str) -> str | None:
        return self.job

    async def image_platforms(self, image: str) -> list[str]:
        return self.published.get(image, ["linux/amd64", "linux/arm64"])


async def _no_sleep(seconds: float) -> None:
    pass


async def _allow(workspace: str) -> None:
    pass


def _serve_archive(monkeypatch, files: dict[str, bytes]) -> None:
    """Serve a task archive holding ``files`` as the task's download, through the runner's own temporary file."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"sort-task/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    async def download(files_client, ref, output, **kwargs) -> None:
        output.write(buffer.getvalue())

    monkeypatch.setattr("nemo_evals.image_builds.runner.download_verified_async", download)


@pytest.fixture
def archive(monkeypatch) -> None:
    _serve_archive(
        monkeypatch, {"environment/Dockerfile": b"FROM python:3.12\n", "tests/Dockerfile": b"FROM python:3.12-slim\n"}
    )


async def _start(
    entity_store, store_task, *, config: dict, allowed: list[str] | None = None, platform: str | None = None
) -> TaskImageBuildSpec:
    digest = await store_task(config=config)
    service = ImageBuildService(entity_store, _Jobs(), _allow)
    request = ImageBuildRequest(allowed_registries=allowed or [], platform=platform)
    await service.request_build("default", "sort-task", digest, request)
    return TaskImageBuildSpec(workspace="default", task="sort-task", revision_digest=digest, attempt=1)


async def _stored_build(entity_store, spec: TaskImageBuildSpec) -> TaskImageBuildEntity:
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    name = build_entity_name(spec.revision_digest, 1)
    return await entity_store.get(TaskImageBuildEntity, name=name, workspace="default", parent=head.id)


async def _stored_task_definition(entity_store, spec: TaskImageBuildSpec) -> HarborTaskDefinition:
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    revision = await get_revision(entity_store, TaskRevisionEntity, head, spec.revision_digest)
    assert isinstance(revision.spec, HarborTaskDefinition)
    return revision.spec


async def _run(entity_store, spec: TaskImageBuildSpec, builder: _Builder, **kwargs) -> TaskImageBuildEntity:
    return await run_task_image_build(
        spec, entities=entity_store, files=Mock(), builder=builder, sleep=_no_sleep, **kwargs
    )


async def test_main_and_verifier_built_makes_the_build_ready(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"})

    build = await _run(entity_store, spec, builder)

    assert build.status == "ready"
    assert build.error is None
    assert [(image.type, image.status) for image in build.images] == [("main", "ready"), ("verifier", "ready")]
    assert all(image.image_ref and image.builder_image and image.build_job for image in build.images)
    # How each image starts comes with it, so the task Job can restore its ENTRYPOINT.
    assert all(image.runtime_config == RUNTIME_CONFIG for image in build.images)
    assert await _stored_build(entity_store, spec) == build


async def test_one_failed_image_fails_the_build_with_its_reason(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "failed"})

    build = await _run(entity_store, spec, builder)

    assert build.status == "failed"
    assert build.error == "verifier: kaniko exited 1"


async def test_a_prebuilt_image_is_copied_alongside_built_ones(entity_store, store_task, archive) -> None:
    config = {"environment": {"docker_image": "ghcr.io/org/sort:1"}, "verifier": {"environment": {"cpus": 1}}}
    spec = await _start(entity_store, store_task, config=config)
    builder = _Builder({"ghcr.io/org/sort:1": "ready", "sort-task/tests": "ready"})

    build = await _run(entity_store, spec, builder)

    assert build.status == "ready"
    assert builder.submits[0] == ImageSource(image="ghcr.io/org/sort:1")
    main = build.images[0]
    assert (main.source, main.status) == ("prebuilt", "ready")
    assert main.builder_image is not None and main.builder_image.startswith("harbor-img-")


async def test_a_copy_is_submitted_by_its_fully_qualified_reference(entity_store, store_task, archive) -> None:
    """Two spellings of one image share a Builder name, so they must send the same request."""
    spec = await _start(entity_store, store_task, config={"environment": {"docker_image": "ubuntu:24.04"}})
    builder = _Builder({"docker.io/library/ubuntu:24.04": "ready"})

    await _run(entity_store, spec, builder)

    assert builder.submits == [ImageSource(image="docker.io/library/ubuntu:24.04")]


async def test_an_image_whose_builder_job_ended_without_it_fails(entity_store, store_task, archive) -> None:
    """Builder leaves a failed build's image pending, so its ended job is what says it won't finish."""
    spec = await _start(entity_store, store_task, config={})

    class _Abandoned(_Builder):
        async def get_image(self, name: str) -> ContainerImage | None:
            return None if name not in self.images else _container_image(name)

    ticks = iter(range(0, 10_000, 100))
    build = await _run(entity_store, spec, _Abandoned({}, job="error"), timeout=250, clock=lambda: next(ticks))

    assert build.status == "failed"
    assert build.error is not None and build.error.startswith("main: Builder job builder-harbor-env-")
    assert build.error.endswith("ended (error) without the image")


async def test_an_unbuildable_task_fails_with_the_plan_error(entity_store, store_task, monkeypatch) -> None:
    _serve_archive(monkeypatch, {"environment/Dockerfile": b"FROM python:3.12\n"})
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)

    build = await _run(entity_store, spec, _Builder({}))

    assert build.status == "failed"
    assert build.error is not None and build.error.startswith("verifier_dockerfile_missing")


async def test_builder_refusing_a_build_fails_the_image(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config={})

    build = await _run(entity_store, spec, _Builder({}, refuse=True))

    assert build.status == "failed"
    assert build.error is not None and "not allowed to build" in build.error


async def test_an_unexpected_error_fails_the_build_instead_of_leaving_it_running(
    entity_store, store_task, monkeypatch
) -> None:
    spec = await _start(entity_store, store_task, config={})

    async def corrupt(files_client, ref, output, **kwargs) -> None:
        raise ValueError("Downloaded object checksum mismatch")

    monkeypatch.setattr("nemo_evals.image_builds.runner.download_verified_async", corrupt)

    build = await _run(entity_store, spec, _Builder({}))

    assert build.status == "failed"
    assert build.error == "task-builder error: Downloaded object checksum mismatch"
    assert (await _stored_build(entity_store, spec)).status == "failed"


async def test_a_build_that_never_finishes_times_out(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config={})

    class _Stuck(_Builder):
        async def get_image(self, name: str) -> ContainerImage | None:
            return None if name not in self.images else _container_image(name)

    ticks = iter(range(0, 10_000, 100))
    build = await _run(entity_store, spec, _Stuck({}), timeout=250, clock=lambda: next(ticks))

    assert build.status == "failed"
    assert build.error == "main: no result from Builder within 250 seconds"


async def test_a_finished_build_is_left_alone(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config={})
    first = await _run(entity_store, spec, _Builder({"sort-task/environment": "ready"}))
    builder = _Builder({"sort-task/environment": "failed"})

    again = await _run(entity_store, spec, builder)

    assert again == first
    assert builder.submits == []


class _RecordingJobs(_Jobs):
    def __init__(self) -> None:
        self.created: list[CreateHelixJobRequest] = []

    async def create(self, request: CreateHelixJobRequest) -> None:
        self.created.append(request)

    async def get(self, name: str) -> BuildJob | None:
        return next(
            (BuildJob(status="active", custom_fields=job.custom_fields) for job in self.created if job.name == name),
            None,
        )


async def _republish(entity_store, harbor_digest: str) -> None:
    """Publish the task again with different contents, as Harbor hashes them."""
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    harbor_hash = head.spec.harbor_hash.model_copy(update={"digest": harbor_digest})
    head.spec = head.spec.model_copy(update={"harbor_hash": harbor_hash})
    await publish_revision(entity_store, entity_store, head, TaskRevisionEntity)


async def test_revisions_with_the_same_contents_share_one_build(entity_store, store_task, archive) -> None:
    """Publishing A, B, then A again makes revisions 1, 2 and 3; 1 and 3 hold the same contents, so the same images."""
    digest = await store_task(config={})
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    await apply_tag(entity_store, entity_store, TaskRevisionEntity, head, "v1", digest)
    await _republish(entity_store, "c" * 64)
    await _republish(entity_store, "b" * 64)
    jobs = _RecordingJobs()
    service = ImageBuildService(entity_store, jobs, _allow)

    first = await service.request_build("default", "sort-task", "v1", ImageBuildRequest())
    again = await service.request_build("default", "sort-task", "latest", ImageBuildRequest())
    assert first is not None and again == first
    # The service creates the Job each time and relies on Jobs refusing a name it already holds.
    assert len({job.name for job in jobs.created}) == 1
    job = jobs.created[0]

    build = await _run(
        entity_store, TaskImageBuildSpec.model_validate(job.spec), _Builder({"sort-task/environment": "ready"})
    )

    assert build.status == "ready"
    for fragment in ("v1", "latest", digest):
        page = await service.list_builds("default", "sort-task", fragment)
        assert page is not None
        [listed] = page.data
        assert listed.status == "ready"


@pytest.mark.parametrize(("status", "job_status"), [("ready", "completed"), ("failed", "failed")])
def test_job_fails_when_the_build_fails(monkeypatch, status: str, job_status: str) -> None:
    """The Job's exit status is what Jobs shows, so a failed build must not exit as success."""
    from nemo_evals.jobs.task_image_build import TaskImageBuildJob
    from nemo_helix_plugin.client.client import AsyncNemoClient

    async def run(spec, **clients):
        return Mock(status=status)

    monkeypatch.setattr("nemo_evals.jobs.task_image_build.run_task_image_build", run)
    config = TaskImageBuildSpec(workspace="default", task="t", revision_digest="a" * 64, attempt=1)

    result = TaskImageBuildJob().run(
        config.model_dump(), ctx=Mock(), async_client=AsyncNemoClient(base_url="http://platform.test")
    )

    assert result["status"] == job_status


def test_a_job_left_by_a_build_kept_under_its_revision_is_never_adopted() -> None:
    """Builds used to be kept under a revision; a Job one of them left must not stand in for a build kept by contents.

    Adopted, the old Job would never write the new record, and the next request would fail it as abandoned.
    """
    digest = "a" * 64
    spec = TaskImageBuildSpec(workspace="default", task="sort-task", revision_digest=digest, attempt=1)
    old_name = f"task-image-build-{hashlib.sha256(f'default/sort-task#{digest}'.encode()).hexdigest()[:16]}-1"

    assert build_job_name("default/sort-task", digest, 1) != old_name
    assert not is_job_for({"task_image_build": f"default/sort-task#{digest}/1"}, spec)
    assert is_job_for(task_image_build_job_request(spec, name="any").custom_fields, spec)


@pytest.fixture
def pinned_archive(monkeypatch) -> None:
    """A task whose environment Dockerfile pins linux/amd64."""
    _serve_archive(
        monkeypatch,
        {
            "environment/Dockerfile": b"FROM --platform=linux/amd64 python:3.12\n",
            "tests/Dockerfile": b"FROM python:3.12\n",
        },
    )


async def test_the_requested_platform_builds_every_image(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, platform="linux/arm64")
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"})

    build = await _run(entity_store, spec, builder)

    assert build.status == "ready"
    assert [image.platform for image in build.images] == ["linux/arm64", "linux/arm64"]
    assert list(builder.platforms.values()) == ["linux/arm64", "linux/arm64"]


async def test_without_a_platform_builder_builds_for_its_default(entity_store, store_task, archive) -> None:
    """A build for the deployment's default platform is named by its context's contents and Dockerfile alone."""
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"})

    build = await _run(entity_store, spec, builder)

    assert builder.platforms == {
        builder_name(context_digest=image.context_digest or "", dockerfile="Dockerfile"): None for image in build.images
    }


async def test_a_dockerfile_pinned_platform_is_built_for(entity_store, store_task, pinned_archive) -> None:
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"})

    build = await _run(entity_store, spec, builder)

    assert [(image.type, image.platform) for image in build.images] == [("main", "linux/amd64"), ("verifier", None)]


async def test_a_request_for_another_platform_than_the_task_pins_fails(
    entity_store, store_task, pinned_archive
) -> None:
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, platform="linux/arm64")
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"})

    build = await _run(entity_store, spec, builder)

    assert build.status == "failed"
    assert build.error is not None and build.error.startswith("platform_conflict: main is pinned to linux/amd64")
    assert builder.submits == []


async def test_a_prebuilt_image_published_for_one_platform_is_copied_for_it(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config={"environment": {"docker_image": "ubuntu:24.04"}})
    reference = "docker.io/library/ubuntu:24.04"
    builder = _Builder({reference: "ready"}, published={reference: ["linux/arm64", "unknown/unknown"]})

    build = await _run(entity_store, spec, builder)

    assert build.images[0].platform == "linux/arm64"
    assert builder.platforms == {copy_builder_name(reference, "linux/arm64"): "linux/arm64"}


async def test_a_prebuilt_image_published_for_several_platforms_follows_the_default(
    entity_store, store_task, archive
) -> None:
    spec = await _start(entity_store, store_task, config={"environment": {"docker_image": "ubuntu:24.04"}})
    reference = "docker.io/library/ubuntu:24.04"
    builder = _Builder({reference: "ready"}, published={reference: ["linux/amd64", "linux/arm64"]})

    await _run(entity_store, spec, builder)

    assert builder.platforms == {copy_builder_name(reference): None}


async def test_an_archive_read_for_its_listing_does_not_point_at_its_deleted_download(monkeypatch) -> None:
    _serve_archive(monkeypatch, {"environment/Dockerfile": b"FROM python:3.12\n"})

    archive = await read_task_archive(
        Mock(), HarborArchiveSource(fileset_ref="default/f#a/task_archive", files_hash="a" * 64)
    )

    assert archive.has("environment/Dockerfile")
    assert archive.text("environment/Dockerfile") is None


def test_build_recorded_with_role_still_loads() -> None:
    """Builds recorded before `role` was renamed to `type` keep loading, and are saved back with `type`."""
    image = BuildImage.model_validate({"role": "service/db", "source": "prebuilt", "image": "redis:7"})

    assert image.type == "service/db"
    assert "role" not in image.model_dump()


async def _build_named(entity_store, store_task, name: str, builder: _Builder) -> TaskImageBuildEntity:
    """Store task ``name`` and run its first image build over the archive being served."""
    digest = await store_task(name)
    await ImageBuildService(entity_store, _Jobs(), _allow).request_build("default", name, digest, ImageBuildRequest())
    spec = TaskImageBuildSpec(workspace="default", task=name, revision_digest=digest, attempt=1)
    return await _run(entity_store, spec, builder)


def _two_tasks_archive(instruction: bytes, dockerfile: bytes = b"FROM python:3.12\n") -> dict[str, bytes]:
    return {"instruction.md": instruction, "environment/Dockerfile": dockerfile, "environment/setup.sh": b"echo hi\n"}


async def test_a_task_whose_context_holds_the_same_files_uses_the_existing_build(
    entity_store, store_task, monkeypatch
) -> None:
    """Tasks that differ only outside their build context share one Builder build."""
    builder = _Builder({"sort-task/environment": "ready"})
    _serve_archive(monkeypatch, _two_tasks_archive(b"Sort the list.\n"))
    first = await _build_named(entity_store, store_task, "sort-a", builder)

    _serve_archive(monkeypatch, _two_tasks_archive(b"Sort the list in reverse.\n"))
    second = await _build_named(entity_store, store_task, "sort-b", builder)

    assert len(builder.submits) == 1
    assert first.status == second.status == "ready"
    assert first.images[0].image_ref == second.images[0].image_ref
    assert first.images[0].builder_image == second.images[0].builder_image


async def test_a_context_with_different_files_gets_its_own_build(entity_store, store_task, monkeypatch) -> None:
    builder = _Builder({"sort-task/environment": "ready"})
    _serve_archive(monkeypatch, _two_tasks_archive(b"Sort.\n"))
    first = await _build_named(entity_store, store_task, "sort-a", builder)

    _serve_archive(monkeypatch, _two_tasks_archive(b"Sort.\n", dockerfile=b"FROM python:3.13\n"))
    second = await _build_named(entity_store, store_task, "sort-b", builder)

    assert len(builder.submits) == 2
    assert first.images[0].context_digest != second.images[0].context_digest


async def test_main_and_verifier_with_identical_contexts_share_one_build(entity_store, store_task, monkeypatch) -> None:
    _serve_archive(
        monkeypatch, {"environment/Dockerfile": b"FROM python:3.12\n", "tests/Dockerfile": b"FROM python:3.12\n"}
    )
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"})

    build = await _run(entity_store, await _start(entity_store, store_task, config=_SEPARATE_VERIFIER), builder)

    assert build.status == "ready"
    assert len(builder.submits) == 1
    assert build.images[0].image_ref == build.images[1].image_ref
