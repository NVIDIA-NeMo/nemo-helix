# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task-builder Job's work: plan, submit and follow every image of one build."""

from __future__ import annotations

from typing import Literal
from unittest.mock import Mock

import httpx
import pytest
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.schema import FileSetSource, ImageSource, SubmitBuildResponse
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.image_builds.models import ImageBuildRequest, TaskImageBuildEntity
from nemo_evals.image_builds.planning import TaskArchive
from nemo_evals.image_builds.runner import TaskImageBuildSpec, run_task_image_build
from nemo_evals.image_builds.service import BuildJob, ImageBuildService
from nemo_evals.revisions import get_revision
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

Outcome = Literal["ready", "failed"]
_SEPARATE_VERIFIER = {"verifier": {"environment_mode": "separate"}}


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

    def __init__(self, outcomes: dict[str, Outcome], *, refuse: bool = False, job: str = "active") -> None:
        self.outcomes = outcomes
        self.refuse = refuse
        self.job = job
        self.images: dict[str, str] = {}
        self.submits: list[FileSetSource | ImageSource] = []

    async def get_image(self, name: str) -> ContainerImage | None:
        context = self.images.get(name)
        if context is None:
            return None
        outcome = self.outcomes[context]
        return _container_image(name, outcome, detail="kaniko exited 1" if outcome == "failed" else None)

    async def submit(
        self, *, name: str, revision: int, source: FileSetSource | ImageSource, dockerfile: str | None
    ) -> SubmitBuildResponse:
        if self.refuse:
            raise PermissionDeniedError(httpx.Response(403, json={"detail": "not allowed to build"}))
        what = source.image if isinstance(source, ImageSource) else source.context_path
        assert what is not None
        self.submits.append(source)
        self.images[f"{name}-{revision}"] = what
        return _submitted(name, revision)

    async def job_status(self, job: str) -> str | None:
        return self.job


async def _no_sleep(seconds: float) -> None:
    pass


async def _allow(workspace: str) -> None:
    pass


@pytest.fixture
def archive(monkeypatch) -> list[str]:
    files = ["environment/Dockerfile", "tests/Dockerfile"]

    async def read(files_client, definition):
        return TaskArchive(root="sort-task", files=frozenset(f"sort-task/{f}" for f in files), compose=None)

    monkeypatch.setattr("nemo_evals.image_builds.runner._read_archive", read)
    return files


async def _start(entity_store, store_task, *, config: dict, allowed: list[str] | None = None) -> TaskImageBuildSpec:
    digest = await store_task(config=config)
    service = ImageBuildService(entity_store, _Jobs(), _allow)
    await service.request_build("default", "sort-task", digest, ImageBuildRequest(allowed_registries=allowed or []))
    return TaskImageBuildSpec(workspace="default", task="sort-task", revision_digest=digest, attempt=1)


async def _stored_build(entity_store, spec: TaskImageBuildSpec) -> TaskImageBuildEntity:
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    revision = await get_revision(entity_store, TaskRevisionEntity, head, spec.revision_digest)
    return await entity_store.get(TaskImageBuildEntity, name="build.1", workspace="default", parent=revision.id)


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
    assert [(image.role, image.status) for image in build.images] == [("main", "ready"), ("verifier", "ready")]
    assert all(image.image_ref and image.builder_image and image.build_job for image in build.images)
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


async def test_an_unbuildable_task_fails_with_the_plan_error(entity_store, store_task, archive) -> None:
    archive.remove("tests/Dockerfile")
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

    async def corrupt(files_client, definition):
        raise ValueError("Downloaded object checksum mismatch")

    monkeypatch.setattr("nemo_evals.image_builds.runner._read_archive", corrupt)

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
