# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task-builder Job's work: plan, submit and follow every image of one build."""

from __future__ import annotations

from typing import Literal
from unittest.mock import Mock

import httpx
import pytest
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.image_builds.builder import ArchiveSource, BuilderImage, SubmittedBuild
from nemo_evals.image_builds.models import ImageBuildRequest, TaskImageBuildEntity
from nemo_evals.image_builds.planning import TaskArchive
from nemo_evals.image_builds.runner import TaskImageBuildSpec, run_task_image_build
from nemo_evals.image_builds.service import BuildJob, ImageBuildService
from nemo_evals.revisions import get_revision
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

Outcome = Literal["ready", "failed"]
_SEPARATE_VERIFIER = {"verifier": {"environment_mode": "separate"}}


class _Jobs:
    async def create(self, request: CreateHelixJobRequest) -> None:
        pass

    async def get(self, name: str) -> BuildJob | None:
        return None


class _Builder:
    """Builder whose images finish with ``outcomes[context_path]`` on the first read after submit."""

    def __init__(self, outcomes: dict[str, Outcome], *, refuse: bool = False) -> None:
        self.outcomes = outcomes
        self.refuse = refuse
        self.images: dict[str, str] = {}
        self.submits = 0

    async def get_image(self, name: str) -> BuilderImage | None:
        context = self.images.get(name)
        if context is None:
            return None
        outcome = self.outcomes[context]
        return BuilderImage(
            name=name,
            status=outcome,
            image_ref=f"registry.test/{name}@sha256:{'c' * 64}" if outcome == "ready" else None,
            status_detail="kaniko exited 1" if outcome == "failed" else None,
        )

    async def submit(self, *, name: str, revision: int, source: ArchiveSource, dockerfile: str) -> SubmittedBuild:
        if self.refuse:
            raise PermissionDeniedError(httpx.Response(403, json={"detail": "not allowed to build"}))
        self.submits += 1
        self.images[f"{name}-{revision}"] = source.context_path
        return SubmittedBuild(image=BuilderImage(name=f"{name}-{revision}", status="pending"), job=f"builder-{name}")


async def _no_sleep(seconds: float) -> None:
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
    service = ImageBuildService(entity_store, _Jobs())
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


async def test_agent_and_verifier_built_makes_the_build_ready(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"})

    build = await _run(entity_store, spec, builder)

    assert build.status == "ready"
    assert build.error is None
    assert [(image.role, image.status) for image in build.images] == [("agent", "ready"), ("verifier", "ready")]
    assert all(image.image_ref and image.builder_image and image.build_job for image in build.images)
    assert await _stored_build(entity_store, spec) == build


async def test_one_failed_image_fails_the_build_with_its_reason(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "failed"})

    build = await _run(entity_store, spec, builder)

    assert build.status == "failed"
    assert build.error == "verifier: kaniko exited 1"


async def test_prebuilt_images_fail_the_build_without_submitting_anything(entity_store, store_task, archive) -> None:
    spec = await _start(
        entity_store, store_task, config={"environment": {"docker_image": "ghcr.io/org/sort:1"}}, allowed=["ghcr.io"]
    )
    builder = _Builder({})

    build = await _run(entity_store, spec, builder)

    assert build.status == "failed"
    assert build.error == "prebuilt_import_unsupported: agent"
    assert builder.submits == 0


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
        async def get_image(self, name: str) -> BuilderImage | None:
            return None if name not in self.images else BuilderImage(name=name, status="pending")

    ticks = iter(range(0, 10_000, 100))
    build = await _run(entity_store, spec, _Stuck({}), timeout=250, clock=lambda: next(ticks))

    assert build.status == "failed"
    assert build.error == "agent: no result from Builder within 250 seconds"


async def test_a_finished_build_is_left_alone(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config={})
    first = await _run(entity_store, spec, _Builder({"sort-task/environment": "ready"}))
    builder = _Builder({"sort-task/environment": "failed"})

    again = await _run(entity_store, spec, builder)

    assert again == first
    assert builder.submits == 0


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
