# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task-builder Job's work: plan, submit and follow every image of one build."""

from __future__ import annotations

from typing import Literal
from unittest.mock import Mock

import httpx
import pytest
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.schema import BuildSet, BuildSpec, FileSetSource, ImageSource, SubmitBuildResponse
from nemo_builder_plugin.submit import request_digest
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.image_builds.models import ImageBuildRequest, TaskImageBuildEntity
from nemo_evals.image_builds.planning import TaskArchive
from nemo_evals.image_builds.runner import TaskImageBuildSpec, run_task_image_build
from nemo_evals.image_builds.service import BuildJob, ImageBuildService
from nemo_evals.jobs.task_image_build import is_job_for
from nemo_evals.revisions import get_revision
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
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
    def __init__(self) -> None:
        self.created: list[CreateHelixJobRequest] = []

    async def create(self, request: CreateHelixJobRequest) -> None:
        self.created.append(request)

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
        self.requests: list[BuildSet] = []

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
        spec = BuildSpec(source=source) if dockerfile is None else BuildSpec(source=source, dockerfile=dockerfile)
        self.requests.append(BuildSet(name=name, revision=revision, build_specs=[spec]))
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
    jobs = _Jobs()
    service = ImageBuildService(entity_store, jobs, _allow)
    await service.request_build("default", "sort-task", digest, ImageBuildRequest(allowed_registries=allowed or []))
    return TaskImageBuildSpec.model_validate(jobs.created[0].spec)


async def _stored_build(entity_store, spec: TaskImageBuildSpec) -> TaskImageBuildEntity:
    return await entity_store.get(
        TaskImageBuildEntity, name=f"build.{spec.attempt}", workspace=spec.workspace, parent=spec.revision_id
    )


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


@pytest.mark.parametrize("republish_before_request", [True, False])
async def test_republished_content_does_not_redirect_the_serialized_job(
    entity_store, store_task, archive, republish_before_request: bool
) -> None:
    digest = await store_task(tags=["v1"])
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    original = await get_revision(entity_store, TaskRevisionEntity, head, "v1")

    async def republish() -> None:
        assert await store_task(archive_digest="c" * 64) != digest
        assert await store_task() == digest

    if republish_before_request:
        await republish()
    jobs = _Jobs()
    service = ImageBuildService(entity_store, jobs, _allow)
    first = await service.request_build("default", "sort-task", "v1", ImageBuildRequest())
    spec = TaskImageBuildSpec.model_validate(jobs.created[0].spec)
    if not republish_before_request:
        await republish()
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    latest = await get_revision(entity_store, TaskRevisionEntity, head)
    assert original.id != latest.id and original.content_hash == latest.content_hash

    built = await _run(entity_store, spec, _Builder({"sort-task/environment": "ready"}))

    assert built.status == "ready" and built.parent == original.id
    assert await service.get_build("default", "sort-task", "latest", "1") is None
    second = await service.request_build("default", "sort-task", "latest", ImageBuildRequest())
    assert first is not None and second is not None and first.job != second.job
    assert not is_job_for(jobs.created[0].custom_fields, TaskImageBuildSpec.model_validate(jobs.created[-1].spec))


@pytest.mark.parametrize("failure", ["missing_head", "missing_revision", "wrong_digest", "mutated_content"])
async def test_revision_validation_failure_is_recorded_on_the_attempt(
    entity_store, store_task, archive, failure: str
) -> None:
    spec = await _start(entity_store, store_task, config={})
    build = await _stored_build(entity_store, spec)
    revision = await entity_store.get_by_id(TaskRevisionEntity, build.parent)
    if failure == "missing_head":
        await entity_store.delete(TaskEntity, "sort-task", workspace="default")
    elif failure == "wrong_digest":
        spec = spec.model_copy(update={"revision_digest": "d" * 64})
    elif failure == "mutated_content":
        revision.spec = revision.spec.model_copy(update={"instruction": "modified after publication"})
        await entity_store.update(revision)
    else:
        await entity_store.delete(TaskRevisionEntity, revision.name, workspace="default", parent=revision.parent)
    builder = _Builder({"sort-task/environment": "ready"})

    result = await _run(entity_store, spec, builder)

    assert result.status == "failed"
    assert result.error is not None and result.error.startswith("revision_validation_failed:")
    stored = await entity_store.get(TaskImageBuildEntity, "build.1", workspace="default", parent=build.parent)
    assert stored.status == "failed" and builder.submits == []


async def test_a_spec_for_another_task_does_not_modify_the_attempt(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config={})
    original = await _stored_build(entity_store, spec)
    await store_task(name="other-task")
    foreign_spec = spec.model_copy(update={"task": "other-task"})
    builder = _Builder({"sort-task/environment": "ready"})

    with pytest.raises((ValueError, NemoEntityNotFoundError)):
        await _run(entity_store, foreign_spec, builder)

    assert await _stored_build(entity_store, spec) == original
    assert builder.submits == []


async def test_missing_attempt_is_not_recreated(entity_store, store_task, archive) -> None:
    spec = await _start(entity_store, store_task, config={})
    build = await _stored_build(entity_store, spec)
    await entity_store.delete(TaskImageBuildEntity, build.name, workspace=build.workspace, parent=build.parent)
    builder = _Builder({"sort-task/environment": "ready"})

    with pytest.raises(NemoEntityNotFoundError):
        await _run(entity_store, spec, builder)

    with pytest.raises(NemoEntityNotFoundError):
        await _stored_build(entity_store, spec)
    assert builder.submits == []


async def test_revision_error_preserves_a_concurrently_terminal_result(
    entity_store, store_task, archive, monkeypatch
) -> None:
    spec = await _start(entity_store, store_task, config={})

    async def read_by_id(entity_type, entity_id):
        current = await _stored_build(entity_store, spec)
        current.status = "ready"
        await entity_store.update(current)
        raise ValueError("revision read failed after another worker finished")

    monkeypatch.setattr(entity_store, "get_by_id", read_by_id)
    builder = _Builder({})

    result = await _run(entity_store, spec, builder)

    assert result.status == "ready" and result.error is None
    assert builder.submits == []


@pytest.mark.parametrize("failure", ["workspace", "parent", "typed_name_missing", "typed_id_mismatch"])
async def test_exact_revision_lookup_checks_ownership_and_stored_type(
    entity_store, store_task, monkeypatch, failure: str
) -> None:
    from nemo_evals.revisions import RevisionNotFoundError, get_revision_by_id

    await store_task()
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    revision = await get_revision(entity_store, TaskRevisionEntity, head)
    candidate = revision.model_copy(deep=True)
    requested_id = revision.id
    if failure == "workspace":
        candidate.workspace = "other"
    elif failure == "parent":
        candidate._parent = "other-task-id"
    elif failure == "typed_name_missing":
        candidate.name = "impostor"
    else:
        requested_id = "impostor-id"
        candidate._id = requested_id

    async def read_by_id(entity_type, entity_id):
        return candidate

    monkeypatch.setattr(entity_store, "get_by_id", read_by_id)

    with pytest.raises(RevisionNotFoundError):
        await get_revision_by_id(entity_store, TaskRevisionEntity, head, requested_id)


async def test_runner_keeps_builder_names_and_submitted_sources_consistent(entity_store, store_task, archive) -> None:
    builder = _Builder({"sort-task/environment": "ready"})
    sources = [
        "default/harbor-tasks#team-a/task_archive",
        "default/harbor-tasks#team-b/task_archive",
        "default/harbor-tasks#team-b/task_archive",
    ]
    for name, ref in zip(["first", "second", "third"], sources, strict=True):
        digest = await store_task(name=name, fileset_ref=ref)
        jobs = _Jobs()
        service = ImageBuildService(entity_store, jobs, _allow)
        await service.request_build("default", name, digest, ImageBuildRequest())
        build = await _run(entity_store, TaskImageBuildSpec.model_validate(jobs.created[0].spec), builder)
        assert build.status == "ready"

    first, second, repeated = builder.requests
    first_source, second_source = first.build_specs[0].source, second.build_specs[0].source
    assert isinstance(first_source, FileSetSource) and isinstance(second_source, FileSetSource)
    assert first_source.archive == "team-a/task_archive"
    assert second_source.archive == "team-b/task_archive"
    assert first.name != second.name
    assert request_digest(first) != request_digest(second)
    assert repeated == second


async def test_unsupported_compose_option_fails_before_any_builder_submission(
    entity_store, store_task, monkeypatch
) -> None:
    spec = await _start(entity_store, store_task, config={})

    async def read(files_client, definition):
        return TaskArchive(
            root="sort-task",
            files=frozenset({"sort-task/environment/Dockerfile"}),
            compose="services:\n  main:\n    build: .\n  db:\n    build: {context: ., args: {MODE: test}}\n",
        )

    monkeypatch.setattr("nemo_evals.image_builds.runner._read_archive", read)
    builder = _Builder({"sort-task/environment": "ready"})

    build = await _run(entity_store, spec, builder)

    assert build.status == "failed"
    assert build.error is not None and build.error.startswith("unsupported_compose_build_option:")
    assert "db" in build.error and builder.submits == []
    assert (await _stored_build(entity_store, spec)).status == "failed"


@pytest.mark.parametrize(("status", "job_status"), [("ready", "completed"), ("failed", "failed")])
def test_job_fails_when_the_build_fails(monkeypatch, status: str, job_status: str) -> None:
    """The Job's exit status is what Jobs shows, so a failed build must not exit as success."""
    from nemo_evals.jobs.task_image_build import TaskImageBuildJob
    from nemo_helix_plugin.client.client import AsyncNemoClient

    async def run(spec, **clients):
        return Mock(status=status)

    monkeypatch.setattr("nemo_evals.jobs.task_image_build.run_task_image_build", run)
    config = TaskImageBuildSpec(
        workspace="default", task="t", revision_id="revision-id", revision_digest="a" * 64, attempt=1
    )

    result = TaskImageBuildJob().run(
        config.model_dump(), ctx=Mock(), async_client=AsyncNemoClient(base_url="http://platform.test")
    )

    assert result["status"] == job_status
