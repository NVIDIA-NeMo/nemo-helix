# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task-builder Job's work: plan, submit and follow every image of one build."""

from __future__ import annotations

import hashlib
import io
import tarfile
from collections.abc import Callable, Sequence
from typing import Literal
from unittest.mock import Mock

import httpx
import pytest
from nemo_builder_plugin.entities import (
    IN_CLUSTER_BACKEND,
    ContainerImage,
    ExternalImage,
    ImageRuntimeConfig,
    Provenance,
)
from nemo_builder_plugin.remote import RemoteBackend
from nemo_builder_plugin.schema import (
    CODE_BACKEND_REJECTED,
    BuildMount,
    FileSetSource,
    ImageSource,
    SubmitBuildResponse,
)
from nemo_evals.api.task_definitions.harbor import HarborArchiveSource, HarborTaskDefinition
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.harbor.archive import pack_task
from nemo_evals.image_builds.builder import builder_name, copy_builder_name
from nemo_evals.image_builds.models import (
    BuildImage,
    ImageBuildRequest,
    TaskImageBuildEntity,
    build_entity_name,
    build_job_name,
)
from nemo_evals.image_builds.runner import (
    TaskImageBuildSpec,
    apply_builder_image,
    read_task_archive,
    run_task_image_build,
)
from nemo_evals.image_builds.service import BuildJob, ImageBuildService
from nemo_evals.jobs.task_image_build import is_job_for, task_image_build_job_request
from nemo_evals.revisions import apply_tag, get_revision, publish_revision
from nemo_helix_plugin.client.errors import ConflictError, NemoClientError, NemoTransportError, PermissionDeniedError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

Outcome = Literal["ready", "failed"]
_SEPARATE_VERIFIER = {"verifier": {"environment_mode": "separate"}}
# A task whose main image is built from its Dockerfile and whose verifier runs in an image the task names.
_PREBUILT_VERIFIER = {"verifier": {"environment": {"docker_image": "ghcr.io/org/verify:1"}}}


# How every ready image starts, as Builder records it from the image's config.
RUNTIME_CONFIG = ImageRuntimeConfig(entrypoint=["/bin/bash", "-l", "-c"], working_dir="/app")


REMOTE_BACKEND = RemoteBackend.name

# What Builder says when it refuses a request itself, as a deployment that can't build it does.
REFUSAL = "this deployment has no registry to push an image to"


def _handle(name: str) -> ExternalImage:
    """What a remote build service calls the image of Builder row ``name``: a handle and a build of its own."""
    return ExternalImage(image_id=f"image-{name}", build_id=f"build-{name}")


def _no_handle(name: str) -> ExternalImage:
    """What a remote build service that has no handle of its own reports for row ``name``."""
    return ExternalImage(build_id=f"build-{name}")


def _refused(code: str | None = None) -> NemoClientError:
    """Builder refusing a build: by permission, or with a 409 that names ``code``."""
    if code is None:
        return PermissionDeniedError(httpx.Response(403, json={"detail": "not allowed to build"}))
    return ConflictError(httpx.Response(409, json={"detail": "the backend refuses it", "code": code}))


def _container_image(
    name: str,
    status: Literal["pending", "ready", "failed"] = "pending",
    *,
    detail: str | None = None,
    backend: str = IN_CLUSTER_BACKEND,
    external: ExternalImage | None = None,
) -> ContainerImage:
    """A Builder row. ``backend`` built it; ``external`` is what a remote build service calls the image.

    Builder writes ``external`` when a row becomes ready. It is kept here at any status, to show when it is read.
    """
    return ContainerImage(
        name=name,
        workspace="default",
        status=status,
        status_detail=detail,
        registry="registry.test",
        repository=f"default/{name}",
        digest="sha256:" + "c" * 64 if status == "ready" else None,
        runtime_config=RUNTIME_CONFIG if status == "ready" else None,
        external=external,
        provenance=Provenance(
            build_set=name.rsplit("-", 1)[0],
            revision=int(name.rsplit("-", 1)[1]),
            job=f"default/builder-{name}",
            request_digest="sha256:" + "a" * 64,
            backend=backend,
        ),
    )


def _submitted(name: str, revision: int, *, backend: str = IN_CLUSTER_BACKEND) -> SubmitBuildResponse:
    return SubmitBuildResponse(
        job=f"builder-{name}-{revision}", images=[_container_image(f"{name}-{revision}", backend=backend)]
    )


def _refusal(code: str | None = "backend_rejected") -> ConflictError:
    """Builder's 409 that refuses the request itself. It says so by its ``code``; an older Builder gives none."""
    return ConflictError(httpx.Response(409, json={"detail": REFUSAL} | ({"code": code} if code else {})))


class _Jobs:
    async def create(self, request: CreateHelixJobRequest) -> None:
        pass

    async def get(self, name: str) -> BuildJob | None:
        return None


class _Builder:
    """Builder whose images finish with ``outcomes[context_path]`` on the first read after submit.

    It builds in-cluster unless ``backend`` names another; ``external`` then gives what its build service calls the
    image of each row, by the row's name. ``refusals`` is what Builder answers the submit of each image it names,
    by context path or prebuilt image, in place of taking it: a refusal, or another error. ``refuse`` refuses
    every submit for lack of permission.
    """

    def __init__(
        self,
        outcomes: dict[str, Outcome],
        *,
        refuse: bool = False,
        refusals: dict[str, NemoClientError] | None = None,
        job: str = "active",
        published: dict[str, list[str]] | None = None,
        backend: str = IN_CLUSTER_BACKEND,
        external: Callable[[str], ExternalImage] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.refuse = refuse
        self.refusals = refusals or {}
        self.job = job
        self.backend = backend
        self.external = external
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
        return _container_image(
            name,
            outcome,
            detail="kaniko exited 1" if outcome == "failed" else None,
            backend=self.backend,
            external=None if self.external is None else self.external(name),
        )

    async def submit(
        self,
        *,
        name: str,
        revision: int,
        source: FileSetSource | ImageSource,
        dockerfile: str | None,
        platform: str | None,
        mounts: Sequence[BuildMount] = (),
    ) -> SubmitBuildResponse:
        self.platforms[name] = platform
        if self.refuse:
            raise _refused()
        what = source.image if isinstance(source, ImageSource) else source.context_path
        assert what is not None
        if what in self.refusals:
            raise self.refusals[what]
        self.submits.append(source)
        self.images[f"{name}-{revision}"] = what
        return _submitted(name, revision, backend=self.backend)

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
    name = build_entity_name(spec.revision_digest, spec.attempt)
    return await entity_store.get(TaskImageBuildEntity, name=name, workspace="default", parent=head.id)


async def _stored_task_definition(entity_store, spec: TaskImageBuildSpec) -> HarborTaskDefinition:
    head = await entity_store.get(TaskEntity, name="sort-task", workspace="default")
    revision = await get_revision(entity_store, TaskRevisionEntity, head, spec.revision_digest)
    assert isinstance(revision.spec, HarborTaskDefinition)
    return revision.spec


async def _run(entity_store, spec: TaskImageBuildSpec, builder: _Builder, **kwargs) -> TaskImageBuildEntity:
    """Run a build to its end. ``_Builder``'s images finish on the first read after their submit.

    Unless a test brings its own, the clock moves 100 seconds each time it is read and the build is given 250.
    A build left unsettled, as by an image that was never submitted, then ends as timed out instead of being
    waited for on the real clock.
    """
    ticks = iter(range(0, 10_000, 100))
    kwargs.setdefault("timeout", 250)
    kwargs.setdefault("clock", lambda: next(ticks))
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


async def test_an_image_a_remote_build_service_built_records_its_handle_and_backend(
    entity_store, store_task, archive, build_image_keys
) -> None:
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder(
        {"sort-task/environment": "ready", "sort-task/tests": "ready"}, backend=REMOTE_BACKEND, external=_handle
    )

    build = await _run(entity_store, spec, builder)

    assert build.status == "ready"
    # Each image gets the handle of its own Builder row.
    assert [(image.external_image_id, image.builder_backend) for image in build.images] == [
        (f"image-{image.builder_image}", REMOTE_BACKEND) for image in build.images
    ]
    assert len({image.external_image_id for image in build.images}) == 2
    stored = await _stored_build(entity_store, spec)
    assert stored == build
    # What the entity store is sent holds both, beside the keys a build image always had.
    for image, sent in zip(build.images, stored._get_data_fields()["images"], strict=True):
        assert set(sent) == {*build_image_keys, "external_image_id", "builder_backend"}
        assert (sent["external_image_id"], sent["builder_backend"]) == (image.external_image_id, REMOTE_BACKEND)


async def test_a_remote_image_is_recorded_with_its_backend_from_its_submit(
    entity_store, store_task, archive, build_image_keys
) -> None:
    """When another image is refused at submit, the build settles there: the submit is the only record of the rest."""
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder(
        {"sort-task/environment": "ready"},
        refusals={"sort-task/tests": _refused()},
        backend=REMOTE_BACKEND,
        external=_handle,
    )

    build = await _run(entity_store, spec, builder)

    assert build.status == "failed"
    main, verifier = (await _stored_build(entity_store, spec))._get_data_fields()["images"]
    assert (main["status"], main["builder_backend"]) == ("building", REMOTE_BACKEND)
    assert set(main) == {*build_image_keys, "builder_backend"}
    assert (verifier["status"], verifier["builder_image"]) == ("failed", None)
    assert list(verifier) == build_image_keys


async def test_a_ready_remote_image_without_a_handle_is_ready_with_none(
    entity_store, store_task, archive, build_image_keys
) -> None:
    """A build service needn't have a handle of its own, so the backend is what says who built the image."""
    spec = await _start(entity_store, store_task, config={})
    builder = _Builder({"sort-task/environment": "ready"}, backend=REMOTE_BACKEND, external=_no_handle)

    build = await _run(entity_store, spec, builder)

    [image] = build.images
    assert (build.status, image.status, image.image_ref is not None) == ("ready", "ready", True)
    assert (image.external_image_id, image.builder_backend) == (None, REMOTE_BACKEND)
    [stored] = (await _stored_build(entity_store, spec))._get_data_fields()["images"]
    assert set(stored) == {*build_image_keys, "builder_backend"}


async def test_an_image_built_in_cluster_is_recorded_key_for_key_as_before(
    entity_store, store_task, archive, build_image_keys
) -> None:
    """A deployment that builds in-cluster must write the record it always wrote: an older reader refuses a new key."""
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)

    build = await _run(entity_store, spec, _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"}))

    assert build.status == "ready"
    stored = await _stored_build(entity_store, spec)
    for image, sent in zip(build.images, stored._get_data_fields()["images"], strict=True):
        assert (image.external_image_id, image.builder_backend) == (None, None)
        assert list(sent) == build_image_keys


@pytest.mark.parametrize("status", ["pending", "failed", "ready"])
def test_a_remote_image_names_its_backend_at_every_status_and_its_handle_only_once_ready(
    status: Literal["pending", "ready", "failed"],
) -> None:
    """The backend is known at submit, so a build that is still running or has failed says who was building it.

    The handle is the finished image's: a row that isn't ready records none, even when it carries one.
    """
    image = BuildImage(type="main", source="dockerfile")
    row = _container_image("harbor-env-abc-1", status, backend=REMOTE_BACKEND, external=_handle("harbor-env-abc-1"))

    apply_builder_image(image, row)

    assert row.external is not None and row.external.image_id == "image-harbor-env-abc-1"
    assert image.builder_backend == REMOTE_BACKEND
    assert image.external_image_id == ("image-harbor-env-abc-1" if status == "ready" else None)


@pytest.mark.parametrize("status", ["pending", "failed", "ready"])
def test_an_in_cluster_image_names_no_backend(status: Literal["pending", "ready", "failed"]) -> None:
    """A row this cluster builds leaves the image as it was recorded before there was another backend to name."""
    image = BuildImage(type="main", source="dockerfile")

    apply_builder_image(image, _container_image("harbor-env-abc-1", status))

    assert image.builder_backend is None
    assert "builder_backend" not in image.model_dump()


@pytest.mark.parametrize("status", ["pending", "failed", "ready"])
def test_a_read_without_the_backend_keeps_the_one_recorded(status: Literal["pending", "ready", "failed"]) -> None:
    """A Builder from before the backend existed reports every row as this cluster's, and without a handle."""
    image = BuildImage(type="main", source="dockerfile")
    apply_builder_image(
        image,
        _container_image("harbor-env-abc-1", "ready", backend=REMOTE_BACKEND, external=_handle("harbor-env-abc-1")),
    )

    apply_builder_image(image, _container_image("harbor-env-abc-1", status))

    assert (image.external_image_id, image.builder_backend) == ("image-harbor-env-abc-1", REMOTE_BACKEND)


@pytest.mark.parametrize(
    ("backend", "config", "refusal"),
    [
        # This cluster refuses a user who may not build.
        (IN_CLUSTER_BACKEND, {}, _refused()),
        # A remote build service can't copy a prebuilt image, so Builder refuses it by the backend.
        (REMOTE_BACKEND, {"environment": {"docker_image": "ghcr.io/org/sort:1"}}, _refused(CODE_BACKEND_REJECTED)),
    ],
    ids=["in-cluster", "remote"],
)
async def test_an_image_builder_refused_names_no_backend_whichever_backend_refused_it(
    entity_store, store_task, archive, build_image_keys, backend: str, config: dict, refusal: NemoClientError
) -> None:
    """The backend is read from the image's Builder row, and Builder writes no row for a build it refuses.

    So a missing backend doesn't by itself say this cluster built the image: such an image has no ``builder_image``.
    """
    spec = await _start(entity_store, store_task, config=config)
    builder = _Builder({}, refusals={"sort-task/environment": refusal, "ghcr.io/org/sort:1": refusal}, backend=backend)

    build = await _run(entity_store, spec, builder)

    [image] = build.images
    assert (build.status, image.status, image.builder_image) == ("failed", "failed", None)
    # A 403 is recorded as the client reports it; a 409 that refuses the request is recorded as Builder's own sentence.
    assert image.error is not None and image.error.startswith("Builder refused the build: ")
    assert image.builder_backend is None
    [stored] = (await _stored_build(entity_store, spec))._get_data_fields()["images"]
    assert list(stored) == build_image_keys


def test_a_build_image_recorded_before_the_handle_existed_still_loads_and_is_saved_unchanged(build_image_keys) -> None:
    recorded = dict.fromkeys(build_image_keys) | {
        "type": "main",
        "source": "dockerfile",
        "context_path": "sort-task/environment",
        "dockerfile": "Dockerfile",
        "context_digest": "d" * 64,
        "status": "ready",
        "builder_image": "harbor-env-abc-1",
        "build_job": "builder-harbor-env-abc-1",
        "image_ref": "registry.test/default/harbor-env-abc-1@sha256:" + "c" * 64,
    }

    image = BuildImage.model_validate(recorded)

    assert (image.external_image_id, image.builder_backend) == (None, None)
    assert image.model_dump(mode="json") == recorded


def test_a_build_image_keeps_a_key_a_later_release_records() -> None:
    """A later release's record still loads here, and a read-modify-write here doesn't drop what it added."""
    recorded = {"type": "main", "source": "dockerfile", "status": "ready", "later_key": {"kept": True}}

    image = BuildImage.model_validate(recorded)

    assert image.model_dump(mode="json")["later_key"] == {"kept": True}


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


@pytest.mark.parametrize("code", ["backend_rejected", "missing_secrets", None], ids=["rejected", "secrets", "no-code"])
async def test_a_dockerfile_build_builder_refuses_fails_its_image_with_builders_sentence(
    entity_store, store_task, archive, code: str | None
) -> None:
    """The record is stored with its images: the refused one says why, and the one submitted before it is named.

    An older Builder gives its refusal no code and writes no row for it, which reads the same.
    """
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder({"sort-task/environment": "ready"}, refusals={"sort-task/tests": _refusal(code)})

    build = await _run(entity_store, spec, builder)

    assert (build.status, build.error) == ("failed", f"verifier: Builder refused the build: {REFUSAL}")
    main, verifier = build.images
    assert (verifier.type, verifier.status, verifier.builder_image) == ("verifier", "failed", None)
    assert verifier.error == f"Builder refused the build: {REFUSAL}"
    # The build submitted before the refusal goes on at Builder, and the record says which it is.
    assert (main.type, main.status) == ("main", "building")
    assert main.builder_image is not None and main.build_job is not None
    assert await _stored_build(entity_store, spec) == build


async def test_nothing_is_submitted_after_a_refusal(entity_store, store_task, archive) -> None:
    """The build has failed, so an image not yet submitted is left as it was planned."""
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder({}, refusals={"sort-task/environment": _refusal()})

    build = await _run(entity_store, spec, builder)

    assert builder.submits == []
    assert build.error == f"main: Builder refused the build: {REFUSAL}"
    assert [(image.type, image.status, image.builder_image) for image in build.images] == [
        ("main", "failed", None),
        ("verifier", "pending", None),
    ]
    assert await _stored_build(entity_store, spec) == build


async def test_a_task_whose_copy_builder_refuses_starts_no_dockerfile_build(entity_store, store_task, archive) -> None:
    """A deployment that copies no image can't build a task that names one.

    The copy is asked for first, so the refusal comes before the task's Dockerfile image is submitted.
    """
    spec = await _start(entity_store, store_task, config=_PREBUILT_VERIFIER)
    builder = _Builder({}, refusals={"ghcr.io/org/verify:1": _refusal()})

    build = await _run(entity_store, spec, builder)

    assert builder.submits == []
    assert (build.status, build.error) == ("failed", f"verifier: Builder refused the build: {REFUSAL}")
    assert [(image.type, image.status, image.builder_image) for image in build.images] == [
        ("main", "pending", None),
        ("verifier", "failed", None),
    ]
    assert await _stored_build(entity_store, spec) == build


async def test_a_rebuild_builder_refuses_fails_its_image_and_submits_nothing_after_it(
    entity_store, store_task, archive
) -> None:
    """A rebuild takes each Dockerfile image to Builder again, so a refusal is that submit's own answer.

    It is recorded as a refused copy is: Builder's sentence on the image, and nothing asked of Builder after it.
    """
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"})
    await _run(entity_store, spec, builder)
    service = ImageBuildService(entity_store, _Jobs(), _allow)
    await service.request_build("default", "sort-task", spec.revision_digest, ImageBuildRequest(rebuild=True))
    builder.refusals = {"sort-task/environment": _refusal()}
    again = spec.model_copy(update={"attempt": 2})

    rebuilt = await _run(entity_store, again, builder)

    assert len(builder.submits) == 2
    assert (rebuilt.status, rebuilt.error) == ("failed", f"main: Builder refused the build: {REFUSAL}")
    main, verifier = rebuilt.images
    assert (main.status, main.builder_image, main.error) == ("failed", None, f"Builder refused the build: {REFUSAL}")
    # The verifier's build of the first attempt is ready at Builder. It is not read either: the image stays as planned.
    assert (verifier.status, verifier.builder_image, verifier.image_ref) == ("pending", None, None)
    assert await _stored_build(entity_store, again) == rebuilt


@pytest.fixture
def compose_archive(monkeypatch, tmp_path) -> None:
    """A task whose compose file adds a service it builds and a service it names an image for.

    A compose task is extracted to be planned, so its archive is a whole task, packed as publishing packs one.
    """
    pytest.importorskip("harbor")
    root = tmp_path / "sort-task"
    for name, text in {
        "task.toml": "",
        "instruction.md": "Sort the list.\n",
        "environment/Dockerfile": "FROM python:3.12\n",
        "environment/docker-compose.yaml": "services:\n  db:\n    build: ./db\n  cache:\n    image: redis:7\n",
        "environment/db/Dockerfile": "FROM postgres:16\n",
        "tests/test.sh": "exit 0\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    packed = tmp_path / "task_archive"
    pack_task(root, packed)

    async def download(files_client, ref, output, **kwargs) -> None:
        output.write(packed.read_bytes())

    monkeypatch.setattr("nemo_evals.image_builds.runner.download_verified_async", download)


#: Every image of ``compose_archive``'s task with a prebuilt verifier, by what Builder is sent for it, in plan order.
_COMPOSE_IMAGES = {
    "sort-task/environment": ("main", "dockerfile"),
    "sort-task/environment/db": ("service/db", "dockerfile"),
    "docker.io/library/redis:7": ("service/cache", "prebuilt"),
    "ghcr.io/org/verify:1": ("verifier", "prebuilt"),
}


def _submitted_in_order(builder: _Builder) -> list[str | None]:
    """What each submit Builder took was of: an image to copy, or a context path."""
    return [s.image if isinstance(s, ImageSource) else s.context_path for s in builder.submits]


async def test_prebuilt_images_are_submitted_before_dockerfile_images(
    entity_store, store_task, compose_archive
) -> None:
    """Each kind goes in the order the task plans it."""
    spec = await _start(entity_store, store_task, config=_PREBUILT_VERIFIER)
    builder = _Builder(dict.fromkeys(_COMPOSE_IMAGES, "ready"))

    build = await _run(entity_store, spec, builder)

    assert build.status == "ready"
    assert _submitted_in_order(builder) == [
        "docker.io/library/redis:7",
        "ghcr.io/org/verify:1",
        "sort-task/environment",
        "sort-task/environment/db",
    ]


async def test_the_record_lists_images_in_the_order_the_task_plans_them(
    entity_store, store_task, compose_archive
) -> None:
    """Only the order of submitting changes: a reader finds ``main`` first and the verifier last, as before."""
    spec = await _start(entity_store, store_task, config=_PREBUILT_VERIFIER)

    build = await _run(entity_store, spec, _Builder(dict.fromkeys(_COMPOSE_IMAGES, "ready")))

    assert build.status == "ready"
    assert [(image.type, image.source) for image in build.images] == list(_COMPOSE_IMAGES.values())
    stored = await _stored_build(entity_store, spec)
    assert [(image.type, image.source) for image in stored.images] == list(_COMPOSE_IMAGES.values())


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (PermissionDeniedError(httpx.Response(403, json={"detail": "not allowed"})), "HTTP 403: not allowed"),
        (ConflictError(httpx.Response(409, json={"detail": "held by another"})), "HTTP 409: held by another"),
        (NemoTransportError(httpx.ConnectError("no route to Builder")), "no route to Builder"),
    ],
    ids=["forbidden", "a-409-with-no-code", "unreachable"],
)
async def test_an_error_that_is_no_refusal_fails_its_image_alone_and_the_rest_are_submitted(
    entity_store, store_task, archive, error: NemoClientError, reason: str
) -> None:
    """Only a 409 by which Builder refuses the request itself ends the submitting, and a copy's says so by its code.

    A copy's 409 that names no code stays the client error it was. Another request may have taken the revision
    meanwhile, or an older Builder may have refused the copy: no row is read to tell them apart, as one is for a
    Dockerfile build, so the task's Dockerfile images are still submitted.
    """
    config = {"environment": {"docker_image": "ghcr.io/org/sort:1"}, "verifier": {"environment": {"cpus": 1}}}
    spec = await _start(entity_store, store_task, config=config)
    builder = _Builder({}, refusals={"ghcr.io/org/sort:1": error})

    build = await _run(entity_store, spec, builder)

    main, verifier = build.images
    assert (main.status, main.builder_image) == ("failed", None)
    assert main.error == f"Builder refused the build: {reason}"
    assert _submitted_in_order(builder) == ["sort-task/tests"]
    assert (verifier.status, verifier.builder_image is not None) == ("building", True)
    assert (build.status, build.error) == ("failed", f"main: Builder refused the build: {reason}")


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


async def test_a_rebuild_takes_a_ready_build_to_builder_again(entity_store, store_task, archive) -> None:
    """A ready build may be for the default platform before it changed; only Builder knows the default now."""
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"})
    await _run(entity_store, spec, builder)
    service = ImageBuildService(entity_store, _Jobs(), _allow)
    await service.request_build("default", "sort-task", spec.revision_digest, ImageBuildRequest(rebuild=True))

    rebuilt = await _run(entity_store, spec.model_copy(update={"attempt": 2}), builder)

    assert rebuilt.status == "ready" and len(builder.submits) == 4


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
