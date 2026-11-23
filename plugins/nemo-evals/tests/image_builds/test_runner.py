# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task-builder Job's work: plan, submit and follow every image of one build."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Callable, Sequence
from pathlib import Path
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
from nemo_builder_plugin.schema import (
    CODE_BACKEND_CHANGED,
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
    backends_disagree,
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


# A Builder backend other than this cluster's. Evals records whichever name a row carries.
REMOTE_BACKEND = "remote"

# What Builder says when it refuses a request itself, as a backend that can't build this one request does.
REFUSAL = "the backend can't build this request"


def _handle(name: str) -> ExternalImage:
    """What a remote build service calls the image of Builder row ``name``: a handle and a build of its own."""
    return ExternalImage(image_id=f"image-{name}", build_id=f"build-{name}")


def _no_handle(name: str) -> ExternalImage:
    """What a remote build service that has no handle of its own reports for row ``name``."""
    return ExternalImage(build_id=f"build-{name}")


def _forbidden() -> PermissionDeniedError:
    """Builder's 403 for a caller who may not build. It turns the caller away, not the request: see ``_refusal``."""
    return PermissionDeniedError(httpx.Response(403, json={"detail": "not allowed to build"}))


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


def _refusal(code: str | None = CODE_BACKEND_REJECTED) -> ConflictError:
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
    image of each row, by the row's name. A test switches the deployment's backend by setting ``backend``: a row
    keeps the backend it was submitted under, and a submit of a revision another backend holds is refused as
    Builder refuses it. ``refusals`` is what Builder answers the submit of each image it names, by context path or
    prebuilt image, in place of taking it: a refusal, or another error.
    """

    def __init__(
        self,
        outcomes: dict[str, Outcome],
        *,
        refusals: dict[str, NemoClientError] | None = None,
        job: str = "active",
        published: dict[str, list[str]] | None = None,
        backend: str = IN_CLUSTER_BACKEND,
        external: Callable[[str], ExternalImage] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.refusals = refusals or {}
        self.job = job
        self.backend = backend
        self.external = external
        #: The platforms each prebuilt image is published for, by fully qualified reference.
        self.published = published or {}
        self.images: dict[str, str] = {}
        #: The backend each row was submitted under, by row name.
        self.built_by: dict[str, str] = {}
        self.submits: list[FileSetSource | ImageSource] = []
        #: The platform each submit named, by Builder name.
        self.platforms: dict[str, str | None] = {}
        #: The role the newest submit under each Builder name named: None when it named none.
        self.roles: dict[str, str | None] = {}

    async def get_image(self, name: str) -> ContainerImage | None:
        context = self.images.get(name)
        if context is None:
            return None
        outcome = self.outcomes[context]
        built_by = self.built_by[name]
        return _container_image(
            name,
            outcome,
            detail="kaniko exited 1" if outcome == "failed" else None,
            backend=built_by,
            external=None if built_by == IN_CLUSTER_BACKEND or self.external is None else self.external(name),
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
        role: str | None = None,
    ) -> SubmitBuildResponse:
        self.platforms[name] = platform
        self.roles[name] = role
        what = source.image if isinstance(source, ImageSource) else source.context_path
        assert what is not None
        if what in self.refusals:
            raise self.refusals[what]
        row = f"{name}-{revision}"
        existing = await self.get_image(row)
        if existing is not None and existing.provenance.backend != self.backend:
            body = {"detail": f"{row} was already submitted under another backend", "code": CODE_BACKEND_CHANGED}
            raise ConflictError(httpx.Response(409, json=body))
        self.submits.append(source)
        self.images[row] = what
        self.built_by[row] = self.backend
        if existing is not None:
            # A revision submitted again is answered with its row as it stands.
            return SubmitBuildResponse(job=f"builder-{row}", images=[existing])
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
    entity_store,
    store_task,
    *,
    config: dict,
    allowed: list[str] | None = None,
    platform: str | None = None,
    backend: str | None = IN_CLUSTER_BACKEND,
) -> TaskImageBuildSpec:
    """Store the task and request its first build, on a deployment whose task images must come from ``backend``.

    None is a deployment whose backend the API can't tell.
    """
    digest = await store_task(config=config)
    service = ImageBuildService(entity_store, _Jobs(), _allow, builder_backend=backend)
    request = ImageBuildRequest(allowed_registries=allowed or [], platform=platform)
    await service.request_build("default", "sort-task", digest, request)
    return TaskImageBuildSpec(workspace="default", task="sort-task", revision_digest=digest, attempt=1)


async def _request_again(
    entity_store, spec: TaskImageBuildSpec, *, backend: str, rebuild: bool = False
) -> TaskImageBuildSpec:
    """Request the task's build once more, on a deployment whose task images must come from ``backend``.

    Returns the attempt the request answered with: the one given when its build is the answer, the next otherwise.
    A ``rebuild`` is never answered with the build that is there.
    """
    service = ImageBuildService(entity_store, _Jobs(), _allow, builder_backend=backend)
    request = ImageBuildRequest(rebuild=rebuild)
    build = await service.request_build("default", spec.task, spec.revision_digest, request)
    assert build is not None
    return spec.model_copy(update={"attempt": int(build.id)})


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
    A build a broken runner left unsettled, as by stopping its submits without failing an image, then ends as
    timed out instead of being waited for on the real clock.
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
    """A deployment that builds through a remote build service, from the request to the stored record."""
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, backend=REMOTE_BACKEND)
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
    """When another image is refused at submit, for lack of permission, the build settles there.

    The submit is then the only record of the rest.
    """
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, backend=REMOTE_BACKEND)
    builder = _Builder({}, refusals={"sort-task/tests": _forbidden()}, backend=REMOTE_BACKEND)

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
    spec = await _start(entity_store, store_task, config={}, backend=REMOTE_BACKEND)
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


async def test_a_remote_row_an_older_builder_finds_ready_keeps_its_backend(entity_store, store_task, archive) -> None:
    """A Builder from before the backend existed can be the one whose read finds a remote row ready.

    The backend the submit named stays.
    """
    spec = await _start(entity_store, store_task, config={}, backend=REMOTE_BACKEND)

    class _ReadByAnOlderBuilder(_Builder):
        async def get_image(self, name: str) -> ContainerImage | None:
            return None if name not in self.images else _container_image(name, "ready")

    build = await _run(entity_store, spec, _ReadByAnOlderBuilder({}, backend=REMOTE_BACKEND))

    assert build.status == "ready"
    [image] = (await _stored_build(entity_store, spec)).images
    assert (image.status, image.builder_backend) == ("ready", REMOTE_BACKEND)


@pytest.mark.parametrize(
    ("config", "refusal", "error"),
    [
        # This cluster turns away a user who may not build. A 403 is recorded as the client reports it.
        ({}, _forbidden(), "Builder refused the build: HTTP 403: not allowed to build"),
        # A remote build service can't copy a prebuilt image, so Builder refuses it by the backend. A 409 that
        # refuses the request is recorded as Builder's own sentence.
        ({"environment": {"docker_image": "ghcr.io/org/sort:1"}}, _refusal(), f"Builder refused the build: {REFUSAL}"),
    ],
    ids=["in-cluster", "remote"],
)
async def test_an_image_builder_refused_names_no_backend_whichever_backend_refused_it(
    entity_store, store_task, archive, build_image_keys, config: dict, refusal: NemoClientError, error: str
) -> None:
    """The backend is read from the Builder row an image names, and an image whose submit Builder refused names none.

    Builder may still have written a row. So a missing backend doesn't by itself say this cluster built the image:
    such an image has no ``builder_image``.
    """
    spec = await _start(entity_store, store_task, config=config)
    # The double's own backend is left out: a refused submit returns no row to read one from.
    builder = _Builder({}, refusals={"sort-task/environment": refusal, "ghcr.io/org/sort:1": refusal})

    build = await _run(entity_store, spec, builder)

    [image] = build.images
    assert (build.status, image.status, image.builder_image) == ("failed", "failed", None)
    assert image.error == error
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

    build = await _run(entity_store, spec, _Abandoned({}, job="error"))

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

    build = await _run(entity_store, spec, _Builder({}, refusals={"sort-task/environment": _forbidden()}))

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


def _serve_compose_task(monkeypatch, tmp_path: Path, files: dict[str, str]) -> None:
    """Serve a whole task as the task's download: one that holds ``files``, its compose file among them.

    A compose task is extracted to be planned, so its archive is a whole task, packed as publishing packs one.
    """
    root = tmp_path / "sort-task"
    task = {"task.toml": "", "instruction.md": "Sort the list.\n", "tests/test.sh": "exit 0\n"} | files
    for name, text in task.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    packed = tmp_path / "task_archive"
    pack_task(root, packed)

    async def download(files_client, ref, output, **kwargs) -> None:
        output.write(packed.read_bytes())

    monkeypatch.setattr("nemo_evals.image_builds.runner.download_verified_async", download)


@pytest.fixture
def compose_archive(monkeypatch, tmp_path) -> None:
    """A task whose compose file adds a service it builds and a service it names an image for."""
    _serve_compose_task(
        monkeypatch,
        tmp_path,
        {
            "environment/Dockerfile": "FROM python:3.12\n",
            "environment/docker-compose.yaml": "services:\n  db:\n    build: ./db\n  cache:\n    image: redis:7\n",
            "environment/db/Dockerfile": "FROM postgres:16\n",
        },
    )


@pytest.fixture
def built_compose_archive(monkeypatch, tmp_path) -> None:
    """A compose task whose images are all built from Dockerfiles: ``main``, a service, and a separate verifier's."""
    _serve_compose_task(
        monkeypatch,
        tmp_path,
        {
            "environment/Dockerfile": "FROM python:3.12\n",
            "environment/docker-compose.yaml": "services:\n  db:\n    build: ./db\n",
            "environment/db/Dockerfile": "FROM postgres:16\n",
            "tests/Dockerfile": "FROM python:3.12-slim\n",
        },
    )


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


#: The build context of each image of ``built_compose_archive``'s task with a separate verifier, in plan order.
_BUILT_COMPOSE_CONTEXTS = ("sort-task/environment", "sort-task/environment/db", "sort-task/tests")


def _build_set(image: BuildImage) -> str:
    """The Builder name ``image`` was submitted under: its row's name, without the revision."""
    assert image.builder_image is not None
    return image.builder_image.rsplit("-", 1)[0]


def _name_without_a_role(image: BuildImage) -> str:
    """The Builder name of a Dockerfile image that names no role: the one it had before a build could name one."""
    assert image.context_digest is not None and image.dockerfile is not None
    return builder_name(context_digest=image.context_digest, dockerfile=image.dockerfile, platform=image.platform)


def _name_as_a_helper(image: BuildImage) -> str:
    """The Builder name of a Dockerfile image submitted as a helper."""
    assert image.context_digest is not None and image.dockerfile is not None
    return builder_name(
        context_digest=image.context_digest, dockerfile=image.dockerfile, platform=image.platform, role="helper"
    )


async def test_a_build_for_a_remote_backend_submits_a_compose_services_image_as_a_helper_and_no_other(
    entity_store, store_task, built_compose_archive
) -> None:
    """A remote build service may test-run an image it builds as a main container, which a sidecar need not pass.

    ``main`` and a separate verifier's image each run as the main container of a sandbox of their own. They name
    no role, under the names they have without one. The service's image is a helper, under a name that says so.
    """
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, backend=REMOTE_BACKEND)
    builder = _Builder(dict.fromkeys(_BUILT_COMPOSE_CONTEXTS, "ready"), backend=REMOTE_BACKEND)

    build = await _run(entity_store, spec, builder)

    assert build.status == "ready"
    main, db, verifier = build.images
    assert [image.type for image in build.images] == ["main", "service/db", "verifier"]
    assert builder.roles == {
        _name_without_a_role(main): None,
        _name_as_a_helper(db): "helper",
        _name_without_a_role(verifier): None,
    }
    # Each image's row is of the name its submit went under.
    assert [_build_set(image) for image in build.images] == list(builder.roles)
    assert _name_as_a_helper(db) != _name_without_a_role(db)


async def test_a_rebuild_for_a_remote_backend_submits_a_compose_services_image_as_a_helper_again(
    entity_store, store_task, built_compose_archive
) -> None:
    """A rebuild submits each image again as the request it was: the helper as a helper, under its own name."""
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, backend=REMOTE_BACKEND)
    builder = _Builder(dict.fromkeys(_BUILT_COMPOSE_CONTEXTS, "ready"), backend=REMOTE_BACKEND)
    first = await _run(entity_store, spec, builder)
    again = await _request_again(entity_store, spec, backend=REMOTE_BACKEND, rebuild=True)
    builder.roles.clear()

    rebuilt = await _run(entity_store, again, builder)

    assert (again.attempt, rebuilt.status, len(builder.submits)) == (2, "ready", 6)
    assert [image.builder_image for image in rebuilt.images] == [image.builder_image for image in first.images]
    main, db, verifier = rebuilt.images
    assert builder.roles == {
        _name_without_a_role(main): None,
        _name_as_a_helper(db): "helper",
        _name_without_a_role(verifier): None,
    }


async def test_a_compose_service_that_names_a_published_image_is_copied_with_no_role_for_a_remote_backend(
    entity_store, store_task, compose_archive
) -> None:
    """Only a service built from a Dockerfile is a helper: a copy is asked for as it always was.

    A remote build service copies no image today, so what a copy of a sidecar should tell it isn't decided. The
    double takes the copy, to show what Builder is asked for.
    """
    spec = await _start(entity_store, store_task, config={}, backend=REMOTE_BACKEND)
    builder = _Builder(dict.fromkeys(_COMPOSE_IMAGES, "ready"), backend=REMOTE_BACKEND)

    build = await _run(entity_store, spec, builder)

    assert build.status == "ready"
    main, db, cache = build.images
    assert [(image.type, image.source) for image in build.images] == [
        ("main", "dockerfile"),
        ("service/db", "dockerfile"),
        ("service/cache", "prebuilt"),
    ]
    assert builder.roles == {
        copy_builder_name("docker.io/library/redis:7"): None,
        _name_without_a_role(main): None,
        _name_as_a_helper(db): "helper",
    }
    assert _build_set(cache) == copy_builder_name("docker.io/library/redis:7")


@pytest.mark.parametrize("backend", [IN_CLUSTER_BACKEND, None], ids=["in-cluster", "a-backend-the-api-cant-tell"])
async def test_a_build_that_expects_the_in_cluster_backend_submits_a_compose_task_with_no_role_under_the_names_it_had(
    entity_store, store_task, compose_archive, backend: str | None
) -> None:
    """A deployment that builds in-cluster sends Builder what it always sent, under the names its builds are kept by.

    Its build's record names no backend, as does one requested while the API could tell none: both expect the
    in-cluster backend, and no image of either names a role, a compose service's included.
    """
    spec = await _start(entity_store, store_task, config=_PREBUILT_VERIFIER, backend=backend)
    assert (await _stored_build(entity_store, spec)).expected_builder_backend is None
    builder = _Builder(dict.fromkeys(_COMPOSE_IMAGES, "ready"))

    build = await _run(entity_store, spec, builder)

    assert build.status == "ready"
    main, db, cache, verifier = build.images
    assert [image.type for image in build.images] == ["main", "service/db", "service/cache", "verifier"]
    assert builder.roles == {
        copy_builder_name("docker.io/library/redis:7"): None,
        copy_builder_name("ghcr.io/org/verify:1"): None,
        _name_without_a_role(main): None,
        _name_without_a_role(db): None,
    }
    # Each image's row is of the name its submit went under: the copies were asked for first.
    assert [_build_set(image) for image in (cache, verifier, main, db)] == list(builder.roles)


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (_forbidden(), "HTTP 403: not allowed to build"),
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
    Dockerfile build that is not a rebuild, so the task's Dockerfile images are still submitted.
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

    build = await _run(entity_store, spec, _Stuck({}), timeout=250)

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


async def _build_named(
    entity_store, store_task, name: str, builder: _Builder, *, backend: str = IN_CLUSTER_BACKEND
) -> TaskImageBuildEntity:
    """Store task ``name`` and run its first image build over the archive being served.

    The deployment's task images must come from ``backend``.
    """
    digest = await store_task(name)
    service = ImageBuildService(entity_store, _Jobs(), _allow, builder_backend=backend)
    await service.request_build("default", name, digest, ImageBuildRequest())
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


@pytest.mark.parametrize(
    ("backend", "builds"), [(IN_CLUSTER_BACKEND, 1), (REMOTE_BACKEND, 2)], ids=["in-cluster", "remote"]
)
async def test_main_and_a_compose_service_with_identical_contexts_share_one_build_only_in_cluster(
    entity_store, store_task, monkeypatch, tmp_path, backend: str, builds: int
) -> None:
    """This cluster builds an image the same whatever it runs as, so the two share one build, as they always did.

    A remote build service may test-run the one as a main container and not the other. They are two requests,
    and Builder keeps one request under a name: so the helper is built under a name of its own.
    """
    _serve_compose_task(
        monkeypatch,
        tmp_path,
        {
            "environment/docker-compose.yaml": "services:\n  main:\n    build: ./app\n  twin:\n    build: ./twin\n",
            "environment/app/Dockerfile": "FROM python:3.12\n",
            "environment/twin/Dockerfile": "FROM python:3.12\n",
        },
    )
    spec = await _start(entity_store, store_task, config={}, backend=backend)
    builder = _Builder({"sort-task/environment/app": "ready", "sort-task/environment/twin": "ready"}, backend=backend)

    build = await _run(entity_store, spec, builder)

    main, twin = build.images
    assert (build.status, main.type, twin.type) == ("ready", "main", "service/twin")
    assert main.context_digest == twin.context_digest
    assert len(builder.submits) == builds
    assert (main.builder_image == twin.builder_image) == (builds == 1)
    assert (main.image_ref == twin.image_ref) == (builds == 1)


# Every key a build was recorded with before it could name the backend its images must come from, in order.
BUILD_RECORD_KEYS = [
    "task",
    "revision_digest",
    "attempt",
    "status",
    "job",
    "allowed_registries",
    "platform",
    "rebuild",
    "images",
    "error",
]


async def test_a_build_of_a_deployment_that_names_no_backend_is_recorded_key_for_key_as_before(
    entity_store, store_task, archive
) -> None:
    """Its images must come from the in-cluster backend, which the record leaves out: when requested and when done."""
    spec = await _start(entity_store, store_task, config={})
    requested = await _stored_build(entity_store, spec)
    assert (requested.expected_builder_backend, requested.expected_backend) == (None, IN_CLUSTER_BACKEND)
    assert list(requested._get_data_fields()) == BUILD_RECORD_KEYS

    build = await _run(entity_store, spec, _Builder({"sort-task/environment": "ready"}))

    assert build.status == "ready"
    assert list((await _stored_build(entity_store, spec))._get_data_fields()) == BUILD_RECORD_KEYS
    assert "expected_builder_backend" not in json.loads(build.model_dump_json())


async def test_a_build_requested_for_a_remote_backend_names_it_on_its_record(entity_store, store_task) -> None:
    """The Job that runs the build reads the backend there: a Job worker doesn't read the platform config file."""
    spec = await _start(entity_store, store_task, config={}, backend=REMOTE_BACKEND)

    stored = await _stored_build(entity_store, spec)

    assert (stored.expected_builder_backend, stored.expected_backend) == (REMOTE_BACKEND, REMOTE_BACKEND)
    sent = stored._get_data_fields()
    assert set(sent) == {*BUILD_RECORD_KEYS, "expected_builder_backend"}
    assert sent["expected_builder_backend"] == REMOTE_BACKEND


def test_a_build_recorded_before_it_could_name_a_backend_still_loads_and_is_saved_unchanged() -> None:
    recorded = {
        "task": "default/sort-task",
        "revision_digest": "d" * 64,
        "attempt": 1,
        "status": "ready",
        "job": "task-image-build-0123456789abcdef-1",
        "allowed_registries": ["docker.io"],
        "platform": None,
        "rebuild": False,
        "images": [],
        "error": None,
    }
    assert list(recorded) == BUILD_RECORD_KEYS

    build = TaskImageBuildEntity.model_validate({"name": "build.abc.1", "workspace": "default", **recorded})

    # Every build was the in-cluster backend's then, so that is what such a record's Job expects.
    assert (build.expected_builder_backend, build.expected_backend) == (None, IN_CLUSTER_BACKEND)
    assert build._get_data_fields() == recorded


@pytest.mark.parametrize("rebuild", [False, True], ids=["plain-request", "rebuild"])
@pytest.mark.parametrize(
    ("before", "after"),
    [(IN_CLUSTER_BACKEND, REMOTE_BACKEND), (REMOTE_BACKEND, IN_CLUSTER_BACKEND)],
    ids=["to-a-remote-build-service", "back-to-the-cluster"],
)
async def test_after_a_deployment_switches_backend_the_next_request_builds_the_task_again(
    entity_store, store_task, archive, before: str, after: str, rebuild: bool
) -> None:
    """Builder builds with the new backend and evals expects it, and the task's ready build is the old backend's.

    That build is no answer to the request, so the next attempt starts. Its Job reads the new backend from the
    attempt's record: it uses neither image's ready build at Builder, which the old backend published, and submits
    each as the next revision. The build that makes is then the answer.

    A rebuild gets there by another way: its Job submits each image's newest revision again, Builder refuses a
    revision the old backend holds, and the Job submits the next one.
    """
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, backend=before)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"}, backend=before, external=_handle)
    first = await _run(entity_store, spec, builder)
    assert (first.status, [image.built_by for image in first.images]) == ("ready", [before, before])

    builder.backend = after
    again = await _request_again(entity_store, spec, backend=after, rebuild=rebuild)
    assert again.attempt == 2
    assert (await _stored_build(entity_store, again)).expected_backend == after
    rebuilt = await _run(entity_store, again, builder)

    assert rebuilt.status == "ready"
    assert len(builder.submits) == 4
    assert [image.built_by for image in rebuilt.images] == [after, after]
    assert [image.builder_image for image in rebuilt.images] == [
        (image.builder_image or "").removesuffix("-1") + "-2" for image in first.images
    ]
    # Each row has a handle of its own, and only a remote build service gives one.
    assert [image.external_image_id for image in rebuilt.images] == [
        _handle(image.builder_image or "").image_id if after == REMOTE_BACKEND else None for image in rebuilt.images
    ]
    assert await _request_again(entity_store, again, backend=after) == again
    assert len(builder.submits) == 4


@pytest.mark.parametrize(
    ("backend", "other"),
    [(IN_CLUSTER_BACKEND, REMOTE_BACKEND), (REMOTE_BACKEND, IN_CLUSTER_BACKEND)],
    ids=["back-to-the-cluster", "back-to-a-remote-service"],
)
async def test_after_a_switch_back_a_compose_services_image_uses_the_build_its_backend_made_before(
    entity_store, store_task, built_compose_archive, backend: str, other: str
) -> None:
    """A remote backend's build of a helper is under a name of its own, so it is never in front of this cluster's.

    Nor is this cluster's in front of it. After a switch to the other backend and back, ``main`` and the verifier's
    image are built again, as the operator guide and Builder's README say of an image the other backend built in
    between. The service's image is not: the newest build of the name it has for the first backend is still that
    backend's, whichever of the two it is.
    """
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, backend=backend)
    builder = _Builder(dict.fromkeys(_BUILT_COMPOSE_CONTEXTS, "ready"), backend=backend)
    first = await _run(entity_store, spec, builder)
    builder.backend = other
    switched = await _request_again(entity_store, spec, backend=other)
    between = await _run(entity_store, switched, builder)
    assert (between.status, len(builder.submits)) == ("ready", 6)
    assert [image.built_by for image in between.images] == [other] * 3

    builder.backend = backend
    back = await _request_again(entity_store, switched, backend=backend)
    again = await _run(entity_store, back, builder)

    assert (back.attempt, again.status) == (3, "ready")
    assert [image.built_by for image in again.images] == [backend] * 3
    main, db, verifier = again.images
    assert db.builder_image == first.images[1].builder_image
    assert [main.builder_image, verifier.builder_image] == [
        f"{_name_without_a_role(image)}-3" for image in (main, verifier)
    ]
    assert _submitted_in_order(builder)[6:] == ["sort-task/environment", "sort-task/tests"]


async def test_a_deployment_that_did_not_switch_uses_the_build_another_task_made(
    entity_store, store_task, monkeypatch
) -> None:
    """The backend check changes nothing where there is one backend: a second task's Job submits nothing."""
    builder = _Builder({"sort-task/environment": "ready"}, backend=REMOTE_BACKEND, external=_handle)
    _serve_archive(monkeypatch, _two_tasks_archive(b"Sort the list.\n"))
    first = await _build_named(entity_store, store_task, "sort-a", builder, backend=REMOTE_BACKEND)

    _serve_archive(monkeypatch, _two_tasks_archive(b"Sort the list in reverse.\n"))
    second = await _build_named(entity_store, store_task, "sort-b", builder, backend=REMOTE_BACKEND)

    assert len(builder.submits) == 1
    assert first.status == second.status == "ready"
    assert first.images[0].builder_image == second.images[0].builder_image
    handle = _handle(second.images[0].builder_image or "").image_id
    assert (second.images[0].built_by, second.images[0].external_image_id) == (REMOTE_BACKEND, handle)


def test_the_error_of_a_build_whose_backends_differ_names_both_and_says_what_to_do() -> None:
    """The one place the sentence is spelt out: the tests below, and the operator guide's, compare with the function."""
    assert backends_disagree("remote", "execution") == (
        "Builder builds with the 'remote' backend, and this build was requested for 'execution': request the build "
        "again; if it fails the same way, the evals API reads another builder.backend than Builder does, so set both "
        "to one backend"
    )


@pytest.mark.parametrize(
    ("expected", "builders"),
    [(IN_CLUSTER_BACKEND, REMOTE_BACKEND), (REMOTE_BACKEND, IN_CLUSTER_BACKEND)],
    ids=["evals-not-told-of-the-remote-service", "builder-not-switched"],
)
async def test_a_builder_that_builds_with_another_backend_than_expected_fails_the_image_naming_both(
    entity_store, store_task, archive, build_image_keys, expected: str, builders: str
) -> None:
    """Builder answers a submit with a row under its own backend, so one under another says the two differ.

    Taken, the image would make a ready build that no request accepts, and every request would build the task
    again without saying why. The request gets one failed build instead, and nothing more of the task is submitted.
    """
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, backend=expected)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"}, backend=builders)

    build = await _run(entity_store, spec, builder)

    sentence = backends_disagree(builders, expected)
    assert (build.status, build.error) == ("failed", f"main: {sentence}")
    main, verifier = build.images
    assert (main.status, main.error, main.image_ref) == ("failed", sentence, None)
    # The image names the row Builder wrote, and the backend it is under.
    assert main.builder_image is not None and main.build_job is not None
    assert main.built_by == builders
    assert _submitted_in_order(builder) == ["sort-task/environment"]
    assert (verifier.status, verifier.builder_image) == ("pending", None)
    stored = await _stored_build(entity_store, spec)
    assert stored == build
    # An in-cluster row's backend is left out, as on any image: an older reader refuses a key it doesn't know.
    assert main.builder_backend == (None if builders == IN_CLUSTER_BACKEND else builders)
    sent = stored._get_data_fields()["images"][0]
    if builders == IN_CLUSTER_BACKEND:
        assert list(sent) == build_image_keys
    else:
        assert set(sent) == {*build_image_keys, "builder_backend"}


@pytest.mark.parametrize(
    ("expected", "builders"),
    [(IN_CLUSTER_BACKEND, REMOTE_BACKEND), (REMOTE_BACKEND, IN_CLUSTER_BACKEND)],
    ids=["evals-not-told-of-the-remote-service", "builder-not-switched"],
)
async def test_each_request_while_the_backends_differ_fails_the_same_way_and_submits_the_image_once_more(
    entity_store, store_task, archive, expected: str, builders: str
) -> None:
    """The row Builder took stays at Builder, under its backend, so the next attempt uses it no more than the first.

    The attempt submits the image as the next revision, which Builder takes too, and fails on the answer as the
    first did. So each request costs one failed attempt and one more Builder build, until the two backends agree.
    """
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, backend=expected)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"}, backend=builders)
    first = await _run(entity_store, spec, builder)
    assert _submitted_in_order(builder) == ["sort-task/environment"]

    again = await _request_again(entity_store, spec, backend=expected)
    assert again.attempt == 2
    second = await _run(entity_store, again, builder)

    sentence = backends_disagree(builders, expected)
    assert (second.status, second.error) == ("failed", f"main: {sentence}")
    main, verifier = second.images
    assert (main.status, main.error, main.built_by) == ("failed", sentence, builders)
    assert _submitted_in_order(builder) == ["sort-task/environment", "sort-task/environment"]
    assert main.builder_image == (first.images[0].builder_image or "").removesuffix("-1") + "-2"
    assert (verifier.status, verifier.builder_image) == ("pending", None)


@pytest.mark.parametrize(
    ("expected", "builders"),
    [(IN_CLUSTER_BACKEND, REMOTE_BACKEND), (REMOTE_BACKEND, IN_CLUSTER_BACKEND)],
    ids=["evals-not-told-of-the-remote-service", "builder-not-switched"],
)
async def test_once_evals_expects_builders_backend_the_next_request_uses_the_build_builder_took(
    entity_store, store_task, archive, expected: str, builders: str
) -> None:
    """What the error asks for: the build is requested again, by an evals API that now reads Builder's backend.

    The build Builder took for the failed attempt is then the newest one of its name, under the backend the new
    attempt expects: the attempt uses it, and submits only the image the failed attempt never reached.
    """
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, backend=expected)
    builder = _Builder({"sort-task/environment": "ready", "sort-task/tests": "ready"}, backend=builders)
    failed = await _run(entity_store, spec, builder)
    assert failed.status == "failed"

    again = await _request_again(entity_store, spec, backend=builders)
    assert again.attempt == 2
    built = await _run(entity_store, again, builder)

    assert (built.status, built.error) == ("ready", None)
    assert [image.built_by for image in built.images] == [builders, builders]
    assert built.images[0].builder_image == failed.images[0].builder_image
    assert _submitted_in_order(builder) == ["sort-task/environment", "sort-task/tests"]
    assert await _request_again(entity_store, again, backend=builders) == again


@pytest.mark.parametrize(
    ("expected", "builders"),
    [(IN_CLUSTER_BACKEND, REMOTE_BACKEND), (REMOTE_BACKEND, IN_CLUSTER_BACKEND)],
    ids=["evals-not-told-of-the-remote-service", "builder-not-switched"],
)
async def test_once_evals_expects_builders_backend_the_build_builder_took_of_a_compose_services_image_stays_unused(
    entity_store, store_task, built_compose_archive, expected: str, builders: str
) -> None:
    """The exception to the test above, as the operator guide and Builder's README say.

    A service's image is named with a role only in a build requested for a remote backend. So the build Builder
    took of it while the two backends differed is under the name the failed attempt asked for, and an attempt that
    expects Builder's backend asks for the other: it submits the image, and never reads that build.

    A service's image is submitted while the backends differ only when ``main`` is not, as here: the expected
    backend built ``main`` before Builder was switched, and the service's image was not built then.
    """
    spec = await _start(entity_store, store_task, config=_SEPARATE_VERIFIER, backend=expected)
    builder = _Builder(
        dict.fromkeys(_BUILT_COMPOSE_CONTEXTS, "ready"),
        refusals={"sort-task/environment/db": _forbidden()},
        backend=expected,
    )
    await _run(entity_store, spec, builder)
    builder.refusals.clear()
    builder.backend = builders
    again = await _request_again(entity_store, spec, backend=expected)
    failed = await _run(entity_store, again, builder)
    assert (failed.status, failed.error) == ("failed", f"service/db: {backends_disagree(builders, expected)}")
    assert _submitted_in_order(builder)[2:] == ["sort-task/environment/db"]

    agreed = await _request_again(entity_store, again, backend=builders)
    built = await _run(entity_store, agreed, builder)

    assert (agreed.attempt, built.status, built.error) == (3, "ready", None)
    assert [image.built_by for image in built.images] == [builders] * 3
    taken, db = failed.images[1], built.images[1]
    names = {REMOTE_BACKEND: _name_as_a_helper(db), IN_CLUSTER_BACKEND: _name_without_a_role(db)}
    assert (taken.builder_image, taken.built_by) == (f"{names[expected]}-1", builders)
    assert db.builder_image == f"{names[builders]}-1"
    # Every image is submitted, the service's too: `main` and the verifier's because the other backend built them.
    assert _submitted_in_order(builder)[3:] == list(_BUILT_COMPOSE_CONTEXTS)


async def test_a_copy_builder_took_under_another_backend_fails_its_image_the_same_way(
    entity_store, store_task, archive, build_image_keys
) -> None:
    """A copy is asked of Builder each time, so the row it answers with is under Builder's own backend.

    The copy goes first, so the task's Dockerfile image is not submitted to a backend the deployment doesn't expect.
    """
    config = {"environment": {"docker_image": "ghcr.io/org/sort:1"}, "verifier": {"environment": {"cpus": 1}}}
    spec = await _start(entity_store, store_task, config=config, backend=REMOTE_BACKEND)
    builder = _Builder({"ghcr.io/org/sort:1": "ready", "sort-task/tests": "ready"})

    build = await _run(entity_store, spec, builder)

    sentence = backends_disagree(IN_CLUSTER_BACKEND, REMOTE_BACKEND)
    assert (build.status, build.error) == ("failed", f"main: {sentence}")
    main, verifier = build.images
    assert (main.source, main.status, main.error) == ("prebuilt", "failed", sentence)
    assert (main.built_by, main.builder_backend) == (IN_CLUSTER_BACKEND, None)
    assert list((await _stored_build(entity_store, spec))._get_data_fields()["images"][0]) == build_image_keys
    assert builder.submits == [ImageSource(image="ghcr.io/org/sort:1")]
    assert (verifier.status, verifier.builder_image) == ("pending", None)


async def test_a_rebuild_builder_answers_under_another_backend_fails_its_image_without_a_reference(
    entity_store, store_task, archive
) -> None:
    """A rebuild rejoins the newest revision, which may be ready: its reference is not this build's to record.

    The newest revision is one Builder's own backend made, so Builder answers with it and starts no build.
    """
    spec = await _start(entity_store, store_task, config={}, backend=REMOTE_BACKEND)
    builder = _Builder({"sort-task/environment": "ready"}, backend=REMOTE_BACKEND, external=_handle)
    first = await _run(entity_store, spec, builder)
    again = await _request_again(entity_store, spec, backend=IN_CLUSTER_BACKEND, rebuild=True)

    rebuilt = await _run(entity_store, again, builder)

    [image] = rebuilt.images
    assert (again.attempt, rebuilt.status, image.status) == (2, "failed", "failed")
    assert image.error == backends_disagree(REMOTE_BACKEND, IN_CLUSTER_BACKEND)
    assert (image.image_ref, image.runtime_config, image.external_image_id) == (None, None, None)
    assert (image.builder_image, image.built_by) == (first.images[0].builder_image, REMOTE_BACKEND)
    assert list(builder.built_by) == [image.builder_image]


@pytest.mark.parametrize(
    ("expected", "builders"),
    [(IN_CLUSTER_BACKEND, REMOTE_BACKEND), (REMOTE_BACKEND, IN_CLUSTER_BACKEND)],
    ids=["evals-not-told-of-the-remote-service", "builder-not-switched"],
)
async def test_a_rebuild_over_a_build_of_the_backend_evals_expects_starts_one_more_builder_build_and_fails(
    entity_store, store_task, archive, expected: str, builders: str
) -> None:
    """Builder now builds with another backend than evals expects, and the image's newest build is the expected one's.

    A rebuild submits that revision again. Builder refuses a revision its other backend holds, so the Job submits
    the next one, which Builder takes under its own backend: one more Builder build, and the image fails on the
    answer. A rebuild after it finds that build the newest, and rejoins it as in the test above.
    """
    spec = await _start(entity_store, store_task, config={}, backend=expected)
    builder = _Builder({"sort-task/environment": "ready"}, backend=expected, external=_handle)
    first = await _run(entity_store, spec, builder)
    assert (first.status, first.images[0].built_by) == ("ready", expected)

    builder.backend = builders
    again = await _request_again(entity_store, spec, backend=expected, rebuild=True)
    rebuilt = await _run(entity_store, again, builder)

    [image] = rebuilt.images
    assert (again.attempt, rebuilt.status, image.status) == (2, "failed", "failed")
    assert (image.error, image.built_by) == (backends_disagree(builders, expected), builders)
    assert (image.image_ref, image.runtime_config, image.external_image_id) == (None, None, None)
    assert image.builder_image == (first.images[0].builder_image or "").removesuffix("-1") + "-2"
    # Builder holds two builds of the image now, one under each backend. The refused resubmit is not one.
    assert builder.built_by == {first.images[0].builder_image: expected, image.builder_image: builders}
    assert len(builder.submits) == 2
