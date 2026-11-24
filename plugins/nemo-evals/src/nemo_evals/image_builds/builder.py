# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submitting and following NeMo Builder builds for task images."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Protocol

from filesets import parse_fileset_ref
from nemo_builder_plugin.client import AsyncBuilderClient
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.identity import ImageIdentityError, normalize_reference
from nemo_builder_plugin.schema import (
    CODE_BACKEND_REJECTED,
    CODE_MISSING_SECRETS,
    BuildMount,
    BuildSet,
    BuildSpec,
    FileSetSource,
    ImageSource,
    SubmitBuildResponse,
)
from nemo_builder_plugin.upstream import UpstreamError, UpstreamRegistry
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.jobs.client import AsyncJobsClient

logger = logging.getLogger(__name__)

_BUILDER_NAME_PREFIX = "harbor-env-"
_COPY_NAME_PREFIX = "harbor-img-"
#: Builder job statuses after which the job writes nothing more to its images.
ENDED_JOB_STATUSES = frozenset({"completed", "error", "cancelled"})
#: How long a pending image may have no job before it counts as abandoned. Builder writes an image's row first and
#: creates its job after, so a request that reads the row in between (any of many tasks sharing one build) sees no
#: job yet. Longer than any submit takes; short enough that a submit that died between the two is retried soon.
JOB_GRACE = timedelta(minutes=2)


class ImageBuilder(Protocol):
    """The Builder operations a task image build needs."""

    async def get_image(self, name: str) -> ContainerImage | None:
        """The Builder image record ``name``, or None when there is none."""
        ...

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
        """Submit (or rejoin) one build, or copy, of one image; for the deployment's platform when ``platform`` is None."""
        ...

    async def image_platforms(self, image: str) -> list[str]:
        """The platforms a published image has, or none when they can't be read."""
        ...

    async def job_status(self, job: str) -> str | None:
        """The status of Builder's job ``job``, or None when there is none."""
        ...


def builder_name(*, context_digest: str, dockerfile: str, platform: str | None = None) -> str:
    """Builder build-set name for an image built from a context holding ``context_digest``'s files, for one platform.

    Keyed by content, not by which task archive holds it: tasks whose build contexts hold the same files share
    one build (see :func:`adopt_or_build`). A build for the deployment's default platform leaves ``platform``
    out of the key.
    """
    key = {"v": 3, "context": context_digest, "dockerfile": dockerfile}
    return f"{_BUILDER_NAME_PREFIX}{_key_hash(key, platform)}"


def copy_builder_name(image: str, platform: str | None = None) -> str:
    """Builder build-set name for a copy of ``image``, by its fully qualified reference, for one platform."""
    return f"{_COPY_NAME_PREFIX}{_key_hash({'v': 1, 'image': image}, platform)}"


def _key_hash(key: dict[str, object], platform: str | None) -> str:
    if platform is not None:
        key = {**key, "platform": platform}
    return hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def archive_source(fileset_ref: str, context_path: str) -> FileSetSource:
    """Builder source for ``context_path`` inside the archive at ``workspace/fileset#path``."""
    workspace, fileset, path = parse_fileset_ref(fileset_ref, workspace_fallback=None)
    return FileSetSource(fileset=f"{workspace}/{fileset}", archive=path, context_path=context_path)


#: The codes Builder gives a 409 that refuses the request itself, where the next revision would fare no better.
#: A revision another request holds has no code, and one submitted under another backend has one of its own.
_REFUSALS = frozenset({CODE_BACKEND_REJECTED, CODE_MISSING_SECRETS})


def _refuses_the_request(conflict: ConflictError) -> bool:
    """Whether Builder's 409 says so itself, by its ``code``."""
    code = conflict.body.get("code") if isinstance(conflict.body, dict) else None
    return isinstance(code, str) and code in _REFUSALS


class BuildRefused(RuntimeError):
    """Builder refused the request for a build itself, and would refuse it under any revision.

    ``detail`` is Builder's own sentence for why.
    """

    def __init__(self, build: str, detail: str) -> None:
        super().__init__(f"Builder refused build {build}: {detail}")
        self.detail = detail


async def ensure_build(
    builder: ImageBuilder,
    *,
    name: str,
    source: FileSetSource | ImageSource,
    dockerfile: str | None,
    platform: str | None = None,
) -> SubmitBuildResponse:
    """Submit the newest revision of ``name`` again, or the next one when the newest failed.

    A pending revision whose job has ended, or has none, counts as failed: Builder leaves it pending for good.
    So does one Builder refuses as another request's: the same name can mean another platform once the
    deployment's default changes. And so does one submitted under another backend, after the deployment changed its
    backend. A 409 by which Builder refuses the request itself, whatever its revision, is raised instead, as a
    :class:`BuildRefused` with Builder's own detail.
    """
    newest: ContainerImage | None = None
    revision = 1
    while (image := await builder.get_image(f"{name}-{revision}")) is not None:
        newest = image
        revision += 1
    if newest is not None and not await _abandoned(builder, newest):
        try:
            return await _submit_unless_refused(
                builder, name=name, revision=revision - 1, source=source, dockerfile=dockerfile, platform=platform
            )
        except ConflictError:
            logger.info(
                "Builder holds %s-%d for another request, or under another backend; submitting the next revision",
                name,
                revision - 1,
            )
    return await _submit_unless_refused(
        builder, name=name, revision=revision, source=source, dockerfile=dockerfile, platform=platform
    )


async def _submit_unless_refused(
    builder: ImageBuilder,
    *,
    name: str,
    revision: int,
    source: FileSetSource | ImageSource,
    dockerfile: str | None,
    platform: str | None,
) -> SubmitBuildResponse:
    """Submit one revision, raising a 409 by which Builder refuses the request itself as a :class:`BuildRefused`.

    Builder says so by the 409's code. Any other 409 is raised as it is.
    """
    try:
        return await builder.submit(
            name=name, revision=revision, source=source, dockerfile=dockerfile, platform=platform
        )
    except ConflictError as exc:
        if _refuses_the_request(exc):
            raise BuildRefused(f"{name}-{revision}", exc.detail) from exc
        raise


#: Submits that may each lose a race for the next revision before :func:`adopt_or_build` gives up.
_ADOPT_ATTEMPTS = 3


async def adopt_or_build(
    builder: ImageBuilder,
    *,
    name: str,
    source: FileSetSource | ImageSource,
    dockerfile: str | None,
    mounts: Sequence[BuildMount] = (),
    platform: str | None = None,
    built_after: datetime | None = None,
    backend: str | None = None,
) -> ContainerImage:
    """The newest usable build of ``name``, submitting one only when there is none.

    ``name`` is keyed by content (:func:`builder_name`), so another task, run or user may already have built it:
    a ready build, or a pending one whose job still runs, is used as it is. A failed or abandoned one, or one made
    before ``built_after``, gets the next revision. Builder refuses a revision another request holds, so a submit
    that loses that race reads the winner's build instead. A 409 by which Builder refuses the request itself
    is raised as a :class:`BuildRefused`, with Builder's own detail, whether it says so by its code or leaves no row
    at the revision. Giving up on a build other requests keep submitting is not one.

    When the caller names a ``backend``, a build another backend made gets the next revision too: its image is
    where that backend published it. A build returned under another backend is then one Builder took just now, for
    this request or for the one that won the race for the revision: either way Builder builds with another backend
    than the caller expects, and would take the next revision the same way, so none is submitted. With no
    ``backend`` named, a build is used whichever backend made it.
    """
    for _ in range(_ADOPT_ATTEMPTS):
        newest, revision = await _newest(builder, name)
        if newest is not None and not await _unusable(builder, newest, built_after, backend):
            return newest
        try:
            submitted = await builder.submit(
                name=name, revision=revision, source=source, dockerfile=dockerfile, platform=platform, mounts=mounts
            )
        except ConflictError as exc:
            # Builder also answers 409 when it refuses the request before writing anything (no registry, unusable
            # execution profiles, missing secrets). What it says is read first, by its code: a row another request
            # wrote meanwhile must not hide a refusal of this one. A Builder that gives no code leaves no row
            # either: a lost race leaves the winner's revision behind; a refusal doesn't.
            winner = None if _refuses_the_request(exc) else await builder.get_image(f"{name}-{revision}")
            if winner is None:
                raise BuildRefused(f"{name}-{revision}", exc.detail) from exc
            if backend is not None and winner.provenance.backend != backend:
                # Read again, the winner's build would be passed over and the revision after it submitted, which
                # Builder would take under the same backend: one more build for each request that loses.
                logger.info("Another request submitted %s-%d first, under another backend", name, revision)
                return winner
            logger.info("Another request submitted %s-%d first; using its build", name, revision)
            continue
        return submitted.images[0]
    raise RuntimeError(f"Builder build {name} kept being submitted by other requests; try again")


async def usable_build(
    builder: ImageBuilder, name: str, *, built_after: datetime | None = None, backend: str | None = None
) -> ContainerImage | None:
    """The newest build of ``name`` that is ready, or still building; None when there is none to use.

    What :func:`adopt_or_build` would use without submitting, so a caller can skip preparing a source it won't need.
    ``backend`` is as that function takes it.
    """
    newest, _ = await _newest(builder, name)
    return newest if newest is not None and not await _unusable(builder, newest, built_after, backend) else None


async def _newest(builder: ImageBuilder, name: str) -> tuple[ContainerImage | None, int]:
    """The newest revision of ``name`` and the next revision number."""
    newest: ContainerImage | None = None
    revision = 1
    while (image := await builder.get_image(f"{name}-{revision}")) is not None:
        newest = image
        revision += 1
    return newest, revision


async def _unusable(
    builder: ImageBuilder, image: ContainerImage, built_after: datetime | None, backend: str | None
) -> bool:
    if backend is not None and image.provenance.backend != backend:
        return True
    if built_after is not None and (image.created_at is None or _utc(image.created_at) < built_after):
        return True
    return await _abandoned(builder, image)


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


async def _abandoned(builder: ImageBuilder, image: ContainerImage) -> bool:
    if image.status != "pending":
        return image.status == "failed"
    return await job_gone(builder, image)


async def job_gone(builder: ImageBuilder, image: ContainerImage, *, now: datetime | None = None) -> bool:
    """Whether a pending image's job has ended, or never appeared within :data:`JOB_GRACE` of its row."""
    status = await builder.job_status(image.provenance.job.rpartition("/")[2])
    if status is not None:
        return status in ENDED_JOB_STATUSES
    created = image.created_at
    if created is None:
        return True
    return (now or datetime.now(timezone.utc)) - _utc(created) > JOB_GRACE


class HttpImageBuilder:
    """:class:`ImageBuilder` over Builder's and Jobs' APIs, acting as ``client``'s caller."""

    def __init__(self, client: AsyncNemoClient, *, workspace: str) -> None:
        self._builder = AsyncBuilderClient.from_client(client)
        self._jobs = AsyncJobsClient.from_client(client)
        self._workspace = workspace

    async def get_image(self, name: str) -> ContainerImage | None:
        try:
            response = await self._builder.get_container_image(workspace=self._workspace, name=name)
        except NotFoundError:
            return None
        return response.data()

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
        # Only what is set: the client serializes with `exclude_unset`, and a copy takes no dockerfile or mounts.
        options: dict[str, object] = {}
        if dockerfile is not None:
            options["dockerfile"] = dockerfile
        if platform is not None:
            options["platform"] = platform
        if mounts:
            options["mounts"] = list(mounts)
        spec = BuildSpec.model_validate({"source": source, **options})
        build_set = BuildSet(name=name, revision=revision, build_specs=[spec])
        return (await self._builder.submit_build(workspace=self._workspace, body=build_set)).data()

    async def image_platforms(self, image: str) -> list[str]:
        registry = UpstreamRegistry()
        try:
            return await asyncio.to_thread(registry.platforms, normalize_reference(image))
        except (UpstreamError, ImageIdentityError) as exc:
            logger.warning("Can't read the platforms of %s: %s", image, exc)
            return []
        finally:
            registry.close()

    async def job_status(self, job: str) -> str | None:
        try:
            response = await self._jobs.get_job_status(workspace=self._workspace, name=job)
        except NotFoundError:
            return None
        return response.data().status.value
