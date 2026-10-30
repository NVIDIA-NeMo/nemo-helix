# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submitting and following NeMo Builder builds for task images."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Protocol

from filesets import parse_fileset_ref
from nemo_builder_plugin.client import AsyncBuilderClient
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.identity import ImageIdentityError, normalize_reference
from nemo_builder_plugin.schema import BuildSet, BuildSpec, FileSetSource, ImageSource, SubmitBuildResponse
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
    deployment's default changes.
    """
    newest: ContainerImage | None = None
    revision = 1
    while (image := await builder.get_image(f"{name}-{revision}")) is not None:
        newest = image
        revision += 1
    if newest is not None and not await _abandoned(builder, newest):
        try:
            return await builder.submit(
                name=name, revision=revision - 1, source=source, dockerfile=dockerfile, platform=platform
            )
        except ConflictError:
            logger.info("Builder holds %s-%d for another request; submitting the next revision", name, revision - 1)
    return await builder.submit(name=name, revision=revision, source=source, dockerfile=dockerfile, platform=platform)


#: Submits that may each lose a race for the next revision before :func:`adopt_or_build` gives up.
_ADOPT_ATTEMPTS = 3


async def adopt_or_build(
    builder: ImageBuilder,
    *,
    name: str,
    source: FileSetSource | ImageSource,
    dockerfile: str | None,
    platform: str | None = None,
) -> ContainerImage:
    """The newest usable build of ``name``, submitting one only when there is none.

    ``name`` is keyed by content (:func:`builder_name`), so another task, run or user may already have built it:
    a ready build, or a pending one whose job still runs, is used as it is. A failed or abandoned one gets the next
    revision. Builder refuses a revision another request holds, so a submit that loses that race reads the
    winner's build instead.
    """
    for _ in range(_ADOPT_ATTEMPTS):
        newest: ContainerImage | None = None
        revision = 1
        while (image := await builder.get_image(f"{name}-{revision}")) is not None:
            newest = image
            revision += 1
        if newest is not None and not await _abandoned(builder, newest):
            return newest
        try:
            submitted = await builder.submit(
                name=name, revision=revision, source=source, dockerfile=dockerfile, platform=platform
            )
        except ConflictError:
            logger.info("Another request submitted %s-%d first; using its build", name, revision)
            continue
        return submitted.images[0]
    raise RuntimeError(f"Builder build {name} kept being submitted by other requests; try again")


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
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (now or datetime.now(timezone.utc)) - created > JOB_GRACE


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
    ) -> SubmitBuildResponse:
        # Only what is set: the client serializes with `exclude_unset`, and a copy takes no dockerfile.
        options: dict[str, str] = {}
        if dockerfile is not None:
            options["dockerfile"] = dockerfile
        if platform is not None:
            options["platform"] = platform
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
