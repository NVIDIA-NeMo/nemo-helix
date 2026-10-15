# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submitting and following NeMo Builder builds for task images."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from filesets import parse_fileset_ref
from nemo_builder_plugin.client import AsyncBuilderClient
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.schema import BuildSet, BuildSpec, FileSetSource, ImageSource, SubmitBuildResponse
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.jobs.client import AsyncJobsClient

_BUILDER_NAME_PREFIX = "harbor-env-"
_COPY_NAME_PREFIX = "harbor-img-"
#: Builder job statuses after which the job writes nothing more to its images.
ENDED_JOB_STATUSES = frozenset({"completed", "error", "cancelled"})


class ImageBuilder(Protocol):
    """The Builder operations a task image build needs."""

    async def get_image(self, name: str) -> ContainerImage | None:
        """The Builder image record ``name``, or None when there is none."""
        ...

    async def submit(
        self, *, name: str, revision: int, source: FileSetSource | ImageSource, dockerfile: str | None
    ) -> SubmitBuildResponse:
        """Submit (or rejoin) one build, or copy, of one image."""
        ...

    async def job_status(self, job: str) -> str | None:
        """The status of Builder's job ``job``, or None when there is none."""
        ...


def builder_name(*, archive_digest: str, context_path: str, dockerfile: str) -> str:
    """Builder build-set name for one image of one task archive."""
    key = json.dumps(
        {"v": 1, "archive": archive_digest, "context": context_path, "dockerfile": dockerfile},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{_BUILDER_NAME_PREFIX}{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def copy_builder_name(image: str) -> str:
    """Builder build-set name for a copy of ``image``, by its fully qualified reference."""
    key = json.dumps({"v": 1, "image": image}, sort_keys=True, separators=(",", ":"))
    return f"{_COPY_NAME_PREFIX}{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def archive_source(fileset_ref: str, context_path: str) -> FileSetSource:
    """Builder source for ``context_path`` inside the archive at ``workspace/fileset#path``."""
    workspace, fileset, path = parse_fileset_ref(fileset_ref, workspace_fallback=None)
    return FileSetSource(fileset=f"{workspace}/{fileset}", archive=path, context_path=context_path)


async def ensure_build(
    builder: ImageBuilder, *, name: str, source: FileSetSource | ImageSource, dockerfile: str | None
) -> SubmitBuildResponse:
    """Submit the newest revision of ``name`` again, or the next one when the newest failed."""
    newest: ContainerImage | None = None
    revision = 1
    while (image := await builder.get_image(f"{name}-{revision}")) is not None:
        newest = image
        revision += 1
    if newest is not None and newest.status != "failed":
        revision -= 1
    return await builder.submit(name=name, revision=revision, source=source, dockerfile=dockerfile)


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
        self, *, name: str, revision: int, source: FileSetSource | ImageSource, dockerfile: str | None
    ) -> SubmitBuildResponse:
        spec = BuildSpec(source=source) if dockerfile is None else BuildSpec(source=source, dockerfile=dockerfile)
        build_set = BuildSet(name=name, revision=revision, build_specs=[spec])
        return (await self._builder.submit_build(workspace=self._workspace, body=build_set)).data()

    async def job_status(self, job: str) -> str | None:
        try:
            response = await self._jobs.get_job_status(workspace=self._workspace, name=job)
        except NotFoundError:
            return None
        return response.data().status.value
