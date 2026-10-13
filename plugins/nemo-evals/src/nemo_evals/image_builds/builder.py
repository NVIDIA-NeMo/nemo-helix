# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submitting and following NeMo Builder builds for task images."""

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from typing import Literal, Protocol

from filesets import parse_fileset_ref
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.endpoint import get, post
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.client.method import method
from pydantic import BaseModel, ConfigDict, Field

_BUILDER_BASE = "/apis/builder/v2/workspaces/{workspace}"
_BUILDER_NAME_PREFIX = "harbor-env-"


class BuilderImage(BaseModel):
    """The part of a Builder ``ContainerImage`` record this module reads."""

    model_config = ConfigDict(extra="ignore")

    name: str
    status: Literal["pending", "ready", "failed"]
    status_detail: str | None = None
    image_ref: str | None = None


class ArchiveSource(BaseModel):
    """A build context inside a task archive stored in a fileset."""

    model_config = ConfigDict(extra="forbid")

    fileset: str
    archive: str
    context_path: str


class _BuildSpec(BaseModel):
    source: ArchiveSource
    dockerfile: str


class _BuildSet(BaseModel):
    name: str
    revision: int
    build_specs: list[_BuildSpec]


class _SubmitBuildResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job: str
    images: list[BuilderImage] = Field(min_length=1)


class SubmittedBuild(BaseModel):
    image: BuilderImage
    job: str


class ImageBuilder(Protocol):
    """The two Builder operations a task image build needs."""

    async def get_image(self, name: str) -> BuilderImage | None:
        """The Builder image record ``name``, or None when there is none."""
        ...

    async def submit(self, *, name: str, revision: int, source: ArchiveSource, dockerfile: str) -> SubmittedBuild:
        """Submit (or rejoin) one build of one image."""
        ...


def builder_name(*, archive_digest: str, context_path: str, dockerfile: str) -> str:
    """Builder build-set name for one image of one task archive."""
    key = json.dumps(
        {"v": 1, "archive": archive_digest, "context": context_path, "dockerfile": dockerfile},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{_BUILDER_NAME_PREFIX}{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def archive_source(fileset_ref: str, context_path: str) -> ArchiveSource:
    """Builder source for ``context_path`` inside the archive at ``workspace/fileset#path``."""
    workspace, fileset, path = parse_fileset_ref(fileset_ref, workspace_fallback=None)
    return ArchiveSource(fileset=f"{workspace}/{fileset}", archive=path, context_path=context_path)


async def ensure_build(builder: ImageBuilder, *, name: str, source: ArchiveSource, dockerfile: str) -> SubmittedBuild:
    """Submit the newest revision of ``name`` again, or the next one when the newest failed."""
    newest: BuilderImage | None = None
    revision = 1
    while (image := await builder.get_image(f"{name}-{revision}")) is not None:
        newest = image
        revision += 1
    if newest is not None and newest.status != "failed":
        revision -= 1
    return await builder.submit(name=name, revision=revision, source=source, dockerfile=dockerfile)


@post(f"{_BUILDER_BASE}/builds")
@abstractmethod
def _submit_build(*, workspace: str | None = None, body: _BuildSet) -> _SubmitBuildResponse: ...


@get(f"{_BUILDER_BASE}/container-images/{{name}}")
@abstractmethod
def _get_container_image(*, workspace: str | None = None, name: str) -> BuilderImage: ...


class _AsyncBuilderRoutes(AsyncNemoClient):
    submit_build = method(_submit_build)
    get_container_image = method(_get_container_image)


class HttpImageBuilder:
    """:class:`ImageBuilder` over Builder's HTTP API, acting as ``client``'s caller."""

    def __init__(self, client: AsyncNemoClient, *, workspace: str) -> None:
        self._routes = _AsyncBuilderRoutes.from_client(client)
        self._workspace = workspace

    async def get_image(self, name: str) -> BuilderImage | None:
        try:
            response = await self._routes.get_container_image(workspace=self._workspace, name=name)
        except NotFoundError:
            return None
        return response.data()

    async def submit(self, *, name: str, revision: int, source: ArchiveSource, dockerfile: str) -> SubmittedBuild:
        body = _BuildSet(name=name, revision=revision, build_specs=[_BuildSpec(source=source, dockerfile=dockerfile)])
        response = (await self._routes.submit_build(workspace=self._workspace, body=body)).data()
        return SubmittedBuild(image=response.images[0], job=response.job)
