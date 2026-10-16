# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records and API shapes for container image builds of a Harbor task revision."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import ClassVar, Literal

from nemo_evals.content_hash import DIGEST_LENGTH, DIGEST_PATTERN
from nemo_helix_plugin.entities import EntityBase
from pydantic import BaseModel, ConfigDict, Field

BuildStatus = Literal["pending", "running", "ready", "failed"]
ImageStatus = Literal["pending", "building", "ready", "failed"]
ImageSource = Literal["dockerfile", "prebuilt"]

TERMINAL_BUILD_STATUSES: frozenset[BuildStatus] = frozenset({"ready", "failed"})
TERMINAL_IMAGE_STATUSES: frozenset[ImageStatus] = frozenset({"ready", "failed"})

TASK_IMAGE_BUILD_JOB_SOURCE = "nemo-evals.task-image-build"
_BUILD_NAME_PREFIX = "build."


class BuildImage(BaseModel):
    """One container image a task needs, and where its build stands."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(
        description=(
            "The container the image runs as, named as Harbor names it: `main`, the agent's container; a compose "
            "service's own name; or `verifier`, for a separate verifier."
        )
    )
    source: ImageSource = Field(
        description="`dockerfile` when built from the task archive, `prebuilt` when the task names an image."
    )
    context_path: str | None = Field(
        default=None, description="Build context inside the task archive, for a `dockerfile` image."
    )
    dockerfile: str | None = Field(
        default=None, description="Dockerfile path relative to `context_path`, for a `dockerfile` image."
    )
    image: str | None = Field(default=None, description="The image the task names, for a `prebuilt` image.")
    status: ImageStatus = "pending"
    builder_image: str | None = Field(default=None, description="The Builder container image record.")
    build_job: str | None = Field(default=None, description="The Builder job building this image.")
    image_ref: str | None = Field(
        default=None, description="`<registry>/<repository>@<digest>` once the image is ready."
    )
    error: str | None = None


class TaskImageBuildEntity(EntityBase):
    """One attempt at building every container image a task revision needs."""

    __entity_type__: ClassVar[str] = "task_image_build"

    task: str = Field(description="`<workspace>/<name>` of the task.")
    revision_digest: str = Field(min_length=DIGEST_LENGTH, max_length=DIGEST_LENGTH, pattern=DIGEST_PATTERN)
    attempt: int = Field(ge=1, description="1-based attempt for this revision. Matches the `build.<n>` name.")
    status: BuildStatus = "pending"
    job: str = Field(description="Name of the task-builder Job that runs this attempt.")
    allowed_registries: list[str] = Field(default_factory=list)
    images: list[BuildImage] = Field(default_factory=list)
    error: str | None = None


class ImageBuildRequest(BaseModel):
    """Options for an image build of one task revision."""

    model_config = ConfigDict(extra="forbid")

    allowed_registries: list[str] = Field(
        default_factory=list,
        description=(
            "Registry hosts a prebuilt image named by the task may be copied from, e.g. `docker.io`. "
            "A prebuilt image from any other registry fails the build."
        ),
    )


class ImageBuild(BaseModel):
    """An image build of one task revision."""

    id: str = Field(description="The build's attempt number for this revision.")
    task: str
    revision: str = Field(description="Digest of the task revision being built.")
    status: BuildStatus
    job: str
    allowed_registries: list[str]
    images: list[BuildImage]
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_entity(cls, entity: TaskImageBuildEntity) -> ImageBuild:
        return cls(
            id=str(entity.attempt),
            task=entity.task,
            revision=entity.revision_digest,
            status=entity.status,
            job=entity.job,
            allowed_registries=entity.allowed_registries,
            images=entity.images,
            error=entity.error,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )


def build_entity_name(attempt: int) -> str:
    return f"{_BUILD_NAME_PREFIX}{attempt}"


def build_job_name(task: str, revision_digest: str, attempt: int) -> str:
    """Name of the task-builder Job for one attempt, unique per task revision and attempt."""
    key = hashlib.sha256(f"{task}#{revision_digest}".encode()).hexdigest()[:16]
    return f"task-image-build-{key}-{attempt}"
