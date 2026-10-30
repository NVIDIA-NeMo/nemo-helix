# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records and API shapes for container image builds of a Harbor task revision."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import ClassVar, Literal

from nemo_builder_plugin.entities import ImageRuntimeConfig
from nemo_builder_plugin.identity import PLATFORM_PATTERN, ImageIdentityError, normalize_registry, platform_name
from nemo_evals.content_hash import DIGEST_LENGTH, DIGEST_PATTERN
from nemo_helix_plugin.entities import EntityBase
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

BuildStatus = Literal["pending", "running", "ready", "failed"]
ImageStatus = Literal["pending", "building", "ready", "failed"]
BuildImageSource = Literal["dockerfile", "prebuilt"]

TERMINAL_BUILD_STATUSES: frozenset[BuildStatus] = frozenset({"ready", "failed"})
TERMINAL_IMAGE_STATUSES: frozenset[ImageStatus] = frozenset({"ready", "failed"})

TASK_IMAGE_BUILD_JOB_SOURCE = "nemo-evals.task-image-build"
_BUILD_NAME_PREFIX = "build."


class BuildImage(BaseModel):
    """One container image a task needs, and where its build stands."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        # Builds recorded before the rename stored this as `role`; read either so they still load.
        validation_alias=AliasChoices("type", "role"),
        description=(
            "The container the image runs as: `main`, the agent's container; `verifier`, for a separate verifier; "
            "or `service/<name>`, for a compose service."
        ),
    )
    source: BuildImageSource = Field(
        description=(
            "`dockerfile` when built from the task archive, `prebuilt` when the task names an image, which is "
            "copied unchanged into the platform's registry."
        )
    )
    context_path: str | None = Field(
        default=None, description="Build context inside the task archive, for a `dockerfile` image."
    )
    dockerfile: str | None = Field(
        default=None, description="Dockerfile path relative to `context_path`, for a `dockerfile` image."
    )
    context_digest: str | None = Field(
        default=None,
        description=(
            "Digest of every file in the build context, for a `dockerfile` image. Tasks whose contexts digest the "
            "same share one build."
        ),
    )
    image: str | None = Field(default=None, description="The image the task names, for a `prebuilt` image.")
    platform: str | None = Field(
        default=None,
        description=(
            "The platform the image is built or copied for, once the build decides it: the request's `platform`, "
            "the task's own pin, or a prebuilt image's only platform. None means the deployment's default."
        ),
    )
    status: ImageStatus = "pending"
    builder_image: str | None = Field(default=None, description="The Builder container image record.")
    build_job: str | None = Field(default=None, description="The Builder job building this image.")
    image_ref: str | None = Field(
        default=None, description="`<registry>/<repository>@<digest>` once the image is ready."
    )
    runtime_config: ImageRuntimeConfig | None = Field(
        default=None,
        description=(
            "How the image starts by default (its `ENTRYPOINT`, `CMD` and `WORKDIR`), once it's ready. None when "
            "Builder didn't record it, as for an image it built before it started to."
        ),
    )
    error: str | None = None


class TaskImageBuildEntity(EntityBase):
    """One attempt at building every container image a task's contents need.

    A child of the task, keyed by the revision digest: revisions with the same contents need the same images, so
    they share builds. Publishing earlier contents again republishes them as a new revision with the same digest.
    """

    __entity_type__: ClassVar[str] = "task_image_build"

    task: str = Field(description="`<workspace>/<name>` of the task.")
    revision_digest: str = Field(min_length=DIGEST_LENGTH, max_length=DIGEST_LENGTH, pattern=DIGEST_PATTERN)
    attempt: int = Field(ge=1, description="1-based attempt for these contents. Ends the record's name.")
    status: BuildStatus = "pending"
    job: str = Field(description="Name of the task-builder Job that runs this attempt.")
    allowed_registries: list[str] = Field(default_factory=list)
    platform: str | None = Field(default=None, description="The platform the request asked for, if any.")
    images: list[BuildImage] = Field(default_factory=list)
    error: str | None = None


class ImageBuildRequest(BaseModel):
    """Options for an image build of one task revision."""

    model_config = ConfigDict(extra="forbid")

    allowed_registries: list[str] = Field(
        default_factory=list,
        description=(
            "Registry hosts, such as `docker.io`, that a prebuilt image named by the task may be copied from. "
            "Narrows the registries the deployment allows; empty allows all of those. A prebuilt image from "
            "any other registry fails the build."
        ),
    )

    platform: str | None = Field(
        default=None,
        pattern=PLATFORM_PATTERN,
        description=(
            "The platform to build every image for, as `os/architecture`, such as `linux/arm64`. Omit to follow the "
            "task: an image whose Dockerfile or compose service pins a platform is built for it, a prebuilt image "
            "published for one platform is copied for that one, and anything else follows the deployment's default. "
            "An image pinned to another platform fails the build. BuildKit's spellings, such as `linux/x86_64`, are "
            "taken and named as it names them."
        ),
    )

    _platform_name = field_validator("platform", mode="before")(platform_name)

    rebuild: bool = Field(
        default=False,
        description=(
            "Start a new attempt even when the newest build is ready, so the request reaches Builder again: after "
            "the deployment's default platform changes, this builds for the new one. A build still running is "
            "returned as it is. Builder reuses any image it already has for an unchanged request."
        ),
    )

    @field_validator("allowed_registries")
    @classmethod
    def _registries_are_hosts(cls, value: list[str]) -> list[str]:
        try:
            return [normalize_registry(host) for host in value]
        except ImageIdentityError as exc:
            raise ValueError(f"allowed_registries must be registry hosts: {exc}") from exc


class ImageBuild(BaseModel):
    """An image build of one task revision."""

    id: str = Field(description="The build's attempt number for this revision.")
    task: str
    revision: str = Field(description="Digest of the task revision being built.")
    status: BuildStatus
    job: str
    allowed_registries: list[str]
    platform: str | None = None
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
            platform=entity.platform,
            images=entity.images,
            error=entity.error,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )


def build_entity_name(revision_digest: str, attempt: int) -> str:
    """Name of one attempt's record under its task: the digest's first 32 characters keep it a valid entity name."""
    return f"{_BUILD_NAME_PREFIX}{revision_digest[:32]}.{attempt}"


def build_job_name(task: str, revision_digest: str, attempt: int) -> str:
    """Name of the task-builder Job for one attempt, unique per task contents and attempt.

    ``by-contents`` keeps these names apart from the Jobs of builds kept under a revision, before builds were keyed
    by contents: such a Job would never write a build kept by contents.
    """
    key = hashlib.sha256(f"by-contents:{task}#{revision_digest}".encode()).hexdigest()[:16]
    return f"task-image-build-{key}-{attempt}"
