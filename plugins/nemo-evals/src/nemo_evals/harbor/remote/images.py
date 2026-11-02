# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared images of one Harbor task: what submit pins and every task Job of the run uses.

Shared by the run plan (frozen at submit) and the task Job input, so both name the same images.
"""

from nemo_builder_plugin.entities import ImageRuntimeConfig
from nemo_evals.harbor.remote.profile import PINNED_IMAGE_REF_PATTERN, ResourceClass
from pydantic import BaseModel, ConfigDict, Field


class PreparedImage(BaseModel):
    """One verified image, pinned by digest. Never a tag."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_ref: str = Field(pattern=PINNED_IMAGE_REF_PATTERN, description="registry/repo@sha256:<64 hex>.")
    signed_tag: str | None = Field(
        default=None,
        description="registry/repo:tag; required when the policy's image_reference_mode is 'tag' (M2 story 6a).",
    )
    preparation_key: str = Field(min_length=1, description="Content key: same key means same image (M3 story 3).")
    builder_image: str | None = Field(
        default=None, description="Built images only: the Builder ContainerImage record (BuildImage.builder_image)."
    )
    build_job: str | None = Field(
        default=None, description="Built images only: the Builder Job (BuildImage.build_job)."
    )
    source_archive_digest: str | None = Field(default=None, description="Built images only: digest of the build input.")
    runtime_config: ImageRuntimeConfig | None = Field(
        default=None,
        description=(
            "How the image starts by default, as Builder recorded it. The sandbox replaces the container's "
            "entrypoint, so the task Job restores the image's own from here. None when Builder didn't record it."
        ),
    )


class PreparedHarborEnvironment(BaseModel):
    """Everything a task Job needs about its images. Set once on the member; never replaced."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_revision_digest: str = Field(min_length=1)
    archive_digest: str = Field(min_length=1)
    agent: PreparedImage
    verifier: PreparedImage | None = Field(default=None, description="Set only when the task has a separate verifier.")
    resource_class: ResourceClass = "cpu"
    compose_images: dict[str, PreparedImage] = Field(
        default_factory=dict,
        description="Compose service name → pinned image (M3 story 8b); empty for single-container tasks.",
    )
