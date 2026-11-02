# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ContainerImage``: one row per image a build publishes.

The builder writes a row ``pending`` when the build is submitted, or ``queued`` when builds are capped and the job
waits for a slot (see ``QueuedBuild``). It becomes ``ready`` when the image's push
step has pushed and signed the image and completes it, or ``failed`` when the submit is refused. The builder's
routes never change a row after that, but they aren't the only way to write one: see the README's Security
limitations.
"""

from __future__ import annotations

from typing import Literal

from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.refs import ENTITY_REF_PATTERN
from pydantic import BaseModel, Field, PrivateAttr, computed_field


class UpstreamImage(BaseModel):
    """The published image a copy came from, resolved when the build was submitted."""

    image: str = Field(
        description="The image as the request named it, fully qualified: `docker.io/library/ubuntu:24.04`."
    )
    manifest_digest: str = Field(
        pattern=DIGEST_PATTERN,
        description="The manifest for the spec's platform that `image` named at submit. The copy's digest equals it.",
    )


class Provenance(BaseModel):
    """The build that produces the image, written at submit."""

    build_set: str = Field(description="The `BuildSet.name` this row came from.")
    revision: int = Field(ge=1, description="The `BuildSet.revision` this row came from.")
    spec: str | None = Field(
        default=None, description="The `BuildSpec.name` this row came from, or None for the set's unnamed spec."
    )
    job: str = Field(pattern=ENTITY_REF_PATTERN, description="`<workspace>/<name>` of the build job.")
    request_digest: str = Field(
        pattern=DIGEST_PATTERN,
        description=(
            "sha256 of the canonical JSON of the `BuildSet`. Tells a retry of the same request from a "
            "different request that reuses its revision. Every signature carries it."
        ),
    )
    upstream: UpstreamImage | None = Field(default=None, description="For a copy, the image it copies.")


class ImageRuntimeConfig(BaseModel):
    """How the image starts by default, from its image config, including what it inherits from its base image.

    A runtime that replaces the container's entrypoint, such as OpenSandbox, reads this to start the image's own.
    """

    entrypoint: list[str] | None = Field(
        default=None, max_length=256, description="The image's `ENTRYPOINT`, or None when it has none."
    )
    cmd: list[str] | None = Field(
        default=None,
        max_length=256,
        description="The image's `CMD`: the default arguments to its `ENTRYPOINT`, or None when it has none.",
    )
    working_dir: str | None = Field(
        default=None, max_length=4096, description="The image's `WORKDIR`, or None when it sets none."
    )


class ContainerImage(NemoEntity, entity_type="container_image"):
    """One image. ``name`` is ``<set>-<revision>.<spec>``, or ``<set>-<revision>`` for the set's unnamed spec."""

    status: Literal["queued", "pending", "ready", "failed"] = "pending"
    status_detail: str | None = None

    registry: str = Field(description="Registry host the image is published to.")
    repository: str = Field(description="Repository path within the registry.")
    provenance: Provenance

    # Written once, when the row becomes `ready`.
    digest: str | None = Field(
        default=None,
        pattern=DIGEST_PATTERN,
        description="The manifest digest the push step pushed under the system tag, and signed.",
    )
    runtime_config: ImageRuntimeConfig | None = Field(
        default=None,
        description=(
            "How the image starts by default, read from its config when it became `ready`. None for an image "
            "completed before Builder recorded it, or whose config couldn't be read."
        ),
    )

    # In every response, but never stored: the entity store writes a row without its computed fields.
    @computed_field(
        description="`<registry>/<repository>@<digest>`, the reference to pin, once the row is `ready`.",
        json_schema_extra={"nullable": True},
    )
    @property
    def image_ref(self) -> str | None:
        if self.digest is None:
            return None
        return f"{self.registry}/{self.repository}@{self.digest}"


class QueuedBuild(NemoEntity, entity_type="builder_queued_build"):
    """A build job waiting for a slot under ``max_concurrent_builds``. ``name`` is the job's name.

    The submit writes it with its images ``queued``; the queue controller creates the job as ``created_by``,
    moves the images to ``pending`` and marks it ``started``, then deletes it once the job ends.
    """

    request: dict = Field(description="The `CreateHelixJobRequest` the controller creates the job from, as JSON.")
    images: list[str] = Field(description="The `ContainerImage` rows the job builds.")
    state: Literal["queued", "started"] = "queued"

    # The submitter's full principal (id, email, account id, groups, aliases): the controller creates the job as
    # them, and the platform can't look a user's groups up later. The entity store gives it back only to a
    # service-principal client. None with auth off.
    _auth_context: AuthContext | None = PrivateAttr(default=None)

    @property
    def auth_context(self) -> AuthContext | None:
        return self._auth_context
