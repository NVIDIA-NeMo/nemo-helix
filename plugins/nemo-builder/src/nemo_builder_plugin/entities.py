# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ContainerImage``: one row per image a build publishes.

The builder writes a row ``pending`` when the build is submitted, before the job that builds it exists. A row a
remote build service builds is written again once that job does, with the job's ID and its first attempt's (see
``Provenance.job_uid``). It becomes ``ready`` when the step that built the
image completes it: the push step, once it has pushed and signed the image, or a remote-build step, once its build
service has published it. It becomes ``failed`` when the submit is refused, when a submit gives its build up
because the job could not be recorded on its rows, or once its job is found to have ended without completing it
(see ``settle``). The builder never changes a row after that, but it isn't the
only way to write one: see the README's Security limitations.

``queued`` is deprecated: 0.10's builder wrote it when it queued a build itself (see :class:`QueuedBuild`). Rows it
left ``queued`` are settled as ``pending`` ones are; nothing writes it any more.
"""

from __future__ import annotations

from typing import Literal

from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.refs import ENTITY_REF_PATTERN
from pydantic import BaseModel, Field, computed_field, field_validator

#: The entity store filters JSON integers as int64, so a larger revision would be stored but never found.
MAX_REVISION = 2**63 - 1

#: `provenance.backend` of a row this cluster builds, and of every row from before the field existed.
IN_CLUSTER_BACKEND = "execution"

#: The longest ID of a job, or of one of its attempts, a row records. The platform's are a few dozen characters.
MAX_JOB_ID = 256


def _is_unset(value: object) -> bool:
    return value is None


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
    revision: int = Field(ge=1, le=MAX_REVISION, description="The `BuildSet.revision` this row came from.")
    spec: str | None = Field(
        default=None, description="The `BuildSpec.name` this row came from, or None for the set's unnamed spec."
    )
    job: str = Field(
        pattern=ENTITY_REF_PATTERN,
        description=(
            "`<workspace>/<name>` of the build job. A name is not one job for good: another job can take it once "
            "this one is deleted, or before it exists."
        ),
    )
    # Left out of a row that has none, so a row built in this cluster is stored, and read, as it always was.
    job_uid: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_JOB_ID,
        exclude_if=_is_unset,
        description=(
            "The ID Jobs gave the build job this row's submit created, which no later job is given. Written once "
            "that job exists, for an image a remote build service builds. Under workload token exchange, only a "
            "step of that job's first attempt completes the image; with platform auth off, or where the "
            "deployment sets `builder.remote.allow_unverified_completion`, it is compared with nothing. None "
            "until then, and for an image this cluster builds."
        ),
    )
    job_attempt_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_JOB_ID,
        exclude_if=_is_unset,
        description=(
            "The ID of that job's first attempt, the one that runs the request as the builder compiled it. A "
            "rerun of the job is another attempt, and may run a specification changed since. Written with "
            "`job_uid`."
        ),
    )
    request_digest: str = Field(
        pattern=DIGEST_PATTERN,
        description=(
            "sha256 of the canonical JSON of the `BuildSet`. Tells a retry of the same request from a "
            "different request that reuses its revision. Every signature the builder makes carries it."
        ),
    )
    upstream: UpstreamImage | None = Field(default=None, description="For a copy, the image it copies.")
    backend: str = Field(
        default=IN_CLUSTER_BACKEND,
        max_length=64,
        description=(
            "The builder backend the row was submitted under, as `builder.backend` named it. A row written before "
            "this field existed reads as `execution`, the only backend there was."
        ),
    )


#: A remote build service's own facts about an image: a few short strings, never a document.
MAX_EXTERNAL_ATTRIBUTES = 32
MAX_EXTERNAL_VALUE = 512
MAX_EXTERNAL_KEY = 128


class ExternalImage(BaseModel):
    """What a remote build service calls the image it built, written when the row becomes ``ready``.

    The builder stores these for a consumer that uses the same service. It reads none of them for itself. It
    only requires that a completion carries a ``build_id``, and that a second completion carries the same three.
    """

    image_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_EXTERNAL_VALUE,
        description="The service's own handle for the image, if it has one. Not necessarily unique to a row.",
    )
    build_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_EXTERNAL_VALUE,
        description="The service's record of the build, for operators.",
    )
    attributes: dict[str, str] = Field(
        default_factory=dict,
        max_length=MAX_EXTERNAL_ATTRIBUTES,
        description="A few facts the service reports about the build, such as its builder's version.",
    )

    @field_validator("attributes")
    @classmethod
    def _attributes_are_short(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if len(key) > MAX_EXTERNAL_KEY or len(item) > MAX_EXTERNAL_VALUE:
                raise ValueError(
                    f"attribute {key[:32]!r} is too long: a key holds {MAX_EXTERNAL_KEY} characters, "
                    f"a value {MAX_EXTERNAL_VALUE}"
                )
        return value


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
    user: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "The image's `USER`, as `name`, `uid` or either with `:group`; an empty string when it sets none, so it "
            "runs as root. None when Builder didn't record it, as for an image it built before it started to."
        ),
    )


class ContainerImage(NemoEntity, entity_type="container_image"):
    """One image. ``name`` is ``<set>-<revision>.<spec>``, or ``<set>-<revision>`` for the set's unnamed spec."""

    # `queued`: deprecated, only on rows 0.10's builder wrote.
    status: Literal["queued", "pending", "ready", "failed"] = "pending"
    status_detail: str | None = None

    registry: str = Field(description="Registry host the image is published to.")
    repository: str = Field(description="Repository path within the registry.")
    provenance: Provenance

    # Written once, when the row becomes `ready`.
    digest: str | None = Field(
        default=None,
        pattern=DIGEST_PATTERN,
        description=(
            "The manifest digest the image is published at: the one its push step pushed under the system tag and "
            "signed, or the one a remote build service published."
        ),
    )
    runtime_config: ImageRuntimeConfig | None = Field(
        default=None,
        description=(
            "How the image starts by default: read from its config when it became `ready`, or, for an image a "
            "remote build service built, what the service's client reported. None for an image completed before "
            "Builder recorded it, one whose config couldn't be read, or one whose build service's client reported "
            "none."
        ),
    )
    external: ExternalImage | None = Field(
        default=None,
        description=(
            "What a remote build service calls the image, written when the row becomes `ready`. None for an image "
            "this cluster built."
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
    """Deprecated: a build 0.10's builder queued itself, until its queue controller created the job.

    Nothing writes one any more. ``builder-settle`` deletes the ones 0.10 left, since each holds its submitter's
    principal; their images settle as any others do. Its fields are left out: nothing reads them.
    """
