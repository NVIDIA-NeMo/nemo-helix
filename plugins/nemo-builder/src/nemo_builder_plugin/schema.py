# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The request ``POST /builds`` accepts: a ``BuildSet`` of independent ``BuildSpec``s, one per image."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.identity import (
    PLATFORM_PATTERN,
    normalize_reference,
    platform_name,
    validate_caller_tag,
    validate_repository,
)
from nemo_helix_plugin.entity_naming import NAME_MAX_LENGTH, NAME_PATTERN
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator, model_validator

_STRICT = ConfigDict(extra="forbid")

_PLATFORM_NAME = re.compile(NAME_PATTERN)

# A set's name becomes part of row, job, tag and pod names, and this is what all four accept. `\Z`, not `$`:
# `BuildSet` validates with Python's `re`, whose `$` also matches before a final newline.
_BUILD_SET_NAME_PATTERN = r"^[a-z](?:[a-z0-9]|-(?!-))*[a-z0-9]\Z"

# A repository path component that can also be part of an entity name, which refuses `--`.
_SPEC_NAME_PATTERN = r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$"

MAX_BUILD_SPECS = 100

#: A set's job is `builder-<set>-<revision>`. Jobs names are per workspace, shared with every other job there.
JOB_NAME_PREFIX = "builder-"

# Jobs names a job's fileset `job-fileset-<job name>`: the longest name a set's name ends up in.
_JOBS_FILESET_PREFIX = "job-fileset-"


def _relative_path(value: str, *, field: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"{field} must be a relative path, got {value!r}")
    if ".." in path.parts:
        raise ValueError(f"{field} must not climb with '..', got {value!r}")
    if not path.parts:
        raise ValueError(f"{field} must name something, got {value!r}")
    return value


class FileSetSource(BaseModel):
    """A build context stored in a fileset: its files, or an archive among them."""

    model_config = _STRICT

    fileset: str = Field(description="The fileset: `<name>` in this workspace, or `<workspace>/<name>`.")
    archive: str | None = Field(
        default=None,
        description=(
            "A tar archive in the fileset, compressed or not, to unpack and build from, under any file name. "
            "`context_path` is then a directory inside it. A file that isn't a tar archive fails the build."
        ),
    )
    context_path: str | None = Field(
        default=None,
        description=(
            "Directory to use as the build context: within the fileset, or within `archive` if set. "
            "Omit to use the whole fileset or archive."
        ),
    )

    @field_validator("fileset")
    @classmethod
    def _fileset_is_a_reference(cls, value: str) -> str:
        parts = value.split("/")
        if len(parts) > 2 or not all(_PLATFORM_NAME.fullmatch(part) for part in parts):
            raise ValueError(f"fileset must be `<name>` or `<workspace>/<name>`, got {value!r}")
        return value

    @field_validator("archive")
    @classmethod
    def _archive_is_a_tar_in_the_fileset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _relative_path(value, field="archive")
        if PurePosixPath(value).as_posix() != value:
            raise ValueError(f"archive must be written plainly, without '.' or empty segments, got {value!r}")
        return value

    @field_validator("context_path")
    @classmethod
    def _context_path_stays_in_the_fileset(cls, value: str | None) -> str | None:
        return None if value is None else _relative_path(value, field="context_path")


class ImageSource(BaseModel):
    """A published image to copy unchanged, rather than build."""

    model_config = _STRICT

    type: Literal["image"] = "image"
    image: str = Field(
        description=(
            "The image, by tag or digest, as `docker pull` takes it: `ubuntu:24.04` is "
            "`docker.io/library/ubuntu:24.04`. Its registry must be one the deployment allows. The build "
            "copies the image for the spec's `platform` that this names when the build is submitted."
        )
    )

    @field_validator("image")
    @classmethod
    def _image_is_a_reference(cls, value: str) -> str:
        normalize_reference(value)
        return value


def _source_type(value: object) -> str:
    if isinstance(value, dict):
        return "image" if value.get("type") == "image" or "image" in value else "fileset"
    return "image" if isinstance(value, ImageSource) else "fileset"


BuildSource = Annotated[
    Annotated[FileSetSource, Tag("fileset")] | Annotated[ImageSource, Tag("image")],
    Discriminator(_source_type),
]


class BuildOutput(BaseModel):
    """Where a built image is published: ``<repository_prefix>/<workspace>/<repository>:<tag>``.

    The registry and prefix are the deployment's, so a caller publishes only under its own workspace.
    """

    model_config = _STRICT

    repository: str = Field(description="Repository path under the workspace, e.g. `team/app`.")
    tag: str = Field(
        description=(
            "The tag to push, alongside the system tag. Tags containing `--` or starting with `sha256-` "
            "are reserved for system tags and signatures."
        )
    )

    @field_validator("repository")
    @classmethod
    def _repository_is_an_oci_path(cls, value: str) -> str:
        return validate_repository(value)

    @field_validator("tag")
    @classmethod
    def _tag_is_a_caller_tag(cls, value: str) -> str:
        return validate_caller_tag(value)


class BuildSpec(BaseModel):
    """One image to build."""

    model_config = _STRICT

    name: str | None = Field(
        default=None,
        pattern=_SPEC_NAME_PATTERN,
        description=(
            "The spec's name, unique in the set: lowercase letters and digits, joined by single `.`, `_` or `-`. "
            "One spec in a set may leave it out."
        ),
    )
    source: BuildSource = Field(
        description="What to build from: a fileset holding a build context, or a published image to copy."
    )
    output: BuildOutput | None = Field(
        default=None,
        description="Where to publish. Omit to publish under the workspace, with the system tag only (see `name`).",
    )
    dockerfile: str = Field(default="Dockerfile", description="Path to the Dockerfile, relative to the context.")
    platform: str | None = Field(
        default=None,
        pattern=PLATFORM_PATTERN,
        description=(
            "The platform to build or copy, as `os/architecture`. BuildKit's spellings, such as `linux/x86_64`, are "
            "taken and named as it names them. Omit for the deployment's `default_platform`."
        ),
    )

    _platform_name = field_validator("platform", mode="before")(platform_name)

    @model_validator(mode="after")
    def _dockerfile_stays_within_the_context(self) -> Self:
        _relative_path(self.dockerfile, field="dockerfile")
        if isinstance(self.source, ImageSource) and "dockerfile" in self.model_fields_set:
            raise ValueError("a spec that copies an image has no dockerfile")
        return self


class BuildSet(BaseModel):
    """A named, versioned set of images to build in one job."""

    model_config = ConfigDict(extra="forbid", regex_engine="python-re")

    name: str = Field(
        pattern=_BUILD_SET_NAME_PATTERN,
        description=(
            "What is being built, reused on every rebuild: a task or agent name. Lowercase letters, "
            "digits and single hyphens, starting with a letter."
        ),
    )
    revision: int = Field(
        ge=1,
        description=(
            "The caller's version of `name`. Submitting the same request again returns the existing "
            "build; submitting a different one under the same revision is refused."
        ),
    )
    build_specs: list[BuildSpec] = Field(min_length=1, max_length=MAX_BUILD_SPECS)

    def with_default_platform(self, platform: str) -> BuildSet:
        """This set with ``platform`` for every spec that names none: the request as it is built and recorded."""
        specs = [spec if spec.platform else spec.model_copy(update={"platform": platform}) for spec in self.build_specs]
        return self.model_copy(update={"build_specs": specs})

    @model_validator(mode="after")
    def _spec_names_are_unique(self) -> Self:
        names = [spec.name for spec in self.build_specs]
        if names.count(None) > 1:
            raise ValueError("at most one build_spec in a set may leave out its name")
        named = [name for name in names if name is not None]
        if len(named) != len(set(named)):
            raise ValueError("build_spec names must be unique within a set")
        return self

    @model_validator(mode="after")
    def _derived_names_fit(self) -> Self:
        """Valid parts can still compose a name over the limit; checked here, before anything is written."""
        base = f"{self.name}-{self.revision}"
        composed = [
            f"{_JOBS_FILESET_PREFIX}{JOB_NAME_PREFIX}{base}",
            *(f"{base}.{spec.name}" for spec in self.build_specs if spec.name),
        ]
        longest = max(composed, key=len)
        if len(longest) > NAME_MAX_LENGTH:
            raise ValueError(
                f"name {self.name!r} with revision {self.revision} is too long: it would compose {longest!r}, "
                f"over {NAME_MAX_LENGTH} characters"
            )
        return self


class SubmitBuildResponse(BaseModel):
    """The submitted build's job and images."""

    job: str = Field(description="Name of the build job, for logs and debugging.")
    images: list[ContainerImage] = Field(
        description="One row per build spec, created `pending`. POLL THESE, not the job."
    )
