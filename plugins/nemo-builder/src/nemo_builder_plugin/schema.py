# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The request ``POST /builds`` accepts: a ``BuildSet`` of independent ``BuildSpec``s, one per image."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Self

from nemo_builder_plugin.identity import REPOSITORY_COMPONENT_PATTERN, validate_caller_tag, validate_repository
from nemo_helix_plugin.entity_naming import NAME_MAX_LENGTH, NAME_PATTERN
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Unknown fields are refused: a misspelled field, silently dropped, would build something the
# caller didn't ask for.
_STRICT = ConfigDict(extra="forbid")

_PLATFORM_NAME = re.compile(NAME_PATTERN)

# Lowercase letters, digits and single hyphens: a set's name becomes part of row, job, tag and pod
# names, and this is what all four accept.
_BUILD_SET_NAME_PATTERN = r"^[a-z](?:[a-z0-9]|-(?!-))*[a-z0-9]$"

#: The most specs a set may have.
MAX_BUILD_SPECS = 100

# Jobs names a job's fileset `job-fileset-<job name>`: the longest name a set's name ends up in.
_JOBS_FILESET_PREFIX = "job-fileset-"


def _relative_path(value: str, *, field: str) -> str:
    """A path that is relative and does not climb out with ``..``."""
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"{field} must be a relative path, got {value!r}")
    if ".." in path.parts:
        raise ValueError(f"{field} must not climb with '..', got {value!r}")
    if not path.parts:
        raise ValueError(f"{field} must name something, got {value!r}")
    return value


class FileSetSource(BaseModel):
    """A build context stored in a fileset."""

    model_config = _STRICT

    fileset: str = Field(description="The fileset: `<name>` in this workspace, or `<workspace>/<name>`.")
    context_path: str | None = Field(
        default=None,
        description="Directory within the fileset to use as the build context. Omit to use the whole fileset.",
    )

    @field_validator("fileset")
    @classmethod
    def _fileset_is_a_reference(cls, value: str) -> str:
        parts = value.split("/")
        if len(parts) > 2 or not all(_PLATFORM_NAME.fullmatch(part) for part in parts):
            raise ValueError(f"fileset must be `<name>` or `<workspace>/<name>`, got {value!r}")
        return value

    @field_validator("context_path")
    @classmethod
    def _context_path_stays_in_the_fileset(cls, value: str | None) -> str | None:
        return None if value is None else _relative_path(value, field="context_path")


class BuildOutput(BaseModel):
    """Where a built image is published: ``<repository_prefix>/<workspace>/<repository>:<tag>``.

    The registry and the prefix are the deployment's, and the workspace is the submitter's, so a
    caller can publish only under its own workspace.
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

    name: str = Field(
        max_length=63,
        pattern=REPOSITORY_COMPONENT_PATTERN,
        description=(
            "The spec's name within the set: lowercase letters and digits, joined by `.`, `_` or `-`. "
            "A spec that names no output is published at `<set>/<name>` under the workspace."
        ),
    )
    source: FileSetSource
    output: BuildOutput | None = Field(
        default=None,
        description="Where to publish. Omit to publish at `<set>/<name>` under the workspace, with the system tag only.",
    )
    dockerfile: str = Field(default="Dockerfile", description="Path to the Dockerfile, relative to the context.")
    platform: str = Field(default="linux/amd64", pattern=r"^[a-z0-9]+/[a-z0-9]+(/[a-z0-9]+)?$")

    @model_validator(mode="after")
    def _dockerfile_stays_within_the_context(self) -> Self:
        _relative_path(self.dockerfile, field="dockerfile")
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

    @model_validator(mode="after")
    def _spec_names_are_unique(self) -> Self:
        names = [spec.name for spec in self.build_specs]
        if len(names) != len(set(names)):
            raise ValueError("build_spec names must be unique within a set")
        return self

    @model_validator(mode="after")
    def _derived_names_fit(self) -> Self:
        """Every name composed from the set's name must fit, checked before anything is written.

        The row names (``<set>-<revision>-<index>``), the job name and the fileset Jobs creates for
        the job all have a length limit that a name matching the pattern can still exceed.
        """
        longest = max(
            f"{self.name}-{self.revision}-{len(self.build_specs) - 1}",
            f"{_JOBS_FILESET_PREFIX}{self.name}-{self.revision}",
            key=len,
        )
        if len(longest) > NAME_MAX_LENGTH:
            raise ValueError(
                f"name {self.name!r} is too long for revision {self.revision} with "
                f"{len(self.build_specs)} spec(s): it would compose {longest!r}, over "
                f"{NAME_MAX_LENGTH} characters"
            )
        return self
