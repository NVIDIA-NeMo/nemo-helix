# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request schemas: what a caller submits to ``POST /builds``.

These are validated at submit, compiled into a ``HelixJobSpec``, and never stored by us --
Jobs keeps the compiled spec immutably on ``HelixJobAttempt.platform_spec``, which is what
``ContainerImage.provenance.built_by`` resolves against. Plain ``BaseModel``s, deliberately:
the only thing this design stores as an entity is ``ContainerImage``.

A ``BuildSpec`` is one image to build. A ``BuildSet`` is a flat group of N of them with **no
ordering between them** -- the build graph (one image's output as another's ``FROM``) is
deferred, and the flatness is what makes the set parallelisable later without a schema change.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal, Self

from nemo_builder_plugin.identity import validate_caller_tag, validate_repository
from nemo_helix_plugin.entity_naming import NAME_MAX_LENGTH, NAME_PATTERN
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Every request model refuses fields it does not know. A misspelt `dockerFile` or `contextPath`
#: that was silently dropped would build something other than what the caller asked for, and
#: the caller would find out from the image, not from the response.
_STRICT = ConfigDict(extra="forbid")

_PLATFORM_NAME = re.compile(NAME_PATTERN)

#: What a build set name may contain: the intersection of the grammars it ends up in -- a row
#: name, a Jobs job name, a system tag, and a Kubernetes pod name. NAME_PATTERN alone admits
#: `@`, `+`, `_` and `.`, and each of those is illegal in at least one of the others.
BUILD_SET_NAME_PATTERN = r"^[a-z](?:[a-z0-9]|-(?!-))*[a-z0-9]$"

#: Specs per set. Each is one row, one sandbox build and one push, all sequential today.
MAX_BUILD_SPECS = 100

#: The Jobs service names the fileset it creates for a job `job-fileset-<job name>`, and that
#: name must itself be a legal entity name. It is the longest name a set's name ends up in.
_JOBS_FILESET_PREFIX = "job-fileset-"


def _relative_path(value: str, *, field: str) -> str:
    """A path inside something the caller owns: relative, and never climbing out with `..`."""
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"{field} must be a relative path, got {value!r}")
    if ".." in path.parts:
        raise ValueError(f"{field} must not climb with '..', got {value!r}")
    if not path.parts:
        raise ValueError(f"{field} must name something, got {value!r}")
    return value


class FileSetSource(BaseModel):
    """A build context already in Files.

    v1 ships exactly one transport. Git and OCI-artifact sources are deferred rather than
    rejected -- no consumer wants them yet. This stays a named type rather than collapsing into
    ``BuildSpec.source`` so that adding a transport is a new arm here, not a field-type change
    everywhere the type is named.
    """

    model_config = _STRICT

    type: Literal["fileset"] = "fileset"
    fileset: str = Field(
        description="FileSet holding the context: `<name>` in this workspace, or `<workspace>/<name>`."
    )
    context_path: str | None = Field(
        default=None,
        description=(
            "Subtree of the fileset to use as the build context. Selects a directory; it does "
            "not repack. Omit to use the fileset root."
        ),
    )

    @field_validator("fileset")
    @classmethod
    def _fileset_is_a_reference(cls, value: str) -> str:
        """Checked at submit, where a bad name is a 4xx, rather than in the fetch pod."""
        parts = value.split("/")
        if len(parts) > 2 or not all(_PLATFORM_NAME.fullmatch(part) for part in parts):
            raise ValueError(f"fileset must be `<name>` or `<workspace>/<name>`, got {value!r}")
        return value

    @field_validator("context_path")
    @classmethod
    def _context_path_stays_in_the_fileset(cls, value: str | None) -> str | None:
        return None if value is None else _relative_path(value, field="context_path")


class BuildOutput(BaseModel):
    """Where one built image is published: a repository and tag, never a registry.

    Every image goes to the deployment's one registry, at
    ``<repository_prefix>/<workspace>/<repository>``, pushed with the operator's credential. One
    registry is what lets every workload that runs a built image pull it with one credential. The
    workspace in the path is what keeps one tenant out of another's repositories: the credential
    can write anywhere under the prefix, so the registry cannot tell workspaces apart, and the
    compiler is the only thing that can.
    """

    # A caller still sending `registry` gets a 422 rather than having it ignored -- an ignored
    # `registry` would publish somewhere other than where the caller asked.
    model_config = ConfigDict(extra="forbid")

    repository: str = Field(
        description=(
            "Repository path within this workspace's part of the registry, e.g. `team/app`. "
            "Components on the OCI grammar; no `..`, so it cannot leave the workspace's path."
        )
    )
    tag: str = Field(
        description=(
            "The caller's tag. The system tag is pushed alongside it. Tags containing `--` or "
            "starting with `sha256-` are reserved: they are the shapes of system tags and of "
            "cosign's signature tags."
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

    name: str = Field(description="Caller-facing label for this spec within the set.")
    source: FileSetSource
    output: BuildOutput
    dockerfile: str = Field(
        default="Dockerfile",
        description="Path to the Dockerfile, relative to the context directory.",
    )
    platform: str = Field(
        default="linux/amd64",
        pattern=r"^[a-z0-9]+/[a-z0-9]+(/[a-z0-9]+)?$",
    )

    @model_validator(mode="after")
    def _dockerfile_stays_within_the_context(self) -> Self:
        """The one validator here that carries real meaning.

        The Dockerfile path is caller-supplied and is resolved *inside* the sandbox, against a
        mount the sandbox holds read-only. A path that escapes the context would be resolved
        against the work volume -- which, at the mount the sandbox actually has, is that job's
        slice. Reject absolute paths and any `..` component at submit, where the error is a 4xx
        on the caller's own request rather than a build failure twenty minutes later.

        This is cheap and it is not the real control: `supervise` mounts only the group's own
        context subPath, so even a successful escape here reaches nothing else in the set.
        Defence in depth, with the depth stated.
        """
        _relative_path(self.dockerfile, field="dockerfile")
        return self


class BuildSet(BaseModel):
    """A named, revisioned group of images to build in one job."""

    model_config = ConfigDict(extra="forbid", regex_engine="python-re")

    name: str = Field(
        pattern=BUILD_SET_NAME_PATTERN,
        description=(
            "What is being built -- a task name, an agent name. Reused on every rebuild: it is "
            "what makes a second submission supersede the first rather than race it. Lowercase "
            "letters, digits and single hyphens, starting with a letter: it becomes part of row, "
            "job, tag and pod names, and this is what all four accept."
        ),
    )
    revision: int = Field(
        ge=1,
        description=(
            "The caller's own ordinal for `name`. Required, not optional: without it the system "
            "cannot distinguish a resubmission of the same content from a resubmission of "
            "changed content, and guessing wrong returns a stale build. It is never generated "
            "or incremented here. A caller that already versions its content passes what it has."
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
        """Every name composed from this one must itself be a legal name, checked before any write.

        The set's name, revision and spec count become row names (`<set>-<rev>-<index>`), the
        job name (`<set>-<rev>`), and the fileset Jobs creates for the job. A name that fits the
        pattern can still overflow one of those, and discovering that after the first rows are
        written leaves a half-submitted revision behind.
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
