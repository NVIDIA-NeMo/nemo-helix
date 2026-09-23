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

from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class FileSetSource(BaseModel):
    """A build context already in Files.

    v1 ships exactly one transport. Git and OCI-artifact sources are deferred rather than
    rejected -- no consumer wants them yet. This stays a named type rather than collapsing into
    ``BuildSpec.source`` so that adding a transport is a new arm here, not a field-type change
    everywhere the type is named.
    """

    type: Literal["fileset"] = "fileset"
    fileset: str = Field(description="FileSet holding the context.")
    context_path: str | None = Field(
        default=None,
        description=(
            "Subtree of the fileset to use as the build context. Selects a directory; it does "
            "not repack. Omit to use the fileset root."
        ),
    )


class BuildOutput(BaseModel):
    """Where one built image is published."""

    registry: str | None = Field(
        default=None,
        description=(
            "Registry host to push to. Omit to use the deployment's default registry. A "
            "deployment with neither fails the COMPILE, not the build -- an unconfigured "
            "destination is a property of the request being unanswerable, which the caller "
            "should learn at submit."
        ),
    )
    repository: str = Field(description="Repository path within the registry.")
    tag: str = Field(description="The caller's tag. The system tag is pushed alongside it.")


class BuildSpec(BaseModel):
    """One image to build."""

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
        path = PurePosixPath(self.dockerfile)
        if path.is_absolute():
            raise ValueError(f"dockerfile must be relative to the context, got {self.dockerfile!r}")
        if any(part == ".." for part in path.parts):
            raise ValueError(f"dockerfile must not escape the context with '..', got {self.dockerfile!r}")
        return self


class BuildSet(BaseModel):
    """A named, revisioned group of images to build in one job."""

    name: str = Field(
        description=(
            "What is being built -- a task name, an agent name. Reused on every rebuild: it is "
            "what makes a second submission supersede the first rather than race it."
        )
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
    build_specs: list[BuildSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _spec_names_are_unique(self) -> Self:
        names = [spec.name for spec in self.build_specs]
        if len(names) != len(set(names)):
            raise ValueError("build_spec names must be unique within a set")
        return self
