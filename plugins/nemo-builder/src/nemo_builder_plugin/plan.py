# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A build request, resolved once, so the job and its rows agree on every name and destination."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from nemo_builder_plugin.schema import BuildSet, BuildSpec
from nemo_builder_plugin.steps import ContextSource


@dataclass(frozen=True, slots=True)
class Destination:
    registry: str
    repository: str
    #: The tag the build pushes for the image, and no caller can: ``<workspace>--<image>``.
    system_tag: str

    @property
    def system_ref(self) -> str:
        return f"{self.registry}/{self.repository}:{self.system_tag}"


@dataclass(frozen=True, slots=True)
class PlannedImage:
    spec: BuildSpec
    #: The submitting workspace.
    workspace: str
    build_set: str
    #: The ``ContainerImage`` name.
    name: str
    destination: Destination | None = None

    @property
    def source(self) -> ContextSource:
        fileset = self.spec.source.fileset
        return ContextSource(
            fileset=fileset if "/" in fileset else f"{self.workspace}/{fileset}",
            archive=self.spec.source.archive or None,
            context_path=self.spec.source.context_path or None,
        )

    @property
    def placed(self) -> Destination:
        if self.destination is None:
            raise ValueError(f"{self.name} has no destination yet; call plan.with_destinations() first")
        return self.destination

    @property
    def caller_ref(self) -> str | None:
        if self.spec.output is None:
            return None
        return f"{self.placed.registry}/{self.placed.repository}:{self.spec.output.tag}"


@dataclass(frozen=True, slots=True)
class BuildPlan:
    build_set: BuildSet
    workspace: str
    job_name: str
    images: tuple[PlannedImage, ...]

    @classmethod
    def resolve(cls, build_set: BuildSet, *, workspace: str) -> BuildPlan:
        job_name = f"{build_set.name}-{build_set.revision}"
        images = tuple(
            PlannedImage(
                spec=spec,
                workspace=workspace,
                build_set=build_set.name,
                name=f"{job_name}.{spec.name}" if spec.name else job_name,
            )
            for spec in build_set.build_specs
        )
        return cls(build_set=build_set, workspace=workspace, job_name=job_name, images=images)

    def with_destinations(self, destination: Callable[[PlannedImage], Destination]) -> BuildPlan:
        placed = tuple(replace(image, destination=destination(image)) for image in self.images)
        seen: dict[str, str] = {}
        for image in placed:
            ref = image.caller_ref
            if ref is None:
                continue
            if ref in seen:
                raise ValueError(
                    f"images {seen[ref]!r} and {image.name!r} both publish {ref}; the tag could point at only one of them"
                )
            seen[ref] = image.name
        return replace(self, images=placed)

    def groups(self) -> list[tuple[ContextSource, tuple[PlannedImage, ...]]]:
        """Each group builds in its own sandbox, so a Dockerfile never sees another source's context."""
        grouped: dict[tuple[str, str | None, str | None], list[PlannedImage]] = {}
        for image in self.images:
            source = image.source
            grouped.setdefault((source.fileset, source.archive, source.context_path), []).append(image)
        return [
            (ContextSource(fileset=fileset, archive=archive, context_path=context_path), tuple(images))
            for (fileset, archive, context_path), images in grouped.items()
        ]
