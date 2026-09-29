# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A build request, resolved once.

The backend compiles the job from a :class:`BuildPlan`, and the submit route writes the rows from
the same plan, so the two agree on every name, destination and tag.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from nemo_builder_plugin.schema import BuildSet, BuildSpec
from nemo_builder_plugin.steps import ContextSource


@dataclass(frozen=True, slots=True)
class Destination:
    """Where one image is published."""

    #: Registry host.
    registry: str
    #: Repository path within the registry.
    repository: str
    #: The tag the build pushes and the broker signs.
    system_tag: str

    @property
    def system_ref(self) -> str:
        return f"{self.registry}/{self.repository}:{self.system_tag}"


@dataclass(frozen=True, slots=True)
class PlannedImage:
    """One image to build."""

    index: int
    spec: BuildSpec
    #: The submitting workspace.
    workspace: str
    build_set: str
    revision: int
    #: The ``ContainerImage`` row name, ``<job>-<index>``.
    name: str
    #: Set by :meth:`BuildPlan.with_destinations`.
    destination: Destination | None = None

    @property
    def source(self) -> ContextSource:
        """The spec's source, its fileset qualified with the submitting workspace if it isn't already."""
        fileset = self.spec.source.fileset
        return ContextSource(
            fileset=fileset if "/" in fileset else f"{self.workspace}/{fileset}",
            context_path=self.spec.source.context_path or None,
        )

    @property
    def placed(self) -> Destination:
        if self.destination is None:
            raise ValueError(f"{self.name} has no destination yet; call plan.with_destinations() first")
        return self.destination

    @property
    def caller_ref(self) -> str | None:
        """``<registry>/<repository>:<output.tag>``, or None when the spec names no output."""
        if self.spec.output is None:
            return None
        return f"{self.placed.registry}/{self.placed.repository}:{self.spec.output.tag}"


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """What will be built, where each image goes, and what everything is named."""

    build_set: BuildSet
    workspace: str
    #: ``<set>-<revision>``.
    job_name: str
    images: tuple[PlannedImage, ...]

    @classmethod
    def resolve(cls, build_set: BuildSet, *, workspace: str) -> BuildPlan:
        """Name the job and every row. Destinations come later, from the backend."""
        job_name = f"{build_set.name}-{build_set.revision}"
        images = tuple(
            PlannedImage(
                index=index,
                spec=spec,
                workspace=workspace,
                build_set=build_set.name,
                revision=build_set.revision,
                name=f"{job_name}-{index}",
            )
            for index, spec in enumerate(build_set.build_specs)
        )
        return cls(build_set=build_set, workspace=workspace, job_name=job_name, images=images)

    def with_destinations(self, destination: Callable[[PlannedImage], Destination]) -> BuildPlan:
        """Place each image with ``destination``, the backend's rule.

        Raises ``ValueError`` if two specs would publish the same caller reference.
        """
        placed = tuple(replace(image, destination=destination(image)) for image in self.images)
        seen: dict[str, str] = {}
        for image in placed:
            ref = image.caller_ref
            if ref is None:
                continue
            if ref in seen:
                raise ValueError(
                    f"build specs {seen[ref]!r} and {image.spec.name!r} both publish {ref}; "
                    "the tag could point at only one of them"
                )
            seen[ref] = image.spec.name
        return replace(self, images=placed)

    def groups(self) -> list[tuple[ContextSource, tuple[PlannedImage, ...]]]:
        """The images grouped by source, in first-appearance order.

        Each group builds in its own sandbox, so a Dockerfile never sees another source's context.
        """
        grouped: dict[tuple[str, str | None], list[PlannedImage]] = {}
        for image in self.images:
            source = image.source
            grouped.setdefault((source.fileset, source.context_path), []).append(image)
        return [
            (ContextSource(fileset=fileset, context_path=context_path), tuple(images))
            for (fileset, context_path), images in grouped.items()
        ]
