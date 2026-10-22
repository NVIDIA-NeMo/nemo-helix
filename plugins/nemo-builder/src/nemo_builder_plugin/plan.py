# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A build request, resolved once, so the job and its rows agree on every name and destination."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from nemo_builder_plugin.identity import ImageReference, normalize_reference
from nemo_builder_plugin.schema import JOB_NAME_PREFIX, BuildSet, BuildSpec, FileSetSource, ImageSource
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
    #: For a copy, the manifest digest its image named for the spec's platform when the build was submitted.
    upstream_digest: str | None = None

    @property
    def platform(self) -> str:
        if self.spec.platform is None:
            raise ValueError(f"{self.name} names no platform; plan the set after with_default_platform()")
        return self.spec.platform

    @property
    def upstream(self) -> ImageReference | None:
        """For a copy, the image it copies; None for a build."""
        source = self.spec.source
        return normalize_reference(source.image) if isinstance(source, ImageSource) else None

    @property
    def pinned_upstream(self) -> str:
        """``<registry>/<repository>@<manifest digest>``, the one manifest a copy pulls."""
        upstream = self.upstream
        if upstream is None or self.upstream_digest is None:
            raise ValueError(f"{self.name} copies no resolved image; call plan.with_upstream() first")
        return f"{upstream.registry}/{upstream.repository}@{self.upstream_digest}"

    @property
    def source(self) -> ContextSource:
        source = self.spec.source
        if not isinstance(source, FileSetSource):
            raise ValueError(f"{self.name} copies an image and has no build context")
        return ContextSource(
            fileset=source.fileset if "/" in source.fileset else f"{self.workspace}/{source.fileset}",
            archive=source.archive or None,
            context_path=source.context_path or None,
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
        base = f"{build_set.name}-{build_set.revision}"
        images = tuple(
            PlannedImage(
                spec=spec,
                workspace=workspace,
                build_set=build_set.name,
                name=f"{base}.{spec.name}" if spec.name else base,
            )
            for spec in build_set.build_specs
        )
        return cls(build_set=build_set, workspace=workspace, job_name=f"{JOB_NAME_PREFIX}{base}", images=images)

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

    def with_upstream(self, manifest_digests: Mapping[str, str]) -> BuildPlan:
        """Pin each copy to the manifest digest resolved for it, keyed by image name."""
        return replace(
            self,
            images=tuple(
                replace(image, upstream_digest=manifest_digests[image.name]) if image.upstream else image
                for image in self.images
            ),
        )

    @property
    def copies(self) -> tuple[PlannedImage, ...]:
        return tuple(image for image in self.images if image.upstream is not None)

    def groups(self) -> list[tuple[ContextSource, tuple[PlannedImage, ...]]]:
        """Each group builds in its own sandbox, so a Dockerfile never sees another source's context."""
        grouped: dict[tuple[str, str | None, str | None], list[PlannedImage]] = {}
        for image in self.images:
            if image.upstream is not None:
                continue
            source = image.source
            grouped.setdefault((source.fileset, source.archive, source.context_path), []).append(image)
        return [
            (ContextSource(fileset=fileset, archive=archive, context_path=context_path), tuple(images))
            for (fileset, archive, context_path), images in grouped.items()
        ]
