# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A build request resolved against this deployment, once.

Both halves of a submit read from the same :class:`BuildPlan`: the compiler turns it into a job,
and the submit path turns it into ``ContainerImage`` rows. Neither re-derives a name, a
destination or a tag, so the two cannot disagree about which image is which -- the job pushes the
tag each row will resolve because both read it from the same record.

**Every check that can reject a request runs here.** What comes out is facts, not input: the
compiler and the row builder have nothing left to validate.
"""

from __future__ import annotations

from dataclasses import dataclass

from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.identity import compose_system_tag, validate_repository
from nemo_builder_plugin.schema import BuildSet, BuildSpec
from nemo_builder_plugin.steps import ContextSource


class BuildCompileError(ValueError):
    """This deployment cannot build the request."""


@dataclass(frozen=True, slots=True)
class PlannedImage:
    """One image, with everything about it decided at submit."""

    index: int
    spec: BuildSpec
    #: The ``ContainerImage`` row name, ``<job>-<index>``.
    name: str
    registry: str
    repository: str
    system_tag: str

    @property
    def source(self) -> ContextSource:
        return ContextSource(fileset=self.spec.source.fileset, context_path=self.spec.source.context_path or None)

    @property
    def caller_ref(self) -> str:
        return f"{self.registry}/{self.repository}:{self.spec.output.tag}"

    @property
    def system_ref(self) -> str:
        return f"{self.registry}/{self.repository}:{self.system_tag}"


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """What will be built, where it goes, and what it will be called. Built by :meth:`resolve`."""

    build_set: BuildSet
    workspace: str
    #: ``<set>-<revision>``. Deterministic from the request, which is what lets concurrent
    #: submitters settle on create-or-get against the unique name index.
    job_name: str
    images: tuple[PlannedImage, ...]
    #: The three settings with no safe default, proven present. See `_require_buildable`.
    registry: str
    push_credential_secret: str
    signing_key: str

    @classmethod
    def resolve(cls, build_set: BuildSet, *, config: BuilderConfig, workspace: str) -> BuildPlan:
        """Raises :class:`BuildCompileError` if this deployment cannot build anything, or cannot
        build this; ``ValueError`` if the request contradicts itself."""
        registry, push_credential_secret, signing_key = _require_buildable(config)
        job_name = f"{build_set.name}-{build_set.revision}"
        images = tuple(
            _plan_image(
                index,
                spec,
                registry=registry,
                repository_prefix=config.repository_prefix,
                workspace=workspace,
                build_set=build_set,
                job_name=job_name,
            )
            for index, spec in enumerate(build_set.build_specs)
        )
        _require_distinct_destinations(images)
        return cls(
            build_set=build_set,
            workspace=workspace,
            job_name=job_name,
            images=images,
            registry=registry,
            push_credential_secret=push_credential_secret,
            signing_key=signing_key,
        )

    def groups(self) -> list[tuple[ContextSource, tuple[PlannedImage, ...]]]:
        """Images grouped by source, in first-appearance order. One sandbox per group.

        One sandbox per distinct ``(fileset, context_path)``, so a Dockerfile can never read the
        context of a different source in the same set. Images sharing a source share a sandbox,
        which is the common shape and also the cheap one.
        """
        grouped: dict[tuple[str, str | None], list[PlannedImage]] = {}
        for image in self.images:
            source = image.source
            grouped.setdefault((source.fileset, source.context_path), []).append(image)
        return [
            (ContextSource(fileset=fileset, context_path=context_path), tuple(images))
            for (fileset, context_path), images in grouped.items()
        ]


def _require_buildable(config: BuilderConfig) -> tuple[str, str, str]:
    """Reject a deployment that cannot build at all, before looking at what was asked for.

    A deployment missing its registry, its credential or its signing key fails the submit with an
    error the caller can act on, rather than producing a job that dies in a pod later.
    """
    if not config.execution_enabled:
        raise BuildCompileError(
            "the execution backend is disabled on this deployment (builder.execution_enabled). "
            "Existing images remain readable; no new builds will be accepted."
        )
    if not config.registry:
        raise BuildCompileError("builder.registry is not configured; there is nowhere to publish the result")
    if not config.push_credential_secret:
        raise BuildCompileError("builder.push_credential_secret is not configured; nothing could publish the result")
    if not config.signing_key:
        raise BuildCompileError(
            "builder.signing_key is not configured. Signing is required for everything this "
            "system builds, so an unconfigured key fails the compile rather than publishing "
            "unsigned output."
        )
    return config.registry, config.push_credential_secret, config.signing_key


def _plan_image(
    index: int,
    spec: BuildSpec,
    *,
    registry: str,
    repository_prefix: str,
    workspace: str,
    build_set: BuildSet,
    job_name: str,
) -> PlannedImage:
    """Resolve one spec's destination and identity.

    The destination is composed, never chosen: the deployment's registry, and the repository
    ``<repository_prefix>/<workspace>/<output.repository>``. **The workspace component is the
    isolation between tenants.** Every image is pushed with one operator credential that can
    write anywhere under the prefix, so the registry cannot tell workspaces apart -- only this
    can. ``output.repository`` is checked component by component at the schema, so it cannot
    climb out of the workspace's part of the path.

    ``registry`` is a HOST -- what a registry client connects to -- and ``repository`` a path
    within it; they are kept apart because ``https://host/project/repo/v2/...`` is what a joined
    string produces the moment anything resolves it.
    """
    repository = "/".join(part for part in (repository_prefix, workspace, spec.output.repository) if part)
    # Checked whole, because the workspace is a platform name and `NAME_PATTERN` admits spellings
    # -- `@`, `+`, `..` -- that a repository component does not.
    validate_repository(repository)

    return PlannedImage(
        index=index,
        spec=spec,
        # A tracking handle, not a label: stable within a revision, meaningless across them.
        # Not derived from `output.repository`, which is arbitrary, caller-owned, and may be
        # longer than a name may be.
        name=f"{job_name}-{index}",
        registry=registry,
        repository=repository,
        # Per image, not per set: two images in one repository must not push one tag.
        system_tag=compose_system_tag(workspace, build_set.name, build_set.revision, index),
    )


def _require_distinct_destinations(images: tuple[PlannedImage, ...]) -> None:
    """Refuse two specs publishing the same caller reference: the tag could point at only one.

    A plain ``ValueError`` -- a malformed request (400), not a deployment that cannot satisfy a
    well-formed one (409).
    """
    seen: dict[str, str] = {}
    for image in images:
        if image.caller_ref in seen:
            raise ValueError(
                f"build specs {seen[image.caller_ref]!r} and {image.spec.name!r} both publish "
                f"{image.caller_ref}; the tag could point at only one of them"
            )
        seen[image.caller_ref] = image.spec.name
