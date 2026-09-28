# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A build request resolved against this deployment, once.

Both halves of a submit read from the same :class:`BuildPlan`: the compiler turns it into a job,
and the submit path turns it into ``ContainerImage`` rows. Neither re-derives a name, a
destination or a tag, so the two cannot disagree about which image is which -- the job pushes the
tag each row will resolve because both read it from the same record.

**Every check that can reject a request without asking another service runs here** -- or in
the request schema, before it. What comes out is facts, not input: the compiler and the row
builder have nothing left to validate. What only Files can answer -- does this fileset exist, may
this caller read it -- is answered by ``fetch``, as the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.identity import (
    ImageIdentityError,
    ImageReference,
    compose_system_tag,
    normalize_pinned_reference,
    require_allowed_registry,
    validate_repository,
)
from nemo_builder_plugin.schema import BuildSet, BuildSpec, FileSetSource, ImageSource
from nemo_builder_plugin.steps import ContextSource, RuntimeLayerSpec


class BuildCompileError(ValueError):
    """This deployment cannot build the request."""


def qualify_fileset(fileset: str, workspace: str) -> str:
    """``<workspace>/<name>``, whichever way the request spelled it.

    One spelling per fileset: ``fs`` and ``default/fs`` are one download, and no fileset's
    directory on the work volume can nest inside another's.
    """
    return fileset if "/" in fileset else f"{workspace}/{fileset}"


@dataclass(frozen=True, slots=True)
class PlannedImage:
    """One image, with everything about it decided at submit."""

    index: int
    spec: BuildSpec
    #: The submitting workspace, which an unqualified fileset name belongs to.
    workspace: str
    #: The ``ContainerImage`` row name, ``<job>-<index>``.
    name: str
    registry: str
    repository: str
    system_tag: str
    #: For an import: the upstream reference the caller named, normalized. None for a build.
    upstream: ImageReference | None = None
    #: For an import: the platform manifest the submit path resolved. Filled in by
    #: :meth:`BuildPlan.with_upstream_manifests`, the one step of a submit that reads a registry.
    upstream_manifest_digest: str | None = None

    @property
    def source(self) -> ContextSource | None:
        """The fileset context this image builds from, or None for an import."""
        source = self.spec.source
        if isinstance(source, FileSetSource):
            return ContextSource(
                fileset=qualify_fileset(source.fileset, self.workspace),
                context_path=source.context_path or None,
            )
        return None

    @property
    def pinned_upstream(self) -> str:
        """``<registry>/<repository>@<platform manifest digest>`` -- what an import pulls."""
        if self.upstream is None or self.upstream_manifest_digest is None:
            raise ValueError(f"{self.name} is not a resolved import")
        return f"{self.upstream.repository_ref}@{self.upstream_manifest_digest}"

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
    #: The layer of the runtime the set names, or None. It is what makes an import a derived
    #: build rather than a copy, so it is decided here, once, for every image in the set.
    runtime_layer: RuntimeLayerSpec | None = None

    @classmethod
    def resolve(cls, build_set: BuildSet, *, config: BuilderConfig, workspace: str) -> BuildPlan:
        """Raises :class:`BuildCompileError` if this deployment cannot build anything, or cannot
        build this; ``ValueError`` if the request contradicts itself."""
        registry, push_credential_secret, signing_key = _require_buildable(config)
        runtime_layer = _runtime_layer(build_set.runtime, config)
        job_name = f"{build_set.name}-{build_set.revision}"
        images = tuple(
            _plan_image(
                index,
                spec,
                registry=registry,
                repository_prefix=config.repository_prefix,
                import_registries=config.import_registries,
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
            runtime_layer=runtime_layer,
        )

    def is_copy(self, image: PlannedImage) -> bool:
        """An import published byte for byte. Copies never reach the sandbox: nothing executes."""
        return image.upstream is not None and self.runtime_layer is None

    @property
    def imports(self) -> tuple[PlannedImage, ...]:
        return tuple(image for image in self.images if image.upstream is not None)

    def groups(self) -> list[tuple[ContextSource | None, tuple[PlannedImage, ...]]]:
        """What the sandbox builds, grouped, in first-appearance order. One sandbox per group.

        One sandbox per distinct ``(fileset, context_path)``, so a Dockerfile can never read the
        context of a different source in the same set. Images sharing a source share a sandbox,
        which is the common shape and also the cheap one.

        Derived imports come last, **one group each** (``None``). Each context is a Dockerfile
        ``fetch`` wrote, holding nothing secret -- but each build runs its upstream publisher's
        binaries, and in a shared sandbox one publisher's could rewrite another's image before it
        is signed. Copies are in no group at all -- nothing about them runs.
        """
        grouped: dict[tuple[str, str | None], list[PlannedImage]] = {}
        derived: list[PlannedImage] = []
        for image in self.images:
            source = image.source
            if source is not None:
                grouped.setdefault((source.fileset, source.context_path), []).append(image)
            elif not self.is_copy(image):
                derived.append(image)
        groups: list[tuple[ContextSource | None, tuple[PlannedImage, ...]]] = [
            (ContextSource(fileset=fileset, context_path=context_path), tuple(images))
            for (fileset, context_path), images in grouped.items()
        ]
        groups.extend((None, (image,)) for image in derived)
        return groups

    def with_upstream_manifests(self, manifests: dict[str, str]) -> BuildPlan:
        """The plan with every import's platform manifest filled in, keyed by image name.

        The one piece of a plan that needs I/O -- a registry read -- so it is applied to a
        resolved plan rather than done inside :meth:`resolve`, which stays pure.
        """
        missing = [image.name for image in self.imports if image.name not in manifests]
        if missing:
            raise ValueError(f"no upstream manifest resolved for {', '.join(missing)}")
        images = tuple(
            replace(image, upstream_manifest_digest=manifests[image.name]) if image.upstream else image
            for image in self.images
        )
        return replace(self, images=images)


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


def _runtime_layer(name: str | None, config: BuilderConfig) -> RuntimeLayerSpec | None:
    """The layer the set's runtime names, or None. An unknown name is a deployment gap (409)."""
    if name is None:
        return None
    layer = config.runtime_layers.get(name)
    if layer is None:
        known = ", ".join(sorted(config.runtime_layers)) or "none"
        raise BuildCompileError(f"runtime {name!r} is not configured on this deployment (configured: {known})")
    return RuntimeLayerSpec(name=name, version=layer.version, dockerfile=layer.dockerfile)


def _upstream(spec: BuildSpec, import_registries: list[str]) -> ImageReference | None:
    """An import's upstream reference, refused unless its registry is allowed.

    Checked here, before any row exists, and against the NORMALIZED reference: the schema already
    expanded `alpine@...` to `docker.io/library/alpine@...`, so the spelling cannot choose the
    verdict. Refused as a deployment gap (409) rather than a malformed request: the reference is
    well-formed, and this deployment will not sign images from there.
    """
    if not isinstance(spec.source, ImageSource):
        return None
    reference = normalize_pinned_reference(spec.source.image)
    if not import_registries:
        raise BuildCompileError(
            f"build spec {spec.name!r} imports {reference.normalized_ref}, but imports are disabled on "
            "this deployment (builder.import_registries is empty)"
        )
    try:
        return require_allowed_registry(reference, import_registries)
    except ImageIdentityError as exc:
        raise BuildCompileError(f"build spec {spec.name!r}: {exc}") from exc


def _plan_image(
    index: int,
    spec: BuildSpec,
    *,
    registry: str,
    repository_prefix: str,
    import_registries: list[str],
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
        workspace=workspace,
        # A tracking handle, not a label: stable within a revision, meaningless across them.
        # Not derived from `output.repository`, which is arbitrary, caller-owned, and may be
        # longer than a name may be.
        name=f"{job_name}-{index}",
        registry=registry,
        repository=repository,
        # Per image, not per set: two images in one repository must not push one tag.
        system_tag=compose_system_tag(workspace, build_set.name, build_set.revision, index),
        upstream=_upstream(spec, import_registries),
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
