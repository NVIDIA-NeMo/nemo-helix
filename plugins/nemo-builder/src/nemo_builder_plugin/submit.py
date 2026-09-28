# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submit path: rows first, then the job.

Separated from ``service.py`` so the ordering below is testable against fakes rather than only
through a running FastAPI app. The route is a thin adapter over :func:`submit_build_set`.

**The order is the design, not an implementation detail.** Rows are created ``pending`` *before*
the job exists. An earlier shape created the job first and wrote rows on completion, which leaves
a window where a pushed image has nothing pointing at it -- and closing that window needs a sweep
over terminal jobs. Creating the row first means the pointer always exists and the only question
is what state it is in.

**An import is resolved before anything is written.** Its upstream reference is read from the
upstream registry -- the one piece of I/O a submit does besides writing -- and the platform
manifest it names is recorded on the row. That value is what the reconciler later checks a copy
against, so it must come from here, not from any step. A reference that does not resolve is a 400,
before any row exists.

**Concurrency is settled by the database, not by a check.** Row names are deterministic from the
request (``<set>-<revision>-<index>``), so two submitters racing on the same ``(name, revision)``
collide on the entity store's unique index. There is no read-then-write here, because a
read-then-write across replicas is a race.

**A collision is adopted only when it is the same request.** Every row records a digest of the
request that wrote it (``JobOrigin.request_digest``). The same request again -- a client retry
after a lost response, a double submit, a concurrent twin -- adopts the rows and the job and
returns them. A *different* request under a reused revision is refused with a conflict: adopting
its rows would leave each one describing a request the job never built.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from nemo_builder_plugin.compile import compile_build_set
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, JobOrigin, Provenance, UpstreamImage
from nemo_builder_plugin.identity import ImageReference
from nemo_builder_plugin.plan import BuildCompileError, BuildPlan, PlannedImage
from nemo_builder_plugin.registry import ReferenceNotFound
from nemo_builder_plugin.schema import BuildSet
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities import EntityConflictError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

logger = logging.getLogger(__name__)

#: ``CreateHelixJobRequest.source`` -- how the reconciler recognises its own jobs.
JOB_SOURCE = "builder.build"


class _EntityWriter(Protocol):
    """The two operations this function needs, and no more.

    Narrower than ``EntityClient`` on purpose: depending on the full client would make a test
    double implement a surface it never exercises, which is how fakes drift from the thing they
    stand in for.
    """

    async def create(self, entity: ContainerImage) -> ContainerImage: ...

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage: ...


#: Injected so the submit path stays testable without a platform; it is the one piece that must
#: talk to Jobs.
CreateJob = Callable[[CreateHelixJobRequest], Awaitable[object]]
#: The ``custom_fields`` of an existing job, by name. Read only when creating the job collided.
GetJobFields = Callable[[str], Awaitable[Mapping[str, Any]]]


class InvalidBuildRequest(Exception):
    """The request contradicts itself or names something unusable. The caller's to fix (400).

    Deliberately not a ``ValueError``: a route catching ``ValueError`` would also catch every
    validation error raised while talking to other services, and report the platform's failure
    as the caller's.
    """


class BuildConflict(Exception):
    """This ``(name, revision)`` was already submitted with a different request (409)."""


def request_digest(build_set: BuildSet) -> str:
    """sha256 of the request's canonical JSON: defaults included, keys sorted.

    Defaults are included so that two spellings of one request -- one naming ``type:
    fileset``, one leaving it out -- are one request.
    """
    canonical = json.dumps(build_set.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


#: Resolves an import's upstream reference to the manifest digest for a platform. Injected for the
#: same reason as `CreateJob`: it is the one piece of the submit that reads a registry.
ResolveUpstream = Callable[[ImageReference, str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class SubmitResult:
    """What the caller gets back.

    The caller polls the **images**, not the job. A set of ten produces ten rows that reach
    ``ready`` independently, and a job status cannot express that -- a job that exited non-zero
    may still have pushed some of its images, because the build step attempts every spec and does
    not abort the set.
    """

    job: str
    images: list[ContainerImage]


def _upstream(image: PlannedImage) -> UpstreamImage | None:
    if image.upstream is None or image.upstream_manifest_digest is None:
        return None
    return UpstreamImage(image_ref=image.upstream.normalized_ref, manifest_digest=image.upstream_manifest_digest)


async def _resolve_imports(plan: BuildPlan, resolve_upstream: ResolveUpstream | None) -> BuildPlan:
    """Fill in every import's platform manifest, or refuse the request."""
    if not plan.imports:
        return plan
    if resolve_upstream is None:
        raise RuntimeError("this submit path has no upstream resolver, so it cannot accept imports")
    manifests: dict[str, str] = {}
    for image in plan.imports:
        assert image.upstream is not None
        try:
            manifests[image.name] = await resolve_upstream(image.upstream, image.spec.platform)
        except ReferenceNotFound as exc:
            raise ValueError(f"build spec {image.spec.name!r}: {exc}") from exc
    return plan.with_upstream_manifests(manifests)


def _rows_for(plan: BuildPlan, digest: str) -> list[ContainerImage]:
    """The desired state, before anything is built."""
    runtime_layer = plan.runtime_layer.label if plan.runtime_layer else None
    return [
        ContainerImage(
            name=image.name,
            workspace=plan.workspace,
            registry=image.registry,
            repository=image.repository,
            platform=image.spec.platform,
            provenance=Provenance(
                backend="execution",
                built_by=JobOrigin(
                    build_set=plan.build_set.name,
                    revision=plan.build_set.revision,
                    job=f"{plan.workspace}/{plan.job_name}",
                    system_tag=image.system_tag,
                    request_digest=digest,
                    upstream=_upstream(image),
                    runtime_layer=runtime_layer,
                ),
            ),
        )
        for image in plan.images
    ]


async def submit_build_set(
    build_set: BuildSet,
    *,
    config: BuilderConfig,
    workspace: str,
    entity_client: _EntityWriter,
    create_job: CreateJob,
    get_job_fields: GetJobFields,
    resolve_upstream: ResolveUpstream | None = None,
) -> SubmitResult:
    """Resolve, compile, write the rows, create the job. In that order."""
    # Resolve and compile FIRST. Resolving is where the request can be rejected for a reason the
    # caller can act on, so it runs before anything is written. The rows and the job are both
    # projections of this one plan.
    try:
        plan = BuildPlan.resolve(build_set, config=config, workspace=workspace)
        plan = await _resolve_imports(plan, resolve_upstream)
        platform_spec = compile_build_set(plan, config=config)
    except BuildCompileError:
        raise
    except ValueError as exc:
        raise InvalidBuildRequest(str(exc)) from exc
    digest = request_digest(build_set)

    images: list[ContainerImage] = []
    for row in _rows_for(plan, digest):
        try:
            images.append(await entity_client.create(row))
        except EntityConflictError:
            images.append(await _adopt(entity_client, row, digest, build_set, workspace))

    try:
        await create_job(
            CreateHelixJobRequest(
                name=plan.job_name,
                description=f"Container image build for {build_set.name} revision {build_set.revision}",
                source=JOB_SOURCE,
                spec=build_set.model_dump(mode="json"),
                platform_spec=platform_spec,
                custom_fields={"images": [image.name for image in images], "request_digest": digest},
            )
        )
    except ConflictError:
        # The job exists. Every row above is this request's -- created, or adopted with a matching
        # digest -- so the job is ours too if it recorded the same digest: a retry, or a
        # concurrent twin that got there first. Anything else holds the name, and these rows
        # must not be reconciled against it.
        existing = await get_job_fields(plan.job_name)
        if existing.get("request_digest") != digest:
            raise BuildConflict(
                f"a job named {plan.job_name!r} already exists and was not created by this request"
            ) from None
        logger.info("job %s already exists for this request; adopting it", plan.job_name)

    logger.info(
        "submitted build set %s revision %s as job %s with %d image(s)",
        build_set.name,
        build_set.revision,
        plan.job_name,
        len(images),
    )
    return SubmitResult(job=plan.job_name, images=images)


async def _adopt(
    entity_client: _EntityWriter, row: ContainerImage, digest: str, build_set: BuildSet, workspace: str
) -> ContainerImage:
    """The existing row, if the same request wrote it. Otherwise a conflict, and nothing adopted."""
    existing = await entity_client.get(ContainerImage, name=row.name, workspace=workspace)
    origin = existing.provenance.built_by
    if not isinstance(origin, JobOrigin) or origin.request_digest != digest:
        raise BuildConflict(
            f"build set {build_set.name!r} revision {build_set.revision} was already submitted with a "
            "different request. Submit a new revision; a revision names one request."
        )
    return existing
