# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submit path: rows first, then the job, so every image the job pushes already has a row."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from nemo_builder_plugin.backend import Backend, BackendRejectedError
from nemo_builder_plugin.entities import ContainerImage, Provenance, QueuedBuild, UpstreamImage
from nemo_builder_plugin.identity import ImageReference
from nemo_builder_plugin.plan import BuildPlan, PlannedImage
from nemo_builder_plugin.schema import BuildSet
from nemo_builder_plugin.upstream import UpstreamError, UpstreamNotFound
from nemo_helix_plugin.auth import current_auth_context
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities import EntityConflictError, EntityNotFoundError
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

JOB_SOURCE = "builder.build"


_Row = TypeVar("_Row", ContainerImage, QueuedBuild)


class _EntityWriter(Protocol):
    async def create(self, entity: _Row) -> _Row: ...

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage: ...

    async def update(self, entity: ContainerImage) -> ContainerImage: ...


CreateJob = Callable[[CreateHelixJobRequest], Awaitable[object]]
#: The ``custom_fields`` of the job with this name.
GetJobFields = Callable[[str], Awaitable[Mapping[str, Any]]]
#: Whether the caller can read the platform secret ``(workspace, name)``.
SecretReadable = Callable[[str, str], Awaitable[bool]]

#: The manifest digest an image reference names for a platform. Raises :class:`UpstreamNotFound` or
#: :class:`UpstreamError`.
ResolveUpstream = Callable[[ImageReference, str], Awaitable[str]]


class InvalidBuildRequest(Exception):
    """The request contradicts itself or names something unusable (400).

    Not a ``ValueError``: a route catching those would also catch the platform's own errors.
    """


class BuildConflict(Exception):
    """This ``(name, revision)`` was submitted with a different request, or its job name is taken (409)."""


class UpstreamUnavailable(Exception):
    """An image to copy couldn't be resolved because its registry failed (502)."""


class MissingSecrets(Exception):
    """The job would read platform secrets that don't exist, or that the caller can't read (409)."""


def referenced_secrets(spec: HelixJobSpec, workspace: str) -> list[tuple[str, str]]:
    """Each platform secret ``spec`` reads, as ``(workspace, name)``. A bare name is in the job's workspace."""
    found: list[tuple[str, str]] = []
    for step in spec.steps:
        for variable in step.environment or []:
            if variable.from_secret is None:
                continue
            secret_workspace, _, name = variable.from_secret.name.rpartition("/")
            if (ref := (secret_workspace or workspace, name)) not in found:
                found.append(ref)
    return found


def request_digest(build_set: BuildSet) -> str:
    """sha256 of the request's canonical JSON, defaults included, so two spellings of one request match."""
    canonical = json.dumps(build_set.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SubmitResult:
    job: str
    images: list[ContainerImage]


def _rows_for(plan: BuildPlan, digest: str, status: Literal["queued", "pending"]) -> list[ContainerImage]:
    return [
        ContainerImage(
            name=image.name,
            workspace=plan.workspace,
            status=status,
            registry=image.placed.registry,
            repository=image.placed.repository,
            provenance=Provenance(
                build_set=plan.build_set.name,
                revision=plan.build_set.revision,
                spec=image.spec.name,
                job=f"{plan.workspace}/{plan.job_name}",
                request_digest=digest,
                upstream=_upstream_image(image),
            ),
        )
        for image in plan.images
    ]


def _upstream_image(image: PlannedImage) -> UpstreamImage | None:
    if image.upstream is None or image.upstream_digest is None:
        return None
    return UpstreamImage(image=str(image.upstream), manifest_digest=image.upstream_digest)


async def _resolve_copies(
    plan: BuildPlan, entity_client: _EntityWriter, resolve_upstream: ResolveUpstream, build_set: BuildSet, digest: str
) -> BuildPlan:
    """Pin each copy to a manifest digest: the one its row already holds, or the registry's now.

    A resubmit of the same request is the same build, so it keeps the digest pinned first, without asking the
    registry again: a tag that moved since then, or a registry that refuses (a rate limit, say), changes nothing.
    """
    digests: dict[str, str] = {}
    for image in plan.copies:
        upstream = image.upstream
        assert upstream is not None
        pinned = await _pinned_digest(entity_client, image.name, plan.workspace, build_set, digest)
        if pinned is not None:
            digests[image.name] = pinned
            continue
        try:
            digests[image.name] = await resolve_upstream(upstream, image.platform)
        except UpstreamNotFound as exc:
            raise InvalidBuildRequest(f"{image.name}: {exc}") from exc
        except UpstreamError as exc:
            raise UpstreamUnavailable(f"{image.name}: {exc}") from exc
    return plan.with_upstream(digests)


async def _pinned_digest(
    entity_client: _EntityWriter, name: str, workspace: str, build_set: BuildSet, digest: str
) -> str | None:
    """The manifest digest the copy's row already pinned, if it has a row.

    A row written for a different request makes the submit a conflict, whatever the registry would say.
    """
    try:
        row = await entity_client.get(ContainerImage, name=name, workspace=workspace)
    except EntityNotFoundError:
        return None
    if row.provenance.request_digest != digest:
        raise _different_request(build_set)
    upstream = row.provenance.upstream
    return upstream.manifest_digest if upstream is not None else None


async def submit_build_set(
    build_set: BuildSet,
    *,
    backend: Backend,
    workspace: str,
    entity_client: _EntityWriter,
    create_job: CreateJob,
    get_job_fields: GetJobFields,
    secret_readable: SecretReadable,
    resolve_upstream: ResolveUpstream,
    default_platform: str,
    queue: bool = False,
) -> SubmitResult:
    """Resolve, check, place, compile; then write the rows and create the job. In that order.

    With ``queue``, the rows are written ``queued`` and the job is left to the queue controller as a
    :class:`QueuedBuild`, rather than created here.
    """
    # Filled in before anything reads the set, the request digest included: a spec that names no platform is the
    # same request as one naming the default, as it was when the default was fixed at linux/amd64.
    build_set = build_set.with_default_platform(default_platform)
    digest = request_digest(build_set)
    # Everything that can refuse the request runs before anything is written.
    try:
        plan = BuildPlan.resolve(build_set, workspace=workspace)
        backend.check(plan)
        plan = await _resolve_copies(plan, entity_client, resolve_upstream, build_set, digest)
        plan = plan.with_destinations(backend.destination)
        platform_spec = backend.compile(plan)
    except BackendRejectedError:
        raise
    except ValueError as exc:
        raise InvalidBuildRequest(str(exc)) from exc
    # Jobs checks these too, but only once the rows exist, and its refusal looks like a taken job name.
    missing = [
        f"{secret_workspace}/{name}"
        for secret_workspace, name in referenced_secrets(platform_spec, workspace)
        if not await secret_readable(secret_workspace, name)
    ]
    if missing:
        raise MissingSecrets(
            "the push step needs platform secrets that don't exist, or that you can't read: " + ", ".join(missing)
        )

    images: list[ContainerImage] = []
    created: list[ContainerImage] = []
    for row in _rows_for(plan, digest, "queued" if queue else "pending"):
        try:
            created.append(await entity_client.create(row))
            images.append(created[-1])
        except EntityConflictError:
            try:
                images.append(await _adopt(entity_client, row, digest, build_set, workspace))
            except BuildConflict:
                await fail_rows(
                    entity_client,
                    created,
                    f"not built: {build_set.name} revision {build_set.revision} belongs to a different request",
                )
                raise

    request = CreateHelixJobRequest(
        name=plan.job_name,
        description=f"Container image build for {build_set.name} revision {build_set.revision}",
        source=JOB_SOURCE,
        spec=build_set.model_dump(mode="json"),
        platform_spec=platform_spec,
        custom_fields={"images": [image.name for image in images], "request_digest": digest},
    )
    if queue:
        # Rows adopted already past ``queued`` mean the job exists; a resubmit of a queued one is queued already.
        if any(image.status == "queued" for image in images):
            queued = QueuedBuild(
                name=plan.job_name,
                workspace=workspace,
                request=request.model_dump(mode="json"),
                images=[image.name for image in images],
            )
            queued._auth_context = current_auth_context()
            try:
                await entity_client.create(queued)
            except EntityConflictError:
                logger.info("build %s is already queued", sanitize_for_log(plan.job_name))
        return SubmitResult(job=plan.job_name, images=images)

    try:
        await create_job(request)
    except ConflictError:
        existing = await get_job_fields(plan.job_name)
        if existing.get("request_digest") != digest:
            await fail_rows(
                entity_client, images, f"not built: job {plan.job_name!r} is held by a job this request did not create"
            )
            raise BuildConflict(
                f"a job named {plan.job_name!r} already exists and was not created by this request"
            ) from None
        logger.info("job %s already exists for this request; adopting it", sanitize_for_log(plan.job_name))

    logger.info(
        "submitted build set %s revision %s as job %s with %d image(s)",
        sanitize_for_log(build_set.name),
        build_set.revision,
        sanitize_for_log(plan.job_name),
        len(images),
    )
    return SubmitResult(job=plan.job_name, images=images)


async def fail_rows(entity_client: _EntityWriter, rows: list[ContainerImage], detail: str) -> None:
    """Fail each of ``rows`` still waiting, rather than leave it waiting on a job this submit didn't create.

    One that changed since it was read, or can't be written, is logged and left.
    """
    for row in rows:
        if row.status not in ("queued", "pending"):
            continue
        row.status = "failed"
        row.status_detail = detail
        try:
            await entity_client.update(row)
        except EntityConflictError:
            logger.warning(
                "image %s/%s changed while being failed; leaving it as it is",
                sanitize_for_log(row.workspace),
                sanitize_for_log(row.name),
            )
        except Exception:
            logger.exception(
                "image %s/%s could not be failed, and stays pending",
                sanitize_for_log(row.workspace),
                sanitize_for_log(row.name),
            )


async def _adopt(
    entity_client: _EntityWriter, row: ContainerImage, digest: str, build_set: BuildSet, workspace: str
) -> ContainerImage:
    existing = await entity_client.get(ContainerImage, name=row.name, workspace=workspace)
    if existing.provenance.request_digest != digest:
        raise _different_request(build_set)
    return existing


def _different_request(build_set: BuildSet) -> BuildConflict:
    return BuildConflict(
        f"build set {build_set.name!r} revision {build_set.revision} was already submitted with a "
        "different request. Submit a new revision; a revision names one request."
    )
