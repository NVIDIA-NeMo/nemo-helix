# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submit path: rows first, then the job, so every image the job pushes already has a row."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from nemo_builder_plugin.backend import Backend, BackendRefused
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.schema import BuildSet
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities import EntityConflictError
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

JOB_SOURCE = "builder.build"


class _EntityWriter(Protocol):
    async def create(self, entity: ContainerImage) -> ContainerImage: ...

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage: ...

    async def update(self, entity: ContainerImage) -> ContainerImage: ...


CreateJob = Callable[[CreateHelixJobRequest], Awaitable[object]]
#: The ``custom_fields`` of the job with this name.
GetJobFields = Callable[[str], Awaitable[Mapping[str, Any]]]
#: Whether the caller can read the platform secret ``(workspace, name)``.
SecretReadable = Callable[[str, str], Awaitable[bool]]


class InvalidBuildRequest(Exception):
    """The request contradicts itself or names something unusable (400).

    Not a ``ValueError``: a route catching those would also catch the platform's own errors.
    """


class BuildConflict(Exception):
    """This ``(name, revision)`` was submitted with a different request, or its job name is taken (409)."""


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


def _rows_for(plan: BuildPlan, digest: str) -> list[ContainerImage]:
    return [
        ContainerImage(
            name=image.name,
            workspace=plan.workspace,
            registry=image.placed.registry,
            repository=image.placed.repository,
            platform=image.spec.platform,
            provenance=Provenance(
                build_set=plan.build_set.name,
                revision=plan.build_set.revision,
                spec=image.spec.name,
                job=f"{plan.workspace}/{plan.job_name}",
                request_digest=digest,
            ),
        )
        for image in plan.images
    ]


async def submit_build_set(
    build_set: BuildSet,
    *,
    backend: Backend,
    workspace: str,
    entity_client: _EntityWriter,
    create_job: CreateJob,
    get_job_fields: GetJobFields,
    secret_readable: SecretReadable,
) -> SubmitResult:
    """Resolve, check, place, compile; then write the rows and create the job. In that order."""
    # Everything that can refuse the request runs before anything is written.
    try:
        plan = BuildPlan.resolve(build_set, workspace=workspace)
        backend.check(plan)
        plan = plan.with_destinations(backend.destination)
        platform_spec = backend.compile(plan)
    except BackendRefused:
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
    digest = request_digest(build_set)

    images: list[ContainerImage] = []
    created: list[ContainerImage] = []
    for row in _rows_for(plan, digest):
        try:
            created.append(await entity_client.create(row))
            images.append(created[-1])
        except EntityConflictError:
            try:
                images.append(await _adopt(entity_client, row, digest, build_set, workspace))
            except BuildConflict:
                await _fail_rows(
                    entity_client,
                    created,
                    f"not built: {build_set.name} revision {build_set.revision} belongs to a different request",
                )
                raise

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
        existing = await get_job_fields(plan.job_name)
        if existing.get("request_digest") != digest:
            await _fail_rows(
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


async def _fail_rows(entity_client: _EntityWriter, rows: list[ContainerImage], detail: str) -> None:
    """Fail each of ``rows`` still ``pending``, rather than leave it waiting on a job this submit didn't create.

    One that changed since it was read, or can't be written, is logged and left.
    """
    for row in rows:
        if row.status != "pending":
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
        raise BuildConflict(
            f"build set {build_set.name!r} revision {build_set.revision} was already submitted with a "
            "different request. Submit a new revision; a revision names one request."
        )
    return existing
