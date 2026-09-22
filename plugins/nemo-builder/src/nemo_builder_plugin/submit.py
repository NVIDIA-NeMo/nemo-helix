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

**Concurrency is settled by the database, not by a check.** Row names are deterministic from the
request (``<set>-<revision>-<index>``), so two submitters racing on the same ``(name, revision)``
collide on the entity store's unique index and the loser adopts the winner's rows. There is no
read-then-write here, because a read-then-write across replicas is a race.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from nemo_builder_plugin.compile import compile_build_set, image_name_for, job_name_for, resolve_destinations
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, JobOrigin, Provenance
from nemo_builder_plugin.identity import compose_system_tag
from nemo_builder_plugin.schema import BuildSet
from nemo_platform_plugin.entities import EntityConflictError
from nemo_platform_plugin.jobs.types import CreatePlatformJobRequest

logger = logging.getLogger(__name__)

#: ``CreatePlatformJobRequest.source`` -- how the reconciler recognises its own jobs.
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
CreateJob = Callable[[CreatePlatformJobRequest], Awaitable[object]]


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


def _rows_for(build_set: BuildSet, config: BuilderConfig, workspace: str) -> list[ContainerImage]:
    """The desired state, before anything is built."""
    job_name = job_name_for(build_set)
    system_tag = compose_system_tag(workspace, build_set.name, build_set.revision)
    destinations = resolve_destinations(build_set, config)

    return [
        ContainerImage(
            name=image_name_for(job_name, index),
            workspace=workspace,
            registry=destinations[index][0],
            repository=destinations[index][1],
            platform=spec.platform,
            provenance=Provenance(
                backend="execution",
                built_by=JobOrigin(
                    build_set=build_set.name,
                    revision=build_set.revision,
                    job=f"{workspace}/{job_name}",
                    system_tag=system_tag,
                ),
            ),
        )
        for index, spec in enumerate(build_set.build_specs)
    ]


async def submit_build_set(
    build_set: BuildSet,
    *,
    config: BuilderConfig,
    workspace: str,
    entity_client: _EntityWriter,
    create_job: CreateJob,
) -> SubmitResult:
    """Compile, write the rows, create the job. In that order."""
    job_name = job_name_for(build_set)
    system_tag = compose_system_tag(workspace, build_set.name, build_set.revision)

    # Compile FIRST. It is pure and it is the only step that can reject the request for a reason
    # the caller can act on, so it should run before anything is written.
    platform_spec = compile_build_set(build_set, config=config, system_tag=system_tag)

    images: list[ContainerImage] = []
    for row in _rows_for(build_set, config, workspace):
        try:
            images.append(await entity_client.create(row))
        except EntityConflictError:
            # A concurrent submitter won, or this is a resubmission of the same
            # (name, revision). Adopt the existing row rather than failing: the name is
            # deterministic, so the row that exists is the row this request would have made.
            images.append(await entity_client.get(ContainerImage, name=row.name, workspace=workspace))

    await create_job(
        CreatePlatformJobRequest(
            name=job_name,
            description=f"Container image build for {build_set.name} revision {build_set.revision}",
            source=JOB_SOURCE,
            spec=build_set.model_dump(mode="json"),
            platform_spec=platform_spec,
            custom_fields={"images": [image.name for image in images]},
        )
    )

    logger.info(
        "submitted build set %s revision %s as job %s with %d image(s)",
        build_set.name,
        build_set.revision,
        job_name,
        len(images),
    )
    return SubmitResult(job=job_name, images=images)
