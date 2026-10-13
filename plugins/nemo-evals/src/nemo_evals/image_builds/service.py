# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Requesting and reading image builds of a task revision."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from nemo_evals.api.task_definitions.harbor import HarborTaskDefinition
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.image_builds.models import (
    ImageBuild,
    ImageBuildRequest,
    TaskImageBuildEntity,
    build_entity_name,
    build_job_name,
)
from nemo_evals.image_builds.planning import parse_task_config
from nemo_evals.image_builds.runner import TaskImageBuildSpec
from nemo_evals.jobs.task_image_build import is_job_for, task_image_build_job_request
from nemo_evals.revisions import get_revision
from nemo_evals.task_identity import UnsupportedTaskKindError
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.entities import EntityClient, PaginationInfo
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.schema import Page, PaginationData

logger = logging.getLogger(__name__)

_ENDED_JOB_STATUSES = frozenset({"completed", "error", "cancelled"})


class ImageBuildJobConflictError(Exception):
    """The build's Job name is held by a Job that runs something else."""


@dataclass(frozen=True)
class BuildJob:
    status: str
    custom_fields: dict | None


class BuildJobs(Protocol):
    """The Jobs operations an image build request needs."""

    async def create(self, request: CreateHelixJobRequest) -> None: ...

    async def get(self, name: str) -> BuildJob | None: ...


class JobsServiceBuildJobs:
    """:class:`BuildJobs` over the Jobs service, acting as the request's caller."""

    def __init__(self, jobs: AsyncJobsClient, *, workspace: str) -> None:
        self._jobs = jobs
        self._workspace = workspace

    async def create(self, request: CreateHelixJobRequest) -> None:
        await self._jobs.create_job(workspace=self._workspace, body=request)

    async def get(self, name: str) -> BuildJob | None:
        try:
            job = (await self._jobs.get_job(workspace=self._workspace, name=name)).data()
        except NotFoundError:
            return None
        return BuildJob(status=job.status.value, custom_fields=job.custom_fields)


class ImageBuildService:
    """Image builds of task revisions, stored as children of the revision they build."""

    def __init__(self, entity_client: EntityClient, jobs: BuildJobs) -> None:
        self.entity_client = entity_client
        self.jobs = jobs

    async def request_build(
        self, workspace: str, name: str, revision: str, request: ImageBuildRequest
    ) -> ImageBuild | None:
        """Start a build of a revision, or return the one already running or done.

        A new attempt starts only when there is none yet or the newest one failed. Returns ``None``
        when the task doesn't exist.

        Raises:
            RevisionNotFoundError: ``revision`` names no revision of the task.
            UnsupportedTaskKindError: The task isn't a Harbor task.
            ImagePlanError: The task's configuration can't be built at all.
            ImageBuildJobConflictError: The build's Job name is held by another Job.
        """
        found = await self._revision(workspace, name, revision)
        if found is None:
            return None
        head, task_revision = found
        if not isinstance(task_revision.spec, HarborTaskDefinition):
            raise UnsupportedTaskKindError(
                f"Task {workspace}/{name} is not a Harbor task; only Harbor tasks have images"
            )
        parse_task_config(task_revision.spec.config)

        newest = await self._newest(task_revision)
        if newest is not None and newest.status == "ready":
            return ImageBuild.from_entity(newest)
        if newest is not None and newest.status != "failed":
            job = await self.jobs.get(newest.job)
            if job is None or job.status not in _ENDED_JOB_STATUSES:
                await self._ensure_job(newest)
                return ImageBuild.from_entity(newest)
            newest = await self._fail_abandoned(newest, job.status)
            if newest.status != "failed":
                return ImageBuild.from_entity(newest)

        attempt = 1 if newest is None else newest.attempt + 1
        task = f"{workspace}/{name}"
        build = TaskImageBuildEntity(
            name=build_entity_name(attempt),
            workspace=workspace,
            project=head.project,
            task=task,
            revision_digest=task_revision.content_hash,
            attempt=attempt,
            job=build_job_name(task, task_revision.content_hash, attempt),
            allowed_registries=request.allowed_registries,
        )
        build._parent = task_revision.id
        try:
            build = await self.entity_client.create(build)
        except NemoEntityConflictError:
            build = await self.entity_client.get(
                TaskImageBuildEntity, name=build.name, workspace=workspace, parent=task_revision.id
            )
        await self._ensure_job(build)
        return ImageBuild.from_entity(build)

    async def list_builds(
        self, workspace: str, name: str, revision: str, *, page: int = 1, page_size: int = 100
    ) -> Page[ImageBuild] | None:
        """A revision's builds, newest first; ``None`` when the task doesn't exist."""
        found = await self._revision(workspace, name, revision)
        if found is None:
            return None
        _, task_revision = found
        result = await self.entity_client.list(
            TaskImageBuildEntity,
            workspace=workspace,
            filter_operation=ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=task_revision.id),
            sort="-created_at",
            page=page,
            page_size=page_size,
        )
        data = [ImageBuild.from_entity(build) for build in result.data]
        return Page(data=data, pagination=_pagination(result.pagination, len(data)), sort=None, filter=None)

    async def get_build(self, workspace: str, name: str, revision: str, build_id: str) -> ImageBuild | None:
        """One build of a revision; ``None`` when the task or build doesn't exist."""
        found = await self._revision(workspace, name, revision)
        if found is None or not build_id.isdigit():
            return None
        _, task_revision = found
        try:
            build = await self.entity_client.get(
                TaskImageBuildEntity,
                name=build_entity_name(int(build_id)),
                workspace=workspace,
                parent=task_revision.id,
            )
        except NemoEntityNotFoundError:
            return None
        return ImageBuild.from_entity(build)

    async def _revision(self, workspace: str, name: str, revision: str) -> tuple[TaskEntity, TaskRevisionEntity] | None:
        try:
            head = await self.entity_client.get(TaskEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None
        return head, await get_revision(self.entity_client, TaskRevisionEntity, head, revision)

    async def _newest(self, task_revision: TaskRevisionEntity) -> TaskImageBuildEntity | None:
        result = await self.entity_client.list(
            TaskImageBuildEntity,
            workspace=task_revision.workspace,
            filter_operation=ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=task_revision.id),
            sort="-created_at",
            page_size=1,
        )
        return result.data[0] if result.data else None

    async def _fail_abandoned(self, build: TaskImageBuildEntity, job_status: str) -> TaskImageBuildEntity:
        """Fail a build whose Job ended without recording a result, unless it just recorded one."""
        build.status = "failed"
        build.error = f"task-builder job {build.job} ended ({job_status}) without a result"
        try:
            return await self.entity_client.update(build)
        except NemoEntityConflictError:
            return await self.entity_client.get(
                TaskImageBuildEntity, name=build.name, workspace=build.workspace, parent=build.parent
            )

    async def _ensure_job(self, build: TaskImageBuildEntity) -> None:
        """Create the build's Job, or confirm the Job already under its name is this build's."""
        workspace, _, task = build.task.partition("/")
        spec = TaskImageBuildSpec(
            workspace=workspace, task=task, revision_digest=build.revision_digest, attempt=build.attempt
        )
        try:
            await self.jobs.create(task_image_build_job_request(spec, name=build.job))
        except ConflictError:
            existing = await self.jobs.get(build.job)
            if existing is None or not is_job_for(existing.custom_fields, spec):
                raise ImageBuildJobConflictError(f"Job {build.job} exists and does not run this build") from None
            return
        logger.info("Started image build %s of %s as job %s", build.name, build.task, build.job)


def _pagination(src: PaginationInfo, current_page_size: int) -> PaginationData:
    return PaginationData(
        page=src.page,
        page_size=src.page_size,
        current_page_size=current_page_size,
        total_pages=src.total_pages,
        total_results=src.total_results,
    )
