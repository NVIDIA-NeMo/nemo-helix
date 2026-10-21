# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Requesting and reading image builds of a task revision."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

from fastapi import HTTPException, status
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
from nemo_helix_plugin.dependencies import RequestAuthorizer
from nemo_helix_plugin.entities import EntityClient, PaginationInfo
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.schema import Page, PaginationData

logger = logging.getLogger(__name__)

_ENDED_JOB_STATUSES = frozenset({"completed", "error", "cancelled"})


class ImageBuildJobConflictError(Exception):
    """The build's Job name is held by a Job that runs something else."""


class ImageBuildForbiddenError(Exception):
    """The caller may not submit container image builds in the workspace."""


#: Raises :class:`ImageBuildForbiddenError` unless the caller may submit Builder builds in the workspace.
AuthorizeBuilds = Callable[[str], Awaitable[None]]


def builder_authorizer(authorize: RequestAuthorizer) -> AuthorizeBuilds:
    """Check the caller may submit Builder builds in a workspace.

    Builder admits the task-builder Job on evals' own permissions, so the caller's permission to build is checked here.
    """

    async def authorize_builds(build_workspace: str) -> None:
        try:
            await authorize("POST", f"/apis/builder/v2/workspaces/{quote(build_workspace, safe='')}/builds")
        except HTTPException as e:
            if e.status_code != status.HTTP_403_FORBIDDEN:
                raise
            raise ImageBuildForbiddenError(
                f"You may not submit container image builds in workspace {build_workspace}"
            ) from e

    return authorize_builds


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
    """Image builds of task revisions, stored under the task and keyed by revision digest.

    Revisions with the same contents share builds: their images are the same.
    """

    def __init__(self, entity_client: EntityClient, jobs: BuildJobs, authorize_builds: AuthorizeBuilds) -> None:
        self.entity_client = entity_client
        self.jobs = jobs
        self.authorize_builds = authorize_builds

    async def request_build(
        self, workspace: str, name: str, revision: str, request: ImageBuildRequest
    ) -> ImageBuild | None:
        """Start a build of a revision, or return the one already running or done.

        A new attempt starts only when none for the requested platform is pending, running or ready. Returns
        ``None`` when the task doesn't exist.

        Raises:
            RevisionNotFoundError: ``revision`` names no revision of the task.
            UnsupportedTaskKindError: The task isn't a Harbor task.
            ImagePlanError: The task's configuration can't be built at all.
            ImageBuildJobConflictError: The build's Job name is held by another Job.
            ImageBuildForbiddenError: The caller may not submit Builder builds in the workspace.
        """
        found = await self._revision(workspace, name, revision)
        if found is None:
            return None
        await self.authorize_builds(workspace)
        head, task_revision = found
        if not isinstance(task_revision.spec, HarborTaskDefinition):
            raise UnsupportedTaskKindError(
                f"Task {workspace}/{name} is not a Harbor task; only Harbor tasks have images"
            )
        parse_task_config(task_revision.spec.config)

        builds = await self._newest_first(head, task_revision.content_hash)
        latest = builds[0] if builds else None
        # Another platform's build is no answer to this request, so only builds for this one are reused.
        newest = next((build for build in builds if build.platform == request.platform), None)
        if newest is not None and newest.status == "ready" and not request.rebuild:
            return ImageBuild.from_entity(newest)
        if newest is not None and newest.status not in ("ready", "failed"):
            job = await self.jobs.get(newest.job)
            if job is None or job.status not in _ENDED_JOB_STATUSES:
                await self._ensure_job(newest)
                return ImageBuild.from_entity(newest)
            newest = await self._fail_abandoned(newest, job.status)
            if newest.status != "failed":
                return ImageBuild.from_entity(newest)

        attempt = 1 if latest is None else latest.attempt + 1
        task = f"{workspace}/{name}"
        build = TaskImageBuildEntity(
            name=build_entity_name(task_revision.content_hash, attempt),
            workspace=workspace,
            project=head.project,
            task=task,
            revision_digest=task_revision.content_hash,
            attempt=attempt,
            job=build_job_name(task, task_revision.content_hash, attempt),
            allowed_registries=request.allowed_registries,
            platform=request.platform,
        )
        build._parent = head.id
        try:
            build = await self.entity_client.create(build)
        except NemoEntityConflictError:
            build = await self.entity_client.get(
                TaskImageBuildEntity, name=build.name, workspace=workspace, parent=head.id
            )
            if build.platform != request.platform:
                raise ImageBuildJobConflictError(
                    f"attempt {attempt} of this revision was just started for another platform; request again"
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
        head, task_revision = found
        result = await self.entity_client.list(
            TaskImageBuildEntity,
            workspace=workspace,
            filter_operation=_builds_of(head, task_revision.content_hash),
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
        head, task_revision = found
        try:
            build = await self.entity_client.get(
                TaskImageBuildEntity,
                name=build_entity_name(task_revision.content_hash, int(build_id)),
                workspace=workspace,
                parent=head.id,
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

    async def _newest_first(self, head: TaskEntity, revision_digest: str) -> list[TaskImageBuildEntity]:
        """The task's builds of these contents, newest first: as many as one page holds, which is more than they get."""
        result = await self.entity_client.list(
            TaskImageBuildEntity,
            workspace=head.workspace,
            filter_operation=_builds_of(head, revision_digest),
            sort="-created_at",
            page_size=100,
        )
        return list(result.data)

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


def _builds_of(head: TaskEntity, revision_digest: str) -> LogicalOperation:
    """The task's builds of the contents with ``revision_digest``."""
    return LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=head.id),
            ComparisonOperation(field="data.revision_digest", operator=FilterOperator.EQ, value=revision_digest),
        ],
    )


def _pagination(src: PaginationInfo, current_page_size: int) -> PaginationData:
    return PaginationData(
        page=src.page,
        page_size=src.page_size,
        current_page_size=current_page_size,
        total_pages=src.total_pages,
        total_results=src.total_results,
    )
