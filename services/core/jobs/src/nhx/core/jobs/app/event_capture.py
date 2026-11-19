# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort lifecycle recording owned by Jobs, independent of its persistence engine."""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Self

from nemo_helix_plugin.jobs.events import JobEventType
from nemo_helix_plugin.workspaces.client import AsyncWorkspacesClient, WorkspacesClient
from nhx.common.entities import EntityBase, EntityClient, EntityNotFoundError, SyncEntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.event_stream import JobEventStream, LifecycleChange, SyncJobEventStream
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt, HelixJobStep
from pydantic import JsonValue

logger = logging.getLogger(__name__)
LifecycleResource = HelixJob | HelixJobAttempt | HelixJobStep


@dataclass(frozen=True)
class LifecycleImage:
    kind: Literal["job", "attempt", "step"]
    workspace_id: str
    job_id: str
    attempt_id: str | None
    step_id: str | None
    state: str
    occurred_at: datetime
    snapshot: dict[str, JsonValue]
    related_job: Self | None = None

    def change(self, previous: str | None, action: str) -> LifecycleChange:
        return LifecycleChange(
            event_type=JobEventType(self.kind + "." + action),
            workspace_id=self.workspace_id,
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            step_id=self.step_id,
            previous_state=previous,
            new_state=None if action == "deleted" else self.state,
            occurred_at=self.occurred_at,
            snapshot=self.snapshot,
        )


def image(resource: LifecycleResource, job: HelixJob, workspace_id: str, job_state: HelixJobStatus) -> LifecycleImage:
    snapshot: dict[str, JsonValue] = {"name": resource.name, "source": job.source, "project": resource.project}
    if isinstance(resource, HelixJob):
        kind: Literal["job", "attempt", "step"] = "job"
        attempt_id = resource.current_attempt_id
        step_id = None
    elif isinstance(resource, HelixJobAttempt):
        kind, attempt_id, step_id = "attempt", resource.id, None
        snapshot["sequence"] = resource.seq
    else:
        kind, attempt_id, step_id = "step", resource.attempt_id, resource.id
    timestamp = resource.updated_at or resource.created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return LifecycleImage(
        kind,
        workspace_id,
        job.id,
        attempt_id,
        step_id,
        (job_state if isinstance(resource, HelixJob) else resource.status).value,
        timestamp,
        snapshot,
        image(job, job, workspace_id, job_state) if isinstance(resource, HelixJobAttempt) else None,
    )


class JobEventRecorder:
    def __init__(self, store: EntityClient, workspaces: AsyncWorkspacesClient):
        self.store = store
        self.workspaces = workspaces
        self.stream = JobEventStream(store)

    async def _image(self, resource: LifecycleResource) -> LifecycleImage:
        if isinstance(resource, HelixJob):
            job = resource
        elif isinstance(resource, HelixJobAttempt):
            job = await self.store.get_by_id(HelixJob, resource.job)
        else:
            attempt = await self.store.get_by_id(HelixJobAttempt, resource.attempt_id)
            job = await self.store.get_by_id(HelixJob, attempt.job)
        state = job.status or HelixJobStatus.CREATED
        if job.current_attempt_id and job.status not in HelixJobStatus.lifecycle_pending():
            try:
                state = (await self.store.get_by_id(HelixJobAttempt, job.current_attempt_id)).status
            except EntityNotFoundError:
                pass
        workspace = (await self.workspaces.get_workspace(name=resource.workspace)).data()
        return image(resource, job, workspace.id, state)

    async def before(self, entity: EntityBase) -> LifecycleImage | None:
        if not isinstance(entity, (HelixJob, HelixJobAttempt, HelixJobStep)):
            return None
        try:
            async with asyncio.timeout(2):
                return await self._image(await self.store.get_by_id(type(entity), entity.id))
        except Exception:
            logger.warning("Could not capture Jobs lifecycle preimage", exc_info=True)
            return None

    async def created(self, entity: EntityBase) -> None:
        if not isinstance(entity, (HelixJob, HelixJobAttempt, HelixJobStep)):
            return
        try:
            async with asyncio.timeout(2):
                await self.stream.append((await self._image(entity)).change(None, "created"))
        except Exception:
            logger.warning("Jobs lifecycle creation event was dropped", exc_info=True)

    async def updated(self, before: LifecycleImage | None, entity: EntityBase) -> None:
        if before is None or not isinstance(entity, (HelixJob, HelixJobAttempt, HelixJobStep)):
            return
        if not isinstance(entity, HelixJob) and before.state == entity.status.value:
            return
        try:
            async with asyncio.timeout(2):
                after = await self._image(entity)
                if before.state != after.state:
                    await self.stream.append(after.change(before.state, "state_changed"))
                if (
                    before.related_job is not None
                    and after.related_job is not None
                    and before.related_job.state != after.related_job.state
                ):
                    change = after.related_job.change(before.related_job.state, "state_changed")
                    change.occurred_at = after.occurred_at
                    await self.stream.append(change)
        except Exception:
            logger.warning("Jobs lifecycle state event was dropped", exc_info=True)


class SyncJobEventRecorder:
    def __init__(self, store: SyncEntityClient, workspaces: WorkspacesClient):
        self.store = store
        self.workspaces = workspaces
        self.stream = SyncJobEventStream(store)

    def _image(self, resource: LifecycleResource) -> LifecycleImage:
        if isinstance(resource, HelixJob):
            job = resource
        elif isinstance(resource, HelixJobAttempt):
            job = self.store.get_by_id(HelixJob, resource.job)
        else:
            attempt = self.store.get_by_id(HelixJobAttempt, resource.attempt_id)
            job = self.store.get_by_id(HelixJob, attempt.job)
        state = job.status or HelixJobStatus.CREATED
        if job.current_attempt_id and job.status not in HelixJobStatus.lifecycle_pending():
            try:
                state = self.store.get_by_id(HelixJobAttempt, job.current_attempt_id).status
            except EntityNotFoundError:
                pass
        return image(resource, job, self.workspaces.get_workspace(name=resource.workspace).data().id, state)

    def before(self, resource: LifecycleResource) -> LifecycleImage | None:
        try:
            return self._image(resource)
        except Exception:
            logger.warning("Could not capture controller lifecycle preimage", exc_info=True)
            return None

    def created(self, resource: LifecycleResource) -> None:
        try:
            self.stream.append(self._image(resource).change(None, "created"))
        except Exception:
            logger.warning("Controller lifecycle creation event was dropped", exc_info=True)

    def updated(self, before: LifecycleImage | None, resource: LifecycleResource) -> None:
        if before is None or before.state == (resource.status or HelixJobStatus.CREATED).value:
            return
        try:
            self.stream.append(self._image(resource).change(before.state, "state_changed"))
        except Exception:
            logger.warning("Controller lifecycle state event was dropped", exc_info=True)

    def deletion_images(self, job: HelixJob) -> list[LifecycleImage]:
        try:
            images: list[LifecycleImage] = []
            page = 1
            while True:
                attempts = self.store.list(
                    HelixJobAttempt, workspace=job.workspace, filter_obj={"job": job.id}, page=page
                )
                for attempt in attempts.data:
                    step_page = 1
                    while True:
                        steps = self.store.list(
                            HelixJobStep, workspace=job.workspace, filter_obj={"attempt_id": attempt.id}, page=step_page
                        )
                        images.extend(self._image(step) for step in steps.data)
                        if step_page >= steps.pagination.total_pages:
                            break
                        step_page += 1
                    images.append(self._image(attempt))
                if page >= attempts.pagination.total_pages:
                    break
                page += 1
            images.append(self._image(job))
            return images
        except Exception:
            logger.warning("Could not capture Jobs deletion preimages", exc_info=True)
            return []

    def deleted(self, images: list[LifecycleImage]) -> None:
        deadline = time.monotonic() + 2
        for before in images:
            if time.monotonic() >= deadline:
                logger.warning("Remaining Jobs deletion events were dropped after the recording deadline")
                break
            try:
                change = before.change(before.state, "deleted")
                change.occurred_at = datetime.now(timezone.utc)
                self.stream.append(change)
            except Exception:
                logger.warning("Jobs lifecycle deletion event was dropped", exc_info=True)
