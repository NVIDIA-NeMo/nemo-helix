# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settling: a ``pending`` image whose job will never complete it is failed.

An image is ``pending`` while its job may still complete it. Its job decides that: once the job has ended, or if
there is none (never created, or deleted), nothing will. :func:`settle` asks Jobs about each ``pending`` image it is
given, and writes the ones it finds abandoned ``failed``, conditional on the version it read. A row its push step
completed meanwhile conflicts, and is read again. Reading one image settles it, and the ``builder-settle`` controller
settles every ``pending`` one each pass; the two can run at once, since each write is conditional.

A submit writes the rows, then creates the job, so a row without a job is only abandoned once :data:`JOB_GRACE` has
passed. A submit slow enough to create its job after that builds images already failed: their push step's completion
is refused, and the job fails. Either way, the images end ``failed``, and a caller submits the next revision.

A ``queued`` row, which only 0.10's builder wrote, is settled as a ``pending`` one is. If 0.10's queue created its job,
it is completed or failed with that job; if not, nothing will create it now, and it fails once :data:`JOB_GRACE` has
passed.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol

from nemo_builder_plugin.entities import ContainerImage
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.entities import EntityConflictError
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

#: How long after its row was written an image may have no job: far longer than a submit takes between the two.
JOB_GRACE = timedelta(minutes=5)
#: A ``queued`` row with no job: 0.10's queue never started it, and nothing will now.
QUEUE_GONE = "not built: it was queued by an earlier builder, whose queue is gone; submit its next revision"
#: Jobs has no such job: the submit never created it, or it was deleted. Jobs' 404 doesn't say which.
NO_JOB = "not built: its build job was not found (never created, or deleted)"

_ENDED = frozenset(HelixJobStatus.terminals())
#: The statuses a job may still complete. ``queued`` is deprecated (see the module docstring).
WAITING = ("pending", "queued")

#: The status of the job ``(workspace, name)``, or None when there is no such job.
JobStatus = Callable[[str, str], Awaitable[HelixJobStatus | None]]


def job_status_from(jobs: AsyncJobsClient) -> JobStatus:
    """:data:`JobStatus` over ``jobs``, which must act as the builder: a caller who may not see a job must not have it
    taken for missing."""

    async def job_status(workspace: str, name: str) -> HelixJobStatus | None:
        try:
            response = await jobs.get_job_status(workspace=workspace, name=name)
        except NotFoundError:
            return None
        return response.data().status

    return job_status


class _Rows(Protocol):
    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage: ...

    async def update(self, entity: ContainerImage) -> ContainerImage: ...


async def settle(
    rows: _Rows, images: list[ContainerImage], job_status: JobStatus, *, now: datetime | None = None
) -> list[ContainerImage]:
    """``images`` as they stand once each abandoned waiting one is failed. Each job is asked about once.

    A job that can't be read leaves its images as they are, for the next read to settle.
    """
    now = now or datetime.now(UTC)
    statuses: dict[str, HelixJobStatus | None] = {}
    unreadable: set[str] = set()
    settled: list[ContainerImage] = []
    for image in images:
        job = image.provenance.job
        if image.status not in WAITING or job in unreadable:
            settled.append(image)
            continue
        if job not in statuses:
            workspace, _, name = job.partition("/")
            try:
                statuses[job] = await job_status(workspace, name)
            except Exception:
                logger.exception("could not read build job %s; leaving its images pending", sanitize_for_log(job))
                unreadable.add(job)
                settled.append(image)
                continue
        detail = _abandoned(image, statuses[job], now)
        settled.append(image if detail is None else await _fail(rows, image, detail))
    return settled


def _abandoned(image: ContainerImage, status: HelixJobStatus | None, now: datetime) -> str | None:
    """Why no job will complete ``image``, or None if one still may."""
    if status is None:
        created = image.created_at
        if created is None:
            # Undated, it is never taken for old.
            return None
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        if now - created < JOB_GRACE:
            return None
        return QUEUE_GONE if image.status == "queued" else NO_JOB
    if status in _ENDED:
        return f"not built: its build job ended ({status.value}) without publishing it"
    return None


async def _fail(rows: _Rows, image: ContainerImage, detail: str) -> ContainerImage:
    failed = image.model_copy(update={"status": "failed", "status_detail": detail})
    try:
        written = await rows.update(failed)
    except EntityConflictError:
        # Changed since it was read: its push step completed it, or another read failed it first.
        try:
            return await rows.get(ContainerImage, name=image.name, workspace=image.workspace)
        except Exception:
            logger.exception("could not read image %s again", _label(image))
            return image
    except Exception:
        logger.exception("could not fail image %s; it stays pending", _label(image))
        return image
    logger.info("image %s failed: %s", _label(image), detail)
    return written


def _label(image: ContainerImage) -> str:
    return f"{sanitize_for_log(image.workspace)}/{sanitize_for_log(image.name)}"
