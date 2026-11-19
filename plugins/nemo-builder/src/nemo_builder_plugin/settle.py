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

A row failed because its job ended says why, when the job does: the line its step logged for the image (a
remote-build step logs one, see ``run/remote.py``), or else the job's own account, such as a time limit Jobs
enforced.

A ``queued`` row, which only 0.10's builder wrote, is settled as a ``pending`` one is. If 0.10's queue created its job,
it is completed or failed with that job; if not, nothing will create it now, and it fails once :data:`JOB_GRACE` has
passed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.schema import JOB_NAME_PREFIX
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.entities import EntityConflictError
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse
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

#: How many build jobs :func:`settle` asks Jobs about at once. One at a time, a pass over about 100 pending images
#: took longer than the controller's 90-second health window (3 × its 30-second interval) on a loaded cluster.
JOB_READS = 8

#: The line a step logs for an image it didn't build: ``NHX_BUILD_FAILED <row name>: <reason>``. The same marker
#: as ``run/remote.py``'s ``FAILURE_MARKER``, spelt here so the API doesn't load a step's program.
_MARKER = "NHX_BUILD_FAILED "
_FAILURE_LINE = re.compile(r"NHX_BUILD_FAILED (\S+): (.+)")
#: The longest reason a row records.
_MAX_REASON_CHARS = 500
#: The one step whose log a reason is read from, and the backend whose rows take it: ``remote.py``'s ``STEP_NAME``
#: and backend name, spelt here as the marker is. A remote-build step's driver writes each marked line itself,
#: with nothing a submitter chose on a line of its own. An in-cluster build's log carries whatever a Dockerfile
#: prints, a line that looks like the marker included, so it is never read for one.
_REASON_STEP = "remote-build"
_REASON_BACKEND = "remote"


@dataclass(frozen=True, slots=True)
class EndedJob:
    """A job that has ended, and what it said about why."""

    status: HelixJobStatus
    #: Why each image wasn't built, by row name, as its step logged it.
    reasons: Mapping[str, str] = field(default_factory=dict)
    #: The job's own account of how it ended, when it gave one.
    message: str | None = None


#: The status of the job ``(workspace, name)``, an :class:`EndedJob` when it ended and said why, or None when
#: there is no such job.
JobStatus = Callable[[str, str], Awaitable[HelixJobStatus | EndedJob | None]]


def _clip(reason: str) -> str:
    # One line of printable text: a job's log is whatever its step printed.
    reason = "".join(character for character in " ".join(reason.split()) if character.isprintable())
    return reason if len(reason) <= _MAX_REASON_CHARS else reason[: _MAX_REASON_CHARS - 1] + "…"


def _message(details: object) -> str | None:
    message = details.get("message") if isinstance(details, dict) else None
    return _clip(message) if isinstance(message, str) and message.strip() else None


def _marked_lines(log_tail: str) -> Iterator[str]:
    """Each marked line in ``log_tail``, from the marker on, as the step wrote it.

    Each executor keeps a step's log its own way. Kubernetes and Docker hand back the text. The subprocess
    executor keeps JSON lines, with a log line as the string that ends each object: read from there, the line is
    still escaped and ends with the object's ``"}``. The tail is cut by length, so its first line may be the end
    of one; what follows the marker is whole all the same.

    A line ends at a newline and nowhere else. ``str.splitlines`` also breaks on a form feed and six more
    characters, which a reason may hold, and what followed one would read as a line of its own.
    """
    for line in log_tail.split("\n"):
        line = line.removesuffix("\r")
        start = line.find(_MARKER)
        if start < 0:
            continue
        text = line[start:]
        if text.endswith('"}'):
            try:
                decoded = json.loads(f'"{text[:-2]}"')
            except ValueError:
                decoded = None
            if isinstance(decoded, str):
                text = decoded
        yield text


def ended_job(job: HelixJobStatusResponse) -> EndedJob:
    """What an ended ``job`` said: each image's line from the end of its tasks' logs, and the job's own message."""
    reasons: dict[str, str] = {}
    message = _message(job.error_details)
    for step in job.steps:
        message = message or _message(step.error_details)
        if step.name != _REASON_STEP:
            continue
        for task in step.tasks:
            # Jobs keeps the last lines of a failed task's log here. The last line for an image is its reason.
            for line in _marked_lines(task.error_stack or ""):
                if found := _FAILURE_LINE.match(line):
                    reasons[found.group(1)] = _clip(found.group(2))
    return EndedJob(status=job.status, reasons=reasons, message=message)


def job_status_from(jobs: AsyncJobsClient) -> JobStatus:
    """:data:`JobStatus` over ``jobs``, which must act as the builder: a caller who may not see a job must not have it
    taken for missing."""

    async def job_status(workspace: str, name: str) -> HelixJobStatus | EndedJob | None:
        try:
            response = await jobs.get_job_status(workspace=workspace, name=name)
        except NotFoundError:
            return None
        job = response.data()
        return ended_job(job) if job.status in _ENDED else job.status

    return job_status


class _Rows(Protocol):
    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage: ...

    async def update(self, entity: ContainerImage) -> ContainerImage: ...


async def settle(
    rows: _Rows, images: list[ContainerImage], job_status: JobStatus, *, now: datetime | None = None
) -> list[ContainerImage]:
    """``images`` as they stand once each abandoned waiting one is failed. Each job is asked about once.

    The jobs are read first, :data:`JOB_READS` at a time, then the abandoned images are written one by one. A job
    that can't be read leaves its images as they are, for the next read to settle.
    """
    now = now or datetime.now(UTC)
    statuses = await _job_statuses([image.provenance.job for image in images if image.status in WAITING], job_status)
    settled: list[ContainerImage] = []
    for image in images:
        job = image.provenance.job
        if image.status not in WAITING or job not in statuses:
            settled.append(image)
            continue
        detail = _abandoned(image, statuses[job], now)
        settled.append(image if detail is None else await _fail(rows, image, detail))
    return settled


async def _job_statuses(jobs: list[str], job_status: JobStatus) -> dict[str, HelixJobStatus | EndedJob | None]:
    """Each distinct job's status, read :data:`JOB_READS` at a time; a job that can't be read is left out."""
    limit = asyncio.Semaphore(JOB_READS)

    async def read(job: str) -> tuple[str, HelixJobStatus | EndedJob | None] | None:
        workspace, _, name = job.partition("/")
        async with limit:
            try:
                return job, await job_status(workspace, name)
            except Exception:
                logger.exception("could not read build job %s; leaving its images pending", sanitize_for_log(job))
                return None

    read_jobs = await asyncio.gather(*(read(job) for job in dict.fromkeys(jobs)))
    return dict(found for found in read_jobs if found is not None)


def _abandoned(image: ContainerImage, status: HelixJobStatus | EndedJob | None, now: datetime) -> str | None:
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
    ended = status if isinstance(status, EndedJob) else EndedJob(status)
    if ended.status not in _ENDED:
        return None
    detail = f"not built: its build job ended ({ended.status.value}) without publishing it"
    if not _own_job(image):
        # The job was read as the builder, which may read any. What it said is copied only onto a row of its own
        # build: a row written some other way can name any job, in any workspace.
        return detail
    line = ended.reasons.get(image.name) if image.provenance.backend == _REASON_BACKEND else None
    why = line or ended.message
    return f"{detail}: {why}" if why else detail


def _own_job(image: ContainerImage) -> bool:
    """Whether the job ``image`` names is the one a submit creates for it: in its workspace, named for its set."""
    provenance = image.provenance
    return provenance.job == f"{image.workspace}/{JOB_NAME_PREFIX}{provenance.build_set}-{provenance.revision}"


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
