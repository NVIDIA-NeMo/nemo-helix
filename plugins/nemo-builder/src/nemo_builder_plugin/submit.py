# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submit path: rows first, then the job, so every image the job pushes already has a row."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from nemo_builder_plugin.backend import Backend, BackendRejectedError
from nemo_builder_plugin.entities import ContainerImage, Provenance, UpstreamImage
from nemo_builder_plugin.identity import ImageReference
from nemo_builder_plugin.plan import BuildPlan, PlannedImage
from nemo_builder_plugin.schema import BuildSet
from nemo_builder_plugin.upstream import UpstreamError, UpstreamNotFound
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.entities import EntityConflictError, EntityNotFoundError
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.log_utils import sanitize_for_log
from pydantic import ValidationError

logger = logging.getLogger(__name__)

JOB_SOURCE = "builder.build"


class _EntityWriter(Protocol):
    async def create(self, entity: ContainerImage) -> ContainerImage: ...

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage: ...

    async def update(self, entity: ContainerImage) -> ContainerImage: ...


@dataclass(frozen=True, slots=True)
class CreatedJob:
    """What Jobs answered a create this submit made: what a row records of its build job.

    Never filled from a job found under a name. Everything a caller can read back of such a job was written by
    whoever created it, and the attempt it names is its latest, which a rerun replaces.
    """

    #: The job's ID, which Jobs gives no later job. Its name is free again once it is deleted.
    uid: str
    #: The ID of its first attempt, the one that runs the request as the builder compiled it.
    attempt_id: str | None


@dataclass(frozen=True, slots=True)
class ExistingJob:
    """The job found under a name: its ID, and the ``custom_fields`` its creator gave it."""

    uid: str
    custom_fields: Mapping[str, Any]


#: Create the build job, as the caller.
CreateJob = Callable[[CreateHelixJobRequest], Awaitable[CreatedJob]]
#: The job with this name, as the caller reads it.
GetJob = Callable[[str], Awaitable[ExistingJob]]
#: Cancel the job a create of this submit's made, named by the ID Jobs answered that create with.
CancelJob = Callable[[str], Awaitable[None]]
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


class BackendChanged(BuildConflict):
    """This ``(name, revision)`` was submitted under another backend than the deployment builds with now (409)."""


class BuildNotRecorded(Exception):
    """The build's job was created and could not be recorded on its images, so it was given up (503).

    Nothing would have completed an image that does not record its job where a step is verified, and nothing
    restores the record later. The job is cancelled, or the message says it could not be.
    """

    def __init__(self, message: str, *, final: bool = True) -> None:
        super().__init__(message)
        #: Whether the same request, sent again, would build nothing. Not where the job was gone already and
        #: its rows were left as they were: sent again, that request creates a job for them.
        self.final = final


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


def _rows_for(plan: BuildPlan, digest: str, backend: str) -> list[ContainerImage]:
    return [
        ContainerImage(
            name=image.name,
            workspace=plan.workspace,
            status="pending",
            registry=image.placed.registry,
            repository=image.placed.repository,
            provenance=Provenance(
                build_set=plan.build_set.name,
                revision=plan.build_set.revision,
                spec=image.spec.name,
                job=f"{plan.workspace}/{plan.job_name}",
                request_digest=digest,
                upstream=_upstream_image(image),
                backend=backend,
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
    get_job: GetJob,
    cancel_job: CancelJob,
    secret_readable: SecretReadable,
    resolve_upstream: ResolveUpstream,
    default_platform: str,
) -> SubmitResult:
    """Resolve, check, place, compile; then write the rows and create the job. In that order.

    For a backend whose rows are completed on their caller's word, the rows are then written once more, with
    the job the create made: the one whose steps may complete them. If a row that still waits can't be made
    to record it, the job is cancelled, the rows are failed and :class:`BuildNotRecorded` is raised.
    ``cancel_job`` is that cancel, and ``get_job`` is asked, too, whether the job still holds its name.
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
    try:
        # Made here, with the rest of what can refuse the request, though they are written further down. A
        # workspace whose name a row's job reference can't hold gets this far only when no repository path
        # needed its name first.
        rows = _rows_for(plan, digest, backend.name)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise InvalidBuildRequest(
            f"an image's row can't record this request in workspace {workspace!r} ({problems})"
        ) from exc
    # Jobs checks these too, but only once the rows exist, and its refusal looks like a taken job name.
    missing = [
        f"{secret_workspace}/{name}"
        for secret_workspace, name in referenced_secrets(platform_spec, workspace)
        if not await secret_readable(secret_workspace, name)
    ]
    if missing:
        raise MissingSecrets(
            "the build job needs platform secrets that don't exist, or that you can't read: " + ", ".join(missing)
        )

    images: list[ContainerImage] = []
    created: list[ContainerImage] = []
    for row in rows:
        try:
            created.append(await entity_client.create(row))
            images.append(created[-1])
        except EntityConflictError:
            try:
                images.append(await _adopt(entity_client, row, digest, build_set, workspace, backend.name))
            except BuildConflict as exc:
                # What the answer says: the rows this request wrote are failed for the same reason.
                why = (
                    "was already submitted under another backend"
                    if isinstance(exc, BackendChanged)
                    else "belongs to a different request"
                )
                await fail_rows(
                    entity_client, created, f"not built: {build_set.name} revision {build_set.revision} {why}"
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
    try:
        job = await create_job(request)
    except ConflictError:
        # Not proof that a job of that name exists: Jobs refuses a create for other reasons too. A job that
        # can't then be read ends the submit with Jobs' own error, as it always did.
        existing = await get_job(plan.job_name)
        if backend.binds_rows_to_job:
            # Not on what the job says of itself. Its creator wrote that, and anyone who knows the request can
            # work out its digest: only a job the rows record was created by a submit of theirs.
            images = await _rows_under(entity_client, images, existing.uid, plan.job_name)
        elif existing.custom_fields.get("request_digest") == digest:
            logger.info("job %s already exists for this request; adopting it", sanitize_for_log(plan.job_name))
        else:
            await fail_rows(
                entity_client, images, f"not built: job {plan.job_name!r} is held by a job this request did not create"
            )
            raise BuildConflict(
                f"a job named {plan.job_name!r} already exists and was not created by this request"
            ) from None
    else:
        if backend.binds_rows_to_job:
            images = await _record_job(entity_client, images, job, plan.job_name, get_job, cancel_job)

    logger.info(
        "submitted build set %s revision %s as job %s with %d image(s)",
        sanitize_for_log(build_set.name),
        build_set.revision,
        sanitize_for_log(plan.job_name),
        len(images),
    )
    return SubmitResult(job=plan.job_name, images=images)


#: How many times a row is written with its build job, or read again and written, and how long apart, before
#: the submit gives the build up: for a store that fails for a moment, or a row something else is writing.
#: The same for the cancel that follows, should Jobs refuse it for a status that moved meanwhile.
RECORD_TRIES = 3
RECORD_PAUSE_SECONDS = 0.5


class _JobGone(Exception):
    """The job this submit created no longer holds the build's name: it was deleted before a row recorded it."""

    def __init__(self, row: ContainerImage) -> None:
        super().__init__()
        #: The row that was being written, as it is stored.
        self.row = row


async def _record_job(
    entity_client: _EntityWriter,
    rows: list[ContainerImage],
    job: CreatedJob,
    job_name: str,
    get_job: GetJob,
    cancel_job: CancelJob,
) -> list[ContainerImage]:
    """Write on each of ``rows`` still waiting the job this submit created, and return the rows as they then are.

    ``job`` is what Jobs answered this submit's own create with. A row that still waits and can't be made to
    record it would never be completed where a step is verified: its completion is put off until the step
    gives up, with its image built. And nothing would restore the record later, since that answer is held
    here alone and a job found under the name is never taken at its word. So the build is given up at once
    instead: the job is cancelled, the rows of the set that still wait are failed, and
    :class:`BuildNotRecorded` is raised. In any mode: a row that does not record its job is no better where no
    caller is verified, since the same request, submitted again, is refused while that job holds the name.

    If the job is deleted before a row records it, there is nothing to give up, and the rows are returned
    as they are: they wait on no job of this submit's, and are another submit's to record on, or are failed
    as abandoned.
    """

    async def still_this_submits() -> bool:
        """Whether the job under the build's name is still the one this submit created."""
        try:
            return (await get_job(job_name)).uid == job.uid
        except NotFoundError:
            return False

    stored: list[ContainerImage] = []
    if job.uid and job.attempt_id:
        try:
            for row in rows:
                written = await _record_job_on(entity_client, row, job.uid, job.attempt_id, still_this_submits)
                if written is None:
                    break
                stored.append(written)
            else:
                return stored
        except _JobGone as gone:
            logger.warning(
                "job %s was deleted before every image of its set recorded it; the images are left as they are",
                sanitize_for_log(job_name),
            )
            rest = [await _stored_or_held(entity_client, row) for row in rows[len(stored) + 1 :]]
            return [*stored, gone.row, *rest]
        problem = f"image {rows[len(stored)].name} could not be made to record it"
    else:
        # A row records both or neither: one that named a job and no attempt would wait on one for good.
        logger.error("Jobs created job %s and named no ID, or no attempt, for it", sanitize_for_log(job_name))
        problem = "Jobs named no ID, or no attempt, for it"
    said = (
        f"The build's job {job_name!r} was created, but could not be recorded on its images ({problem}), so no "
        "step that is verified could have completed them. "
    )
    cancelled = await _cancel_created(cancel_job, job.uid or job_name, job_name)
    if cancelled is None:
        # Deleted meanwhile. Rows that wait on no job are another submit's to record on, or fail as abandoned.
        raise BuildNotRecorded(
            said + "The job is gone, so there was none to cancel, and the images are left as they are. Submit "
            "the request again.",
            final=False,
        )
    ending = "and was cancelled" if cancelled else "and could not be cancelled"
    failed = await _fail_given_up(
        entity_client,
        rows,
        f"not built: its build job could not be recorded on its images, {ending}",
        # A cancel Jobs took shows the job was there, and so held the build's name, until now.
        held_the_name=None if cancelled else job.uid,
    )
    raise BuildNotRecorded(
        said
        + ("The job was cancelled. " if cancelled else f"The job could not be cancelled: cancel {job_name!r}. ")
        + (
            "Its images that still waited are failed. "
            if failed
            else "An image that could not be failed is failed once the job has ended. "
        )
        + "Submit the next revision."
    )


async def _stored_or_held(entity_client: _EntityWriter, row: ContainerImage) -> ContainerImage:
    """``row`` as it is stored now. As this submit holds it, if it can't be read or is no longer its row."""
    try:
        return await _as_stored(entity_client, row) or row
    except Exception:
        return row


async def _fail_given_up(
    entity_client: _EntityWriter, rows: list[ContainerImage], detail: str, *, held_the_name: str | None
) -> bool:
    """Fail each of ``rows``, as it is stored now, that still waits on the job this submit gave up. Whether
    every one of ``rows`` is then settled.

    With ``held_the_name`` None, the job is known to have held the build's name until it was given up: Jobs
    took its cancel. So whatever a row that still waits says of a job, none will complete it, and it is
    failed. Otherwise that is not known, and ``held_the_name`` is the job's ID: a row that records another
    job's may be waiting on one that took the name since, and is left to it.

    One that is gone, or is no longer the row this submit held, is not this submit's to write. A row that
    can't be failed is logged and left: it is failed once the job has ended, as any abandoned row is.
    """
    waiting: list[ContainerImage] = []
    for row in rows:
        try:
            now = await _as_stored(entity_client, row)
        except Exception:
            # It can't be read. Failed as this submit holds it, should the store take that.
            now = row
        if now is None:
            continue
        if held_the_name is not None and now.provenance.job_uid not in (None, held_the_name):
            continue
        # A copy: a row is failed in place before it is written, and one the store refuses stays as read.
        waiting.append(now.model_copy())
    return await fail_rows(entity_client, waiting, detail) and len(waiting) == len(rows)


async def _cancel_created(cancel_job: CancelJob, job: str, job_name: str) -> bool | None:
    """Cancel the job this submit created and gave up. Whether it is cancelled; None if it is gone already."""
    failure: Exception | None = None
    for tried in range(RECORD_TRIES):
        if tried:
            await asyncio.sleep(RECORD_PAUSE_SECONDS)
        try:
            await cancel_job(job)
        except NotFoundError:
            # Deleted since: there is nothing left to stop.
            return None
        except ConflictError as exc:
            # Its status moved while Jobs was cancelling it, as when its step was being scheduled. Asked again.
            failure = exc
        except Exception as exc:
            failure = exc
            break
        else:
            logger.warning("job %s was cancelled: it could not be recorded on its images", sanitize_for_log(job_name))
            return True
    logger.error(
        "job %s could not be recorded on its images, and could not be cancelled. Its images can't be "
        "completed where a step is verified: cancel it",
        sanitize_for_log(job_name),
        exc_info=failure,
    )
    return False


async def _record_job_on(
    entity_client: _EntityWriter,
    row: ContainerImage,
    uid: str,
    attempt_id: str,
    still_this_submits: Callable[[], Awaitable[bool]],
) -> ContainerImage | None:
    """``row`` as it is stored once the job is recorded on it, on condition of the version it was read at. As
    it is stored, too, if it no longer waits. None if it still waits and could not be made to record the job
    in :data:`RECORD_TRIES` tries, or is no longer the row this submit holds.

    Raises :class:`_JobGone` if a write has to be made again and the job no longer holds the build's name.
    """
    current = row
    #: Whether the row is read before it is written: not the first time, when it is as this submit holds it.
    again = False
    for tried in range(RECORD_TRIES):
        if tried:
            await asyncio.sleep(RECORD_PAUSE_SECONDS)
        try:
            if again:
                # The write was refused, or failed, and time has passed. The row is read once more.
                current = await entity_client.get(ContainerImage, name=row.name, workspace=row.workspace)
                if current.id != row.id:
                    break
                if (current.provenance.job_uid, current.provenance.job_attempt_id) == (uid, attempt_id):
                    # This submit's own write, made though its answer was lost: no one else holds these IDs.
                    return current
                if current.status != "pending":
                    # Settled meanwhile, whatever job it came to record: returned as it settled.
                    return current
                # And Jobs is asked which job holds the build's name. Not the row: what it says of a job may
                # have been written by a submit whose job is gone. While the name is this submit's job's,
                # any other job the row records was deleted, and this one is written in its place. Once it
                # is not, this job is the one that was deleted, and the row is left to whoever has the name.
                if not await still_this_submits():
                    raise _JobGone(current)
            if current.status != "pending":
                # Settled meanwhile, or adopted as it was: no step completes it now.
                return current
            provenance = current.provenance.model_copy(update={"job_uid": uid, "job_attempt_id": attempt_id})
            return await entity_client.update(current.model_copy(update={"provenance": provenance}))
        except EntityConflictError:
            # It changed since it was read.
            again = True
        except EntityNotFoundError:
            # Gone, or another row has its name.
            break
        except _JobGone:
            raise
        except Exception:
            logger.warning(
                "image %s/%s could not record its build job, on try %d of %d",
                sanitize_for_log(row.workspace),
                sanitize_for_log(row.name),
                tried + 1,
                RECORD_TRIES,
                exc_info=True,
            )
            again = True
    logger.error(
        "image %s/%s does not record its build job: it kept changing, could not be written, or is no longer "
        "the row this submit holds",
        sanitize_for_log(row.workspace),
        sanitize_for_log(row.name),
    )
    return None


#: How many times the rows are read, and how long apart, by a submit that finds a job under its build's name
#: on rows that are still waiting and do not record it, before it answers that the job is not known to be
#: theirs. Another submit of the same request, made at the same moment, creates the job first and records it
#: on the rows a moment later.
ADOPT_READS = 5
ADOPT_PAUSE_SECONDS = 0.5


async def _as_stored(entity_client: _EntityWriter, row: ContainerImage) -> ContainerImage | None:
    """``row`` as it is stored now. None if it is gone, or another row has its name."""
    try:
        stored = await entity_client.get(ContainerImage, name=row.name, workspace=row.workspace)
    except EntityNotFoundError:
        return None
    return stored if stored.id == row.id else None


async def _rows_under(
    entity_client: _EntityWriter, rows: list[ContainerImage], uid: str, job_name: str
) -> list[ContainerImage]:
    """``rows`` as they are stored now, for a submit that finds the job with the ID ``uid`` under its build's
    name: returned if that job is the build's own, or nothing is left for a job to do. Raises
    :class:`BuildConflict` if not.

    The job is the build's own if every row still waiting records it. A row that has settled is not asked:
    no step completes it now, whichever job holds the name, and one that settled before its job was recorded
    never records one. So where no row is still waiting, the rows are returned as they are, as a build in
    this cluster returns them: whose job holds the name then decides nothing.

    Read anew: a row is recorded on after the job is created, by the submit that created it. One still
    waiting that does not record the job may be that submit's, a moment before it writes: it records no
    job, or the job an earlier submit of the request created, deleted since, which that write replaces.
    So the rows are read a few times before the answer is a conflict. Such a row is then left as it is.
    Failed here, it would be the build of the submit that created the job that failed: that submit's own
    write would find the row settled, and its job would build nothing. If the write never comes, the row is
    failed once the job has ended, as any abandoned row is, and until then no step of that job that is
    verified completes it: the row records no job for one to be a step of, or the attempt of another job.

    A row that is gone, or is no longer the one this submit holds, is not waited for. It does not come
    back, so the rows this submit holds will never all record the job. No row is written here, whatever
    the answer: one that is no longer this submit's is someone else's now.
    """
    waiting: list[ContainerImage] = []
    for read in range(ADOPT_READS):
        if read:
            await asyncio.sleep(ADOPT_PAUSE_SECONDS)
        stored = [await _as_stored(entity_client, row) for row in rows]
        current = [row for row in stored if row is not None]
        waiting = [row for row in current if row.status == "pending"]
        if len(current) != len(rows):
            # Not the build's own, whatever the rows that are left come to record.
            break
        if all(row.provenance.job_uid == uid for row in waiting):
            if waiting:
                logger.info("job %s already exists for this request; adopting it", sanitize_for_log(job_name))
            else:
                logger.info(
                    "a job named %s exists, and no image of this request still waits on a job; returning them "
                    "as they are",
                    sanitize_for_log(job_name),
                )
            return current
    conflict = f"a job named {job_name!r} already exists, and this request's images do not record it as their build job"
    if any(row.provenance.job_uid != uid for row in waiting):
        conflict += (
            ". Another submit of the same request may be recording it now: submit the request again in a moment. "
            "If the answer is still this, submit the next revision"
        )
    raise BuildConflict(conflict) from None


async def fail_rows(entity_client: _EntityWriter, rows: list[ContainerImage], detail: str) -> bool:
    """Fail each of ``rows`` still waiting, rather than leave it waiting on a job this submit didn't create,
    or one it gave up.

    One that changed since it was read, or can't be written, is logged and left. True if none was left.
    """
    settled = True
    for row in rows:
        if row.status != "pending":
            continue
        row.status = "failed"
        row.status_detail = detail
        try:
            await entity_client.update(row)
        except EntityConflictError:
            settled = False
            logger.warning(
                "image %s/%s changed while being failed; leaving it as it is",
                sanitize_for_log(row.workspace),
                sanitize_for_log(row.name),
            )
        except Exception:
            settled = False
            logger.exception(
                "image %s/%s could not be failed, and stays pending",
                sanitize_for_log(row.workspace),
                sanitize_for_log(row.name),
            )
    return settled


async def _adopt(
    entity_client: _EntityWriter, row: ContainerImage, digest: str, build_set: BuildSet, workspace: str, backend: str
) -> ContainerImage:
    existing = await entity_client.get(ContainerImage, name=row.name, workspace=workspace)
    if existing.provenance.request_digest != digest:
        raise _different_request(build_set)
    # The same request, but another backend holds it: its image is, or would be, somewhere this backend doesn't
    # publish. Whether anything was built is not looked at.
    if existing.provenance.backend != backend:
        raise BackendChanged(
            f"build set {build_set.name!r} revision {build_set.revision} was already submitted under the "
            f"{existing.provenance.backend!r} backend, and this deployment now builds with {backend!r}. "
            "Submit a new revision."
        )
    return existing


def _different_request(build_set: BuildSet) -> BuildConflict:
    return BuildConflict(
        f"build set {build_set.name!r} revision {build_set.revision} was already submitted with a "
        "different request. Submit a new revision; a revision names one request."
    )
