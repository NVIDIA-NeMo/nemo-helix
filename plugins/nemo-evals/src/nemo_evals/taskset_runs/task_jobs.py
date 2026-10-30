# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task Jobs of a run: the request each task's Job is created from, and how the run controller registers one.

Every Job input comes from the stored plan, so a create after a config change is exactly what the first
attempt would have sent. The controller registers a task in three steps: write the create
intent (``task_job``) on its member, send one named create, then mark the member registered. A create whose
outcome is unknown keeps its slot until the Job is found, or until ``create_grace`` proves none is coming.
"""

import logging
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Literal, Protocol

from nemo_evals.jobs.harbor_task_dispatch import TASK_JOB_SOURCE, HarborTaskDispatchJob
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment, build_task_job_input
from nemo_evals.taskset_runs.entities import MemberPhase, TasksetRunEntity, TasksetRunMemberEntity
from nemo_evals.taskset_runs.identifiers import task_job_name
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_evals.taskset_runs.store import TasksetRunStore
from nemo_helix_plugin.client.errors import (
    BadRequestError,
    ConflictError,
    NemoHTTPError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse

logger = logging.getLogger(__name__)


class TaskJobs(Protocol):
    """The Jobs check submit needs, acting as the caller."""

    async def is_subprocess_profile(self, profile: str) -> bool:
        """True when Jobs runs ``profile``'s CPU steps as host subprocesses."""
        ...


class JobsServiceTaskJobs:
    """:class:`TaskJobs` over the Jobs service."""

    def __init__(self, jobs: AsyncJobsClient) -> None:
        self._jobs = jobs

    async def is_subprocess_profile(self, profile: str) -> bool:
        profiles = (await self._jobs.get_execution_profiles()).data()
        return any(isinstance(item, SubprocessJobExecutionProfile) and item.profile == profile for item in profiles)


async def task_job_request(
    plan: TasksetRunPlan, run: TasksetRunEntity, index: int, environment: PreparedHarborEnvironment
) -> CreateHelixJobRequest:
    """The Jobs request for task ``index``: its input as ``spec``, its compiled Job as ``platform_spec``."""
    spec = build_task_job_input(plan, run, index, environment)
    return CreateHelixJobRequest(
        name=task_job_name(run.name, index),
        description=f"Task {index} ({spec.spec.tasks[0].id}) of taskset run {run.name}",
        spec=spec.model_dump(mode="json"),
        platform_spec=await HarborTaskDispatchJob.compile(spec=spec, profile=plan.dispatcher_profile),
        source=TASK_JOB_SOURCE,
    )


def member_environment(plan: TasksetRunPlan, member: TasksetRunMemberEntity) -> PreparedHarborEnvironment:
    """The images ``member``'s task Job runs on: the plan's, with its agent layer as the agent image when it has one."""
    environment = plan.task_environments[member.task_index]
    if member.agent_layer is None:
        return environment
    return environment.model_copy(update={"agent": member.agent_layer, "agent_preinstalled": True})


def runs_request(job: HelixJobResponse, body: CreateHelixJobRequest) -> bool:
    """True when ``job`` has ``body``'s ``source``, ``spec`` and ``platform_spec``: it runs what ``body`` asks for.

    Jobs copies all three from the create request, so a Job someone else made under the same name differs
    in at least one of them unless it runs exactly this work.
    """
    return (
        job.source == body.source
        and job.spec == body.spec
        and job.platform_spec.model_dump(mode="json") == body.platform_spec.model_dump(mode="json")
    )


def past_grace(sent_at: datetime | None, now: datetime, grace: timedelta) -> bool:
    """A 404 or stepless read of this create is conclusive: no POST of ours can still be saving.

    Jobs keeps saving after the controller's client times out, and GET shows nothing until the Job's attempt
    is linked, so only ``grace`` after the last POST with an unknown outcome does a missing Job prove anything.
    """
    if sent_at is None:
        return True
    return now >= _utc(sent_at) + grace


def sent_at_after_refusal(overwritten: datetime | None, this_post: datetime) -> datetime:
    """A refusal (400/403/404/409/422) started nothing: a retry writes back the value it overwrote; the first POST keeps its own."""
    return overwritten or this_post


class RegistrationJobs(Protocol):
    """The Jobs calls registration needs: creates act for the run's creator, reads as the evals service."""

    async def create(self, request: CreateHelixJobRequest) -> HelixJobResponse: ...

    async def get(self, name: str) -> HelixJobResponse | None: ...

    async def has_steps(self, name: str) -> bool: ...


async def adoption(
    job: HelixJobResponse, request: CreateHelixJobRequest, jobs: RegistrationJobs
) -> Literal["ours", "foreign", "stepless"]:
    """Whether the Job under ``request.name`` is the one this request creates.

    ``foreign`` when it runs something else (``runs_request``). ``stepless`` when it is ours but its current
    attempt has no step yet: a create still saving, or one that failed midway and will never run (AIRCORE-1379).
    """
    if not runs_request(job, request):
        return "foreign"
    if not await jobs.has_steps(job.name):
        return "stepless"
    return "ours"


class Registration(StrEnum):
    """How one create of a member's task Job ended."""

    REGISTERED = "registered"  # a Job of ours is under task_job (its id in job_id): created, or adopted on 409
    FAILED = "failed"  # refused, or the name is lost: the member settled and gave its slot back
    PENDING = "pending"  # a refused retry, or a 409 whose Job isn't visible yet: holds its slot until past_grace
    AMBIGUOUS = "ambiguous"  # 401, 429, 5xx, timeout: the create may still be saving; ends this drain


_REFUSALS: tuple[tuple[type[NemoHTTPError], str], ...] = (
    (PermissionDeniedError, "job_forbidden"),
    (BadRequestError, "job_spec_invalid"),
    (NotFoundError, "job_spec_invalid"),
    (UnprocessableEntityError, "job_spec_invalid"),
)


async def register_member(
    member: TasksetRunMemberEntity,
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    jobs: RegistrationJobs,
    store: TasksetRunStore,
    *,
    now: datetime,
    grace: timedelta,
) -> tuple[Registration, TasksetRunMemberEntity]:
    """Create ``member``'s task Job: write the intent, send one named create, then record what happened.

    The intent (``task_job``, ``create_sent_at``) is written before the create, so the slot is counted from
    the moment a create may happen until the Job's fate is known, across timeouts and crashes. The same call
    retries a pending create: the member already holds its slot, and ``task_job`` never changes.

    A refusal proves only that this POST saved nothing. The first POST's refusal fails the member at once; a
    retry's refusal restores the earlier ``create_sent_at`` (that create may still be saving) and fails the
    member only once that is ``past_grace``. A 409 is also a refusal: Jobs answers it before saving anything,
    for a taken name and for a missing secret alike.

    Returns:
        The outcome and the member as last written.

    Raises:
        NemoEntityConflictError: A member write lost to another write; the next cycle re-reads it.
        Exception: Member writes that fail; the member keeps its intent, so the next cycle resolves it.
    """
    request = await task_job_request(plan, run, member.task_index, member_environment(plan, member))
    assert request.name is not None
    first_post = member.task_job is None
    overwritten = member.create_sent_at
    member = await store.update_member(member, task_job=request.name, create_sent_at=now)
    try:
        created = await jobs.create(request)
    except ConflictError as exc:
        return await _after_conflict(
            member, request, jobs, store, now=now, grace=grace, overwritten=overwritten, exc=exc
        )
    except NemoHTTPError as exc:
        code = next((code for refused, code in _REFUSALS if isinstance(exc, refused)), None)
        if code is None:
            logger.warning("Creating task Job %s failed: %s", request.name, exc)
            return Registration.AMBIGUOUS, member
        error = f"{code}: {exc.detail}"
        if first_post:
            return Registration.FAILED, await store.update_member(member, phase=MemberPhase.FAILED, error=error)
        return await _refused_retry(member, store, now=now, grace=grace, overwritten=overwritten, error=error)
    except Exception as exc:
        logger.warning("Creating task Job %s failed: %s", request.name, exc or type(exc).__name__)
        return Registration.AMBIGUOUS, member
    return Registration.REGISTERED, await store.update_member(
        member, phase=MemberPhase.REGISTERED, job_id=created.id, job_status=created.status
    )


async def _after_conflict(
    member: TasksetRunMemberEntity,
    request: CreateHelixJobRequest,
    jobs: RegistrationJobs,
    store: TasksetRunStore,
    *,
    now: datetime,
    grace: timedelta,
    overwritten: datetime | None,
    exc: ConflictError,
) -> tuple[Registration, TasksetRunMemberEntity]:
    """A 409: adopt our own Job, fail on someone else's, and wait out a Job that can't be seen yet."""
    assert request.name is not None
    try:
        job = await jobs.get(request.name)
        found = await adoption(job, request, jobs) if job is not None else None
    except Exception as read_error:
        logger.warning("Reading task Job %s after a 409 failed: %s", request.name, read_error)
        # The 409 saved nothing, so this POST opens no window: restore the earlier create's, or the slot would
        # outlive create_grace while the lookup keeps failing.
        sent_at = sent_at_after_refusal(overwritten, now)
        if sent_at != member.create_sent_at:
            member = await store.update_member(member, create_sent_at=sent_at)
        return Registration.AMBIGUOUS, member
    if job is not None and found == "ours":
        return Registration.REGISTERED, await store.update_member(
            member, phase=MemberPhase.REGISTERED, job_id=job.id, job_status=job.status
        )
    if found == "foreign":
        error = f"job_name_conflict: Job {request.name} exists and runs something else"
        return Registration.FAILED, await store.update_member(member, phase=MemberPhase.FAILED, error=error)
    # 404 or stepless: an earlier create of ours may still be saving, or the name is lost (AIRCORE-1379).
    error = f"job_name_conflict: {exc.detail}"
    return await _refused_retry(member, store, now=now, grace=grace, overwritten=overwritten, error=error)


async def _refused_retry(
    member: TasksetRunMemberEntity,
    store: TasksetRunStore,
    *,
    now: datetime,
    grace: timedelta,
    overwritten: datetime | None,
    error: str,
) -> tuple[Registration, TasksetRunMemberEntity]:
    """This POST saved nothing: restore the window an earlier create opened, then fail once it is past."""
    sent_at = sent_at_after_refusal(overwritten, now)
    if past_grace(sent_at, now, grace):
        return Registration.FAILED, await store.update_member(
            member, phase=MemberPhase.FAILED, error=error, create_sent_at=sent_at
        )
    if sent_at != member.create_sent_at:
        member = await store.update_member(member, create_sent_at=sent_at)
    return Registration.PENDING, member


def _utc(value: datetime) -> datetime:
    """Stored timestamps may come back naive; they are UTC."""
    return value if value.utcoffset() is not None else value.replace(tzinfo=timezone.utc)
