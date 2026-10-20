# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create one task Job per task of an accepted run, or adopt the one an earlier attempt created.

Run by submit right after acceptance. Every Job input comes from the stored plan, so a retry after a
config change creates the remaining Jobs exactly as the first attempt would have.

TODO(M4 story 4): the controller registers task Jobs instead.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from nemo_evals.jobs.harbor_task_dispatch import TASK_JOB_SOURCE, HarborTaskDispatchJob
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment, build_task_job_input
from nemo_evals.taskset_runs.entities import TasksetRunEntity
from nemo_evals.taskset_runs.identifiers import task_job_name
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse

logger = logging.getLogger(__name__)


class TaskJobs(Protocol):
    """The Jobs operations fan-out and its submit checks need, acting as the run's caller."""

    async def is_subprocess_profile(self, profile: str) -> bool:
        """True when Jobs runs ``profile``'s CPU steps as host subprocesses."""
        ...

    async def create(self, workspace: str, request: CreateHelixJobRequest) -> None:
        """Create the Job. Raises ``ConflictError`` on a 409 (name taken, or a secret missing)."""
        ...

    async def get(self, workspace: str, name: str) -> HelixJobResponse | None:
        """The Job, or None when it doesn't exist."""
        ...

    async def has_steps(self, workspace: str, name: str) -> bool:
        """True when the Job's current attempt has at least one step."""
        ...


class JobsServiceTaskJobs:
    """:class:`TaskJobs` over the Jobs service."""

    def __init__(self, jobs: AsyncJobsClient) -> None:
        self._jobs = jobs

    async def is_subprocess_profile(self, profile: str) -> bool:
        profiles = (await self._jobs.get_execution_profiles()).data()
        return any(isinstance(item, SubprocessJobExecutionProfile) and item.profile == profile for item in profiles)

    async def create(self, workspace: str, request: CreateHelixJobRequest) -> None:
        await self._jobs.create_job(workspace=workspace, body=request)

    async def get(self, workspace: str, name: str) -> HelixJobResponse | None:
        try:
            return (await self._jobs.get_job(workspace=workspace, name=name)).data()
        except NotFoundError:
            return None

    async def has_steps(self, workspace: str, name: str) -> bool:
        response = await self._jobs.list_steps(workspace=workspace, name=name, query_params={"page_size": 1})
        return bool(response.page().items)


@dataclass(frozen=True)
class TaskJobsIncompleteError(Exception):
    """Some task Jobs couldn't be created or adopted; a retry with the same idempotency_key completes them."""

    run_id: str
    missing: tuple[tuple[int, str], ...]  # (task index, reason)
    notices: tuple[str, ...] = ()  # what submit had to say about the run before the Jobs failed

    def __str__(self) -> str:
        detail = "; ".join(f"task {index}: {reason}" for index, reason in self.missing)
        message = f"task Jobs of run {self.run_id} not created ({detail})"
        return "; ".join((message, *self.notices))


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


async def create_task_jobs(
    jobs: TaskJobs,
    plan: TasksetRunPlan,
    run: TasksetRunEntity,
    environments: Sequence[PreparedHarborEnvironment | str],
) -> list[str]:
    """Create or adopt every task Job of ``run``; return their names in task order.

    ``environments[i]`` is task i's prepared images, or why they couldn't be prepared.

    A 409 is ours only when the existing Job has the same ``source``, ``spec`` and ``platform_spec``
    and at least one step: Jobs also answers 409 for a missing secret before any Job exists, and it
    saves a Job and its attempt before creating the first step, with no rollback.

    Raises:
        TaskJobsIncompleteError: Any task Job is missing; the others were created or adopted.
    """
    names: list[str] = []
    missing: list[tuple[int, str]] = []
    for index, environment in enumerate(environments):
        name = task_job_name(run.name, index)
        names.append(name)
        if isinstance(environment, str):
            missing.append((index, environment))
            continue
        body = await task_job_request(plan, run, index, environment)
        try:
            await jobs.create(run.workspace, body)
            continue
        except ConflictError:
            pass
        except Exception as exc:
            logger.warning("Creating task Job %s failed", name, exc_info=True)
            missing.append((index, f"create failed: {type(exc).__name__}"))
            continue
        try:
            reason = await _adoption_refused(jobs, run.workspace, body)
        except Exception as exc:
            # Like a failed create: report the task missing so the other tasks still get their Jobs.
            logger.warning("Inspecting existing task Job %s failed", name, exc_info=True)
            missing.append((index, f"adoption check failed: {type(exc).__name__}"))
            continue
        if reason is not None:
            missing.append((index, reason))
    if missing:
        raise TaskJobsIncompleteError(run.name, tuple(missing))
    return names


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


async def _adoption_refused(jobs: TaskJobs, workspace: str, body: CreateHelixJobRequest) -> str | None:
    """Why the Job already under ``body.name`` isn't this request's, or None when it is."""
    assert body.name is not None
    job = await jobs.get(workspace, body.name)
    if job is None:
        # A missing secret, or a create that failed before linking the attempt (AIRCORE-1379). Deleting frees the
        # name only when no attempt was saved; otherwise the Job can't be cancelled or deleted.
        return (
            f"Jobs refused the create of {body.name}: a referenced secret may be missing or unreadable, or an earlier "
            "create failed midway. If `nemo jobs delete` doesn't free the name, resubmit with a new idempotency_key"
        )
    if not runs_request(job, body):
        return f"Job {body.name} exists and runs something else; delete it with `nemo jobs delete`"
    if not await jobs.has_steps(workspace, body.name):
        # Cancel leaves its attempt `created` and delete then answers 409, so the name stays taken (AIRCORE-1379).
        return (
            f"Job {body.name} was saved without its first step and will never run; Jobs can't cancel or delete it, "
            "so resubmit with a new idempotency_key"
        )
    return None
