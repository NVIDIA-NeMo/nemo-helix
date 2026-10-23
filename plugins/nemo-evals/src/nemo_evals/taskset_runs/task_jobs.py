# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task Jobs of a run: the request each task's Job is created from, and submit's one Jobs check.

Every Job input comes from the stored plan, so a create after a config change is exactly what the first
attempt would have sent. TODO(M4 story 4): the controller registers task Jobs.
"""

import logging
from typing import Protocol

from nemo_evals.jobs.harbor_task_dispatch import TASK_JOB_SOURCE, HarborTaskDispatchJob
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment, build_task_job_input
from nemo_evals.taskset_runs.entities import TasksetRunEntity
from nemo_evals.taskset_runs.identifiers import task_job_name
from nemo_evals.taskset_runs.specs import TasksetRunPlan
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
