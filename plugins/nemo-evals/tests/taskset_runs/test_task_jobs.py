# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fan-out creates each task Job once, and adopts an existing one only when it is provably ours."""

import pytest
from nemo_evals.jobs.harbor_task_dispatch import TASK_JOB_SOURCE
from nemo_evals.jobs.harbor_task_spec import HarborTaskJobSpec, PreparedHarborEnvironment, PreparedImage
from nemo_evals.taskset_runs.task_jobs import TaskJobsIncompleteError, create_task_jobs, task_job_request
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

IMAGE = PreparedImage(image_ref="registry.invalid/agent@sha256:" + "a" * 64, preparation_key="k")
ENVIRONMENT = PreparedHarborEnvironment(task_revision_digest="r", archive_digest="d", agent=IMAGE)
NAMES = ["run-0123456789abcdef-task-0", "run-0123456789abcdef-task-1"]


async def test_creates_one_job_per_task(fake_task_jobs, harbor_plan, harbor_run):
    names = await create_task_jobs(fake_task_jobs, harbor_plan, harbor_run, [ENVIRONMENT, ENVIRONMENT])
    assert names == NAMES == [request.name for request in fake_task_jobs.created]


async def test_request_matches_the_jobs_api_contract(harbor_plan, harbor_run):
    request = await task_job_request(harbor_plan, harbor_run, 0, ENVIRONMENT)
    golden = CreateHelixJobRequest.model_validate_json(request.model_dump_json())
    assert golden == request
    assert request.source == TASK_JOB_SOURCE and request.name == NAMES[0]
    assert HarborTaskJobSpec.model_validate(request.spec).task_index == 0
    assert request.platform_spec.steps[-1].config == request.spec


async def test_retry_adopts_its_own_jobs_without_recreating(fake_task_jobs, harbor_plan, harbor_run):
    await create_task_jobs(fake_task_jobs, harbor_plan, harbor_run, [ENVIRONMENT, ENVIRONMENT])
    assert await create_task_jobs(fake_task_jobs, harbor_plan, harbor_run, [ENVIRONMENT, ENVIRONMENT]) == NAMES
    assert len(fake_task_jobs.created) == 2


async def test_lost_create_reply_is_reported_then_adopted(fake_task_jobs, harbor_plan, harbor_run):
    fake_task_jobs.lose_reply = {NAMES[1]}
    with pytest.raises(TaskJobsIncompleteError) as exc:
        await create_task_jobs(fake_task_jobs, harbor_plan, harbor_run, [ENVIRONMENT, ENVIRONMENT])
    assert [index for index, _ in exc.value.missing] == [1]
    assert await create_task_jobs(fake_task_jobs, harbor_plan, harbor_run, [ENVIRONMENT, ENVIRONMENT]) == NAMES


async def test_conflict_without_a_job_is_missing(fake_task_jobs, harbor_plan, harbor_run):
    fake_task_jobs.refuse = {NAMES[0]}
    with pytest.raises(TaskJobsIncompleteError, match="secret may be missing") as exc:
        await create_task_jobs(fake_task_jobs, harbor_plan, harbor_run, [ENVIRONMENT, ENVIRONMENT])
    assert [index for index, _ in exc.value.missing] == [0]
    assert [request.name for request in fake_task_jobs.created] == [NAMES[1]]


async def test_job_saved_without_its_first_step_is_never_adopted(fake_task_jobs, harbor_plan, harbor_run):
    fake_task_jobs.fail_create = {NAMES[0]}
    with pytest.raises(TaskJobsIncompleteError, match="create failed"):
        await create_task_jobs(fake_task_jobs, harbor_plan, harbor_run, [ENVIRONMENT, ENVIRONMENT])
    with pytest.raises(TaskJobsIncompleteError, match="without its first step"):
        await create_task_jobs(fake_task_jobs, harbor_plan, harbor_run, [ENVIRONMENT, ENVIRONMENT])


async def test_foreign_job_under_the_name_is_never_adopted(fake_task_jobs, harbor_plan, harbor_run):
    other = await task_job_request(harbor_plan, harbor_run, 0, ENVIRONMENT)
    other = other.model_copy(update={"spec": {**other.spec, "task_index": 9}})
    await fake_task_jobs.create("default", other)
    with pytest.raises(TaskJobsIncompleteError, match="runs something else"):
        await create_task_jobs(fake_task_jobs, harbor_plan, harbor_run, [ENVIRONMENT, ENVIRONMENT])


async def test_task_without_images_is_missing(fake_task_jobs, harbor_plan, harbor_run):
    with pytest.raises(TaskJobsIncompleteError, match="task 1: image_unavailable"):
        await create_task_jobs(fake_task_jobs, harbor_plan, harbor_run, [ENVIRONMENT, "image_unavailable: gone"])
