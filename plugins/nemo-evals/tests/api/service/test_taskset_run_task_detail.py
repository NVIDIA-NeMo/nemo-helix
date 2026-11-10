# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One task of a run, with its trials read live from its task Job."""

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_evals.taskset_runs.entities import MemberPhase, TasksetRunMemberEntity
from nemo_evals.taskset_runs.trials import TrialCounts, TrialPhase
from nemo_helix_plugin.client.errors import NemoTransportError, PermissionDeniedError


@pytest.fixture
def add_member(entity_store):
    async def add(run, index: int, **fields: Any) -> TasksetRunMemberEntity:
        member = TasksetRunMemberEntity(name=f"task-{index}", workspace="default", task_index=index, **fields)
        member._parent = run.id
        return await entity_store.create(member)

    return add


async def test_a_task_with_a_job_has_its_jobs_trials(service, seed_run, add_member, fake_task_jobs):
    run = await seed_run()
    job = f"{run.name}-task-3"
    await add_member(
        run,
        3,
        task_name="hello",
        phase=MemberPhase.REGISTERED,
        task_job=job,
        job_id="job-3",
        # The controller's last copy, from before the trial started.
        trial_counts=TrialCounts(total=1, phases={TrialPhase.PENDING: 1}),
    )
    fake_task_jobs.details[job] = {"trials": [{"trial": "hello__a", "phase": "agent_running", "attempts": 2}]}
    fake_task_jobs.job_ids[job] = "job-3"

    task = await service.get_task("default", run.name, 3)

    assert task is not None and (task.task, task.task_name, task.job) == ("task-3", "hello", job)

    assert task.trials is not None
    assert [(t.trial, t.phase, t.attempts) for t in task.trials] == [("hello__a", TrialPhase.AGENT_RUNNING, 2)]

    # Counted from the list just read, not the controller's copy.
    assert task.trial_counts == TrialCounts(total=1, phases={TrialPhase.AGENT_RUNNING: 1})


async def test_a_replacement_job_under_the_name_does_not_supply_trials(service, seed_run, add_member, fake_task_jobs):
    """Jobs deletes only ended Jobs, so once ours ended and was deleted another Job can take its name."""
    run = await seed_run()
    job = f"{run.name}-task-3"
    controllers_copy = TrialCounts(total=1, phases={TrialPhase.SUCCEEDED: 1}, rewarded=1, reward=1.0)
    await add_member(run, 3, phase=MemberPhase.REGISTERED, task_job=job, job_id="job-3", trial_counts=controllers_copy)
    fake_task_jobs.details[job] = {"trials": [{"trial": "other__a", "phase": "agent_running", "attempts": 1}]}
    fake_task_jobs.job_ids[job] = "job-9"  # not the Job the member registered

    task = await service.get_task("default", run.name, 3)

    assert task is not None
    assert task.trials is None
    assert task.trial_counts == controllers_copy  # the stored counts stand, not the other Job's


async def test_a_foreign_job_that_rejected_registration_does_not_supply_trials(
    service, seed_run, add_member, fake_task_jobs
):
    """A foreign Job already owned the name, so the member holds the slot (task_job) but never registered a job_id."""
    run = await seed_run()
    job = f"{run.name}-task-0"
    await add_member(run, 0, phase=MemberPhase.FAILED, task_job=job, error="job_name_conflict")
    fake_task_jobs.details[job] = {"trials": [{"trial": "foreign__a", "phase": "agent_running", "attempts": 1}]}
    fake_task_jobs.job_ids[job] = "job-foreign"

    task = await service.get_task("default", run.name, 0)

    assert task is not None and task.error == "job_name_conflict"
    assert task.trials is None


async def test_a_queued_task_never_reads_jobs(service, seed_run, add_member, fake_task_jobs, monkeypatch):
    run = await seed_run()
    await add_member(run, 0, phase=MemberPhase.IMAGE_READY)

    monkeypatch.setattr(fake_task_jobs, "job_details", AsyncMock(side_effect=AssertionError("Jobs call")))

    task = await service.get_task("default", run.name, 0)

    assert task is not None and task.trials is None


@pytest.mark.parametrize("job_state", ["gone", "no trials reported", "denied", "unreachable"])
async def test_a_task_whose_trials_cant_be_read_still_reads(service, seed_run, add_member, fake_task_jobs, job_state):
    run = await seed_run()
    job = f"{run.name}-task-0"
    await add_member(run, 0, phase=MemberPhase.FAILED, task_job=job, job_id="job-0", error="job_failed")

    if job_state == "no trials reported":
        fake_task_jobs.details[job] = {"publication": {}}
        fake_task_jobs.job_ids[job] = "job-0"  # ours, so it's the missing trials that leave the task bare
    elif job_state == "denied":
        response = httpx.Response(403, request=httpx.Request("GET", "http://jobs.test"))
        fake_task_jobs.failures[job] = PermissionDeniedError(response)
    elif job_state == "unreachable":
        fake_task_jobs.failures[job] = NemoTransportError(httpx.ConnectError("refused"))

    task = await service.get_task("default", run.name, 0)

    assert task is not None and task.error == "job_failed" and task.trials is None


async def test_a_missing_run_or_task_is_none(service, seed_run, add_member):
    run = await seed_run()
    await add_member(run, 0, phase=MemberPhase.IMAGE_READY)

    assert await service.get_task("default", "run-missing", 0) is None
    assert await service.get_task("other", run.name, 0) is None
    assert await service.get_task("default", run.name, 1) is None
