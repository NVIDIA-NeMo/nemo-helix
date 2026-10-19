# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit checks the deployment can run the request, prepares images, then creates one task Job per task."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nemo_evals.api.fields import TaskRef
from nemo_evals.api.service.taskset_run_service import MAX_CONCURRENT_IMAGE_PREPARATIONS, TasksetRunResolutionError
from nemo_evals.api.taskset_run_schemas import TasksetRunInput
from nemo_evals.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evals.jobs.harbor_task_spec import HarborTaskJobSpec
from nemo_evals.revisions import publish_revision
from nemo_evals.taskset_runs.entities import TasksetRunEntity, TasksetRunPhase
from nemo_evals.taskset_runs.task_jobs import TaskJobsIncompleteError

pytestmark = pytest.mark.usefixtures("published")


async def submit(service, request_body):
    return await service.submit(TasksetRunInput.model_validate(request_body), workspace="default", caller="user:alice")


async def runs(entity_store):
    return (await entity_store.list(TasksetRunEntity, workspace="default")).data


async def test_runs_off_in_this_deployment(service, request_body, entity_store):
    service._config.runs.harbor = None
    with pytest.raises(TasksetRunResolutionError, match="aren't configured in this deployment"):
        await submit(service, request_body)
    assert not await runs(entity_store)


async def test_unknown_sandbox_policy_lists_the_profiles(service, request_body):
    request_body["execution"]["sandbox_policy"] = "strict"
    with pytest.raises(TasksetRunResolutionError, match="harbor-open, harbor-standard"):
        await submit(service, request_body)


async def test_open_profile_is_frozen_into_the_plan(service, request_body, run_store, fake_task_jobs):
    request_body["execution"]["sandbox_policy"] = "harbor-open"
    outcome = await submit(service, request_body)
    plan = await run_store.load_plan(await run_store.find("default", outcome.status.run_id))
    assert plan.sandbox_policy.egress_allow == ("igw.svc.cluster.local", "inference-api.nvidia.com")
    (job,) = fake_task_jobs.created
    assert HarborTaskJobSpec.model_validate(job.spec).sandbox_policy == plan.sandbox_policy


async def test_dispatcher_profile_must_be_allowed(service, request_body):
    request_body["execution"]["dispatcher_profile"] = "gpu"
    with pytest.raises(TasksetRunResolutionError, match="isn't allowed; use one of: local-k8s, k8s-cpu"):
        await submit(service, request_body)


async def test_omitted_dispatcher_profile_uses_the_operators_first(
    service, request_body, harbor_config, fake_task_jobs, run_store
):
    request_body["execution"].pop("dispatcher_profile", None)
    service._config.runs.harbor = harbor_config.model_copy(update={"dispatcher_profiles": ("k8s-cpu", "local-k8s")})
    outcome = await submit(service, request_body)
    plan = await run_store.load_plan(await run_store.find("default", outcome.status.run_id))
    assert plan.dispatcher_profile == "k8s-cpu"


async def test_subprocess_dispatcher_profile_is_rejected(service, request_body, fake_task_jobs):
    fake_task_jobs.subprocess_profiles = {"local-k8s"}
    with pytest.raises(TasksetRunResolutionError, match="runs as a host subprocess"):
        await submit(service, request_body)


async def test_more_tasks_than_the_concurrency_cap_is_rejected(service, request_body, entity_store, published):
    task_head = await entity_store.get(TaskEntity, name="task-1", workspace="default")
    second = TaskEntity(
        name="task-2", workspace="default", spec=task_head.spec.model_copy(update={"native_task_id": "task-2"})
    )
    second, _, _ = await publish_revision(
        entity_store, entity_store, await entity_store.create(second), TaskRevisionEntity
    )
    pair = TasksetEntity(
        name="pair",
        workspace="default",
        tasks=[*published[1].tasks, TaskRef(f"default/task-2#{second.content_hash}")],
    )
    await publish_revision(entity_store, entity_store, await entity_store.create(pair), TasksetRevisionEntity)
    request_body["spec"]["tasks"] = "default/pair"
    request_body["execution"]["max_concurrent_task_jobs"] = 1
    with pytest.raises(TasksetRunResolutionError, match="2 tasks exceed execution.max_concurrent_task_jobs=1"):
        await submit(service, request_body)
    assert not await runs(entity_store)


async def test_task_without_images_is_rejected_before_acceptance(service, request_body, entity_store, run_preparer):
    run_preparer.fail = {"task-1": "image_unavailable"}
    with pytest.raises(TasksetRunResolutionError, match="tasks without images: task-1: image_unavailable"):
        await submit(service, request_body)
    assert not await runs(entity_store)


async def test_missing_task_job_is_503_and_a_retry_completes_it_from_the_plan(
    service, request_body, fake_task_jobs, entity_store, harbor_config
):
    fake_task_jobs.refuse = {"run-11f77762178b0e05-task-0"}
    with pytest.raises(TaskJobsIncompleteError) as exc:
        await submit(service, request_body)
    assert [index for index, _ in exc.value.missing] == [0]
    assert len(await runs(entity_store)) == 1 and not fake_task_jobs.created

    # The operator changes the dispatcher image before the retry; the Job still uses the plan's.
    service._config.runs.harbor = harbor_config.model_copy(update={"dispatcher_image": "r/x@sha256:" + "9" * 64})
    fake_task_jobs.refuse = set()
    replay = await submit(service, request_body)
    assert replay.created is False
    (job,) = fake_task_jobs.created
    assert job.platform_spec.steps[-1].executor.container.image == harbor_config.dispatcher_image
    assert replay.status.warnings == [f"task 0 (task-1): Job {job.name}"]


async def test_retry_uses_the_images_frozen_at_accept(service, request_body, fake_task_jobs, run_preparer):
    fake_task_jobs.refuse = {"run-11f77762178b0e05-task-0"}
    with pytest.raises(TaskJobsIncompleteError):
        await submit(service, request_body)
    calls = len(run_preparer.calls)

    # Before the retry the task's images change (a moved tag, a newer build), then can't be prepared at all.
    run_preparer.with_verifier = {"task-1"}
    run_preparer.fail = {"task-1": "image_unavailable"}
    fake_task_jobs.refuse = set()
    await submit(service, request_body)
    assert len(run_preparer.calls) == calls  # the retry never prepares images again
    (job,) = fake_task_jobs.created
    environment = HarborTaskJobSpec.model_validate(job.spec).environment
    assert environment.verifier is None and environment.agent.image_ref.startswith(
        "registry.invalid/fake/task-1-agent@"
    )


@pytest.mark.parametrize("miss_replay", [False, True], ids=["replay", "accept-conflict"])
async def test_an_ended_run_creates_no_task_jobs(
    service, request_body, fake_task_jobs, run_store, run_preparer, monkeypatch, miss_replay
):
    fake_task_jobs.refuse = {"run-11f77762178b0e05-task-0"}
    with pytest.raises(TaskJobsIncompleteError):
        await submit(service, request_body)
    # The controller expired the run before the caller retried.
    await run_store.set_state(await run_store.find("default", "run-11f77762178b0e05"), phase=TasksetRunPhase.EXPIRED)
    prepared = len(run_preparer.calls)
    if miss_replay:
        # Stands in for a submit that raced the winner past the replay lookup: accept returns the winner's run.
        monkeypatch.setattr(service._store, "replay", AsyncMock(return_value=None))
    fake_task_jobs.refuse = set()
    outcome = await submit(service, request_body)
    # Only the accept path prepares images again; this proves which path ran.
    assert (len(run_preparer.calls) > prepared) is miss_replay
    assert outcome.created is False and outcome.status.phase == TasksetRunPhase.EXPIRED
    assert outcome.status.warnings == [] and not fake_task_jobs.created


async def test_a_retry_after_the_execution_deadline_creates_no_task_jobs(
    service, request_body, fake_task_jobs, run_store, monkeypatch
):
    fake_task_jobs.refuse = {"run-11f77762178b0e05-task-0"}
    with pytest.raises(TaskJobsIncompleteError):
        await submit(service, request_body)
    run = await run_store.find("default", "run-11f77762178b0e05")
    # The controller hasn't expired the run yet; a Job created now would outlive its expiry pass.
    monkeypatch.setattr(service, "_clock", lambda: run.execution_deadline_at)
    fake_task_jobs.refuse = set()
    outcome = await submit(service, request_body)
    assert outcome.created is False and outcome.status.warnings == [] and not fake_task_jobs.created


async def test_failure_tolerance_is_rejected_until_enforced(service, request_body, entity_store):
    request_body["execution"]["max_failed_tasks"] = 0
    with pytest.raises(TasksetRunResolutionError, match="max_failed_tasks isn't enforced"):
        await submit(service, request_body)
    assert not await runs(entity_store)


@pytest.mark.parametrize("revoke", ["dispatcher_profile", "sandbox_policy", "runs"])
async def test_retry_after_execution_policy_change_starts_no_jobs(
    service, request_body, fake_task_jobs, entity_store, harbor_config, revoke
):
    """Today's operator config gates whether a retry may start the run's missing Jobs (422 at the route)."""
    fake_task_jobs.refuse = {"run-11f77762178b0e05-task-0"}
    with pytest.raises(TaskJobsIncompleteError):
        await submit(service, request_body)
    fake_task_jobs.refuse = set()

    execution = TasksetRunInput.model_validate(request_body).execution
    if revoke == "dispatcher_profile":
        frozen = execution.dispatcher_profile or harbor_config.dispatcher_profiles[0]  # what the plan froze
        profiles = [p for p in harbor_config.dispatcher_profiles if p != frozen]
        service._config.runs.harbor = harbor_config.model_copy(update={"dispatcher_profiles": profiles})
    elif revoke == "sandbox_policy":
        sandboxes = {k: v for k, v in harbor_config.sandbox_profiles.items() if k != execution.sandbox_policy}
        service._config.runs.harbor = harbor_config.model_copy(update={"sandbox_profiles": sandboxes})
    else:
        service._config.runs.harbor = None
    with pytest.raises(TasksetRunResolutionError):
        await submit(service, request_body)
    assert not fake_task_jobs.created and len(await runs(entity_store)) == 1


class SlowPreparer:
    """Prepares every task after a pause, recording how many preparations ran at once."""

    def __init__(self, *, raise_for: str | None = None) -> None:
        self.raise_for = raise_for
        self.running = self.most_running = 0
        self.cancelled: list[str] = []

    async def __call__(self, task, recorded, **_):
        self.running += 1
        self.most_running = max(self.most_running, self.running)
        try:
            await asyncio.sleep(0.01)
            if task.id == self.raise_for:
                raise RuntimeError(f"registry down for {task.id}")
            await asyncio.sleep(0.05)
            return SimpleNamespace(environment=f"images of {task.id}", error=None)  # what submit reads
        except asyncio.CancelledError:
            self.cancelled.append(task.id)
            raise
        finally:
            self.running -= 1


async def prepare(service, preparer, count):
    service._preparer = preparer
    tasks = [SimpleNamespace(id=f"task-{i}") for i in range(count)]
    return await service._prepare_images(tasks, None, "fail", caller="user:alice")


async def test_images_are_prepared_ten_at_a_time_in_task_order(service):
    preparer = SlowPreparer()
    results = await prepare(service, preparer, 25)
    assert preparer.most_running == MAX_CONCURRENT_IMAGE_PREPARATIONS == 10
    assert results == [f"images of task-{i}" for i in range(25)]


async def test_a_failed_preparation_cancels_the_others_and_fails_submit(service):
    preparer = SlowPreparer(raise_for="task-3")
    with pytest.raises(RuntimeError, match="registry down for task-3"):
        await prepare(service, preparer, 25)
    assert preparer.cancelled and preparer.running == 0  # the in-flight ones were stopped, none left running
