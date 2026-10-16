# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit checks the deployment can run the request, prepares images, then creates one task Job per task."""

import pytest
from nemo_evals.api.fields import TaskRef
from nemo_evals.api.service.taskset_run_service import TasksetRunResolutionError
from nemo_evals.api.taskset_run_schemas import TasksetRunInput
from nemo_evals.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evals.jobs.harbor_task_spec import HarborTaskJobSpec
from nemo_evals.revisions import publish_revision
from nemo_evals.taskset_runs.entities import TasksetRunEntity
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
