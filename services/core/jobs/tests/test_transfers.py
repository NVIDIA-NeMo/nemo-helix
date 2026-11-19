# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persisted transfer barriers across concurrent workload Tasks."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from nemo_helix_plugin.jobs.providers import ComputeResources, ContainerSpec, DistributedGPUExecutionProvider
from nemo_helix_plugin.jobs.spec import HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.transfers import (
    FilesetPull,
    FilesetPush,
    FilesetUploadResult,
    StepFilesets,
    TaskTransferProgress,
    TransferPhase,
)
from nemo_helix_plugin.jobs.types import HelixJobTaskUpdate
from nhx.common.entities.client import EntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.dispatcher import JobDispatcher
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt, HelixJobStep
from pydantic import ValidationError


async def execution(
    store: EntityClient, nodes: int = 2, when: str = "always"
) -> tuple[HelixJob, HelixJobAttempt, HelixJobStep]:
    push = FilesetPush.model_validate({"fileset_id": "destination", "directory": "/output", "when": when})
    spec = HelixJobSpec(
        steps=[
            HelixJobStepSpec(
                name="work",
                executor=DistributedGPUExecutionProvider(
                    container=ContainerSpec(image="busybox"), resources=ComputeResources(num_nodes=nodes)
                ),
                filesets=StepFilesets(push=[push]),
            )
        ]
    )
    job = await store.create(
        HelixJob(
            name="transfer-" + uuid4().hex,
            workspace="default",
            source="test",
            platform_spec=spec,
            fileset="logs",
            status=HelixJobStatus.ACTIVE,
        )
    )
    attempt = await store.create(
        HelixJobAttempt(
            name="attempt", workspace="default", job=job.id, platform_spec=spec, status=HelixJobStatus.ACTIVE
        )
    )
    job.current_attempt_id = attempt.id
    job = await store.update(job)
    step = await store.create(
        HelixJobStep(name="work", workspace="default", attempt_id=attempt.id, status=HelixJobStatus.ACTIVE)
    )
    return job, attempt, step


async def progress(
    dispatcher: JobDispatcher,
    job: HelixJob,
    step: HelixJobStep,
    task: str,
    phase: TransferPhase,
    *,
    exit_code: int | None = None,
    leader: bool = False,
    success: bool = True,
    stopped: bool = False,
    runtime_id: str = "",
):
    results = (
        [FilesetUploadResult(fileset_id="destination", success=success)]
        if phase in {"uploaded", "upload_failed"}
        else []
    )
    return await dispatcher.create_or_update_task(
        job.id,
        task,
        "default",
        HelixJobTaskUpdate(
            status=HelixJobStatus.ACTIVE,
            transfer=TaskTransferProgress(
                phase=phase, runtime_id=runtime_id, exit_code=exit_code, uploads=results, leader=leader, stopped=stopped
            ),
        ),
        step,
    )


def commit_response(mock_files_client: AsyncMock) -> None:
    metadata = MagicMock()
    metadata.id = "destination"
    metadata.commit_revision = "commit-one"
    response = MagicMock()
    response.data.return_value = metadata
    mock_files_client.get_fileset_by_id.return_value = response
    mock_files_client.commit_fileset.return_value = response


async def test_all_uploaders_acknowledge_before_commit_and_terminal_status(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, mock_files_client: AsyncMock
):
    commit_response(mock_files_client)
    job, _, step = await execution(mock_store)
    assert (await progress(mock_dispatcher, job, step, "left", "ready")).transfer_action == "wait"
    assert (await progress(mock_dispatcher, job, step, "right", "ready")).transfer_action == "run"
    assert (await progress(mock_dispatcher, job, step, "left", "ready")).transfer_action == "run"
    assert (await progress(mock_dispatcher, job, step, "left", "finished", exit_code=0)).transfer_action == "wait"
    assert (
        await progress(mock_dispatcher, job, step, "right", "finished", exit_code=0)
    ).transfer_action == "upload_success"
    assert (await progress(mock_dispatcher, job, step, "left", "uploaded", exit_code=0)).transfer_action == "wait"
    mock_files_client.commit_fileset.assert_not_awaited()
    saved, _ = await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.COMPLETED)
    assert saved.status == HelixJobStatus.ACTIVE
    assert (await progress(mock_dispatcher, job, step, "right", "uploaded", exit_code=0)).transfer_action == "done"
    mock_files_client.commit_fileset.assert_awaited_once_with(workspace="default", name="id:destination")
    assert (await progress(mock_dispatcher, job, step, "left", "uploaded", exit_code=0)).transfer_action == "done"
    mock_files_client.commit_fileset.assert_awaited_once()
    saved, _ = await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.COMPLETED)
    assert saved.status == HelixJobStatus.COMPLETED


@pytest.mark.parametrize("leader", [True, False])
async def test_recorded_workload_result_stops_other_processes_before_upload(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, leader: bool
):
    job, _, step = await execution(mock_store)
    await progress(mock_dispatcher, job, step, "left", "ready")
    await progress(mock_dispatcher, job, step, "right", "ready")
    code = 0 if leader else 9
    await progress(mock_dispatcher, job, step, "left", "finished", exit_code=code, leader=leader)
    assert (await progress(mock_dispatcher, job, step, "right", "running")).transfer_action == "stop"
    result = await progress(mock_dispatcher, job, step, "right", "finished", exit_code=-1, stopped=True)
    assert result.transfer_action == ("upload_success" if leader else "upload_failure")


async def test_one_upload_failure_suppresses_commit_and_fails_step(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, mock_files_client: AsyncMock
):
    job, _, step = await execution(mock_store)
    await progress(mock_dispatcher, job, step, "left", "ready")
    await progress(mock_dispatcher, job, step, "right", "ready")
    for task in ("left", "right"):
        await progress(mock_dispatcher, job, step, task, "finished", exit_code=0)
    await progress(mock_dispatcher, job, step, "left", "uploaded", exit_code=0)
    assert (
        await progress(mock_dispatcher, job, step, "right", "upload_failed", exit_code=0, success=False)
    ).transfer_action == "done"
    mock_files_client.commit_fileset.assert_not_awaited()
    saved, _ = await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.COMPLETED)
    assert saved.status == HelixJobStatus.ERROR


async def test_pull_failure_aborts_all_without_compute(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    job, _, step = await execution(mock_store)
    await progress(mock_dispatcher, job, step, "left", "ready")
    assert (await progress(mock_dispatcher, job, step, "right", "pull_failed")).transfer_action == "abort"
    assert (await progress(mock_dispatcher, job, step, "left", "ready")).transfer_action == "abort"
    saved, _ = await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.ERROR)
    assert saved.status == HelixJobStatus.ERROR


async def test_concurrent_steps_have_independent_barriers(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, mock_files_client: AsyncMock
):
    commit_response(mock_files_client)
    first, _, step_a = await execution(mock_store, nodes=1)
    second, _, step_b = await execution(mock_store, nodes=1)
    await progress(mock_dispatcher, first, step_a, "task", "ready")
    await progress(mock_dispatcher, second, step_b, "task", "ready")
    await progress(mock_dispatcher, first, step_a, "task", "finished", exit_code=0)
    assert (await progress(mock_dispatcher, first, step_a, "task", "uploaded", exit_code=0)).transfer_action == "done"
    assert (await progress(mock_dispatcher, second, step_b, "task", "running")).transfer_action == "wait"
    mock_files_client.commit_fileset.assert_awaited_once()


async def test_controller_reconstruction_uses_persisted_acknowledgements(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    mock_files_client: AsyncMock,
    mock_secrets_client: AsyncMock,
):
    commit_response(mock_files_client)
    job, _, step = await execution(mock_store, nodes=1)
    await progress(mock_dispatcher, job, step, "task", "ready")
    await progress(mock_dispatcher, job, step, "task", "finished", exit_code=0)
    replacement = JobDispatcher(mock_store, mock_files_client, mock_secrets_client)
    assert (await progress(replacement, job, step, "task", "uploaded", exit_code=0)).transfer_action == "done"
    assert (await progress(mock_dispatcher, job, step, "task", "uploaded", exit_code=0)).transfer_action == "done"
    mock_files_client.commit_fileset.assert_awaited_once()


@pytest.mark.parametrize("path", ["/absolute", "../escape", "a/../../escape", "a\\\\b"])
def test_remote_path_containment(path: str):
    with pytest.raises(ValidationError):
        FilesetPull(fileset_id="source", directory="/work", path=path)


def test_overlapping_destinations_are_valid_and_default_to_always():
    config = StepFilesets(
        push=[
            FilesetPush(fileset_id="dst", directory="/work", path=""),
            FilesetPush(fileset_id="dst", directory="/work/other", path="nested"),
        ]
    )
    assert all(entry.when == "always" for entry in config.push)


async def test_native_restart_does_not_reuse_prior_task_progress(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, mock_files_client: AsyncMock
):
    commit_response(mock_files_client)
    job, _, step = await execution(mock_store, nodes=1)
    await progress(mock_dispatcher, job, step, "task", "pulling", runtime_id="old")
    await progress(mock_dispatcher, job, step, "task", "ready", runtime_id="old")
    await progress(mock_dispatcher, job, step, "task", "finished", exit_code=0, runtime_id="old")
    assert (
        await progress(mock_dispatcher, job, step, "task", "uploaded", exit_code=0, runtime_id="old")
    ).transfer_action == "done"
    assert (await progress(mock_dispatcher, job, step, "task", "pulling", runtime_id="new")).transfer_action == "wait"
    assert (await progress(mock_dispatcher, job, step, "task", "ready", runtime_id="old")).transfer_action == "abort"
    assert (await progress(mock_dispatcher, job, step, "task", "ready", runtime_id="new")).transfer_action == "run"
    mock_files_client.commit_fileset.assert_awaited_once()


async def test_native_restart_during_workload_retires_old_generation(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient
):
    job, _, step = await execution(mock_store, nodes=1)
    await progress(mock_dispatcher, job, step, "task", "pulling", runtime_id="old")
    await progress(mock_dispatcher, job, step, "task", "ready", runtime_id="old")
    assert (await progress(mock_dispatcher, job, step, "task", "pulling", runtime_id="new")).transfer_action == "wait"
    assert (
        await progress(mock_dispatcher, job, step, "task", "finished", exit_code=0, runtime_id="old")
    ).transfer_action == "abort"
    assert (await progress(mock_dispatcher, job, step, "task", "ready", runtime_id="new")).transfer_action == "run"


async def test_commit_retry_budget_survives_repeated_reports(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, mock_files_client: AsyncMock
):
    commit_response(mock_files_client)
    mock_files_client.commit_fileset.side_effect = OSError("response lost")
    job, _, step = await execution(mock_store, nodes=1)
    await progress(mock_dispatcher, job, step, "task", "ready")
    await progress(mock_dispatcher, job, step, "task", "finished", exit_code=0)
    for _ in range(3):
        assert (await progress(mock_dispatcher, job, step, "task", "uploaded", exit_code=0)).transfer_action == "wait"
        saved = await mock_store.get_by_id(HelixJobStep, step.id)
        assert saved.transfer_state is not None
        saved.transfer_state.retry_at.clear()
        await mock_store.update(saved)
    assert (await progress(mock_dispatcher, job, step, "task", "uploaded", exit_code=0)).transfer_action == "done"
    assert mock_files_client.commit_fileset.await_count == 3
    saved, _ = await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.COMPLETED)
    assert saved.status == HelixJobStatus.ERROR


async def test_replacement_pod_with_new_task_name_stages_a_new_generation(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient
):
    job, _, step = await execution(mock_store, nodes=1)
    await progress(mock_dispatcher, job, step, "old-pod", "pulling", runtime_id="old")
    await progress(mock_dispatcher, job, step, "old-pod", "ready", runtime_id="old")
    assert (
        await progress(mock_dispatcher, job, step, "new-pod", "pulling", runtime_id="new")
    ).transfer_action == "wait"
    assert (
        await progress(mock_dispatcher, job, step, "old-pod", "finished", exit_code=0, runtime_id="old")
    ).transfer_action == "abort"
    assert (await progress(mock_dispatcher, job, step, "new-pod", "ready", runtime_id="new")).transfer_action == "run"
