# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JobDispatcher operations using EntityStore.

These tests verify that the JobDispatcher correctly manages job lifecycle operations
(create, cancel, pause, resume, rerun, delete) using the EntityStore pattern.
"""

import asyncio
import gc
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_helix_plugin.entity_naming import NAME_MAX_LENGTH, NAME_PATTERN
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.telemetry import build_job_telemetry_custom_fields
from nemo_helix_plugin.jobs.types import JobControl, derive_job_fileset_name
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation, parse_json_filter
from nhx.common.api.parsed_filter import ParsedFilter
from nhx.common.auth import AuthContext
from nhx.common.entities import (
    ALL_WORKSPACES,
    DEFAULT_WORKSPACE,
    SYSTEM_WORKSPACE,
    EntityClient,
    EntityConflictError,
    EntityNotFoundError,
)
from nhx.common.jobs.schemas import FileStorageType, HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import (
    CreateHelixJobRequest,
    HelixJobAttemptSortField,
    HelixJobListSortField,
    HelixJobResponse,
    HelixJobSortField,
    HelixJobStepsListFilter,
    HelixJobTaskUpdate,
    UpdateHelixJobRequest,
)
from nhx.core.jobs.app.constants import (
    IMAGE_DIGEST,
    IMAGE_DIGEST_AT_SAVE,
    IMAGE_DIGEST_STEP,
    IMAGE_DIGEST_WARNING,
    JOB_TASK_CLEANUP_PENDING_KEY,
)
from nhx.core.jobs.app.dispatcher import (
    JobDeletionConflictError,
    JobDispatcher,
    JobOperationConflictError,
    JobStatusUpdateSkippedError,
    JobStorageReclaimPendingError,
    generate_job_name,
)
from nhx.core.jobs.app.schemas import (
    HelixJobEnvironmentVariable,
    HelixJobStepSpec,
    StepLifecycle,
)
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.entities import (
    HelixJob,
    HelixJobAttempt,
    HelixJobResult,
    HelixJobStep,
    HelixJobStorageReclaim,
    HelixJobTask,
)

WorkerResultT = TypeVar("WorkerResultT")

# Upper bound for a worker to reach a patched checkpoint; it normally takes milliseconds.
CHECKPOINT_TIMEOUT_SECONDS = 30


async def create_job_with_attempt(dispatcher: JobDispatcher, job_request: CreateHelixJobRequest) -> HelixJobResponse:
    """Helper to create a job through the dispatcher."""
    return await dispatcher.create_job(job_request, DEFAULT_WORKSPACE)


async def create_test_job_data(
    store: EntityClient,
    job_name: str = "test-job-123",
    fileset: str = "test-logs-fileset",
    output_location: str | None = None,
) -> tuple[str, str, str, str, str, str]:
    """Create test data for a job with all related entities using EntityStore.

    With parent-scoped uniqueness, child entities use simple names that are
    unique within their parent scope (e.g., "attempt-1" is unique per job).
    """
    # Valid platform spec with at least one step
    platform_spec = TestConstants.PLATFORM_SPEC

    # Create job (root entity - must be unique in workspace)
    job = HelixJob(
        name=job_name,
        workspace=DEFAULT_WORKSPACE,
        source="test",
        spec={},
        platform_spec=platform_spec,
        fileset=fileset,
        output_location=output_location,
    )
    saved_job = await store.add(job)

    # Create attempt (parent-scoped: unique per job)
    attempt = HelixJobAttempt(
        name="attempt-1",
        workspace=DEFAULT_WORKSPACE,
        job=saved_job.id,
        seq=1,
        status=HelixJobStatus.COMPLETED,
        spec={},
        platform_spec=platform_spec,
    )
    saved_attempt = await store.add(attempt)

    # Update job with current attempt
    saved_job.current_attempt_id = saved_attempt.id
    await store.update(saved_job)

    # Create step (parent-scoped: unique per attempt)
    step = HelixJobStep(
        name="step-1",
        workspace=DEFAULT_WORKSPACE,
        attempt_id=saved_attempt.id,
        status=HelixJobStatus.COMPLETED,
    )
    saved_step = await store.add(step)

    # Create task (parent-scoped: unique per step)
    task = HelixJobTask(
        name="task-1",
        workspace=DEFAULT_WORKSPACE,
        step_id=saved_step.id,
        status=HelixJobStatus.COMPLETED,
    )
    saved_task = await store.add(task)

    # Create result (parent-scoped: unique per job)
    result = HelixJobResult(
        name="result-1",
        workspace=DEFAULT_WORKSPACE,
        job=saved_job.id,
        artifact_url="default/test-fileset#artifact",
        artifact_storage_type=FileStorageType.FILESET,
    )
    saved_result = await store.add(result)

    return saved_job.id, saved_job.name, saved_attempt.id, saved_step.id, saved_task.id, saved_result.id


async def wait_until_reached(checkpoint: asyncio.Event, worker: asyncio.Task[WorkerResultT]) -> None:
    """Wait for ``worker`` to set ``checkpoint``, raising the worker's own error if it stops first.

    The bound is generous on purpose: a one-second bound flaked on loaded CI runners and reported
    a bare ``TimeoutError`` even when the worker had failed. It only catches a worker that stalls.
    """
    reached = asyncio.create_task(checkpoint.wait())
    try:
        await asyncio.wait({reached, worker}, timeout=CHECKPOINT_TIMEOUT_SECONDS, return_when=asyncio.FIRST_COMPLETED)
    finally:
        reached.cancel()
    if checkpoint.is_set():
        return
    coro = worker.get_coro()
    worker_name = coro.__qualname__ if coro is not None else worker.get_name()
    if not worker.done():
        raise AssertionError(f"{worker_name} did not reach its checkpoint within {CHECKPOINT_TIMEOUT_SECONDS}s")
    worker.result()
    raise AssertionError(f"{worker_name} finished without reaching its checkpoint")


async def cancellation_test_data(
    store: EntityClient, task_status: HelixJobStatus = HelixJobStatus.ACTIVE
) -> tuple[str, HelixJobStep, HelixJobTask]:
    _, job_name, attempt_id, step_id, task_id, _ = await create_test_job_data(store, "cancellation-job")
    attempt = await store.get_by_id(HelixJobAttempt, attempt_id)
    attempt.status = HelixJobStatus.CANCELLING
    await store.update(attempt)
    step = await store.get_by_id(HelixJobStep, step_id)
    step.status = HelixJobStatus.CANCELLING
    step = await store.update(step)
    task = await store.get_by_id(HelixJobTask, task_id)
    task.status = task_status
    task.status_details = {"percentage_done": 25}
    task = await store.update(task)
    return job_name, step, task


@pytest.mark.asyncio
@pytest.mark.parametrize("task_status", list(HelixJobStatus))
async def test_cancellation_finalizes_only_nonterminal_tasks(mock_dispatcher, mock_store, task_status):
    job_name, step, task = await cancellation_test_data(mock_store, task_status)

    await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.CANCELLED)

    stored_task = await mock_store.get_by_id(HelixJobTask, task.id)
    expected = task_status if task_status.is_terminal() else HelixJobStatus.CANCELLED
    assert stored_task.status == expected
    assert stored_task.status_details["percentage_done"] == 25
    status = await mock_dispatcher.get_job_status(job_name, DEFAULT_WORKSPACE)
    assert status.status == HelixJobStatus.CANCELLED
    assert status.steps[0].tasks[0].status == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["list", "update"])
async def test_cancellation_cleanup_failure_leaves_step_reconcilable(mock_dispatcher, mock_store, failure_kind):
    _, step, task = await cancellation_test_data(mock_store)
    method = mock_store.list if failure_kind == "list" else mock_store.update

    async def fail_task_cleanup(entity, *args, **kwargs):
        if entity is HelixJobTask or isinstance(entity, HelixJobTask):
            raise RuntimeError("cleanup unavailable")
        return await method(entity, *args, **kwargs)

    with (
        patch.object(mock_store, failure_kind, side_effect=fail_task_cleanup),
        pytest.raises(RuntimeError, match="cleanup unavailable"),
    ):
        await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.CANCELLED)

    stored_step = await mock_store.get_by_id(HelixJobStep, step.id)
    stored_attempt = await mock_store.get_by_id(HelixJobAttempt, step.attempt_id)
    assert stored_step.status == HelixJobStatus.CANCELLING
    assert stored_attempt.status == HelixJobStatus.CANCELLING
    await mock_dispatcher.update_job_status_from_step(stored_step, HelixJobStatus.CANCELLED)
    assert (await mock_store.get_by_id(HelixJobTask, task.id)).status == HelixJobStatus.CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent_status", [HelixJobStatus.ACTIVE, HelixJobStatus.COMPLETED, HelixJobStatus.ERROR])
async def test_cancellation_retries_task_conflicts_and_preserves_terminal_winner(
    mock_dispatcher, mock_store, concurrent_status
):
    _, step, task = await cancellation_test_data(mock_store)
    original_update = mock_store.update
    conflicted = False

    async def concurrent_task_update(entity, *args, **kwargs):
        nonlocal conflicted
        if isinstance(entity, HelixJobTask) and not conflicted:
            conflicted = True
            concurrent_task = await mock_store.get_by_id(HelixJobTask, task.id)
            concurrent_task.status = concurrent_status
            concurrent_task.status_details["step"] = 42
            await original_update(concurrent_task)
            raise EntityConflictError("concurrent task report")
        return await original_update(entity, *args, **kwargs)

    with patch.object(mock_store, "update", side_effect=concurrent_task_update):
        await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.CANCELLED)

    stored_task = await mock_store.get_by_id(HelixJobTask, task.id)
    expected = concurrent_status if concurrent_status.is_terminal() else HelixJobStatus.CANCELLED
    assert stored_task.status == expected
    assert stored_task.status_details["step"] == 42


@pytest.mark.asyncio
async def test_cancellation_repeated_conflicts_do_not_terminalize_step(mock_dispatcher, mock_store):
    _, step, _ = await cancellation_test_data(mock_store)
    original_update = mock_store.update

    async def conflict_task_updates(entity, *args, **kwargs):
        if isinstance(entity, HelixJobTask):
            raise EntityConflictError("task keeps changing")
        return await original_update(entity, *args, **kwargs)

    with (
        patch.object(mock_store, "update", side_effect=conflict_task_updates),
        pytest.raises(EntityConflictError),
    ):
        await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.CANCELLED)

    assert (await mock_store.get_by_id(HelixJobStep, step.id)).status == HelixJobStatus.CANCELLING


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_task", [False, True])
@pytest.mark.parametrize("report_status", [HelixJobStatus.ACTIVE, HelixJobStatus.ERROR, HelixJobStatus.COMPLETED])
async def test_late_task_reports_use_fresh_cancelled_parent(mock_dispatcher, mock_store, existing_task, report_status):
    job_name, stale_step, task = await cancellation_test_data(mock_store)
    stored_step = await mock_store.get_by_id(HelixJobStep, stale_step.id)
    stored_step.status = HelixJobStatus.CANCELLED
    await mock_store.update(stored_step)
    task_name = task.name if existing_task else "late-task"

    updated_task = await mock_dispatcher.create_or_update_task(
        job_name,
        task_name,
        DEFAULT_WORKSPACE,
        HelixJobTaskUpdate(status=report_status, status_details={"percentage_done": 90}),
        stale_step,
    )

    expected_status = report_status if report_status.is_terminal() else HelixJobStatus.CANCELLED
    assert updated_task.status == expected_status
    if not report_status.is_terminal():
        assert updated_task.status_details.get("percentage_done") != 90
    attempt = await mock_store.get_by_id(HelixJobAttempt, stale_step.attempt_id)
    assert attempt.status_details.get("percentage_done") != 90


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", HelixJobStatus.terminals())
async def test_cancelled_parent_preserves_terminal_task_and_metadata(mock_dispatcher, mock_store, terminal_status):
    job_name, step, task = await cancellation_test_data(mock_store, terminal_status)
    await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.CANCELLED)

    updated_task = await mock_dispatcher.create_or_update_task(
        job_name,
        task.name,
        DEFAULT_WORKSPACE,
        HelixJobTaskUpdate(status=HelixJobStatus.ACTIVE, status_details={"percentage_done": 90}),
        step,
    )

    assert updated_task.status == terminal_status
    assert updated_task.status_details == {"percentage_done": 25}


@pytest.mark.asyncio
@pytest.mark.parametrize("final_status", [HelixJobStatus.CANCELLED, HelixJobStatus.ERROR])
async def test_teardown_error_preserves_error_when_cancellation_finishes(mock_dispatcher, mock_store, final_status):
    job_name, step, task = await cancellation_test_data(mock_store)

    updated_task = await mock_dispatcher.create_or_update_task(
        job_name,
        task.name,
        DEFAULT_WORKSPACE,
        HelixJobTaskUpdate(status=HelixJobStatus.ERROR, error_details={"message": "terminated pod"}),
        step,
    )
    assert updated_task.status == HelixJobStatus.ERROR
    assert updated_task.error_details == {"message": "terminated pod"}
    assert (await mock_store.get_by_id(HelixJobStep, step.id)).status == HelixJobStatus.CANCELLING
    other_task = await mock_store.create(
        HelixJobTask(name="other-pod", workspace=step.workspace, step_id=step.id, status=HelixJobStatus.ACTIVE)
    )
    await mock_dispatcher.update_job_status_from_step(step, final_status)
    stored_task = await mock_store.get_by_id(HelixJobTask, task.id)
    assert stored_task.status == HelixJobStatus.ERROR
    assert stored_task.error_details == {"message": "terminated pod"}
    assert (await mock_store.get_by_id(HelixJobTask, other_task.id)).status == final_status


@pytest.mark.asyncio
async def test_cancel_request_does_not_finalize_running_task(mock_dispatcher, mock_store):
    job_name, step, task = await cancellation_test_data(mock_store)

    await mock_dispatcher.cancel_job(job_name, DEFAULT_WORKSPACE)

    assert (await mock_store.get_by_id(HelixJobTask, task.id)).status == HelixJobStatus.ACTIVE
    assert (await mock_store.get_by_id(HelixJobStep, step.id)).status == HelixJobStatus.CANCELLING


@pytest.mark.asyncio
@pytest.mark.parametrize("final_status", [HelixJobStatus.CANCELLED, HelixJobStatus.ERROR])
async def test_repeat_cancel_repairs_only_current_attempt_tasks(mock_dispatcher, mock_store, final_status):
    job_name, step, task = await cancellation_test_data(mock_store)
    _, other_job_name, _, other_step_id, _, _ = await create_test_job_data(mock_store, "unrelated-job")
    other_step = await mock_store.get_by_id(HelixJobStep, other_step_id)
    other_task = await mock_store.create(
        HelixJobTask(name="orphan", workspace=DEFAULT_WORKSPACE, step_id=other_step.id, status=HelixJobStatus.ACTIVE)
    )
    archived_attempt = await mock_store.create(
        HelixJobAttempt(
            name="old-attempt",
            workspace=DEFAULT_WORKSPACE,
            job=(await mock_dispatcher.get_job(job_name, DEFAULT_WORKSPACE)).id,
            seq=0,
            status=HelixJobStatus.CANCELLED,
            platform_spec=TestConstants.PLATFORM_SPEC,
        )
    )
    archived_step = await mock_store.create(
        HelixJobStep(
            name="step-1", workspace=DEFAULT_WORKSPACE, attempt_id=archived_attempt.id, status=HelixJobStatus.CANCELLED
        )
    )
    archived_task = await mock_store.create(
        HelixJobTask(name="orphan", workspace=DEFAULT_WORKSPACE, step_id=archived_step.id, status=HelixJobStatus.ACTIVE)
    )
    step.status = final_status
    await mock_store.update(step)
    attempt = await mock_store.get_by_id(HelixJobAttempt, step.attempt_id)
    attempt.status = final_status
    await mock_store.update(attempt)

    for _ in range(2):
        response = await mock_dispatcher.cancel_job(job_name, DEFAULT_WORKSPACE)
        assert response.status == final_status

    assert (await mock_store.get_by_id(HelixJobTask, task.id)).status == final_status
    assert (await mock_store.get_by_id(HelixJobTask, other_task.id)).status == HelixJobStatus.ACTIVE
    assert (await mock_store.get_by_id(HelixJobTask, archived_task.id)).status == HelixJobStatus.ACTIVE
    assert (await mock_dispatcher.get_job(other_job_name, DEFAULT_WORKSPACE)).status == HelixJobStatus.COMPLETED


@pytest.mark.asyncio
async def test_task_listing_includes_more_than_one_thousand_tasks(mock_dispatcher, mock_store):
    tasks = [
        HelixJobTask(name=f"task-{index}", workspace=DEFAULT_WORKSPACE, step_id="step-id") for index in range(1001)
    ]
    responses = [
        SimpleNamespace(data=tasks[:1000], pagination=SimpleNamespace(page=1, total_pages=2)),
        SimpleNamespace(data=tasks[1000:], pagination=SimpleNamespace(page=2, total_pages=2)),
    ]

    with patch.object(mock_store, "list", side_effect=responses) as list_entities:
        result = await mock_dispatcher.list_tasks("step-id", "other-workspace")

    assert result == tasks
    assert [call.kwargs["page"] for call in list_entities.call_args_list] == [1, 2]
    for call in list_entities.call_args_list:
        assert call.kwargs["workspace"] == "other-workspace"
        assert call.kwargs["filter_obj"] == {"step_id": "step-id"}


@pytest.mark.asyncio
async def test_cancellation_cleans_all_pages_and_retries_partial_success(mock_dispatcher, mock_store):
    _, step, first_task = await cancellation_test_data(mock_store)
    extra_tasks = [
        await mock_store.create(
            HelixJobTask(
                name=f"task-extra-{index}", workspace=step.workspace, step_id=step.id, status=HelixJobStatus.ACTIVE
            )
        )
        for index in range(3)
    ]
    original_list = mock_store.list
    original_update = mock_store.update
    task_updates = 0

    async def small_pages(entity_type, *args, **kwargs):
        if entity_type is HelixJobTask:
            kwargs["page_size"] = 2
        return await original_list(entity_type, *args, **kwargs)

    async def fail_second_task(entity, *args, **kwargs):
        nonlocal task_updates
        if isinstance(entity, HelixJobTask):
            task_updates += 1
            if task_updates == 2:
                raise RuntimeError("temporary task failure")
        return await original_update(entity, *args, **kwargs)

    with (
        patch.object(mock_store, "list", side_effect=small_pages),
        patch.object(mock_store, "update", side_effect=fail_second_task),
        pytest.raises(RuntimeError, match="temporary task failure"),
    ):
        await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.CANCELLED)

    assert (await mock_store.get_by_id(HelixJobStep, step.id)).status == HelixJobStatus.CANCELLING
    with patch.object(mock_store, "list", side_effect=small_pages):
        await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.CANCELLED)
    for task in [first_task, *extra_tasks]:
        assert (await mock_store.get_by_id(HelixJobTask, task.id)).status == HelixJobStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancellation_serializes_with_late_task_creation(mock_dispatcher, mock_store, mock_files_client):
    job_name, step, _ = await cancellation_test_data(mock_store)
    other_dispatcher = JobDispatcher(store=mock_store, files=mock_files_client, secrets=AsyncMock())
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    original_update = mock_store.update

    async def block_cleanup(entity, *args, **kwargs):
        if isinstance(entity, HelixJobTask) and not cleanup_started.is_set():
            cleanup_started.set()
            await allow_cleanup.wait()
        return await original_update(entity, *args, **kwargs)

    with patch.object(mock_store, "update", side_effect=block_cleanup):
        cancellation = asyncio.create_task(mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.CANCELLED))
        late_report = None
        try:
            await wait_until_reached(cleanup_started, cancellation)
            late_report = asyncio.create_task(
                other_dispatcher.create_or_update_task(
                    job_name, "late-task", step.workspace, HelixJobTaskUpdate(status=HelixJobStatus.ACTIVE), step
                )
            )
            await asyncio.sleep(0)
            assert not late_report.done()
            allow_cleanup.set()
            await cancellation
            assert (await late_report).status == HelixJobStatus.CANCELLED
        finally:
            allow_cleanup.set()
            await asyncio.gather(
                cancellation, *([late_report] if late_report is not None else []), return_exceptions=True
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("final_status", [HelixJobStatus.CANCELLED, HelixJobStatus.ERROR])
async def test_terminal_post_save_sweep_catches_other_replica_task(
    mock_dispatcher, mock_store, mock_files_client, final_status
):
    """Bypass the process-local lock to simulate an independently running API worker."""
    job_name, step, _ = await cancellation_test_data(mock_store)
    replica = JobDispatcher(store=mock_store, files=mock_files_client, secrets=AsyncMock())
    original_update = mock_store.update
    late_task = None

    async def interleave_task_before_step_save(entity, *args, **kwargs):
        nonlocal late_task
        if isinstance(entity, HelixJobStep) and entity.status == final_status and late_task is None:
            late_task = await replica._create_or_update_task_locked(
                job_name, "other-replica", step.workspace, HelixJobTaskUpdate(status=HelixJobStatus.ACTIVE), step
            )
            assert late_task.status == HelixJobStatus.ACTIVE
        return await original_update(entity, *args, **kwargs)

    with patch.object(mock_store, "update", side_effect=interleave_task_before_step_save):
        await mock_dispatcher.update_job_status_from_step(step, final_status)

    assert late_task is not None
    assert (await mock_store.get_by_id(HelixJobTask, late_task.id)).status == final_status
    stored_step = await mock_store.get_by_id(HelixJobStep, step.id)
    assert stored_step.status == final_status
    assert stored_step.status_details[JOB_TASK_CLEANUP_PENDING_KEY] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("final_status", [HelixJobStatus.CANCELLED, HelixJobStatus.ERROR])
async def test_post_save_cleanup_failure_is_durable_and_queryable(mock_dispatcher, mock_store, final_status):
    job_name, step, _ = await cancellation_test_data(mock_store)
    attempt = await mock_store.get_by_id(HelixJobAttempt, step.attempt_id)
    attempt.platform_spec = attempt.platform_spec.model_copy(
        update={"steps": [attempt.platform_spec.steps[0].model_copy(update={"name": step.name})]}
    )
    await mock_store.update(attempt)
    original_update = mock_store.update
    late_task = None

    async def fail_post_save_cleanup(entity, *args, **kwargs):
        nonlocal late_task
        if isinstance(entity, HelixJobStep) and entity.status == final_status and late_task is None:
            saved_step = await original_update(entity, *args, **kwargs)
            late_task = await mock_store.create(
                HelixJobTask(
                    name="racing-task", workspace=step.workspace, step_id=step.id, status=HelixJobStatus.ACTIVE
                )
            )
            return saved_step
        if isinstance(entity, HelixJobTask) and late_task is not None and entity.id == late_task.id:
            raise RuntimeError("post-save cleanup unavailable")
        return await original_update(entity, *args, **kwargs)

    with (
        patch.object(mock_store, "update", side_effect=fail_post_save_cleanup),
        pytest.raises(RuntimeError, match="post-save cleanup unavailable"),
    ):
        await mock_dispatcher.update_job_status_from_step(step, final_status)

    stored_step = await mock_store.get_by_id(HelixJobStep, step.id)
    assert stored_step.status == final_status
    assert stored_step.status_details[JOB_TASK_CLEANUP_PENDING_KEY] is True
    pending, count = await mock_dispatcher.list_steps(
        HelixJobStepsListFilter(status=[final_status], task_cleanup_pending=True),
        HelixJobSortField.UPDATED_AT_ASC,
        100,
        0,
        step.workspace,
    )
    assert count == 1
    assert [candidate.id for candidate in pending] == [step.id]

    response = await mock_dispatcher.cancel_job(job_name, step.workspace)
    assert response.status == final_status

    assert (await mock_store.get_by_id(HelixJobTask, late_task.id)).status == final_status
    assert (await mock_store.get_by_id(HelixJobStep, step.id)).status_details[JOB_TASK_CLEANUP_PENDING_KEY] is False
    pending, count = await mock_dispatcher.list_steps(
        HelixJobStepsListFilter(status=[final_status], task_cleanup_pending=True),
        HelixJobSortField.UPDATED_AT_ASC,
        100,
        0,
        step.workspace,
    )
    assert pending == []
    assert count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("final_status", [HelixJobStatus.CANCELLED, HelixJobStatus.ERROR])
async def test_other_replica_task_write_after_final_sweep_rechecks_parent(
    mock_dispatcher, mock_store, mock_files_client, final_status
):
    """A worker with a stale parent read must not commit an orphan after the last sweep."""
    job_name, step, _ = await cancellation_test_data(mock_store)
    replica = JobDispatcher(store=mock_store, files=mock_files_client, secrets=AsyncMock())
    creation_started = asyncio.Event()
    allow_creation = asyncio.Event()
    original_create = mock_store.create

    async def delay_task_write(entity, *args, **kwargs):
        if isinstance(entity, HelixJobTask) and entity.name == "delayed-task":
            creation_started.set()
            await allow_creation.wait()
        return await original_create(entity, *args, **kwargs)

    with patch.object(mock_store, "create", side_effect=delay_task_write):
        late_report = asyncio.create_task(
            replica._create_or_update_task_locked(
                job_name, "delayed-task", step.workspace, HelixJobTaskUpdate(status=HelixJobStatus.ACTIVE), step
            )
        )
        try:
            await wait_until_reached(creation_started, late_report)
            await mock_dispatcher.update_job_status_from_step(step, final_status)
            allow_creation.set()
            assert (await late_report).status == final_status
        finally:
            allow_creation.set()
            await asyncio.gather(late_report, return_exceptions=True)


@pytest.mark.asyncio
async def test_error_parent_does_not_accept_active_task_reports(mock_dispatcher, mock_store):
    job_name, step, _ = await cancellation_test_data(mock_store)
    await mock_dispatcher.update_job_status_from_step(
        step, HelixJobStatus.ERROR, error_details={"message": "Job failed"}
    )
    task = await mock_dispatcher.create_or_update_task(
        job_name, "late-task", step.workspace, HelixJobTaskUpdate(status=HelixJobStatus.ACTIVE), step
    )
    assert task.status == HelixJobStatus.ERROR
    assert task.error_details == {"message": "Job failed"}


@pytest.mark.asyncio
async def test_rerun_completes_pending_child_cleanup_before_archiving(mock_dispatcher, mock_store):
    # Rerun applies only to a job that ended in error.
    final_status = HelixJobStatus.ERROR
    job_name, step, task = await cancellation_test_data(mock_store)
    step.status = final_status
    step.status_details[JOB_TASK_CLEANUP_PENDING_KEY] = True
    await mock_store.update(step)
    attempt = await mock_store.get_by_id(HelixJobAttempt, step.attempt_id)
    attempt.status = final_status
    await mock_store.update(attempt)
    original_update = mock_store.update

    async def fail_cleanup(entity, *args, **kwargs):
        if isinstance(entity, HelixJobTask):
            raise RuntimeError("cleanup unavailable during rerun")
        return await original_update(entity, *args, **kwargs)

    with (
        patch.object(mock_store, "update", side_effect=fail_cleanup),
        pytest.raises(RuntimeError, match="cleanup unavailable during rerun"),
    ):
        await mock_dispatcher.rerun_job(job_name, step.workspace)

    current_attempt = await mock_dispatcher.get_current_attempt(job_name, step.workspace)
    assert current_attempt.id == step.attempt_id
    rerun = await mock_dispatcher.rerun_job(job_name, step.workspace)
    assert rerun.attempt_id != step.attempt_id
    assert rerun.status == HelixJobStatus.CREATED
    assert (await mock_store.get_by_id(HelixJobTask, task.id)).status == final_status
    assert (await mock_store.get_by_id(HelixJobStep, step.id)).status_details[JOB_TASK_CLEANUP_PENDING_KEY] is False


async def verify_job_data_exists(store: EntityClient, job_id: str, should_exist: bool = True) -> None:
    """Verify that job data exists or does not exist."""
    if should_exist:
        job = await store.get_by_id(HelixJob, job_id)
        assert job is not None
    else:
        with pytest.raises(EntityNotFoundError):
            await store.get_by_id(HelixJob, job_id)


async def count_entities(store: EntityClient, entity_type, filter_obj: dict) -> int:
    """Count entities matching a filter."""
    response = await store.list(entity_type, filter_obj=filter_obj)
    return len(response.data)


@pytest.mark.asyncio
async def test_delete_job_success(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """Test successful job deletion using EntityStore."""
    # Create test data
    job_id, job_name, attempt_id, step_id, _, _ = await create_test_job_data(mock_store, "delete-test-job")

    # Verify data exists before deletion
    await verify_job_data_exists(mock_store, job_id, should_exist=True)
    assert await count_entities(mock_store, HelixJobAttempt, {"job": job_id}) == 1
    assert await count_entities(mock_store, HelixJobStep, {"attempt_id": attempt_id}) == 1
    assert await count_entities(mock_store, HelixJobTask, {"step_id": step_id}) == 1
    assert await count_entities(mock_store, HelixJobResult, {"job": job_id}) == 1

    # Execute deletion - should return True when job exists
    deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)
    assert deleted is True

    # Verify all data is deleted
    await verify_job_data_exists(mock_store, job_id, should_exist=False)
    assert await count_entities(mock_store, HelixJobAttempt, {"job": job_id}) == 0
    assert await count_entities(mock_store, HelixJobStep, {"attempt_id": attempt_id}) == 0
    assert await count_entities(mock_store, HelixJobTask, {"step_id": step_id}) == 0
    assert await count_entities(mock_store, HelixJobResult, {"job": job_id}) == 0


@pytest.mark.asyncio
async def test_delete_job_multiple_jobs(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """Test deletion only affects the specified job."""
    # Create test data for two jobs
    job1_id, job1_name, _, _, _, _ = await create_test_job_data(mock_store, "job-1")
    job2_id, _, _, _, _, _ = await create_test_job_data(mock_store, "job-2")

    # Delete only job-1
    deleted = await mock_dispatcher.delete_job(job1_name, DEFAULT_WORKSPACE)
    assert deleted is True

    # Verify job-1 is deleted but job-2 remains
    await verify_job_data_exists(mock_store, job1_id, should_exist=False)
    await verify_job_data_exists(mock_store, job2_id, should_exist=True)

    # Verify job2's related data still exists
    assert await count_entities(mock_store, HelixJobAttempt, {"job": job2_id}) == 1


@pytest.mark.asyncio
async def test_delete_job_nonexistent_job(mock_dispatcher: JobDispatcher):
    """Test deletion of non-existent job returns False."""
    # This should return False when the job doesn't exist
    deleted = await mock_dispatcher.delete_job("nonexistent-job-id", DEFAULT_WORKSPACE)
    assert deleted is False


@pytest.mark.asyncio
async def test_delete_job_non_terminal_job_raises_without_deleting_data(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    mock_files_client,
):
    """Deleting an active job is refused before metadata or fileset cleanup."""
    job_id, job_name, attempt_id, step_id, _, _ = await create_test_job_data(mock_store, "active-delete-test-job")

    attempt = await mock_store.get_by_id(HelixJobAttempt, attempt_id)
    attempt.status = HelixJobStatus.ACTIVE
    await mock_store.update(attempt)

    step = await mock_store.get_by_id(HelixJobStep, step_id)
    step.status = HelixJobStatus.ACTIVE
    await mock_store.update(step)

    with pytest.raises(JobDeletionConflictError, match="Cancel the job and wait"):
        await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)

    await verify_job_data_exists(mock_store, job_id, should_exist=True)
    assert await count_entities(mock_store, HelixJobAttempt, {"job": job_id}) == 1
    assert await count_entities(mock_store, HelixJobStep, {"attempt_id": attempt_id}) == 1
    assert await count_entities(mock_store, HelixJobTask, {"step_id": step_id}) == 1
    assert await count_entities(mock_store, HelixJobResult, {"job": job_id}) == 1
    mock_files_client.delete_fileset.assert_not_called()


@pytest.mark.asyncio
async def test_delete_job_deletes_related_entities_across_all_pages(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """Deleting a terminal job visits every attempt, step, task, and result page."""
    from nhx.core.jobs.app import dispatcher as dispatcher_module

    monkeypatch.setattr(dispatcher_module, "_DELETE_PAGE_SIZE", 1)

    job_id, job_name, attempt_id, step_id, task_id, result_id = await create_test_job_data(
        mock_store, "paginated-delete-test-job"
    )
    job = await mock_store.get_by_id(HelixJob, job_id)

    second_attempt = await mock_store.add(
        HelixJobAttempt(
            name="attempt-2",
            workspace=DEFAULT_WORKSPACE,
            job=job.id,
            seq=2,
            status=HelixJobStatus.COMPLETED,
            spec={},
            platform_spec=job.platform_spec,
        )
    )
    second_step = await mock_store.add(
        HelixJobStep(
            name="step-2",
            workspace=DEFAULT_WORKSPACE,
            attempt_id=second_attempt.id,
            status=HelixJobStatus.COMPLETED,
        )
    )
    second_task = await mock_store.add(
        HelixJobTask(
            name="task-2",
            workspace=DEFAULT_WORKSPACE,
            step_id=second_step.id,
            status=HelixJobStatus.COMPLETED,
        )
    )
    second_result = await mock_store.add(
        HelixJobResult(
            name="result-2",
            workspace=DEFAULT_WORKSPACE,
            job=job.id,
            artifact_url="default/test-fileset#artifact-2",
            artifact_storage_type=FileStorageType.FILESET,
        )
    )

    deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)

    assert deleted is True
    for entity_type, entity_id in (
        (HelixJob, job_id),
        (HelixJobAttempt, attempt_id),
        (HelixJobAttempt, second_attempt.id),
        (HelixJobStep, step_id),
        (HelixJobStep, second_step.id),
        (HelixJobTask, task_id),
        (HelixJobTask, second_task.id),
        (HelixJobResult, result_id),
        (HelixJobResult, second_result.id),
    ):
        with pytest.raises(EntityNotFoundError):
            await mock_store.get_by_id(entity_type, entity_id)


@pytest.mark.asyncio
async def test_delete_job_serializes_with_rerun_job(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    mock_files_client,
):
    """A rerun request cannot create a new attempt while deletion is cleaning up."""
    _, job_name, _, _, _, _ = await create_test_job_data(mock_store, "delete-rerun-lock-test-job")
    other_dispatcher = JobDispatcher(store=mock_store, files=mock_files_client, secrets=AsyncMock())

    delete_started = asyncio.Event()
    allow_delete = asyncio.Event()
    original_delete_by_id = mock_store.delete_by_id

    async def blocking_delete_by_id(entity_type, entity_id):
        if entity_type is HelixJobResult and not delete_started.is_set():
            delete_started.set()
            await allow_delete.wait()
        return await original_delete_by_id(entity_type, entity_id)

    with patch.object(mock_store, "delete_by_id", side_effect=blocking_delete_by_id):
        delete_task = asyncio.create_task(mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE))
        rerun_task = None
        try:
            await wait_until_reached(delete_started, delete_task)

            rerun_task = asyncio.create_task(other_dispatcher.rerun_job(job_name, DEFAULT_WORKSPACE))
            await asyncio.sleep(0.05)
            assert not rerun_task.done()

            allow_delete.set()
            assert await delete_task is True
            assert await rerun_task is None
        finally:
            allow_delete.set()
            if rerun_task is not None:
                await asyncio.gather(delete_task, rerun_task, return_exceptions=True)
            else:
                await asyncio.gather(delete_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_delete_job_serializes_with_same_name_create(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    mock_files_client,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A same-name create waits until delete finishes all cleanup for the old job."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "delete-create-lock-test-job")
    other_dispatcher = JobDispatcher(store=mock_store, files=mock_files_client, secrets=AsyncMock())
    create_request = sample_platform_job_request.model_copy(update={"name": job_name})

    delete_started = asyncio.Event()
    allow_delete = asyncio.Event()
    original_delete_by_id = mock_store.delete_by_id

    async def blocking_delete_by_id(entity_type, entity_id):
        if entity_type is HelixJobResult and not delete_started.is_set():
            delete_started.set()
            await allow_delete.wait()
        return await original_delete_by_id(entity_type, entity_id)

    with patch.object(mock_store, "delete_by_id", side_effect=blocking_delete_by_id):
        delete_task = asyncio.create_task(mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE))
        create_task = None
        try:
            await wait_until_reached(delete_started, delete_task)

            create_task = asyncio.create_task(other_dispatcher.create_job(create_request, DEFAULT_WORKSPACE))
            await asyncio.sleep(0.05)
            assert not create_task.done()

            allow_delete.set()
            assert await delete_task is True
            recreated = await create_task
        finally:
            allow_delete.set()
            if create_task is not None:
                await asyncio.gather(delete_task, create_task, return_exceptions=True)
            else:
                await asyncio.gather(delete_task, return_exceptions=True)

    await verify_job_data_exists(mock_store, job_id, should_exist=False)
    assert recreated.name == job_name
    assert recreated.id != job_id


@pytest.mark.asyncio
async def test_delete_job_serializes_with_task_creation(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    mock_files_client,
):
    """A task update cannot create a late child row after delete cleanup starts."""
    job_id, job_name, _, step_id, _, _ = await create_test_job_data(mock_store, "delete-task-lock-test-job")
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    other_dispatcher = JobDispatcher(store=mock_store, files=mock_files_client, secrets=AsyncMock())

    delete_started = asyncio.Event()
    allow_delete = asyncio.Event()
    original_delete_by_id = mock_store.delete_by_id

    async def blocking_delete_by_id(entity_type, entity_id):
        if entity_type is HelixJobResult and not delete_started.is_set():
            delete_started.set()
            await allow_delete.wait()
        return await original_delete_by_id(entity_type, entity_id)

    with patch.object(mock_store, "delete_by_id", side_effect=blocking_delete_by_id):
        delete_task = asyncio.create_task(mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE))
        task_create_task = None
        try:
            await wait_until_reached(delete_started, delete_task)

            task_create_task = asyncio.create_task(
                other_dispatcher.create_or_update_task(
                    job_name,
                    "late-task",
                    DEFAULT_WORKSPACE,
                    HelixJobTaskUpdate(status=HelixJobStatus.ACTIVE),
                    step,
                )
            )
            await asyncio.sleep(0.05)
            assert not task_create_task.done()

            allow_delete.set()
            assert await delete_task is True
            with pytest.raises(EntityNotFoundError):
                await task_create_task
        finally:
            allow_delete.set()
            if task_create_task is not None:
                await asyncio.gather(delete_task, task_create_task, return_exceptions=True)
            else:
                await asyncio.gather(delete_task, return_exceptions=True)

    await verify_job_data_exists(mock_store, job_id, should_exist=False)
    assert await count_entities(mock_store, HelixJobTask, {"step_id": step_id}) == 0


@pytest.mark.asyncio
async def test_delete_job_serializes_with_result_creation(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    mock_files_client,
):
    """A result create cannot recreate associated data after delete cleanup starts."""
    job_id, job_name, attempt_id, _, _, _ = await create_test_job_data(mock_store, "delete-result-lock-test-job")
    other_dispatcher = JobDispatcher(store=mock_store, files=mock_files_client, secrets=AsyncMock())

    delete_started = asyncio.Event()
    allow_delete = asyncio.Event()
    original_delete_by_id = mock_store.delete_by_id

    async def blocking_delete_by_id(entity_type, entity_id):
        if entity_type is HelixJobResult and not delete_started.is_set():
            delete_started.set()
            await allow_delete.wait()
        return await original_delete_by_id(entity_type, entity_id)

    with patch.object(mock_store, "delete_by_id", side_effect=blocking_delete_by_id):
        delete_task = asyncio.create_task(mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE))
        result_create_task = None
        try:
            await wait_until_reached(delete_started, delete_task)

            result_create_task = asyncio.create_task(
                other_dispatcher.create_result(
                    job_id=job_id,
                    job_name=job_name,
                    result_name="late-result",
                    artifact_url="default/test-fileset#late",
                    artifact_storage_type=FileStorageType.FILESET,
                    workspace=DEFAULT_WORKSPACE,
                    attempt_id=attempt_id,
                )
            )
            await asyncio.sleep(0.05)
            assert not result_create_task.done()

            allow_delete.set()
            assert await delete_task is True
            with pytest.raises(EntityNotFoundError):
                await result_create_task
        finally:
            allow_delete.set()
            if result_create_task is not None:
                await asyncio.gather(delete_task, result_create_task, return_exceptions=True)
            else:
                await asyncio.gather(delete_task, return_exceptions=True)

    await verify_job_data_exists(mock_store, job_id, should_exist=False)
    assert await count_entities(mock_store, HelixJobResult, {"job": job_id}) == 0


@pytest.mark.asyncio
async def test_delete_job_does_not_leave_idle_mutation_lock(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """Idle job mutation locks are weakly held and do not grow forever."""
    from nhx.core.jobs.app import dispatcher as dispatcher_module

    _, job_name, _, _, _, _ = await create_test_job_data(mock_store, "delete-lock-prune-test-job")

    deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)

    assert deleted is True
    gc.collect()
    assert (DEFAULT_WORKSPACE, job_name) not in dispatcher_module._JOB_MUTATION_LOCKS


@pytest.mark.asyncio
async def test_create_or_update_task_ignores_invalid_terminal_transition(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient
):
    """Terminal task states must not be overwritten by stale backend polls."""
    _, job_name, _, step_id, task_id, _ = await create_test_job_data(mock_store, "terminal-task-job")
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    task = await mock_store.get_by_id(HelixJobTask, task_id)

    updated = await mock_dispatcher.create_or_update_task(
        job_name,
        task.name,
        DEFAULT_WORKSPACE,
        HelixJobTaskUpdate(
            status=HelixJobStatus.ACTIVE,
            status_details={"message": "Job is running"},
        ),
        step,
    )

    assert updated.id == task.id
    assert updated.status == HelixJobStatus.COMPLETED
    assert updated.status_details == {}

    stored_task = await mock_store.get_by_id(HelixJobTask, task_id)
    assert stored_task.status == HelixJobStatus.COMPLETED
    assert stored_task.status_details == {}


@pytest.mark.asyncio
async def test_create_or_update_task_records_progress_reported_after_error(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient
):
    """A workload still running past an errored task keeps reporting; the status stays terminal."""
    _, job_name, _, step_id, task_id, _ = await create_test_job_data(mock_store, "errored-task-job")
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    task = await mock_store.get_by_id(HelixJobTask, task_id)
    task.status = HelixJobStatus.ERROR
    await mock_store.update(task)

    updated = await mock_dispatcher.create_or_update_task(
        job_name,
        task.name,
        DEFAULT_WORKSPACE,
        HelixJobTaskUpdate(
            status=HelixJobStatus.ACTIVE,
            status_details={"phase": "training", "step": 2000},
        ),
        step,
    )

    assert updated.status == HelixJobStatus.ERROR
    assert updated.status_details["phase"] == "training"
    assert updated.status_details["step"] == 2000

    stored_task = await mock_store.get_by_id(HelixJobTask, task_id)
    assert stored_task.status == HelixJobStatus.ERROR
    assert stored_task.status_details["step"] == 2000


@pytest.mark.asyncio
async def test_create_or_update_task_keeps_the_task_phase_when_the_controller_reports_the_pod(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient
):
    """The controller and the task's own code write to the same details, merged key by key."""
    _, job_name, _, step_id, task_id, _ = await create_test_job_data(mock_store, "pod-details-job")
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    task = await mock_store.get_by_id(HelixJobTask, task_id)
    task.status = HelixJobStatus.ACTIVE
    task.status_details = {"phase": "training", "step": 12}
    await mock_store.update(task)

    updated = await mock_dispatcher.create_or_update_task(
        job_name,
        task.name,
        DEFAULT_WORKSPACE,
        HelixJobTaskUpdate(
            status=HelixJobStatus.ACTIVE,
            status_details={"pod": {"name": "pod-1", "phase": "Running"}},
        ),
        step,
    )

    assert updated.status_details == {
        "phase": "training",
        "step": 12,
        "pod": {"name": "pod-1", "phase": "Running"},
    }


CLASSIFIED_ERROR = {
    "message": "A GPU ran out of memory.",
    "type": "GpuOutOfMemoryError",
    "detail": "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "cause": "gpu_out_of_memory",
}

# A failure no rule matched. The task still reported it, with the traceback the controller never sees.
UNMATCHED_ERROR = {
    "message": "An internal error occurred during training. (ValueError: something new)",
    "type": "InternalError",
    "detail": "ValueError: something new",
    "cause": None,
}

TASK_ERRORS = pytest.mark.parametrize("reported", [CLASSIFIED_ERROR, UNMATCHED_ERROR], ids=["matched", "unmatched"])


async def _set_running(store: EntityClient, attempt_id: str, step_id: str, task_id: str) -> None:
    for entity_type, entity_id in ((HelixJobAttempt, attempt_id), (HelixJobStep, step_id), (HelixJobTask, task_id)):
        entity = await store.get_by_id(entity_type, entity_id)
        entity.status = HelixJobStatus.ACTIVE
        await store.update(entity)


@TASK_ERRORS
@pytest.mark.asyncio
async def test_a_task_keeps_the_error_it_reported_when_the_controller_reports_the_failure(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, reported: dict
):
    """The controller only sees the process exit, so its error adds to the task's own instead of replacing it."""
    _, job_name, attempt_id, step_id, task_id, _ = await create_test_job_data(mock_store, "classified-task-job")
    await _set_running(mock_store, attempt_id, step_id, task_id)
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    task = await mock_store.get_by_id(HelixJobTask, task_id)

    for error_details in (reported, {"message": "Pod pod-1 is in error state", "failed": "Back-off"}):
        await mock_dispatcher.create_or_update_task(
            job_name,
            task.name,
            DEFAULT_WORKSPACE,
            HelixJobTaskUpdate(status=HelixJobStatus.ERROR, error_details=error_details),
            step,
        )

    stored_task = await mock_store.get_by_id(HelixJobTask, task_id)
    assert stored_task.error_details == {**reported, "failed": "Back-off"}


@TASK_ERRORS
@pytest.mark.asyncio
async def test_a_failed_step_reports_the_error_its_task_classified(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, reported: dict
):
    _, _, attempt_id, step_id, task_id, _ = await create_test_job_data(mock_store, "classified-step-job")
    await _set_running(mock_store, attempt_id, step_id, task_id)
    task = await mock_store.get_by_id(HelixJobTask, task_id)
    task.status = HelixJobStatus.ERROR
    task.error_details = dict(reported)
    await mock_store.update(task)
    step = await mock_store.get_by_id(HelixJobStep, step_id)

    saved_step, attempt = await mock_dispatcher.update_job_status_from_step(
        step, HelixJobStatus.ERROR, error_details={"message": "Job failed; One or more tasks are in error state"}
    )

    assert saved_step.error_details == reported
    assert attempt.error_details == reported


@pytest.mark.asyncio
async def test_a_failed_step_keeps_the_controller_error_when_no_task_classified_its_failure(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient
):
    _, _, attempt_id, step_id, task_id, _ = await create_test_job_data(mock_store, "unclassified-step-job")
    await _set_running(mock_store, attempt_id, step_id, task_id)
    task = await mock_store.get_by_id(HelixJobTask, task_id)
    task.status = HelixJobStatus.ERROR
    task.error_details = {"message": "Pod pod-1 is in error state"}
    await mock_store.update(task)
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    controller_error = {"message": "Job timed out after reaching max TTL of 3600 seconds"}

    saved_step, attempt = await mock_dispatcher.update_job_status_from_step(
        step, HelixJobStatus.ERROR, error_details=controller_error
    )

    assert saved_step.error_details == controller_error
    assert attempt.error_details == controller_error


@pytest.mark.asyncio
async def test_create_or_update_task_ignores_statusless_poll_after_error(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient
):
    """A rejected transition carrying no progress leaves the task untouched."""
    _, job_name, _, step_id, task_id, _ = await create_test_job_data(mock_store, "errored-task-noop-job")
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    task = await mock_store.get_by_id(HelixJobTask, task_id)
    task.status = HelixJobStatus.ERROR
    await mock_store.update(task)

    updated = await mock_dispatcher.create_or_update_task(
        job_name,
        task.name,
        DEFAULT_WORKSPACE,
        HelixJobTaskUpdate(status=HelixJobStatus.ACTIVE),
        step,
    )

    assert updated.status == HelixJobStatus.ERROR
    assert updated.status_details == {}


@pytest.mark.asyncio
async def test_delete_job_missing_fileset_succeeds(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """Test that delete succeeds even if the job fileset was already deleted.

    The fileset may have been cleaned up by workspace cleanup before the explicit
    delete call arrives, so delete_job must tolerate a 404 on the fileset.
    """
    from unittest.mock import AsyncMock

    from nemo_helix_plugin.client.errors import NotFoundError

    # output_location defaults to None -> job owns its fileset -> delete_fileset is attempted.
    job_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "delete-missing-fileset-job")

    # Simulate the fileset already being gone by making the mock files client raise NotFoundError
    mock_dispatcher.files.delete_fileset = AsyncMock(side_effect=NotFoundError.__new__(NotFoundError))

    deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)
    assert deleted is True

    # The job entity itself should be gone
    await verify_job_data_exists(mock_store, job_id, should_exist=False)


@pytest.mark.asyncio
async def test_create_job_uses_existing_output_location(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    mock_files_client,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A supplied output_location becomes the fileset and is persisted as provenance; nothing created."""
    request = sample_platform_job_request.model_copy(update={"output_location": "my-eval-fileset"})

    job = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    assert job.fileset == "my-eval-fileset"
    mock_files_client.get_fileset.assert_awaited_once()
    mock_files_client.create_fileset.assert_not_called()

    stored = await mock_store.get(HelixJob, job.name, workspace=DEFAULT_WORKSPACE)
    assert stored.output_location == "my-eval-fileset"


@pytest.mark.asyncio
async def test_create_job_output_location_not_found_raises(
    mock_dispatcher: JobDispatcher,
    mock_files_client,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A supplied output_location that does not exist is rejected and never auto-created."""
    from nemo_helix_plugin.client.errors import NotFoundError
    from nhx.core.jobs.app.dispatcher import JobOutputLocationError

    mock_files_client.get_fileset.side_effect = NotFoundError.__new__(NotFoundError)
    request = sample_platform_job_request.model_copy(update={"output_location": "ghost-fileset"})

    with pytest.raises(JobOutputLocationError):
        await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    mock_files_client.create_fileset.assert_not_called()


@pytest.mark.asyncio
async def test_create_job_output_location_forbidden_raises(
    mock_dispatcher: JobDispatcher,
    mock_files_client,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A supplied output_location the caller cannot access is rejected, never auto-created."""
    from nemo_helix_plugin.client.errors import PermissionDeniedError
    from nhx.core.jobs.app.dispatcher import JobOutputLocationError

    mock_files_client.get_fileset.side_effect = PermissionDeniedError.__new__(PermissionDeniedError)
    request = sample_platform_job_request.model_copy(update={"output_location": "forbidden-fileset"})

    with pytest.raises(JobOutputLocationError):
        await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    mock_files_client.create_fileset.assert_not_called()


@pytest.mark.asyncio
async def test_create_job_without_output_location_auto_creates_fileset(
    mock_dispatcher: JobDispatcher,
    mock_files_client,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """With no output_location, the dispatcher auto-creates a per-job fileset (unchanged behavior)."""
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    assert job.fileset == "test-fileset-id"
    mock_files_client.create_fileset.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_job_leaves_caller_supplied_fileset(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, mock_files_client
):
    """Deleting a job whose fileset was caller-supplied removes the job but leaves the fileset."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(
        mock_store, "delete-shared-fs-job", fileset="shared-eval-fileset", output_location="shared-eval-fileset"
    )

    deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)

    assert deleted is True
    mock_files_client.delete_fileset.assert_not_called()
    await verify_job_data_exists(mock_store, job_id, should_exist=False)


@pytest.mark.asyncio
async def test_delete_job_deletes_owned_fileset(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, mock_files_client
):
    """Deleting a job that owns its fileset (output_location unset) removes that fileset too."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(
        mock_store, "delete-owned-job", fileset="job-fileset-delete-owned-job"
    )

    deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)

    assert deleted is True
    mock_files_client.delete_fileset.assert_awaited_once()
    await verify_job_data_exists(mock_store, job_id, should_exist=False)


@pytest.mark.asyncio
async def test_delete_job_leaves_caller_fileset_named_like_owned(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, mock_files_client
):
    """A caller-supplied fileset named exactly like an auto-created one is still left intact.

    Regression for the name-collision the old name-heuristic was vulnerable to: ownership comes
    from the persisted output_location, not from matching ``job-fileset-<name>``.
    """
    job_id, job_name, _, _, _, _ = await create_test_job_data(
        mock_store,
        "collide-job",
        fileset="job-fileset-collide-job",
        output_location="job-fileset-collide-job",
    )

    deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)

    assert deleted is True
    mock_files_client.delete_fileset.assert_not_called()
    await verify_job_data_exists(mock_store, job_id, should_exist=False)


@pytest.mark.asyncio
async def test_create_then_delete_auto_fileset_round_trip(
    mock_dispatcher: JobDispatcher,
    mock_files_client,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """The auto-created fileset the dispatcher makes is exactly the one delete removes.

    Guards the load-bearing ``job.fileset == f"job-fileset-{job.name}"`` predicate: the real
    files service echoes the requested fileset name, so the mock does too here.
    """

    def _echo_create(*, body, workspace):
        fileset = MagicMock()
        fileset.name = body.name
        response = MagicMock()
        response.data.return_value = fileset
        return response

    mock_files_client.create_fileset.side_effect = _echo_create

    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert job.fileset == f"job-fileset-{job.name}"

    cancelled = await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)
    assert cancelled is not None
    assert cancelled.status == HelixJobStatus.CANCELLED

    deleted = await mock_dispatcher.delete_job(job.name, DEFAULT_WORKSPACE)

    assert deleted is True
    mock_files_client.delete_fileset.assert_awaited_once_with(
        name=f"job-fileset-{job.name}", workspace=DEFAULT_WORKSPACE
    )


@pytest.mark.asyncio
async def test_full_length_caller_name_derives_within_limit_fileset_and_round_trips(
    mock_dispatcher: JobDispatcher,
    mock_files_client,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A caller name at the entity-name limit submits and its derived fileset name stays within the limit.

    Guards the end-to-end contract: the fileset name the service derives is a valid, within-limit
    entity name, is stored on the job, and is the exact name delete resolves (no re-derivation).
    """

    def _echo_create(*, body, workspace):
        fileset = MagicMock()
        fileset.name = body.name
        response = MagicMock()
        response.data.return_value = fileset
        return response

    mock_files_client.create_fileset.side_effect = _echo_create

    full_name = "a" * NAME_MAX_LENGTH
    request = sample_platform_job_request.model_copy(update={"name": full_name})

    job = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    assert job.name == full_name
    assert job.fileset == derive_job_fileset_name(full_name, DEFAULT_WORKSPACE)
    assert len(job.fileset) <= NAME_MAX_LENGTH
    assert re.fullmatch(NAME_PATTERN, job.fileset)

    cancelled = await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)
    assert cancelled is not None
    assert cancelled.status == HelixJobStatus.CANCELLED

    deleted = await mock_dispatcher.delete_job(job.name, DEFAULT_WORKSPACE)

    assert deleted is True
    mock_files_client.delete_fileset.assert_awaited_once_with(name=job.fileset, workspace=DEFAULT_WORKSPACE)


@pytest.mark.asyncio
async def test_get_job_updated_at_tracks_attempt_status_change(
    mock_dispatcher: JobDispatcher,
    sample_platform_job_request: CreateHelixJobRequest,
):
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert step is not None

    step, _ = await mock_dispatcher.update_job_status_from_step(step=step, status=HelixJobStatus.ACTIVE)
    active_job = await mock_dispatcher.get_job(job.name, DEFAULT_WORKSPACE)
    assert active_job is not None
    await asyncio.sleep(0.01)
    _, attempt = await mock_dispatcher.update_job_status_from_step(
        step=step,
        status=HelixJobStatus.COMPLETED,
        error_details={},
    )
    assert attempt.status == HelixJobStatus.COMPLETED

    completed_job = await mock_dispatcher.get_job(job.name, DEFAULT_WORKSPACE)
    assert completed_job is not None
    assert completed_job.updated_at > active_job.updated_at


@pytest.mark.asyncio
async def test_rerun_job_success(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Test successful job rerun creates a new attempt."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to active
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.ACTIVE
    await mock_store.update(current_step)

    # Rerun while the job is still running is rejected.
    with pytest.raises(JobOperationConflictError):
        await mock_dispatcher.rerun_job(job.name, DEFAULT_WORKSPACE)
    attempts_response = await mock_store.list(HelixJobAttempt, filter_obj={"job": job.id})
    assert len(attempts_response.data) == 1

    # Progress lives on the attempt. Rerun copies that map and warns when it is not resumable.
    await mock_dispatcher.update_attempt_status_details(
        job.name,
        job.attempt_id,
        DEFAULT_WORKSPACE,
        {
            "resumable": False,
            "non_resumable_reason": "out of memory",
            "percentage_done": 10,
            "events": [{"type": "Warning"}],
            "metrics": {"train_loss": [{"step": 1, "value": 0.5}]},
        },
    )
    await mock_dispatcher.update_job_status_from_step(
        step=current_step,
        status=HelixJobStatus.ERROR,
        error_details={"message": "failed"},
    )

    # Rerun (now that it failed)
    updated_job = await mock_dispatcher.get_job(job.name, DEFAULT_WORKSPACE)
    assert updated_job is not None
    rerun = await mock_dispatcher.rerun_job(updated_job.name, DEFAULT_WORKSPACE)
    assert rerun is not None
    assert rerun.status_details["rerun_warning"] == "out of memory"
    assert rerun.status_details["percentage_done"] == 10
    assert rerun.status_details["metrics"]["train_loss"][0]["value"] == 0.5
    assert "resumable" not in rerun.status_details
    assert "events" not in rerun.status_details

    # Verify a new attempt was created
    attempts_response = await mock_store.list(HelixJobAttempt, filter_obj={"job": updated_job.id})
    assert len(attempts_response.data) == 2

    # Get the new attempt (highest seq)
    sorted_attempts = sorted(attempts_response.data, key=lambda a: a.seq, reverse=True)
    new_attempt = sorted_attempts[0]
    assert new_attempt is not None
    assert new_attempt.seq == 1
    assert new_attempt.status == HelixJobStatus.CREATED

    # Verify the new attempt has a first step created
    steps_response = await mock_store.list(HelixJobStep, filter_obj={"attempt_id": new_attempt.id})
    assert len(steps_response.data) == 1
    # With parent-scoped uniqueness, step names are simple (unique per attempt)
    assert steps_response.data[0].name == "basic"


@pytest.mark.asyncio
async def test_pause_job_success(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Test successful job pause sets step to PAUSING status."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to active
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.ACTIVE
    await mock_store.update(current_step)

    # Pause the job
    await mock_dispatcher.pause_job(job.name, DEFAULT_WORKSPACE)

    # Verify the step was set to pausing
    updated_step = await mock_store.get_by_id(HelixJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == HelixJobStatus.PAUSING
    assert updated_step.status_details["pause_requested_at"]


@pytest.mark.asyncio
async def test_pause_job_no_active_step(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Test pause job when the job has already finished is rejected."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to completed
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.COMPLETED
    await mock_store.update(current_step)

    with pytest.raises(JobOperationConflictError):
        await mock_dispatcher.pause_job(job.name, DEFAULT_WORKSPACE)

    updated_step = await mock_store.get_by_id(HelixJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == HelixJobStatus.COMPLETED


@pytest.mark.asyncio
async def test_cancel_job_success(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Test successful job cancel sets step to CANCELLING status."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to active
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.ACTIVE
    await mock_store.update(current_step)

    # Cancel the job
    await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)

    # Verify the step was set to cancelling
    updated_step = await mock_store.get_by_id(HelixJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == HelixJobStatus.CANCELLING

    # Verify the attempt status was updated (CANCELLING status propagates to attempt)
    attempt = await mock_dispatcher.get_current_attempt(job.name, DEFAULT_WORKSPACE)
    assert attempt is not None
    assert attempt.status == HelixJobStatus.CANCELLING


@pytest.mark.asyncio
async def test_update_job_status_from_step_emits_job_run_telemetry_on_terminal_transition(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    request = sample_platform_job_request.model_copy(
        update={"custom_fields": build_job_telemetry_custom_fields("session-123")}
    )
    job = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.ACTIVE
    current_step = await mock_store.update(current_step)

    attempt = await mock_dispatcher.get_current_attempt(job.name, DEFAULT_WORKSPACE)
    assert attempt is not None
    attempt.status = HelixJobStatus.ACTIVE
    attempt.status_details = {"model": "attempt-model", "input_tokens": 1}
    await mock_store.update(attempt)

    with patch("nhx.core.jobs.app.dispatcher.emit_job_run_event") as emit_event:
        await mock_dispatcher.update_job_status_from_step(
            current_step,
            HelixJobStatus.COMPLETED,
            status_details={"model": "terminal-model", "input_tokens": 3, "output_tokens": 5},
        )

    emit_event.assert_called_once()
    event = emit_event.call_args.args[0]
    assert event.session_id == "session-123"
    assert event.event.task_status == "completed"
    assert event.event.job_type == "custom"
    assert event.event.model == "defined"
    assert event.event.input_tokens == 3
    assert event.event.output_tokens == 5


@pytest.mark.asyncio
async def test_update_job_status_from_step_emits_job_run_telemetry_on_pause(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    request = sample_platform_job_request.model_copy(
        update={"custom_fields": build_job_telemetry_custom_fields("session-123")}
    )
    job = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.ACTIVE
    current_step = await mock_store.update(current_step)

    attempt = await mock_dispatcher.get_current_attempt(job.name, DEFAULT_WORKSPACE)
    assert attempt is not None
    attempt.status = HelixJobStatus.ACTIVE
    await mock_store.update(attempt)

    with patch("nhx.core.jobs.app.dispatcher.emit_job_run_event") as emit_event:
        await mock_dispatcher.update_job_status_from_step(
            current_step,
            HelixJobStatus.PAUSED,
            status_details={"message": "Job is paused"},
        )

    emit_event.assert_called_once()
    event = emit_event.call_args.args[0]
    assert event.session_id == "session-123"
    assert event.event.task_status == "undefined"
    assert event.event.job_type == "custom"


@pytest.mark.asyncio
async def test_update_job_status_from_step_skips_job_run_telemetry_without_session_id(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.ACTIVE
    current_step = await mock_store.update(current_step)

    attempt = await mock_dispatcher.get_current_attempt(job.name, DEFAULT_WORKSPACE)
    assert attempt is not None
    attempt.status = HelixJobStatus.ACTIVE
    await mock_store.update(attempt)

    with patch("nhx.core.jobs.app.dispatcher.emit_job_run_event") as emit_event:
        await mock_dispatcher.update_job_status_from_step(current_step, HelixJobStatus.COMPLETED)

    emit_event.assert_not_called()


@pytest.mark.asyncio
async def test_update_job_status_from_step_retries_on_entity_conflict(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Test that update_job_status_from_step refetches and retries on EntityConflictError."""
    # Create a job and set step to ACTIVE (same setup as cancel flow)
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.ACTIVE
    await mock_store.update(current_step)

    # Simulate reconciler/API race: first store.update(step) raises conflict, second succeeds
    step_update_calls = []
    original_update = mock_store.update

    async def update_side_effect(entity, *args, **kwargs):
        if isinstance(entity, HelixJobStep):
            step_update_calls.append(entity.id)
            if len(step_update_calls) == 1:
                raise EntityConflictError("version conflict (simulated)")
        return await original_update(entity, *args, **kwargs)

    with patch.object(mock_store, "update", side_effect=update_side_effect):
        await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)

    # First update raised; refetch and retry should have succeeded on second update
    assert len(step_update_calls) == 2
    updated_step = await mock_store.get_by_id(HelixJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == HelixJobStatus.CANCELLING


@pytest.mark.asyncio
async def test_update_job_status_from_step_skips_step_store_when_noop(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Identical PENDING + status_details should not persist a HelixJobStep update."""
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.PENDING
    current_step.status_details = {"message": "pulling"}
    await mock_store.update(current_step)

    attempt = await mock_store.get_by_id(HelixJobAttempt, job.attempt_id)
    assert attempt is not None
    attempt.status = HelixJobStatus.PENDING
    await mock_store.update(attempt)

    current_step = await mock_store.get_by_id(HelixJobStep, current_step.id)
    assert current_step is not None

    step_update_count = 0
    original_update = mock_store.update

    async def counting_update(entity, *args, **kwargs):
        nonlocal step_update_count
        if isinstance(entity, HelixJobStep):
            step_update_count += 1
        return await original_update(entity, *args, **kwargs)

    with patch.object(mock_store, "update", side_effect=counting_update):
        await mock_dispatcher.update_job_status_from_step(
            current_step,
            HelixJobStatus.PENDING,
            status_details={"message": "pulling"},
        )

    assert step_update_count == 0


@pytest.mark.asyncio
async def test_update_job_status_from_step_queues_created_attempt_for_first_step(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None

    await mock_dispatcher.update_job_status_from_step(
        current_step, HelixJobStatus.QUEUED, status_details={"message": "waiting for capacity"}
    )

    attempt = await mock_store.get_by_id(HelixJobAttempt, job.attempt_id)
    assert attempt is not None
    assert attempt.status == HelixJobStatus.QUEUED


@pytest.mark.asyncio
async def test_update_job_status_from_step_queued_later_step_leaves_running_attempt_unchanged(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    multi_step_spec = sample_platform_job_request.platform_spec.model_copy(deep=True)
    multi_step_spec.steps.append(
        HelixJobStepSpec(
            name="second-step", executor=sample_platform_job_request.platform_spec.steps[0].executor, config={}
        )
    )
    sample_platform_job_request.platform_spec = multi_step_spec
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    attempt = await mock_store.get_by_id(HelixJobAttempt, job.attempt_id)
    assert attempt is not None
    attempt.status = HelixJobStatus.ACTIVE
    await mock_store.update(attempt)

    second_step = await mock_store.add(
        HelixJobStep(
            attempt_id=job.attempt_id,
            name="second-step",
            workspace=DEFAULT_WORKSPACE,
            config={},
            status=HelixJobStatus.CREATED,
        )
    )

    await mock_dispatcher.update_job_status_from_step(second_step, HelixJobStatus.QUEUED)

    attempt = await mock_store.get_by_id(HelixJobAttempt, job.attempt_id)
    assert attempt is not None
    assert attempt.status == HelixJobStatus.ACTIVE


@pytest.mark.asyncio
async def test_update_job_status_from_step_missing_job_raises_typed_skip(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """A step status update racing a job delete does not raise a generic Exception."""
    job_id, _, _, step_id, _, _ = await create_test_job_data(mock_store, "missing-job-status-update")
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    await mock_store.delete_by_id(HelixJob, job_id)

    with pytest.raises(JobStatusUpdateSkippedError):
        await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.COMPLETED)


@pytest.mark.asyncio
async def test_cancel_job_multiple_steps_only_cancels_active(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Test cancel job only cancels the active/non-terminal step."""
    # Modify the job to have multiple steps
    multi_step_spec = sample_platform_job_request.platform_spec.model_copy(deep=True)
    multi_step_spec.steps.append(
        HelixJobStepSpec(
            name="second-step", executor=sample_platform_job_request.platform_spec.steps[0].executor, config={}
        )
    )
    sample_platform_job_request.platform_spec = multi_step_spec

    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the first step and set it to completed
    first_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert first_step is not None
    first_step.status = HelixJobStatus.COMPLETED
    await mock_store.update(first_step)

    # Create and save a second step as active (parent-scoped: unique per attempt)
    second_step = HelixJobStep(
        attempt_id=job.attempt_id,
        name="second-step",  # Simple name, unique within this attempt
        workspace=DEFAULT_WORKSPACE,
        config={},
        status=HelixJobStatus.ACTIVE,
    )
    second_step = await mock_store.add(second_step)

    # Cancel the job
    await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)

    # Verify only the active step was cancelled
    updated_first_step = await mock_store.get_by_id(HelixJobStep, first_step.id)
    assert updated_first_step is not None
    assert updated_first_step.status == HelixJobStatus.COMPLETED  # Should remain completed

    updated_second_step = await mock_store.get_by_id(HelixJobStep, second_step.id)
    assert updated_second_step is not None
    assert updated_second_step.status == HelixJobStatus.CANCELLING  # Should be cancelling


@pytest.mark.asyncio
async def test_cancel_job_no_active_step(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Test cancel job when no active step exists returns job unchanged."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to completed
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.COMPLETED
    await mock_store.update(current_step)

    # Cancel the job
    await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)

    # Verify the step status didn't change
    updated_step = await mock_store.get_by_id(HelixJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == HelixJobStatus.COMPLETED


@pytest.mark.asyncio
async def test_resume_job_success(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Test successful job resume sets paused step to RESUMING status."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to paused
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.PAUSED
    await mock_store.update(current_step)
    attempt = await mock_store.get_by_id(HelixJobAttempt, job.attempt_id)
    assert attempt is not None
    attempt.status = HelixJobStatus.PAUSED
    await mock_store.update(attempt)

    # Resume the job
    await mock_dispatcher.resume_job(job.name, DEFAULT_WORKSPACE)

    # Verify the step was set to resuming
    updated_step = await mock_store.get_by_id(HelixJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == HelixJobStatus.RESUMING


@pytest.mark.asyncio
async def test_resume_clears_the_previous_image_comparison(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """The resumed pod is compared with this pause's digest, not left behind by an earlier resume's result."""
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert step is not None
    step.status = HelixJobStatus.PAUSED
    step.status_details = {
        IMAGE_DIGEST: "sha256:b",
        IMAGE_DIGEST_AT_SAVE: "sha256:b",
        IMAGE_DIGEST_WARNING: "The training image changed since this state was saved (was sha256:a, now sha256:b).",
    }
    await mock_store.update(step)
    attempt = await mock_store.get_by_id(HelixJobAttempt, job.attempt_id)
    assert attempt is not None
    attempt.status = HelixJobStatus.PAUSED
    await mock_store.update(attempt)

    await mock_dispatcher.resume_job(job.name, DEFAULT_WORKSPACE)

    resumed = await mock_store.get_by_id(HelixJobStep, step.id)
    assert resumed is not None
    assert resumed.status == HelixJobStatus.RESUMING
    assert IMAGE_DIGEST_WARNING not in resumed.status_details
    assert resumed.status_details[IMAGE_DIGEST_AT_SAVE] == "sha256:b"


@pytest.mark.asyncio
async def test_resume_job_no_paused_step(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Resume of an active job is a no-op."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set the attempt to active. Resume is a no-op.
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = HelixJobStatus.ACTIVE
    await mock_store.update(current_step)
    attempt = await mock_store.get_by_id(HelixJobAttempt, job.attempt_id)
    assert attempt is not None
    attempt.status = HelixJobStatus.ACTIVE
    await mock_store.update(attempt)

    resumed = await mock_dispatcher.resume_job(job.name, DEFAULT_WORKSPACE)
    assert resumed is not None
    assert resumed.status == HelixJobStatus.ACTIVE

    updated_step = await mock_store.get_by_id(HelixJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == HelixJobStatus.ACTIVE


@pytest.mark.asyncio
async def test_pause_targets_a_live_step_before_an_idempotent_return(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A created, pending, or active step is paused even if another step is already paused."""
    spec = sample_platform_job_request.platform_spec.model_copy(deep=True)
    spec.steps[0].lifecycle = StepLifecycle(pause_deadline_seconds=60)
    spec.steps.append(
        HelixJobStepSpec(
            name="second-step",
            executor=spec.steps[0].executor,
            config={},
            lifecycle=StepLifecycle(pause_deadline_seconds=60),
        )
    )
    request = sample_platform_job_request.model_copy(update={"name": "pause-live-step", "platform_spec": spec})
    job = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    first_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert first_step is not None
    await mock_dispatcher.update_job_status_from_step(first_step, HelixJobStatus.PAUSED)
    second_step = await mock_store.add(
        HelixJobStep(
            attempt_id=job.attempt_id,
            name="second-step",
            workspace=DEFAULT_WORKSPACE,
            config={},
            status=HelixJobStatus.ACTIVE,
        )
    )

    paused = await mock_dispatcher.pause_job(job.name, DEFAULT_WORKSPACE)
    assert paused is not None
    updated_second = await mock_store.get_by_id(HelixJobStep, second_step.id)
    assert updated_second is not None
    assert updated_second.status == HelixJobStatus.PAUSING
    updated_first = await mock_store.get_by_id(HelixJobStep, first_step.id)
    assert updated_first is not None
    assert updated_first.status == HelixJobStatus.PAUSED


@pytest.mark.asyncio
async def test_get_job_status_returns_steps_in_spec_order(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """Test that get_job_status returns steps in the same order as defined in platform_job_spec."""
    # Create a multi-step job with steps in a specific order
    multi_step_spec = TestConstants.PLATFORM_SPEC.model_copy(deep=True)
    multi_step_spec.steps = [
        HelixJobStepSpec(
            name="first-step",
            executor=TestConstants.PLATFORM_SPEC.steps[0].executor,
            config={"order": 1},
        ),
        HelixJobStepSpec(
            name="second-step",
            executor=TestConstants.PLATFORM_SPEC.steps[0].executor,
            config={"order": 2},
        ),
        HelixJobStepSpec(
            name="third-step",
            executor=TestConstants.PLATFORM_SPEC.steps[0].executor,
            config={"order": 3},
        ),
    ]

    job_request = CreateHelixJobRequest(
        name="test-job-step-order",
        description="Test job for step ordering",
        project="test-project",
        source=TestConstants.SOURCE,
        spec=TestConstants.SPEC_BASIC,
        platform_spec=multi_step_spec,
        ownership=TestConstants.OWNERSHIP_BASIC,
        custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
    )

    # Create the job
    job = await mock_dispatcher.create_job(job_request, DEFAULT_WORKSPACE)

    # Complete the first step and create subsequent steps
    first_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "first-step", DEFAULT_WORKSPACE)
    assert first_step is not None
    first_step.status = HelixJobStatus.COMPLETED
    await mock_store.update(first_step)

    # Create second step
    second_step = HelixJobStep(
        attempt_id=job.attempt_id,
        name="second-step",
        workspace=DEFAULT_WORKSPACE,
        config={"order": 2},
        status=HelixJobStatus.COMPLETED,
    )
    second_step = await mock_store.add(second_step)

    # Create third step
    third_step = HelixJobStep(
        attempt_id=job.attempt_id,
        name="third-step",
        workspace=DEFAULT_WORKSPACE,
        config={"order": 3},
        status=HelixJobStatus.ACTIVE,
    )
    third_step = await mock_store.add(third_step)

    # Get the job status
    job_status = await mock_dispatcher.get_job_status(job.name, DEFAULT_WORKSPACE)

    # Verify the job status was returned
    assert job_status is not None
    assert len(job_status.steps) == 3

    # Verify steps are in the same order as defined in platform_spec
    assert job_status.steps[0].name == "first-step", "First step should be 'first-step'"
    assert job_status.steps[1].name == "second-step", "Second step should be 'second-step'"
    assert job_status.steps[2].name == "third-step", "Third step should be 'third-step'"

    # Also verify the order matches the platform_spec
    expected_step_names = [step.name for step in multi_step_spec.steps]
    actual_step_names = [step.name for step in job_status.steps]
    assert actual_step_names == expected_step_names, (
        f"Steps should be in platform_spec order. Expected: {expected_step_names}, Got: {actual_step_names}"
    )


@pytest.mark.asyncio
async def test_list_steps_across_multiple_workspaces(
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Test list_steps respects workspace filtering."""
    from unittest.mock import AsyncMock, MagicMock

    from nhx.common.entities.client import EntityClient
    from nhx.core.jobs.api.v2.jobs.schemas import HelixJobSortField, HelixJobStepsListFilter
    from nhx.testing import create_test_client

    # Create entity store with multiple workspaces and projects
    projects = ["default/test-project", "other-workspace/test-project"]
    with create_test_client(client_type=EntityClient, projects=projects) as mock_store:
        mock_files = AsyncMock()
        mock_fileset_obj = MagicMock()
        mock_fileset_obj.name = "test-fileset-id"
        mock_resp = MagicMock()
        mock_resp.data.return_value = mock_fileset_obj
        mock_files.create_fileset.return_value = mock_resp

        # Create dispatcher with the multi-workspace store
        mock_dispatcher = JobDispatcher(store=mock_store, files=mock_files, secrets=AsyncMock())

        # Create jobs in workspace "default"
        job1 = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        job2_request = CreateHelixJobRequest(
            name="test-job-2",
            description="Second test job",
            project="test-project",
            source=TestConstants.SOURCE,
            spec=TestConstants.SPEC_BASIC,
            platform_spec=TestConstants.PLATFORM_SPEC,
            ownership=TestConstants.OWNERSHIP_BASIC,
            custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
        )
        job2 = await mock_dispatcher.create_job(job2_request, DEFAULT_WORKSPACE)

        # Create jobs in workspace "other-workspace"
        job3_request = CreateHelixJobRequest(
            name="test-job-3",
            description="Third test job in other workspace",
            project="test-project",
            source=TestConstants.SOURCE,
            spec=TestConstants.SPEC_BASIC,
            platform_spec=TestConstants.PLATFORM_SPEC,
            ownership=TestConstants.OWNERSHIP_BASIC,
            custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
        )
        job3 = await mock_dispatcher.create_job(job3_request, "other-workspace")

        job4_request = CreateHelixJobRequest(
            name="test-job-4",
            description="Fourth test job in other workspace",
            project="test-project",
            source=TestConstants.SOURCE,
            spec=TestConstants.SPEC_BASIC,
            platform_spec=TestConstants.PLATFORM_SPEC,
            ownership=TestConstants.OWNERSHIP_BASIC,
            custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
        )
        job4 = await mock_dispatcher.create_job(job4_request, "other-workspace")

        # List steps in "default" workspace
        step_filter = HelixJobStepsListFilter()
        steps_default, count_default = await mock_dispatcher.list_steps(
            filter=step_filter,
            sort=HelixJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace=DEFAULT_WORKSPACE,
        )

        # Should return only steps from "default" workspace
        assert len(steps_default) == 2, f"Expected 2 steps in 'default' workspace, but got {len(steps_default)}"
        assert count_default == 2

        # Verify the steps belong to jobs in default workspace
        job_names_default = {step.job for step in steps_default}
        assert {step.job_id for step in steps_default} == {job1.id, job2.id}
        assert job1.name in job_names_default
        assert job2.name in job_names_default
        assert job3.name not in job_names_default
        assert job4.name not in job_names_default

        # List steps in "other-workspace"
        steps_other, count_other = await mock_dispatcher.list_steps(
            filter=step_filter,
            sort=HelixJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace="other-workspace",
        )

        # Should return only steps from "other-workspace"
        assert len(steps_other) == 2, f"Expected 2 steps in 'other-workspace', but got {len(steps_other)}"
        assert count_other == 2

        # Verify the steps belong to jobs in other-workspace
        job_names_other = {step.job for step in steps_other}
        assert job3.name in job_names_other
        assert job4.name in job_names_other
        assert job1.name not in job_names_other
        assert job2.name not in job_names_other

        # Now query across both workspaces using ALL_WORKSPACES
        steps_all, count_all = await mock_dispatcher.list_steps(
            filter=step_filter,
            sort=HelixJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace=ALL_WORKSPACES,
        )

        # Should return all 4 steps
        assert len(steps_all) == 4, f"Expected 4 steps across all workspaces, but got {len(steps_all)}"
        assert count_all == 4


@pytest.mark.asyncio
async def test_list_steps_with_status_filter(
    sample_platform_job_request: CreateHelixJobRequest,
):
    """list_steps with filter.status set forwards a filter_str to the entity store.

    Regression: this branch used to pass ``search=`` to ``EntityClient.list``;
    after the kwarg was renamed to ``filter_str``, an unguarded call would
    ``TypeError`` at runtime. Existing tests left ``filter.status`` empty so
    the filter_str branch was never exercised.
    """
    from unittest.mock import AsyncMock, MagicMock

    from nhx.common.entities.client import EntityClient
    from nhx.core.jobs.api.v2.jobs.schemas import (
        HelixJobSortField,
        HelixJobStatus,
        HelixJobStepsListFilter,
    )
    from nhx.testing import create_test_client

    with create_test_client(client_type=EntityClient) as mock_store:
        mock_files = AsyncMock()
        mock_fileset_obj = MagicMock()
        mock_fileset_obj.name = "test-fileset-id"
        mock_resp = MagicMock()
        mock_resp.data.return_value = mock_fileset_obj
        mock_files.create_fileset.return_value = mock_resp

        mock_dispatcher = JobDispatcher(store=mock_store, files=mock_files, secrets=AsyncMock())
        await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        # Filtering by a status that no step has should return nothing — the
        # important assertion is that the call doesn't raise.
        step_filter = HelixJobStepsListFilter(status=[HelixJobStatus.ACTIVE])
        steps, count = await mock_dispatcher.list_steps(
            filter=step_filter,
            sort=HelixJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace=DEFAULT_WORKSPACE,
        )
        assert steps == []
        assert count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [100, 2])
async def test_list_steps_looks_up_attempts_and_jobs_in_batches(
    sample_platform_job_request: CreateHelixJobRequest, batch_size: int
):
    """A page of steps costs a store query per batch of IDs, not a request per step.

    Each per-ID request was an authorized call back into the platform, so enriching a page of 50
    steps made about 100 of them.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from nhx.common.entities.client import EntityClient
    from nhx.core.jobs.api.v2.jobs.schemas import HelixJobSortField, HelixJobStepsListFilter
    from nhx.core.jobs.entities import HelixJob, HelixJobAttempt
    from nhx.testing import create_test_client

    with create_test_client(client_type=EntityClient) as mock_store:
        mock_files = AsyncMock()
        mock_fileset_obj = MagicMock()
        mock_fileset_obj.name = "test-fileset-id"
        mock_resp = MagicMock()
        mock_resp.data.return_value = mock_fileset_obj
        mock_files.create_fileset.return_value = mock_resp
        dispatcher = JobDispatcher(store=mock_store, files=mock_files, secrets=AsyncMock())
        jobs = [
            await dispatcher.create_job(
                sample_platform_job_request.model_copy(update={"name": f"batched-job-{index}"}), DEFAULT_WORKSPACE
            )
            for index in range(3)
        ]

        with (
            patch("nhx.core.jobs.app.dispatcher._ENRICHMENT_BATCH_SIZE", batch_size),
            patch.object(mock_store, "get_by_id", wraps=mock_store.get_by_id) as get_by_id,
            patch.object(mock_store, "list", wraps=mock_store.list) as list_entities,
        ):
            steps, count = await dispatcher.list_steps(
                filter=HelixJobStepsListFilter(),
                sort=HelixJobSortField.CREATED_AT_ASC,
                limit=100,
                offset=0,
                workspace=DEFAULT_WORKSPACE,
            )

        assert count == 3
        assert {step.job_id for step in steps} == {job.id for job in jobs}
        get_by_id.assert_not_called()
        listed = [call.args[0] for call in list_entities.call_args_list]
        expected_queries = 1 if batch_size >= 3 else 2
        assert listed.count(HelixJobAttempt) == expected_queries
        assert listed.count(HelixJob) == expected_queries


@pytest.mark.asyncio
async def test_list_steps_drops_a_step_whose_job_no_longer_exists(sample_platform_job_request: CreateHelixJobRequest):
    """A step is listed only with its attempt and job, as when each was fetched by ID."""
    from unittest.mock import AsyncMock, MagicMock

    from nhx.common.entities.client import EntityClient
    from nhx.core.jobs.api.v2.jobs.schemas import HelixJobSortField, HelixJobStepsListFilter
    from nhx.core.jobs.entities import HelixJob
    from nhx.testing import create_test_client

    with create_test_client(client_type=EntityClient) as mock_store:
        mock_files = AsyncMock()
        mock_fileset_obj = MagicMock()
        mock_fileset_obj.name = "test-fileset-id"
        mock_resp = MagicMock()
        mock_resp.data.return_value = mock_fileset_obj
        mock_files.create_fileset.return_value = mock_resp
        dispatcher = JobDispatcher(store=mock_store, files=mock_files, secrets=AsyncMock())
        kept = await dispatcher.create_job(
            sample_platform_job_request.model_copy(update={"name": "kept-job"}), DEFAULT_WORKSPACE
        )
        gone = await dispatcher.create_job(
            sample_platform_job_request.model_copy(update={"name": "gone-job"}), DEFAULT_WORKSPACE
        )
        await mock_store.delete(HelixJob, gone.name, workspace=DEFAULT_WORKSPACE)

        steps, _ = await dispatcher.list_steps(
            filter=HelixJobStepsListFilter(),
            sort=HelixJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace=DEFAULT_WORKSPACE,
        )

        assert {step.job_id for step in steps} == {kept.id}


@pytest.mark.asyncio
async def test_get_job_by_name(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """get_job resolves a job by its name."""
    _, job_name, _, _, _, _ = await create_test_job_data(mock_store, "get-by-name-job")

    job = await mock_dispatcher.get_job(job_name, DEFAULT_WORKSPACE)

    assert job is not None
    assert job.name == job_name


@pytest.mark.asyncio
async def test_get_job_by_id(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """get_job resolves a job by its ID (the value ``nemo jobs list`` leads with)."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "get-by-id-job")
    assert job_id != job_name

    job = await mock_dispatcher.get_job(job_id, DEFAULT_WORKSPACE)

    assert job is not None
    assert job.id == job_id
    assert job.name == job_name


@pytest.mark.asyncio
async def test_get_job_resolves_id_before_name(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """A successful ID lookup does not perform a fallback name lookup."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "canonical-job")

    get_by_name = AsyncMock(side_effect=AssertionError("name fallback must not run after an ID match"))
    with patch.object(mock_store, "get", new=get_by_name):
        job = await mock_dispatcher.get_job(job_id, DEFAULT_WORKSPACE)

    assert job is not None
    assert job.id == job_id
    assert job.name == job_name
    get_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_job_unknown_identifier_returns_none(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """get_job returns None when neither the name nor the ID matches."""
    await create_test_job_data(mock_store, "present-job")

    assert await mock_dispatcher.get_job("does-not-exist", DEFAULT_WORKSPACE) is None


@pytest.mark.asyncio
async def test_cancel_job_by_id(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """cancel_job resolves a job by its ID, not just its name."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "cancel-by-id")
    assert job_id != job_name

    result = await mock_dispatcher.cancel_job(job_id, DEFAULT_WORKSPACE)

    assert result is not None
    assert result.id == job_id


@pytest.mark.asyncio
async def test_pause_job_by_id(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """pause_job resolves a job by its ID, not just its name."""
    job_id, job_name, attempt_id, step_id, _, _ = await create_test_job_data(mock_store, "pause-by-id")
    assert job_id != job_name
    attempt = await mock_store.get_by_id(HelixJobAttempt, attempt_id)
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    assert attempt is not None and step is not None
    attempt.status = HelixJobStatus.ACTIVE
    step.status = HelixJobStatus.ACTIVE
    step.config = {"_step_spec_name": "basic"}
    await mock_store.update(attempt)
    await mock_store.update(step)

    result = await mock_dispatcher.pause_job(job_id, DEFAULT_WORKSPACE)

    assert result is not None
    assert result.id == job_id


@pytest.mark.asyncio
async def test_resume_job_by_id(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """resume_job resolves a job by its ID, not just its name."""
    job_id, job_name, attempt_id, step_id, _, _ = await create_test_job_data(mock_store, "resume-by-id")
    assert job_id != job_name
    attempt = await mock_store.get_by_id(HelixJobAttempt, attempt_id)
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    assert attempt is not None and step is not None
    attempt.status = HelixJobStatus.PAUSED
    step.status = HelixJobStatus.PAUSED
    await mock_store.update(attempt)
    await mock_store.update(step)

    result = await mock_dispatcher.resume_job(job_id, DEFAULT_WORKSPACE)

    assert result is not None
    assert result.id == job_id


@pytest.mark.asyncio
async def test_rerun_job_by_id(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """rerun_job resolves a job by its ID (and locks on the resolved name)."""
    job_id, job_name, attempt_id, step_id, _, _ = await create_test_job_data(mock_store, "rerun-by-id")
    assert job_id != job_name
    attempt = await mock_store.get_by_id(HelixJobAttempt, attempt_id)
    step = await mock_store.get_by_id(HelixJobStep, step_id)
    assert attempt is not None and step is not None
    attempt.status = HelixJobStatus.ERROR
    step.status = HelixJobStatus.ERROR
    await mock_store.update(attempt)
    await mock_store.update(step)

    result = await mock_dispatcher.rerun_job(job_id, DEFAULT_WORKSPACE)

    assert result is not None
    assert result.id == job_id


@pytest.mark.asyncio
async def test_delete_job_by_id(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """delete_job resolves a job by its ID (and locks on the resolved name)."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "delete-by-id")
    assert job_id != job_name

    deleted = await mock_dispatcher.delete_job(job_id, DEFAULT_WORKSPACE)

    assert deleted is True
    await verify_job_data_exists(mock_store, job_id, should_exist=False)


@pytest.mark.asyncio
async def test_delete_job_locked_rejects_same_name_replacement(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient
):
    """The ID guard stops a stale id-based delete from hitting a same-name replacement.

    Simulates the TOCTOU window: the original job (resolved by id) is gone and a
    new job now holds the same name. _delete_job_locked, given the ORIGINAL id,
    must refuse to delete the replacement.
    """
    original_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "racing-del-job")
    await mock_dispatcher.delete_job(original_id, DEFAULT_WORKSPACE)
    # A replacement job reuses the same name but has a different id.
    replacement_id, _, _, _, _, _ = await create_test_job_data(mock_store, job_name)
    assert replacement_id != original_id

    deleted = await mock_dispatcher._delete_job_locked(job_name, DEFAULT_WORKSPACE, expected_id=original_id)

    assert deleted is False
    # The replacement must be untouched.
    await verify_job_data_exists(mock_store, replacement_id, should_exist=True)


@pytest.mark.asyncio
async def test_rerun_job_locked_rejects_same_name_replacement(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """The ID guard stops a stale id-based rerun from hitting a same-name replacement."""
    original_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "racing-rerun-job")
    await mock_dispatcher.delete_job(original_id, DEFAULT_WORKSPACE)
    replacement_id, _, _, _, _, _ = await create_test_job_data(mock_store, job_name)
    assert replacement_id != original_id

    result = await mock_dispatcher._rerun_job_locked(job_name, DEFAULT_WORKSPACE, expected_id=original_id)

    assert result is None
    # The replacement must not have gained a new attempt.
    attempts = await mock_store.list(HelixJobAttempt, filter_obj={"job": replacement_id})
    assert len(attempts.data) == 1


@pytest.mark.asyncio
async def test_mutation_methods_unknown_identifier(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """Mutation methods return the not-found sentinel for an unknown identifier."""
    await create_test_job_data(mock_store, "present-for-mutations")

    assert await mock_dispatcher.cancel_job("nope", DEFAULT_WORKSPACE) is None
    assert await mock_dispatcher.pause_job("nope", DEFAULT_WORKSPACE) is None
    assert await mock_dispatcher.resume_job("nope", DEFAULT_WORKSPACE) is None
    assert await mock_dispatcher.rerun_job("nope", DEFAULT_WORKSPACE) is None
    assert await mock_dispatcher.delete_job("nope", DEFAULT_WORKSPACE) is False


@pytest.mark.asyncio
async def test_get_job_by_id_does_not_cross_workspaces(
    sample_platform_job_request: CreateHelixJobRequest,
):
    """The ID fallback is workspace-agnostic, so a foreign-workspace ID must not resolve."""
    from unittest.mock import MagicMock

    from nhx.testing import create_test_client

    projects = ["default/test-project", "other-workspace/test-project"]
    with create_test_client(client_type=EntityClient, projects=projects) as mock_store:
        mock_files = AsyncMock()
        mock_fileset_obj = MagicMock()
        mock_fileset_obj.name = "test-fileset-id"
        mock_resp = MagicMock()
        mock_resp.data.return_value = mock_fileset_obj
        mock_files.create_fileset.return_value = mock_resp

        dispatcher = JobDispatcher(store=mock_store, files=mock_files, secrets=AsyncMock())

        # A job that lives in "other-workspace".
        other_job = await dispatcher.create_job(sample_platform_job_request, "other-workspace")

        # Looking it up by ID from "default" must not leak it.
        assert await dispatcher.get_job(other_job.id, DEFAULT_WORKSPACE) is None
        # But it resolves by ID from its own workspace.
        resolved = await dispatcher.get_job(other_job.id, "other-workspace")
        assert resolved is not None
        assert resolved.id == other_job.id


@pytest.mark.asyncio
async def test_list_jobs_across_multiple_workspaces(
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Test list_jobs respects workspace filtering."""
    from unittest.mock import MagicMock

    from nhx.common.entities.client import EntityClient
    from nhx.core.jobs.api.v2.jobs.schemas import HelixJobListSortField
    from nhx.testing import create_test_client

    # Create entity store with multiple workspaces and projects
    projects = ["default/test-project", "other-workspace/test-project"]
    with create_test_client(client_type=EntityClient, projects=projects) as mock_store:
        mock_files = AsyncMock()
        mock_fileset_obj = MagicMock()
        mock_fileset_obj.name = "test-fileset-id"
        mock_resp = MagicMock()
        mock_resp.data.return_value = mock_fileset_obj
        mock_files.create_fileset.return_value = mock_resp

        # Create dispatcher with the multi-workspace store
        mock_dispatcher = JobDispatcher(store=mock_store, files=mock_files, secrets=AsyncMock())

        # Create jobs in workspace "default"
        job1 = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        job2_request = CreateHelixJobRequest(
            name="test-job-2",
            description="Second test job",
            project="test-project",
            source=TestConstants.SOURCE,
            spec=TestConstants.SPEC_BASIC,
            platform_spec=TestConstants.PLATFORM_SPEC,
            ownership=TestConstants.OWNERSHIP_BASIC,
            custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
        )
        job2 = await mock_dispatcher.create_job(job2_request, DEFAULT_WORKSPACE)

        # Create jobs in workspace "other-workspace"
        job3_request = CreateHelixJobRequest(
            name="test-job-3",
            description="Third test job in other workspace",
            project="test-project",
            source=TestConstants.SOURCE,
            spec=TestConstants.SPEC_BASIC,
            platform_spec=TestConstants.PLATFORM_SPEC,
            ownership=TestConstants.OWNERSHIP_BASIC,
            custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
        )
        job3 = await mock_dispatcher.create_job(job3_request, "other-workspace")

        job4_request = CreateHelixJobRequest(
            name="test-job-4",
            description="Fourth test job in other workspace",
            project="test-project",
            source=TestConstants.SOURCE,
            spec=TestConstants.SPEC_BASIC,
            platform_spec=TestConstants.PLATFORM_SPEC,
            ownership=TestConstants.OWNERSHIP_BASIC,
            custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
        )
        job4 = await mock_dispatcher.create_job(job4_request, "other-workspace")

        # List jobs in "default" workspace
        jobs_default, count_default = await mock_dispatcher.list_jobs(
            parsed=ParsedFilter(operation=None),
            sort=HelixJobListSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace=DEFAULT_WORKSPACE,
        )

        # Should return only jobs from "default" workspace
        assert len(jobs_default) == 2, (
            f"Expected 2 jobs in '{DEFAULT_WORKSPACE}' workspace, but got {len(jobs_default)}"
        )
        assert count_default == 2

        # Verify the jobs belong to default workspace
        job_ids_default = {job.id for job in jobs_default}
        assert job1.id in job_ids_default
        assert job2.id in job_ids_default
        assert job3.id not in job_ids_default
        assert job4.id not in job_ids_default

        # List jobs in "other-workspace"
        jobs_other, count_other = await mock_dispatcher.list_jobs(
            parsed=ParsedFilter(operation=None),
            sort=HelixJobListSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace="other-workspace",
        )

        # Should return only jobs from "other-workspace"
        assert len(jobs_other) == 2, f"Expected 2 jobs in 'other-workspace', but got {len(jobs_other)}"
        assert count_other == 2

        # Verify the jobs belong to other-workspace
        job_ids_other = {job.id for job in jobs_other}
        assert job3.id in job_ids_other
        assert job4.id in job_ids_other
        assert job1.id not in job_ids_other
        assert job2.id not in job_ids_other

        # Now query across both workspaces using ALL_WORKSPACES
        jobs_all, count_all = await mock_dispatcher.list_jobs(
            parsed=ParsedFilter(operation=None),
            sort=HelixJobListSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace=ALL_WORKSPACES,
        )

        # Should return all 4 jobs
        assert len(jobs_all) == 4, f"Expected 4 jobs across all workspaces, but got {len(jobs_all)}"
        assert count_all == 4


@pytest.mark.asyncio
async def test_list_jobs_sort_by_source(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """list_jobs sorts by the source field ascending and descending."""
    from nhx.core.jobs.api.v2.jobs.schemas import HelixJobListSortField

    for source in ("zebra-source", "alpha-source", "middle-source"):
        await mock_dispatcher.create_job(
            CreateHelixJobRequest(
                name=f"job-{source}",
                source=source,
                project=TestConstants.PROJECT,
                spec=TestConstants.SPEC_BASIC,
                platform_spec=TestConstants.PLATFORM_SPEC,
            ),
            DEFAULT_WORKSPACE,
        )

    jobs_asc, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(operation=None),
        sort=HelixJobListSortField.SOURCE_ASC,
        workspace=DEFAULT_WORKSPACE,
    )
    assert [j.source for j in jobs_asc] == ["alpha-source", "middle-source", "zebra-source"]

    jobs_desc, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(operation=None),
        sort=HelixJobListSortField.SOURCE_DESC,
        workspace=DEFAULT_WORKSPACE,
    )
    assert [j.source for j in jobs_desc] == ["zebra-source", "middle-source", "alpha-source"]


# =============================================================================
# list_jobs: status in-memory filtering via ParsedFilter
# =============================================================================


async def _set_job_status(dispatcher: JobDispatcher, job_name: str, status: HelixJobStatus) -> None:
    step = await dispatcher.get_current_job_step_by_name(
        job_name, TestConstants.PLATFORM_SPEC.steps[0].name, DEFAULT_WORKSPACE
    )
    assert step is not None
    for next_status in dict.fromkeys([HelixJobStatus.ACTIVE, status]):
        step, _ = await dispatcher.update_job_status_from_step(step=step, status=next_status, error_details={})


async def _make_job(
    dispatcher: JobDispatcher,
    store: EntityClient,
    name: str,
    status: HelixJobStatus,
) -> HelixJobResponse:
    """Create a job via the dispatcher and set its attempt to the given status."""
    request = CreateHelixJobRequest(
        name=name,
        source=TestConstants.SOURCE,
        project=TestConstants.PROJECT,
        spec=TestConstants.SPEC_BASIC,
        platform_spec=TestConstants.PLATFORM_SPEC,
    )
    job = await dispatcher.create_job(request, DEFAULT_WORKSPACE)
    await _set_job_status(dispatcher, job.name, status)
    return job


@pytest.mark.asyncio
async def test_list_jobs_filter_status_single_excludes_non_matching(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """filter status with a single value returns only jobs whose attempt matches."""
    active_job = await _make_job(mock_dispatcher, mock_store, "job-active", HelixJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "job-completed", HelixJobStatus.COMPLETED)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=ComparisonOperation(field="data.status", operator=FilterOperator.EQ, value="active"),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    assert len(jobs) == 1
    assert jobs[0].id == active_job.id


@pytest.mark.asyncio
async def test_list_jobs_filter_updated_at_matches_attempt_change(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    job = await _make_job(mock_dispatcher, mock_store, "job-updated", HelixJobStatus.ACTIVE)
    await asyncio.sleep(0.01)
    cutoff = datetime.now(timezone.utc)
    await asyncio.sleep(0.01)
    await _set_job_status(mock_dispatcher, job.name, HelixJobStatus.COMPLETED)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=ComparisonOperation(field="updated_at", operator=FilterOperator.GTE, value=cutoff.isoformat()),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    assert [j.id for j in jobs] == [job.id]


@pytest.mark.asyncio
async def test_list_jobs_filter_status_list_is_or(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """filter status with multiple values matches any of the given statuses (OR logic)."""
    active_job = await _make_job(mock_dispatcher, mock_store, "job-active", HelixJobStatus.ACTIVE)
    completed_job = await _make_job(mock_dispatcher, mock_store, "job-completed", HelixJobStatus.COMPLETED)
    await _make_job(mock_dispatcher, mock_store, "job-error", HelixJobStatus.ERROR)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=LogicalOperation(
                operator=FilterOperator.OR,
                operations=[
                    ComparisonOperation(field="data.status", operator=FilterOperator.EQ, value="active"),
                    ComparisonOperation(field="data.status", operator=FilterOperator.EQ, value="completed"),
                ],
            ),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_ids = {j.id for j in jobs}
    assert active_job.id in returned_ids
    assert completed_job.id in returned_ids
    assert len(returned_ids) == 2


@pytest.mark.asyncio
async def test_list_jobs_filter_status_not_eq_returns_complement(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """$not/$eq on status returns jobs whose status is NOT the value."""
    await _make_job(mock_dispatcher, mock_store, "job-active", HelixJobStatus.ACTIVE)
    completed = await _make_job(mock_dispatcher, mock_store, "job-completed", HelixJobStatus.COMPLETED)
    error = await _make_job(mock_dispatcher, mock_store, "job-error", HelixJobStatus.ERROR)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=parse_json_filter('{"data.status": {"$not": {"$eq": "active"}}}'),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_ids = {j.id for j in jobs}
    assert returned_ids == {completed.id, error.id}


@pytest.mark.asyncio
async def test_list_jobs_filter_status_in(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """$in on status matches any of the listed values."""

    active = await _make_job(mock_dispatcher, mock_store, "job-active", HelixJobStatus.ACTIVE)
    completed = await _make_job(mock_dispatcher, mock_store, "job-completed", HelixJobStatus.COMPLETED)
    await _make_job(mock_dispatcher, mock_store, "job-error", HelixJobStatus.ERROR)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=parse_json_filter('{"data.status": {"$in": ["active", "completed"]}}'),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_ids = {j.id for j in jobs}
    assert returned_ids == {active.id, completed.id}


@pytest.mark.asyncio
async def test_list_jobs_filter_status_nin_returns_complement(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """$nin on status returns jobs whose status is none of the values."""
    await _make_job(mock_dispatcher, mock_store, "job-active", HelixJobStatus.ACTIVE)
    completed = await _make_job(mock_dispatcher, mock_store, "job-completed", HelixJobStatus.COMPLETED)
    error = await _make_job(mock_dispatcher, mock_store, "job-error", HelixJobStatus.ERROR)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=parse_json_filter('{"data.status": {"$nin": ["active"]}}'),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_ids = {j.id for j in jobs}
    assert returned_ids == {completed.id, error.id}


@pytest.mark.asyncio
async def test_list_jobs_filter_or_status_with_non_status(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """$or mixing status with a non-status field returns the union.

    Matches jobs that are ACTIVE *or* whose name contains "special", regardless
    of the other condition.
    """
    active = await _make_job(mock_dispatcher, mock_store, "ordinary-active", HelixJobStatus.ACTIVE)
    special_completed = await _make_job(mock_dispatcher, mock_store, "special-completed", HelixJobStatus.COMPLETED)
    await _make_job(mock_dispatcher, mock_store, "ordinary-completed", HelixJobStatus.COMPLETED)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=parse_json_filter(
                '{"$or": [{"data.status": {"$eq": "active"}}, {"name": {"$like": "special"}}]}'
            ),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_ids = {j.id for j in jobs}
    assert returned_ids == {active.id, special_completed.id}


@pytest.mark.asyncio
async def test_list_jobs_filter_not_and_status_with_non_status(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """$not wrapping a status+name subtree returns the negation.

    NOT (status == active AND name ~ "eval") keeps every job except the one that
    is both ACTIVE and name-matches "eval".
    """
    await _make_job(mock_dispatcher, mock_store, "eval-active", HelixJobStatus.ACTIVE)
    eval_completed = await _make_job(mock_dispatcher, mock_store, "eval-completed", HelixJobStatus.COMPLETED)
    other_active = await _make_job(mock_dispatcher, mock_store, "train-active", HelixJobStatus.ACTIVE)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=parse_json_filter(
                '{"$not": {"$and": [{"data.status": {"$eq": "active"}}, {"name": {"$like": "eval"}}]}}'
            ),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_ids = {j.id for j in jobs}
    assert returned_ids == {eval_completed.id, other_active.id}


@pytest.mark.asyncio
async def test_list_jobs_filter_or_with_status_in_each_branch(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """$or where each branch mixes status with a name term.

    (active AND name~foo) OR (completed AND name~bar) returns exactly the jobs
    matching either full branch.
    """
    foo_active = await _make_job(mock_dispatcher, mock_store, "foo-job", HelixJobStatus.ACTIVE)
    bar_completed = await _make_job(mock_dispatcher, mock_store, "bar-job", HelixJobStatus.COMPLETED)
    # foo but wrong status; bar but wrong status — both excluded.
    await _make_job(mock_dispatcher, mock_store, "foo-completed", HelixJobStatus.COMPLETED)
    await _make_job(mock_dispatcher, mock_store, "bar-active", HelixJobStatus.ACTIVE)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=parse_json_filter(
                '{"$or": ['
                '  {"$and": [{"data.status": {"$eq": "active"}}, {"name": {"$like": "foo"}}]},'
                '  {"$and": [{"data.status": {"$eq": "completed"}}, {"name": {"$like": "bar"}}]}'
                "]}"
            ),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_ids = {j.id for j in jobs}
    assert returned_ids == {foo_active.id, bar_completed.id}


@pytest.mark.asyncio
async def test_list_jobs_status_filter_fills_pages_and_counts_only_matches(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    active_ids = set()
    for i in range(5):
        active_ids.add((await _make_job(mock_dispatcher, mock_store, f"active-{i}", HelixJobStatus.ACTIVE)).id)
        await _make_job(mock_dispatcher, mock_store, f"completed-{i}", HelixJobStatus.COMPLETED)
    status_filter = ParsedFilter(
        operation=ComparisonOperation(field="data.status", operator=FilterOperator.EQ, value="active")
    )

    pages = [
        await mock_dispatcher.list_jobs(parsed=status_filter, workspace=DEFAULT_WORKSPACE, limit=2, offset=offset)
        for offset in (0, 2, 4)
    ]

    assert [len(jobs) for jobs, _ in pages] == [2, 2, 1]
    assert [total for _, total in pages] == [5, 5, 5]
    assert {job.id for jobs, _ in pages for job in jobs} == active_ids


@pytest.mark.asyncio
async def test_list_jobs_status_filter_with_no_matches_reports_zero_total(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    await _make_job(mock_dispatcher, mock_store, "job-active", HelixJobStatus.ACTIVE)

    jobs, total = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=ComparisonOperation(field="data.status", operator=FilterOperator.EQ, value="error")
        ),
        workspace=DEFAULT_WORKSPACE,
        limit=10,
        offset=0,
    )

    assert jobs == []
    assert total == 0


@pytest.mark.asyncio
async def test_list_jobs_updated_at_filter_counts_only_matches(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    for i in range(3):
        await _make_job(mock_dispatcher, mock_store, f"stale-{i}", HelixJobStatus.ACTIVE)
    changed = await _make_job(mock_dispatcher, mock_store, "changed", HelixJobStatus.ACTIVE)
    await asyncio.sleep(0.01)
    cutoff = datetime.now(timezone.utc)
    await asyncio.sleep(0.01)
    await _set_job_status(mock_dispatcher, changed.name, HelixJobStatus.COMPLETED)

    jobs, total = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=ComparisonOperation(field="updated_at", operator=FilterOperator.GTE, value=cutoff.isoformat())
        ),
        workspace=DEFAULT_WORKSPACE,
        limit=1,
        offset=0,
    )

    assert [job.id for job in jobs] == [changed.id]
    assert total == 1


@pytest.mark.asyncio
async def test_list_jobs_sort_by_updated_at_spans_pages(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    oldest = await _make_job(mock_dispatcher, mock_store, "oldest", HelixJobStatus.ACTIVE)
    for i in range(3):
        await _make_job(mock_dispatcher, mock_store, f"newer-{i}", HelixJobStatus.ACTIVE)
    await asyncio.sleep(0.01)
    await _set_job_status(mock_dispatcher, oldest.name, HelixJobStatus.COMPLETED)

    first_page, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(operation=None),
        workspace=DEFAULT_WORKSPACE,
        limit=1,
        offset=0,
        sort=HelixJobListSortField.UPDATED_AT_DESC,
    )
    last_page, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(operation=None),
        workspace=DEFAULT_WORKSPACE,
        limit=1,
        offset=3,
        sort=HelixJobListSortField.UPDATED_AT_ASC,
    )

    assert [job.id for job in first_page] == [oldest.id]
    assert [job.id for job in last_page] == [oldest.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_entity_type", [HelixJob, HelixJobAttempt])
async def test_retried_step_update_repairs_job_status_after_a_failed_save(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    failing_entity_type: type,
):
    job = await _make_job(mock_dispatcher, mock_store, "flaky-store", HelixJobStatus.ACTIVE)
    original_update = EntityClient.update

    async def fail_once(self, entity, **kwargs):
        if isinstance(entity, failing_entity_type) and not fail_once.failed:
            fail_once.failed = True
            raise RuntimeError("store unavailable")
        return await original_update(self, entity, **kwargs)

    fail_once.failed = False

    async def complete_step():
        step = await mock_dispatcher.get_current_job_step_by_name(
            job.name, TestConstants.PLATFORM_SPEC.steps[0].name, DEFAULT_WORKSPACE
        )
        assert step is not None
        await mock_dispatcher.update_job_status_from_step(step=step, status=HelixJobStatus.COMPLETED, error_details={})

    with patch.object(EntityClient, "update", fail_once):
        with pytest.raises(RuntimeError, match="store unavailable"):
            await complete_step()
        await complete_step()

    jobs, total = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=ComparisonOperation(field="data.status", operator=FilterOperator.EQ, value="completed")
        ),
        workspace=DEFAULT_WORKSPACE,
    )
    assert [(listed.id, listed.status) for listed in jobs] == [(job.id, HelixJobStatus.COMPLETED)]
    assert total == 1


@pytest.mark.asyncio
async def test_rerun_job_resets_job_status_for_list_filters(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    job = await _make_job(mock_dispatcher, mock_store, "rerun-me", HelixJobStatus.ERROR)

    await mock_dispatcher.rerun_job(job.name, DEFAULT_WORKSPACE)

    jobs, total = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=ComparisonOperation(field="data.status", operator=FilterOperator.EQ, value="error")
        ),
        workspace=DEFAULT_WORKSPACE,
    )
    assert jobs == []
    assert total == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("job_count", [1, 6])
async def test_list_jobs_store_calls_do_not_grow_with_page_size(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    job_count: int,
):
    for i in range(job_count):
        await _make_job(mock_dispatcher, mock_store, f"job-{i}", HelixJobStatus.ACTIVE)

    with (
        patch.object(mock_store, "list", side_effect=mock_store.list) as list_spy,
        patch.object(mock_store, "get_by_id", side_effect=mock_store.get_by_id) as get_by_id_spy,
    ):
        jobs, _ = await mock_dispatcher.list_jobs(
            parsed=ParsedFilter(operation=None), workspace=DEFAULT_WORKSPACE, limit=job_count, offset=0
        )

    assert len(jobs) == job_count
    assert [call.args[0] for call in list_spy.call_args_list] == [HelixJob, HelixJobAttempt]
    get_by_id_spy.assert_not_called()


@pytest.mark.asyncio
async def test_list_jobs_filter_name_like(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """filter name with $like applies substring matching."""
    match1 = await _make_job(mock_dispatcher, mock_store, "eval-training-run", HelixJobStatus.ACTIVE)
    match2 = await _make_job(mock_dispatcher, mock_store, "training-eval-v2", HelixJobStatus.COMPLETED)
    await _make_job(mock_dispatcher, mock_store, "finetune-run", HelixJobStatus.ACTIVE)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=ComparisonOperation(field="name", operator=FilterOperator.LIKE, value="eval"),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_ids = {j.id for j in jobs}
    assert match1.id in returned_ids
    assert match2.id in returned_ids
    assert len(returned_ids) == 2


@pytest.mark.asyncio
async def test_list_jobs_filter_name_multiple_like_terms_any_match(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """Multiple $like name terms in an OR return a job matching any term."""
    eval_job = await _make_job(mock_dispatcher, mock_store, "eval-run", HelixJobStatus.ACTIVE)
    train_job = await _make_job(mock_dispatcher, mock_store, "training-run", HelixJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "finetune-run", HelixJobStatus.ACTIVE)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=LogicalOperation(
                operator=FilterOperator.OR,
                operations=[
                    ComparisonOperation(field="name", operator=FilterOperator.LIKE, value="eval"),
                    ComparisonOperation(field="name", operator=FilterOperator.LIKE, value="training"),
                ],
            ),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_ids = {j.id for j in jobs}
    assert eval_job.id in returned_ids
    assert train_job.id in returned_ids
    assert len(returned_ids) == 2


# =============================================================================
# list_jobs: full JSON filter operator integration
# These tests verify operator expressions are pushed to the entity store and
# produce correct results against real data.
# =============================================================================


@pytest.mark.asyncio
async def test_list_jobs_filter_name_not_eq_excludes_exact_match(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """$not/$eq on name excludes the job with that exact name.

    This is the primary use case from the UI:
    filter={"name":{"$not":{"$eq":"evaluator-metrics-ybaefjl7"}}}
    """
    await _make_job(mock_dispatcher, mock_store, "target-job", HelixJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "other-job", HelixJobStatus.ACTIVE)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=parse_json_filter('{"name": {"$not": {"$eq": "target-job"}}}'),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_names = {j.name for j in jobs}
    assert "other-job" in returned_names
    assert "target-job" not in returned_names


@pytest.mark.asyncio
async def test_list_jobs_filter_name_eq_exact_match(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """$eq on name returns only the job with that exact name (no substring matching)."""
    target = await _make_job(mock_dispatcher, mock_store, "exact-match-job", HelixJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "exact-match-job-v2", HelixJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "unrelated-job", HelixJobStatus.ACTIVE)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=ComparisonOperation(field="name", operator=FilterOperator.EQ, value="exact-match-job"),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    assert len(jobs) == 1
    assert jobs[0].id == target.id


@pytest.mark.asyncio
async def test_list_jobs_filter_name_not_eq_with_status(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """$not/$eq on name combined with status filter returns correct intersection."""

    await _make_job(mock_dispatcher, mock_store, "excluded-job", HelixJobStatus.ACTIVE)
    included = await _make_job(mock_dispatcher, mock_store, "included-job", HelixJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "included-but-wrong-status", HelixJobStatus.COMPLETED)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=parse_json_filter(
                '{"$and": [{"name": {"$not": {"$eq": "excluded-job"}}}, {"data.status": {"$eq": "active"}}]}'
            ),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_names = {j.name for j in jobs}
    assert returned_names == {"included-job"}
    assert jobs[0].id == included.id


@pytest.mark.asyncio
async def test_list_jobs_filter_name_operator_and_project_like(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """$not/$eq on name combined with project $like applies both conditions."""

    # This job is excluded by name operator
    await _make_job(mock_dispatcher, mock_store, "excluded-job", HelixJobStatus.ACTIVE)
    # This job passes name filter and has matching project
    await _make_job(mock_dispatcher, mock_store, "included-job", HelixJobStatus.ACTIVE)
    # This job passes name filter but has no project, so project $like won't match
    no_project = CreateHelixJobRequest(
        name="no-project-job",
        source=TestConstants.SOURCE,
        project=None,
        spec=TestConstants.SPEC_BASIC,
        platform_spec=TestConstants.PLATFORM_SPEC,
    )
    no_proj_job = await mock_dispatcher.create_job(no_project, DEFAULT_WORKSPACE)
    await _set_job_status(mock_dispatcher, no_proj_job.name, HelixJobStatus.ACTIVE)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=parse_json_filter(
                '{"$and": [{"name": {"$not": {"$eq": "excluded-job"}}}, {"project": {"$like": "'
                + TestConstants.PROJECT
                + '"}}]}'
            ),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    returned_names = {j.name for j in jobs}
    assert "included-job" in returned_names
    assert "excluded-job" not in returned_names
    assert "no-project-job" not in returned_names
    assert len(returned_names) == 1


@pytest.mark.parametrize(
    "source",
    [
        "nemo-evals.retrieve-eval",
        "nemo-evals.agent-eval",
        "abcdefghijklmnopqrs-tail",
        "abcdefghijklmnopqrst-tail",
        "nemo--double-hyphen",
        "nemo-evals.retrieve--",
        "-leading-hyphen",
        "123-starts-with-digit",
        "Mixed Case/Path:Source",
        "a" * 80,
        "a",
        "---",
        "",
    ],
)
def test_generate_job_name_yields_valid_job_and_fileset_names(source: str):
    """The generated name and its derived ``job-fileset-`` name satisfy the entity name rule for any source."""
    name = generate_job_name(source, "bfbbh5ui")

    assert re.fullmatch(NAME_PATTERN, name)
    assert name.endswith("-bfbbh5ui")
    CreateFilesetRequest(name=f"job-fileset-{name}")


def test_generate_job_name_keeps_truncated_source_prefix():
    assert generate_job_name("nemo-evals.retrieve-eval", "bfbbh5ui") == "nemo-evals.retrieve-bfbbh5ui"


@pytest.mark.asyncio
async def test_create_job_without_name_derives_valid_fileset_from_source(
    mock_dispatcher: JobDispatcher,
    mock_files_client,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """NVBug 6912264: the 20-char prefix of ``nemo-evals.retrieve-eval`` ends in a hyphen."""
    request = sample_platform_job_request.model_copy(update={"name": None, "source": "nemo-evals.retrieve-eval"})

    job = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    assert job.name.startswith("nemo-evals.retrieve-")
    assert "--" not in job.name
    fileset_request = mock_files_client.create_fileset.await_args.kwargs["body"]
    assert fileset_request.name == f"job-fileset-{job.name}"


@pytest.mark.asyncio
async def test_create_job_invalid_generated_name_is_not_a_client_error(
    mock_dispatcher: JobDispatcher,
    mock_files_client,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A generator defect must surface as a server error, never as a client-facing ValueError (409/422)."""
    request = sample_platform_job_request.model_copy(update={"name": None})

    with patch("nhx.core.jobs.app.dispatcher.generate_job_name", return_value="bad--name"):
        with pytest.raises(RuntimeError) as exc_info:
            await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    assert not isinstance(exc_info.value, ValueError)
    mock_files_client.create_fileset.assert_not_called()


def _two_step_request(request: CreateHelixJobRequest, name: str) -> CreateHelixJobRequest:
    spec = request.platform_spec.model_copy(deep=True)
    spec.steps[0].lifecycle = StepLifecycle(pause_deadline_seconds=60)
    spec.steps.append(HelixJobStepSpec(name="upload", executor=spec.steps[0].executor, config={}))
    return request.model_copy(update={"name": name, "platform_spec": spec})


@pytest.mark.asyncio
async def test_pausing_step_that_completes_lets_the_job_finish(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Training finished before it saw the pause: the job runs again, then completes."""
    job = await mock_dispatcher.create_job(
        _two_step_request(sample_platform_job_request, "pause-lost-race"), DEFAULT_WORKSPACE
    )
    training = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert training is not None
    training, _ = await mock_dispatcher.update_job_status_from_step(training, HelixJobStatus.ACTIVE)
    await mock_dispatcher.pause_job(job.name, DEFAULT_WORKSPACE)
    training = await mock_store.get_by_id(HelixJobStep, training.id)
    assert training.status == HelixJobStatus.PAUSING

    _, attempt = await mock_dispatcher.update_job_status_from_step(training, HelixJobStatus.COMPLETED)

    assert attempt.status == HelixJobStatus.ACTIVE
    upload = await mock_dispatcher.get_current_job_step_by_name(job.name, "upload", DEFAULT_WORKSPACE)
    assert upload is not None
    with pytest.raises(JobOperationConflictError, match="does not support pause"):
        await mock_dispatcher.pause_job(job.name, DEFAULT_WORKSPACE)
    resumed = await mock_dispatcher.resume_job(job.name, DEFAULT_WORKSPACE)
    assert resumed is not None and resumed.status == HelixJobStatus.ACTIVE

    upload, _ = await mock_dispatcher.update_job_status_from_step(upload, HelixJobStatus.ACTIVE)
    _, attempt = await mock_dispatcher.update_job_status_from_step(upload, HelixJobStatus.COMPLETED)
    assert attempt.status == HelixJobStatus.COMPLETED


@pytest.mark.asyncio
async def test_replayed_completion_of_an_earlier_step_keeps_a_later_pause(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A repeated completed report for a finished step must not set a pausing job back to active."""
    request = _two_step_request(sample_platform_job_request, "pause-later-step")
    request.platform_spec.steps[1].lifecycle = StepLifecycle(pause_deadline_seconds=60)
    job = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)
    training = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert training is not None
    training, _ = await mock_dispatcher.update_job_status_from_step(training, HelixJobStatus.ACTIVE)
    training, _ = await mock_dispatcher.update_job_status_from_step(training, HelixJobStatus.COMPLETED)
    upload = await mock_dispatcher.get_current_job_step_by_name(job.name, "upload", DEFAULT_WORKSPACE)
    assert upload is not None
    upload, _ = await mock_dispatcher.update_job_status_from_step(upload, HelixJobStatus.ACTIVE)
    paused = await mock_dispatcher.pause_job(job.name, DEFAULT_WORKSPACE)
    assert paused is not None and paused.status == HelixJobStatus.PAUSING

    _, attempt = await mock_dispatcher.update_job_status_from_step(training, HelixJobStatus.COMPLETED)

    assert attempt.status == HelixJobStatus.PAUSING
    assert (await mock_store.get_by_id(HelixJobStep, upload.id)).status == HelixJobStatus.PAUSING


@pytest.mark.asyncio
async def test_repeated_pause_keeps_the_first_request_time(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A second pause request must not push the pause deadline back."""
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert step is not None
    step, _ = await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.ACTIVE)
    await mock_dispatcher.pause_job(job.name, DEFAULT_WORKSPACE)
    first = (await mock_store.get_by_id(HelixJobStep, step.id)).status_details["pause_requested_at"]

    # A second request that read the step while it was still active.
    await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.PAUSING)

    again = await mock_store.get_by_id(HelixJobStep, step.id)
    assert again.status == HelixJobStatus.PAUSING
    assert again.status_details["pause_requested_at"] == first


@pytest.mark.asyncio
async def test_attempt_update_retries_after_a_concurrent_write(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A progress write to the attempt between read and save must not leave the attempt behind its step."""
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert step is not None
    original_update = mock_store.update
    conflicts: list[str] = []

    async def update_with_one_attempt_conflict(entity):
        if isinstance(entity, HelixJobAttempt) and not conflicts:
            conflicts.append(entity.id)
            # Another writer saves progress first, so this save loses the version race.
            fresh = await mock_store.get_by_id(HelixJobAttempt, entity.id)
            fresh.status_details = {**fresh.status_details, "percentage_done": 40}
            await original_update(fresh)
            raise EntityConflictError("version conflict")
        return await original_update(entity)

    with patch.object(mock_store, "update", side_effect=update_with_one_attempt_conflict):
        _, attempt = await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.ACTIVE)

    assert conflicts
    assert attempt.status == HelixJobStatus.ACTIVE
    stored = await mock_store.get_by_id(HelixJobAttempt, job.attempt_id)
    assert stored.status == HelixJobStatus.ACTIVE
    assert stored.status_details["percentage_done"] == 40


@pytest.mark.asyncio
async def test_rerun_saves_the_callers_auth_context(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A regular user reads the job with auth_context stripped. Rerun must not save it back as empty."""
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert step is not None
    await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.ERROR, error_details={"message": "boom"})
    caller = AuthContext(principal_id="user@example.com", principal_email="user@example.com")
    original_update = mock_store.update
    saved_auth: list[AuthContext | None] = []

    async def record_job_auth(entity):
        if isinstance(entity, HelixJob):
            saved_auth.append(entity.auth_context)
        return await original_update(entity)

    with patch.object(mock_store, "update", side_effect=record_job_auth):
        rerun = await mock_dispatcher.rerun_job(job.name, DEFAULT_WORKSPACE, auth_context=caller)

    assert rerun is not None
    assert saved_auth == [caller]


@pytest.mark.asyncio
async def test_rerun_without_a_caller_keeps_the_jobs_auth_context(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """The job is read with auth_context stripped. Without a caller's, rerun saves back the job's own."""
    owner = AuthContext(principal_id="owner@example.com", principal_email="owner@example.com")
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE, auth_context=owner)
    step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert step is not None
    await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.ERROR, error_details={"message": "boom"})
    original_get = mock_store.get
    original_update = mock_store.update
    saved_auth: list[AuthContext | None] = []

    async def caller_scoped_get(*args, **kwargs):
        entity = await original_get(*args, **kwargs)
        if isinstance(entity, HelixJob):
            entity.with_auth_context(None)
        return entity

    async def record_job_auth(entity):
        if isinstance(entity, HelixJob):
            saved_auth.append(entity.auth_context)
        return await original_update(entity)

    with (
        patch.object(mock_store, "get", side_effect=caller_scoped_get),
        patch.object(mock_store, "update", side_effect=record_job_auth),
    ):
        rerun = await mock_dispatcher.rerun_job(job.name, DEFAULT_WORKSPACE)

    assert rerun is not None
    assert saved_auth == [owner]


def _download_and_train_request(request: CreateHelixJobRequest, name: str) -> CreateHelixJobRequest:
    spec = request.platform_spec.model_copy(deep=True)
    executor = spec.steps[0].executor
    storage = [HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/storage")]
    spec.steps = [
        HelixJobStepSpec(name="download", executor=executor, config={}, environment=storage),
        HelixJobStepSpec(name="train", executor=executor, config={}, environment=storage),
    ]
    return request.model_copy(update={"name": name, "platform_spec": spec})


@pytest.mark.asyncio
async def test_rerun_compares_the_image_on_the_step_that_saved_it(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """The failed training step's digest seeds the next training step, not the download step."""
    job = await mock_dispatcher.create_job(
        _download_and_train_request(sample_platform_job_request, "digest-rerun"), DEFAULT_WORKSPACE
    )
    download = await mock_dispatcher.get_current_job_step_by_name(job.name, "download", DEFAULT_WORKSPACE)
    assert download is not None
    download, _ = await mock_dispatcher.update_job_status_from_step(download, HelixJobStatus.ACTIVE)
    await mock_dispatcher.update_job_status_from_step(download, HelixJobStatus.COMPLETED)
    train = await mock_dispatcher.get_current_job_step_by_name(job.name, "train", DEFAULT_WORKSPACE)
    assert train is not None
    train, _ = await mock_dispatcher.update_job_status_from_step(train, HelixJobStatus.ACTIVE)
    _, attempt = await mock_dispatcher.update_job_status_from_step(
        train, HelixJobStatus.ERROR, status_details={IMAGE_DIGEST: "sha256:train"}
    )
    assert attempt.status_details[IMAGE_DIGEST_AT_SAVE] == "sha256:train"
    assert attempt.status_details[IMAGE_DIGEST_STEP] == "train"

    rerun = await mock_dispatcher.rerun_job(job.name, DEFAULT_WORKSPACE)
    assert rerun is not None
    download = await mock_dispatcher.get_current_job_step_by_name(job.name, "download", DEFAULT_WORKSPACE)
    assert download is not None
    assert IMAGE_DIGEST_AT_SAVE not in (download.status_details or {})
    download, _ = await mock_dispatcher.update_job_status_from_step(download, HelixJobStatus.ACTIVE)
    await mock_dispatcher.update_job_status_from_step(download, HelixJobStatus.COMPLETED)
    train = await mock_dispatcher.get_current_job_step_by_name(job.name, "train", DEFAULT_WORKSPACE)
    assert train is not None
    assert train.status_details[IMAGE_DIGEST_AT_SAVE] == "sha256:train"


@pytest.mark.asyncio
async def test_delete_queues_storage_reclaim_and_holds_the_name(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Storage is keyed by name, so the name is free again only after the controller reclaimed it."""
    request = _download_and_train_request(sample_platform_job_request, "delete-reclaim")
    job = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)
    download = await mock_dispatcher.get_current_job_step_by_name(job.name, "download", DEFAULT_WORKSPACE)
    assert download is not None
    await mock_dispatcher.update_job_status_from_step(download, HelixJobStatus.ERROR)

    assert await mock_dispatcher.delete_job(job.name, DEFAULT_WORKSPACE)

    reclaims = await mock_store.list(HelixJobStorageReclaim, filter_obj={}, workspace=SYSTEM_WORKSPACE)
    assert [(r.job_id, r.job_workspace, r.job_name) for r in reclaims.data] == [(job.id, DEFAULT_WORKSPACE, job.name)]
    assert [t.model_dump() for t in reclaims.data[0].targets] == [
        {"provider": request.platform_spec.steps[0].executor.provider, "profile": "default"}
    ]
    with pytest.raises(JobStorageReclaimPendingError):
        await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    await mock_store.delete_by_id(HelixJobStorageReclaim, reclaims.data[0].id)
    recreated = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)
    assert recreated.name == job.name


@pytest.mark.asyncio
async def test_delete_without_storage_queues_nothing(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert step is not None
    step, _ = await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.ACTIVE)
    await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.COMPLETED)

    assert await mock_dispatcher.delete_job(job.name, DEFAULT_WORKSPACE)

    reclaims = await mock_store.list(HelixJobStorageReclaim, filter_obj={}, workspace=SYSTEM_WORKSPACE)
    assert reclaims.data == []


@pytest.mark.asyncio
async def test_delete_queues_storage_the_sweep_already_marked(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """The sweep marks a job before it queues the storage, and can stop in between."""
    job = await mock_dispatcher.create_job(
        _download_and_train_request(sample_platform_job_request, "marked-delete"), DEFAULT_WORKSPACE
    )
    download = await mock_dispatcher.get_current_job_step_by_name(job.name, "download", DEFAULT_WORKSPACE)
    assert download is not None
    await mock_dispatcher.update_job_status_from_step(download, HelixJobStatus.ERROR)
    await mock_dispatcher.update_attempt_status_details(
        job.name,
        job.attempt_id,
        DEFAULT_WORKSPACE,
        {"storage_reclaimed_at": "2026-10-08T00:00:00+00:00"},
    )

    assert await mock_dispatcher.delete_job(job.name, DEFAULT_WORKSPACE)

    reclaims = await mock_store.list(HelixJobStorageReclaim, filter_obj={}, workspace=SYSTEM_WORKSPACE)
    assert [r.job_name for r in reclaims.data] == [job.name]


def _failing_for(entity_type, method):
    """Fail store calls on one entity type only, as a store outage would."""

    async def call(entity_or_type, *args, **kwargs):
        if entity_or_type is entity_type or isinstance(entity_or_type, entity_type):
            raise RuntimeError("store unavailable")
        return await method(entity_or_type, *args, **kwargs)

    return call


@pytest.mark.asyncio
async def test_delete_fails_when_storage_reclaim_cannot_be_queued(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """Without the record, a new job with the same name would mount this job's storage."""
    request = _download_and_train_request(sample_platform_job_request, "unqueued-delete")
    job = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)
    download = await mock_dispatcher.get_current_job_step_by_name(job.name, "download", DEFAULT_WORKSPACE)
    assert download is not None
    await mock_dispatcher.update_job_status_from_step(download, HelixJobStatus.ERROR)

    with (
        patch.object(mock_store, "create", side_effect=_failing_for(HelixJobStorageReclaim, mock_store.create)),
        pytest.raises(RuntimeError, match="store unavailable"),
    ):
        await mock_dispatcher.delete_job(job.name, DEFAULT_WORKSPACE)

    assert await mock_store.get(HelixJob, job.name, workspace=DEFAULT_WORKSPACE) is not None


@pytest.mark.asyncio
async def test_create_refuses_a_name_when_pending_reclaims_cannot_be_checked(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """A record may still exist, and its reclaim would delete the new job's storage."""
    with (
        patch.object(mock_store, "list", side_effect=_failing_for(HelixJobStorageReclaim, mock_store.list)),
        pytest.raises(JobStorageReclaimPendingError, match="Could not confirm"),
    ):
        await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)


@pytest.mark.asyncio
async def test_resume_and_rerun_reject_reclaimed_storage(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    await mock_dispatcher.update_job_status_from_step(current_step, HelixJobStatus.PAUSED)
    await mock_dispatcher.update_attempt_status_details(
        job.name,
        job.attempt_id,
        DEFAULT_WORKSPACE,
        {"storage_reclaimed_at": "2026-10-08T00:00:00+00:00"},
    )

    with pytest.raises(JobOperationConflictError, match="storage has been reclaimed"):
        await mock_dispatcher.resume_job(job.name, DEFAULT_WORKSPACE)
    with pytest.raises(JobOperationConflictError, match="storage has been reclaimed"):
        await mock_dispatcher.update_job_control(job.name, DEFAULT_WORKSPACE, JobControl(pause_ttl_seconds=10))

    failed = await mock_dispatcher.create_job(
        sample_platform_job_request.model_copy(update={"name": "reclaimed-error"}), DEFAULT_WORKSPACE
    )
    failed_step = await mock_dispatcher.get_current_job_step_by_name(failed.name, "basic", DEFAULT_WORKSPACE)
    assert failed_step is not None
    await mock_dispatcher.update_job_status_from_step(failed_step, HelixJobStatus.ERROR)
    await mock_dispatcher.update_attempt_status_details(
        failed.name,
        failed.attempt_id,
        DEFAULT_WORKSPACE,
        {"storage_reclaimed_at": "2026-10-08T00:00:00+00:00"},
    )
    with pytest.raises(JobOperationConflictError, match="storage has been reclaimed"):
        await mock_dispatcher.rerun_job(failed.name, DEFAULT_WORKSPACE)


@pytest.mark.asyncio
async def test_list_attempts_paginates_and_marks_current(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    for seq in (1, 2):
        await mock_store.add(
            HelixJobAttempt(
                name=f"attempt-{seq}",
                workspace=DEFAULT_WORKSPACE,
                job=job.id,
                seq=seq,
                spec={"seq": seq},
                platform_spec=TestConstants.PLATFORM_SPEC,
            )
        )

    result = await mock_dispatcher.list_attempts(
        job.name,
        DEFAULT_WORKSPACE,
        ParsedFilter(operation=None),
        page=1,
        page_size=2,
        sort=HelixJobAttemptSortField.SEQ_DESC,
    )

    assert result is not None
    attempts, total = result
    assert total == 3
    assert [attempt.seq for attempt in attempts] == [2, 1]
    assert not any(attempt.is_current for attempt in attempts)

    ascending = await mock_dispatcher.list_attempts(
        job.id,
        DEFAULT_WORKSPACE,
        ParsedFilter(operation=None),
        page=1,
        page_size=3,
        sort=HelixJobAttemptSortField.SEQ_ASC,
    )
    assert ascending is not None
    assert [attempt.seq for attempt in ascending[0]] == [0, 1, 2]
    assert ascending[0][0].is_current is True


@pytest.mark.asyncio
async def test_update_job_preserves_attempt_snapshot(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    created = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    original_attempt = await mock_store.get_by_id(HelixJobAttempt, created.attempt_id)
    original_spec = dict(original_attempt.spec)

    updated = await mock_dispatcher.update_job(
        created.name,
        DEFAULT_WORKSPACE,
        UpdateHelixJobRequest(
            description="updated description",
            spec={"revision": 2},
            platform_spec=TestConstants.PLATFORM_SPEC,
        ),
    )

    assert updated is not None
    assert updated.description == "updated description"
    assert updated.spec == {"revision": 2}
    stored_attempt = await mock_store.get_by_id(HelixJobAttempt, created.attempt_id)
    assert stored_attempt.spec == original_spec


@pytest.mark.asyncio
async def test_update_job_returns_none_when_job_is_deleted_after_resolution(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    created = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    resolved = await mock_store.get(HelixJob, created.name, workspace=DEFAULT_WORKSPACE)

    with (
        patch.object(mock_dispatcher, "_resolve_job", new=AsyncMock(return_value=resolved)),
        patch.object(mock_store, "get", new=AsyncMock(side_effect=EntityNotFoundError("deleted"))),
    ):
        updated = await mock_dispatcher.update_job(
            created.name,
            DEFAULT_WORKSPACE,
            UpdateHelixJobRequest(description="updated description"),
        )

    assert updated is None


@pytest.mark.asyncio
async def test_update_job_restores_callers_auth_context_before_persisting(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    created = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    auth_context = AuthContext(principal_id="updater@example.com")
    persisted_auth_contexts: list[AuthContext | None] = []
    original_update = mock_store.update

    async def capture_update(entity, *args, **kwargs):
        if isinstance(entity, HelixJob):
            persisted_auth_contexts.append(entity.auth_context)
        return await original_update(entity, *args, **kwargs)

    with patch.object(mock_store, "update", side_effect=capture_update):
        updated = await mock_dispatcher.update_job(
            created.name,
            DEFAULT_WORKSPACE,
            UpdateHelixJobRequest(description="updated description"),
            auth_context=auth_context,
        )

    assert updated is not None
    assert persisted_auth_contexts == [auth_context]


@pytest.mark.asyncio
async def test_results_remain_job_owned_and_filter_by_producing_attempt(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    second = await mock_store.add(
        HelixJobAttempt(
            name="attempt-2",
            workspace=DEFAULT_WORKSPACE,
            job=job.id,
            seq=1,
            spec=job.spec,
            platform_spec=job.platform_spec,
        )
    )
    first_result = await mock_dispatcher.create_result(
        job.id,
        job.name,
        "metrics",
        "default/results#first",
        FileStorageType.FILESET,
        DEFAULT_WORKSPACE,
        attempt_id=job.attempt_id,
    )
    with pytest.raises(EntityConflictError):
        await mock_dispatcher.create_result(
            job.id,
            job.name,
            "metrics",
            "default/results#duplicate",
            FileStorageType.FILESET,
            DEFAULT_WORKSPACE,
            attempt_id=second.id,
        )
    second_result = await mock_dispatcher.create_result(
        job.id,
        job.name,
        "summary",
        "default/results#second",
        FileStorageType.FILESET,
        DEFAULT_WORKSPACE,
        attempt_id=second.id,
    )
    legacy = await mock_store.add(
        HelixJobResult(
            name="legacy",
            workspace=DEFAULT_WORKSPACE,
            job=job.id,
            artifact_url="default/results#legacy",
            artifact_storage_type=FileStorageType.FILESET,
        )
    )

    assert first_result.parent == job.id
    assert second_result.parent == job.id
    resolved_first = await mock_dispatcher.get_result(job.name, "metrics", DEFAULT_WORKSPACE)
    resolved_second = await mock_dispatcher.get_result(job.name, "summary", DEFAULT_WORKSPACE)
    resolved_legacy = await mock_dispatcher.get_result(job.name, "legacy", DEFAULT_WORKSPACE)
    all_results, total = await mock_dispatcher.list_results(job.id, DEFAULT_WORKSPACE)
    first_results, first_total = await mock_dispatcher.list_results(
        job.id,
        DEFAULT_WORKSPACE,
        attempt_id=job.attempt_id,
    )
    second_results, second_total = await mock_dispatcher.list_results(
        job.id,
        DEFAULT_WORKSPACE,
        attempt_id=second.id,
    )

    assert resolved_first is not None
    assert resolved_second is not None
    assert resolved_legacy is not None
    assert resolved_first.id == first_result.id
    assert resolved_second.id == second_result.id
    assert resolved_legacy.id == legacy.id
    assert {result.name for result in all_results} == {"metrics", "summary", "legacy"}
    assert total == 3
    assert [result.name for result in first_results] == ["metrics"]
    assert first_total == 1
    assert [result.name for result in second_results] == ["summary"]
    assert second_total == 1


@pytest.mark.asyncio
async def test_attempt_step_lookup_rejects_cross_job_attempts(
    mock_dispatcher: JobDispatcher,
    sample_platform_job_request: CreateHelixJobRequest,
):
    first = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    second = await mock_dispatcher.create_job(
        sample_platform_job_request.model_copy(update={"name": "other-job"}),
        DEFAULT_WORKSPACE,
    )

    assert await mock_dispatcher.resolve_attempt(first.name, second.attempt_id, DEFAULT_WORKSPACE) is None
    assert (
        await mock_dispatcher.get_attempt_step_by_name(
            first.name,
            second.attempt_id,
            "basic",
            DEFAULT_WORKSPACE,
        )
        is None
    )


@pytest.mark.asyncio
async def test_current_attempt_uses_pointer_not_highest_sequence(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    original = await mock_store.get_by_id(HelixJobAttempt, job.attempt_id)
    newer = await mock_store.add(
        HelixJobAttempt(
            name="detached-newer",
            workspace=DEFAULT_WORKSPACE,
            job=job.id,
            seq=99,
            spec={"detached": True},
            platform_spec=TestConstants.PLATFORM_SPEC,
        )
    )

    current = await mock_dispatcher.get_current_attempt(job.name, DEFAULT_WORKSPACE)
    resolved = await mock_dispatcher.resolve_attempt(job.name, "current", DEFAULT_WORKSPACE)

    assert current is not None
    assert current.id == original.id
    assert resolved is not None
    assert resolved[1].id == original.id
    assert resolved[1].id != newer.id


@pytest.mark.asyncio
async def test_resolve_attempt_rejects_internal_name(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    attempt = await mock_store.get_by_id(HelixJobAttempt, job.attempt_id)

    assert await mock_dispatcher.resolve_attempt(job.name, attempt.name, DEFAULT_WORKSPACE) is None


STORAGE_MARK = {"storage_reclaimed_at": "2026-10-08T00:00:00+00:00"}


async def _fail_current_step(dispatcher: JobDispatcher, job_name: str) -> HelixJobStep:
    step = await dispatcher.get_current_job_step_by_name(job_name, "basic", DEFAULT_WORKSPACE)
    assert step is not None
    step, _ = await dispatcher.update_job_status_from_step(step, HelixJobStatus.ACTIVE)
    step, _ = await dispatcher.update_job_status_from_step(step, HelixJobStatus.ERROR)
    return step


@pytest.mark.asyncio
async def test_storage_mark_applies_only_while_the_step_is_current_and_stopped(
    mock_dispatcher: JobDispatcher,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """The storage sweep marks a job this way, so a resume or rerun that took the job first refuses the mark."""
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    failed = await _fail_current_step(mock_dispatcher, job.name)

    with pytest.raises(JobOperationConflictError, match="no longer"):
        await mock_dispatcher.update_attempt_status_details(
            job.name,
            failed.attempt_id,
            DEFAULT_WORKSPACE,
            STORAGE_MARK,
            expected_step_id=failed.id,
            expected_step_status=HelixJobStatus.PAUSED,
        )

    # A rerun replaced the attempt after the sweep listed the failed step. The mark must not land on the rerun.
    await mock_dispatcher.rerun_job(job.name, DEFAULT_WORKSPACE)
    with pytest.raises(JobOperationConflictError, match="no longer"):
        await mock_dispatcher.update_attempt_status_details(
            job.name,
            failed.attempt_id,
            DEFAULT_WORKSPACE,
            STORAGE_MARK,
            expected_step_id=failed.id,
            expected_step_status=HelixJobStatus.ERROR,
        )
    rerun = await mock_dispatcher.get_current_attempt(job.name, DEFAULT_WORKSPACE)
    assert rerun is not None
    assert "storage_reclaimed_at" not in (rerun.status_details or {})

    failed_again = await _fail_current_step(mock_dispatcher, job.name)
    marked = await mock_dispatcher.update_attempt_status_details(
        job.name,
        failed_again.attempt_id,
        DEFAULT_WORKSPACE,
        STORAGE_MARK,
        expected_step_id=failed_again.id,
        expected_step_status=HelixJobStatus.ERROR,
    )
    assert marked is not None
    assert marked.status_details["storage_reclaimed_at"] == STORAGE_MARK["storage_reclaimed_at"]


@pytest.mark.asyncio
async def test_resume_refuses_storage_marked_after_its_check(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """The sweep marks under the job's lock, and resume checks again under it before moving the step."""
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert step is not None
    paused, _ = await mock_dispatcher.update_job_status_from_step(step, HelixJobStatus.PAUSED)
    await mock_dispatcher.update_attempt_status_details(job.name, "current", DEFAULT_WORKSPACE, STORAGE_MARK)

    with pytest.raises(JobOperationConflictError, match="storage has been reclaimed"):
        await mock_dispatcher.update_job_status_from_step(paused, HelixJobStatus.RESUMING)

    assert (await mock_store.get_by_id(HelixJobStep, paused.id)).status == HelixJobStatus.PAUSED


@pytest.mark.asyncio
async def test_a_delete_that_fails_after_queueing_leaves_a_job_rerun_refuses(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
):
    """The controller reclaims queued storage even when the rest of the delete fails."""
    job = await mock_dispatcher.create_job(
        _download_and_train_request(sample_platform_job_request, "partial-delete"), DEFAULT_WORKSPACE
    )
    download = await mock_dispatcher.get_current_job_step_by_name(job.name, "download", DEFAULT_WORKSPACE)
    assert download is not None
    await mock_dispatcher.update_job_status_from_step(download, HelixJobStatus.ERROR)

    with (
        patch.object(mock_store, "list", side_effect=_failing_for(HelixJobResult, mock_store.list)),
        pytest.raises(RuntimeError, match="store unavailable"),
    ):
        await mock_dispatcher.delete_job(job.name, DEFAULT_WORKSPACE)

    with pytest.raises(JobOperationConflictError, match="storage has been reclaimed"):
        await mock_dispatcher.rerun_job(job.name, DEFAULT_WORKSPACE)
