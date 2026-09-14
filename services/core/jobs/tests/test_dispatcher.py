# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JobDispatcher operations using EntityStore.

These tests verify that the JobDispatcher correctly manages job lifecycle operations
(create, cancel, pause, resume, rerun, delete) using the EntityStore pattern.
"""

import asyncio
import gc
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nmp.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation, parse_json_filter
from nmp.common.api.parsed_filter import ParsedFilter
from nmp.common.entities import (
    ALL_WORKSPACES,
    DEFAULT_WORKSPACE,
    EntityClient,
    EntityConflictError,
    EntityNotFoundError,
)
from nmp.common.jobs.schemas import FileStorageType, PlatformJobStatus
from nmp.core.jobs.api.v2.jobs.schemas import (
    CreatePlatformJobRequest,
    PlatformJobResponse,
    PlatformJobTaskUpdate,
)
from nmp.core.jobs.app.dispatcher import JobDeletionConflictError, JobDispatcher, JobStatusUpdateSkippedError
from nmp.core.jobs.app.schemas import (
    PlatformJobStepSpec,
)
from nmp.core.jobs.app.test_helpers import TestConstants
from nmp.core.jobs.entities import (
    PlatformJob,
    PlatformJobAttempt,
    PlatformJobResult,
    PlatformJobStep,
    PlatformJobTask,
)


async def create_job_with_attempt(
    dispatcher: JobDispatcher, job_request: CreatePlatformJobRequest
) -> PlatformJobResponse:
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
    job = PlatformJob(
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
    attempt = PlatformJobAttempt(
        name="attempt-1",
        workspace=DEFAULT_WORKSPACE,
        job=saved_job.id,
        seq=1,
        status=PlatformJobStatus.COMPLETED,
        spec={},
        platform_spec=platform_spec,
    )
    saved_attempt = await store.add(attempt)

    # Update job with current attempt
    saved_job.current_attempt_id = saved_attempt.id
    await store.update(saved_job)

    # Create step (parent-scoped: unique per attempt)
    step = PlatformJobStep(
        name="step-1",
        workspace=DEFAULT_WORKSPACE,
        attempt_id=saved_attempt.id,
        status=PlatformJobStatus.COMPLETED,
    )
    saved_step = await store.add(step)

    # Create task (parent-scoped: unique per step)
    task = PlatformJobTask(
        name="task-1",
        workspace=DEFAULT_WORKSPACE,
        step_id=saved_step.id,
        status=PlatformJobStatus.COMPLETED,
    )
    saved_task = await store.add(task)

    # Create result (parent-scoped: unique per job)
    result = PlatformJobResult(
        name="result-1",
        workspace=DEFAULT_WORKSPACE,
        job=saved_job.id,
        artifact_url="default/test-fileset#artifact",
        artifact_storage_type=FileStorageType.FILESET,
    )
    saved_result = await store.add(result)

    return saved_job.id, saved_job.name, saved_attempt.id, saved_step.id, saved_task.id, saved_result.id


async def verify_job_data_exists(store: EntityClient, job_id: str, should_exist: bool = True) -> None:
    """Verify that job data exists or does not exist."""
    if should_exist:
        job = await store.get_by_id(PlatformJob, job_id)
        assert job is not None
    else:
        with pytest.raises(EntityNotFoundError):
            await store.get_by_id(PlatformJob, job_id)


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
    assert await count_entities(mock_store, PlatformJobAttempt, {"job": job_id}) == 1
    assert await count_entities(mock_store, PlatformJobStep, {"attempt_id": attempt_id}) == 1
    assert await count_entities(mock_store, PlatformJobTask, {"step_id": step_id}) == 1
    assert await count_entities(mock_store, PlatformJobResult, {"job": job_id}) == 1

    # Execute deletion - should return True when job exists
    deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)
    assert deleted is True

    # Verify all data is deleted
    await verify_job_data_exists(mock_store, job_id, should_exist=False)
    assert await count_entities(mock_store, PlatformJobAttempt, {"job": job_id}) == 0
    assert await count_entities(mock_store, PlatformJobStep, {"attempt_id": attempt_id}) == 0
    assert await count_entities(mock_store, PlatformJobTask, {"step_id": step_id}) == 0
    assert await count_entities(mock_store, PlatformJobResult, {"job": job_id}) == 0


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
    assert await count_entities(mock_store, PlatformJobAttempt, {"job": job2_id}) == 1


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
    _mock_files_client,
):
    """Deleting an active job is refused before metadata or fileset cleanup."""
    job_id, job_name, attempt_id, step_id, _, _ = await create_test_job_data(mock_store, "active-delete-test-job")

    attempt = await mock_store.get_by_id(PlatformJobAttempt, attempt_id)
    attempt.status = PlatformJobStatus.ACTIVE
    await mock_store.update(attempt)

    step = await mock_store.get_by_id(PlatformJobStep, step_id)
    step.status = PlatformJobStatus.ACTIVE
    await mock_store.update(step)

    with pytest.raises(JobDeletionConflictError, match="Cancel the job and wait"):
        await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)

    await verify_job_data_exists(mock_store, job_id, should_exist=True)
    assert await count_entities(mock_store, PlatformJobAttempt, {"job": job_id}) == 1
    assert await count_entities(mock_store, PlatformJobStep, {"attempt_id": attempt_id}) == 1
    assert await count_entities(mock_store, PlatformJobTask, {"step_id": step_id}) == 1
    assert await count_entities(mock_store, PlatformJobResult, {"job": job_id}) == 1
    _mock_files_client.delete_fileset.assert_not_called()


@pytest.mark.asyncio
async def test_delete_job_deletes_related_entities_across_all_pages(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """Deleting a terminal job visits every attempt, step, task, and result page."""
    from nmp.core.jobs.app import dispatcher as dispatcher_module

    monkeypatch.setattr(dispatcher_module, "_DELETE_PAGE_SIZE", 1)

    job_id, job_name, attempt_id, step_id, task_id, result_id = await create_test_job_data(
        mock_store, "paginated-delete-test-job"
    )
    job = await mock_store.get_by_id(PlatformJob, job_id)

    second_attempt = await mock_store.add(
        PlatformJobAttempt(
            name="attempt-2",
            workspace=DEFAULT_WORKSPACE,
            job=job.id,
            seq=2,
            status=PlatformJobStatus.COMPLETED,
            spec={},
            platform_spec=job.platform_spec,
        )
    )
    second_step = await mock_store.add(
        PlatformJobStep(
            name="step-2",
            workspace=DEFAULT_WORKSPACE,
            attempt_id=second_attempt.id,
            status=PlatformJobStatus.COMPLETED,
        )
    )
    second_task = await mock_store.add(
        PlatformJobTask(
            name="task-2",
            workspace=DEFAULT_WORKSPACE,
            step_id=second_step.id,
            status=PlatformJobStatus.COMPLETED,
        )
    )
    second_result = await mock_store.add(
        PlatformJobResult(
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
        (PlatformJob, job_id),
        (PlatformJobAttempt, attempt_id),
        (PlatformJobAttempt, second_attempt.id),
        (PlatformJobStep, step_id),
        (PlatformJobStep, second_step.id),
        (PlatformJobTask, task_id),
        (PlatformJobTask, second_task.id),
        (PlatformJobResult, result_id),
        (PlatformJobResult, second_result.id),
    ):
        with pytest.raises(EntityNotFoundError):
            await mock_store.get_by_id(entity_type, entity_id)


@pytest.mark.asyncio
async def test_delete_job_serializes_with_rerun_job(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    mock_nmp_client,
):
    """A rerun request cannot create a new attempt while deletion is cleaning up."""
    _, job_name, _, _, _, _ = await create_test_job_data(mock_store, "delete-rerun-lock-test-job")
    other_dispatcher = JobDispatcher(store=mock_store, sdk=mock_nmp_client)

    delete_started = asyncio.Event()
    allow_delete = asyncio.Event()
    original_delete_by_id = mock_store.delete_by_id

    async def blocking_delete_by_id(entity_type, entity_id):
        if entity_type is PlatformJobResult and not delete_started.is_set():
            delete_started.set()
            await allow_delete.wait()
        return await original_delete_by_id(entity_type, entity_id)

    with patch.object(mock_store, "delete_by_id", side_effect=blocking_delete_by_id):
        delete_task = asyncio.create_task(mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE))
        rerun_task = None
        try:
            await asyncio.wait_for(delete_started.wait(), timeout=1.0)

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
    mock_nmp_client,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """A same-name create waits until delete finishes all cleanup for the old job."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "delete-create-lock-test-job")
    other_dispatcher = JobDispatcher(store=mock_store, sdk=mock_nmp_client)
    create_request = sample_platform_job_request.model_copy(update={"name": job_name})

    delete_started = asyncio.Event()
    allow_delete = asyncio.Event()
    original_delete_by_id = mock_store.delete_by_id

    async def blocking_delete_by_id(entity_type, entity_id):
        if entity_type is PlatformJobResult and not delete_started.is_set():
            delete_started.set()
            await allow_delete.wait()
        return await original_delete_by_id(entity_type, entity_id)

    with patch.object(mock_store, "delete_by_id", side_effect=blocking_delete_by_id):
        delete_task = asyncio.create_task(mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE))
        create_task = None
        try:
            await asyncio.wait_for(delete_started.wait(), timeout=1.0)

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
    mock_nmp_client,
):
    """A task update cannot create a late child row after delete cleanup starts."""
    job_id, job_name, _, step_id, _, _ = await create_test_job_data(mock_store, "delete-task-lock-test-job")
    step = await mock_store.get_by_id(PlatformJobStep, step_id)
    other_dispatcher = JobDispatcher(store=mock_store, sdk=mock_nmp_client)

    delete_started = asyncio.Event()
    allow_delete = asyncio.Event()
    original_delete_by_id = mock_store.delete_by_id

    async def blocking_delete_by_id(entity_type, entity_id):
        if entity_type is PlatformJobResult and not delete_started.is_set():
            delete_started.set()
            await allow_delete.wait()
        return await original_delete_by_id(entity_type, entity_id)

    with patch.object(mock_store, "delete_by_id", side_effect=blocking_delete_by_id):
        delete_task = asyncio.create_task(mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE))
        task_create_task = None
        try:
            await asyncio.wait_for(delete_started.wait(), timeout=1.0)

            task_create_task = asyncio.create_task(
                other_dispatcher.create_or_update_task(
                    job_name,
                    "late-task",
                    DEFAULT_WORKSPACE,
                    PlatformJobTaskUpdate(status=PlatformJobStatus.ACTIVE),
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
    assert await count_entities(mock_store, PlatformJobTask, {"step_id": step_id}) == 0


@pytest.mark.asyncio
async def test_delete_job_serializes_with_result_creation(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    mock_nmp_client,
):
    """A result create cannot recreate associated data after delete cleanup starts."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "delete-result-lock-test-job")
    other_dispatcher = JobDispatcher(store=mock_store, sdk=mock_nmp_client)

    delete_started = asyncio.Event()
    allow_delete = asyncio.Event()
    original_delete_by_id = mock_store.delete_by_id

    async def blocking_delete_by_id(entity_type, entity_id):
        if entity_type is PlatformJobResult and not delete_started.is_set():
            delete_started.set()
            await allow_delete.wait()
        return await original_delete_by_id(entity_type, entity_id)

    with patch.object(mock_store, "delete_by_id", side_effect=blocking_delete_by_id):
        delete_task = asyncio.create_task(mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE))
        result_create_task = None
        try:
            await asyncio.wait_for(delete_started.wait(), timeout=1.0)

            result_create_task = asyncio.create_task(
                other_dispatcher.create_result(
                    job_id=job_id,
                    job_name=job_name,
                    result_name="late-result",
                    artifact_url="default/test-fileset#late",
                    artifact_storage_type=FileStorageType.FILESET,
                    workspace=DEFAULT_WORKSPACE,
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
    assert await count_entities(mock_store, PlatformJobResult, {"job": job_id}) == 0


@pytest.mark.asyncio
async def test_delete_job_does_not_leave_idle_mutation_lock(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """Idle job mutation locks are weakly held and do not grow forever."""
    from nmp.core.jobs.app import dispatcher as dispatcher_module

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
    step = await mock_store.get_by_id(PlatformJobStep, step_id)
    task = await mock_store.get_by_id(PlatformJobTask, task_id)

    updated = await mock_dispatcher.create_or_update_task(
        job_name,
        task.name,
        DEFAULT_WORKSPACE,
        PlatformJobTaskUpdate(
            status=PlatformJobStatus.ACTIVE,
            status_details={"message": "Job is running"},
        ),
        step,
    )

    assert updated.id == task.id
    assert updated.status == PlatformJobStatus.COMPLETED
    assert updated.status_details == {}

    stored_task = await mock_store.get_by_id(PlatformJobTask, task_id)
    assert stored_task.status == PlatformJobStatus.COMPLETED
    assert stored_task.status_details == {}


@pytest.mark.asyncio
async def test_delete_job_missing_fileset_succeeds(mock_dispatcher: JobDispatcher, mock_store: EntityClient):
    """Test that delete succeeds even if the job fileset was already deleted.

    The fileset may have been cleaned up by workspace cleanup before the explicit
    delete call arrives, so delete_job must tolerate a 404 on the fileset.
    """
    from unittest.mock import AsyncMock

    from nemo_platform_plugin.client.errors import NotFoundError

    # output_location defaults to None -> job owns its fileset -> delete_fileset is attempted.
    job_id, job_name, _, _, _, _ = await create_test_job_data(mock_store, "delete-missing-fileset-job")

    # Simulate the fileset already being gone by making the mock files client raise NotFoundError
    mock_files = AsyncMock()
    mock_files.delete_fileset = AsyncMock(side_effect=NotFoundError.__new__(NotFoundError))

    with patch("nmp.core.jobs.app.dispatcher.client_from_platform", return_value=mock_files):
        deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)
    assert deleted is True

    # The job entity itself should be gone
    await verify_job_data_exists(mock_store, job_id, should_exist=False)


@pytest.mark.asyncio
async def test_create_job_uses_existing_output_location(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    _mock_files_client,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """A supplied output_location becomes the fileset and is persisted as provenance; nothing created."""
    request = sample_platform_job_request.model_copy(update={"output_location": "my-eval-fileset"})

    job = await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    assert job.fileset == "my-eval-fileset"
    _mock_files_client.get_fileset.assert_awaited_once()
    _mock_files_client.create_fileset.assert_not_called()

    stored = await mock_store.get(PlatformJob, job.name, workspace=DEFAULT_WORKSPACE)
    assert stored.output_location == "my-eval-fileset"


@pytest.mark.asyncio
async def test_create_job_output_location_not_found_raises(
    mock_dispatcher: JobDispatcher,
    _mock_files_client,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """A supplied output_location that does not exist is rejected and never auto-created."""
    from nemo_platform_plugin.client.errors import NotFoundError
    from nmp.core.jobs.app.dispatcher import JobOutputLocationError

    _mock_files_client.get_fileset.side_effect = NotFoundError.__new__(NotFoundError)
    request = sample_platform_job_request.model_copy(update={"output_location": "ghost-fileset"})

    with pytest.raises(JobOutputLocationError):
        await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    _mock_files_client.create_fileset.assert_not_called()


@pytest.mark.asyncio
async def test_create_job_output_location_forbidden_raises(
    mock_dispatcher: JobDispatcher,
    _mock_files_client,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """A supplied output_location the caller cannot access is rejected, never auto-created."""
    from nemo_platform_plugin.client.errors import PermissionDeniedError
    from nmp.core.jobs.app.dispatcher import JobOutputLocationError

    _mock_files_client.get_fileset.side_effect = PermissionDeniedError.__new__(PermissionDeniedError)
    request = sample_platform_job_request.model_copy(update={"output_location": "forbidden-fileset"})

    with pytest.raises(JobOutputLocationError):
        await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)

    _mock_files_client.create_fileset.assert_not_called()


@pytest.mark.asyncio
async def test_create_job_without_output_location_auto_creates_fileset(
    mock_dispatcher: JobDispatcher,
    _mock_files_client,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """With no output_location, the dispatcher auto-creates a per-job fileset (unchanged behavior)."""
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    assert job.fileset == "test-fileset-id"
    _mock_files_client.create_fileset.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_job_leaves_caller_supplied_fileset(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, _mock_files_client
):
    """Deleting a job whose fileset was caller-supplied removes the job but leaves the fileset."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(
        mock_store, "delete-shared-fs-job", fileset="shared-eval-fileset", output_location="shared-eval-fileset"
    )

    deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)

    assert deleted is True
    _mock_files_client.delete_fileset.assert_not_called()
    await verify_job_data_exists(mock_store, job_id, should_exist=False)


@pytest.mark.asyncio
async def test_delete_job_deletes_owned_fileset(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, _mock_files_client
):
    """Deleting a job that owns its fileset (output_location unset) removes that fileset too."""
    job_id, job_name, _, _, _, _ = await create_test_job_data(
        mock_store, "delete-owned-job", fileset="job-fileset-delete-owned-job"
    )

    deleted = await mock_dispatcher.delete_job(job_name, DEFAULT_WORKSPACE)

    assert deleted is True
    _mock_files_client.delete_fileset.assert_awaited_once()
    await verify_job_data_exists(mock_store, job_id, should_exist=False)


@pytest.mark.asyncio
async def test_delete_job_leaves_caller_fileset_named_like_owned(
    mock_dispatcher: JobDispatcher, mock_store: EntityClient, _mock_files_client
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
    _mock_files_client.delete_fileset.assert_not_called()
    await verify_job_data_exists(mock_store, job_id, should_exist=False)


@pytest.mark.asyncio
async def test_create_then_delete_auto_fileset_round_trip(
    mock_dispatcher: JobDispatcher,
    _mock_files_client,
    sample_platform_job_request: CreatePlatformJobRequest,
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

    _mock_files_client.create_fileset.side_effect = _echo_create

    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert job.fileset == f"job-fileset-{job.name}"

    cancelled = await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)
    assert cancelled is not None
    assert cancelled.status == PlatformJobStatus.CANCELLED

    deleted = await mock_dispatcher.delete_job(job.name, DEFAULT_WORKSPACE)

    assert deleted is True
    _mock_files_client.delete_fileset.assert_awaited_once_with(
        name=f"job-fileset-{job.name}", workspace=DEFAULT_WORKSPACE
    )


@pytest.mark.asyncio
async def test_rerun_job_success(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test successful job rerun creates a new attempt."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to active
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = PlatformJobStatus.ACTIVE
    await mock_store.update(current_step)

    # Rerun the job, when in active state. This should be a no op
    await mock_dispatcher.rerun_job(job.name, DEFAULT_WORKSPACE)
    attempts_response = await mock_store.list(PlatformJobAttempt, filter_obj={"job": job.id})
    assert len(attempts_response.data) == 1

    # Cancel the job
    await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)
    updated_step = await mock_store.get_by_id(PlatformJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == PlatformJobStatus.CANCELLING

    # Mark as Cancelled
    await mock_dispatcher.update_job_status_from_step(
        step=updated_step,
        status=PlatformJobStatus.CANCELLED,
        error_details={},
    )

    # Rerun (now that it's cancelled)
    updated_job = await mock_dispatcher.get_job(job.name, DEFAULT_WORKSPACE)
    assert updated_job is not None
    await mock_dispatcher.rerun_job(updated_job.name, DEFAULT_WORKSPACE)

    # Verify a new attempt was created
    attempts_response = await mock_store.list(PlatformJobAttempt, filter_obj={"job": updated_job.id})
    assert len(attempts_response.data) == 2

    # Get the new attempt (highest seq)
    sorted_attempts = sorted(attempts_response.data, key=lambda a: a.seq, reverse=True)
    new_attempt = sorted_attempts[0]
    assert new_attempt is not None
    assert new_attempt.seq == 1
    assert new_attempt.status == PlatformJobStatus.CREATED

    # Verify the new attempt has a first step created
    steps_response = await mock_store.list(PlatformJobStep, filter_obj={"attempt_id": new_attempt.id})
    assert len(steps_response.data) == 1
    # With parent-scoped uniqueness, step names are simple (unique per attempt)
    assert steps_response.data[0].name == "basic"


@pytest.mark.asyncio
async def test_pause_job_success(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test successful job pause sets step to PAUSING status."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to active
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = PlatformJobStatus.ACTIVE
    await mock_store.update(current_step)

    # Pause the job
    await mock_dispatcher.pause_job(job.name, DEFAULT_WORKSPACE)

    # Verify the step was set to pausing
    updated_step = await mock_store.get_by_id(PlatformJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == PlatformJobStatus.PAUSING


@pytest.mark.asyncio
async def test_pause_job_no_active_step(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test pause job when no active step exists returns job unchanged."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to completed
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = PlatformJobStatus.COMPLETED
    await mock_store.update(current_step)

    # Pause the job
    await mock_dispatcher.pause_job(job.name, DEFAULT_WORKSPACE)

    # Verify the step status didn't change
    updated_step = await mock_store.get_by_id(PlatformJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == PlatformJobStatus.COMPLETED


@pytest.mark.asyncio
async def test_cancel_job_success(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test successful job cancel sets step to CANCELLING status."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to active
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = PlatformJobStatus.ACTIVE
    await mock_store.update(current_step)

    # Cancel the job
    await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)

    # Verify the step was set to cancelling
    updated_step = await mock_store.get_by_id(PlatformJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == PlatformJobStatus.CANCELLING

    # Verify the attempt status was updated (CANCELLING status propagates to attempt)
    attempt = await mock_dispatcher.get_current_attempt(job.name, DEFAULT_WORKSPACE)
    assert attempt is not None
    assert attempt.status == PlatformJobStatus.CANCELLING


@pytest.mark.asyncio
async def test_update_job_status_from_step_retries_on_entity_conflict(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test that update_job_status_from_step refetches and retries on EntityConflictError."""
    # Create a job and set step to ACTIVE (same setup as cancel flow)
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = PlatformJobStatus.ACTIVE
    await mock_store.update(current_step)

    # Simulate reconciler/API race: first store.update(step) raises conflict, second succeeds
    step_update_calls = []
    original_update = mock_store.update

    async def update_side_effect(entity, *args, **kwargs):
        if isinstance(entity, PlatformJobStep):
            step_update_calls.append(entity.id)
            if len(step_update_calls) == 1:
                raise EntityConflictError("version conflict (simulated)")
        return await original_update(entity, *args, **kwargs)

    with patch.object(mock_store, "update", side_effect=update_side_effect):
        await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)

    # First update raised; refetch and retry should have succeeded on second update
    assert len(step_update_calls) == 2
    updated_step = await mock_store.get_by_id(PlatformJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == PlatformJobStatus.CANCELLING


@pytest.mark.asyncio
async def test_update_job_status_from_step_skips_step_store_when_noop(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Identical PENDING + status_details should not persist a PlatformJobStep update."""
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = PlatformJobStatus.PENDING
    current_step.status_details = {"message": "pulling"}
    await mock_store.update(current_step)

    attempt = await mock_store.get_by_id(PlatformJobAttempt, job.attempt_id)
    assert attempt is not None
    attempt.status = PlatformJobStatus.PENDING
    await mock_store.update(attempt)

    current_step = await mock_store.get_by_id(PlatformJobStep, current_step.id)
    assert current_step is not None

    step_update_count = 0
    original_update = mock_store.update

    async def counting_update(entity, *args, **kwargs):
        nonlocal step_update_count
        if isinstance(entity, PlatformJobStep):
            step_update_count += 1
        return await original_update(entity, *args, **kwargs)

    with patch.object(mock_store, "update", side_effect=counting_update):
        await mock_dispatcher.update_job_status_from_step(
            current_step,
            PlatformJobStatus.PENDING,
            status_details={"message": "pulling"},
        )

    assert step_update_count == 0


@pytest.mark.asyncio
async def test_update_job_status_from_step_missing_job_raises_typed_skip(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """A step status update racing a job delete does not raise a generic Exception."""
    job_id, _, _, step_id, _, _ = await create_test_job_data(mock_store, "missing-job-status-update")
    step = await mock_store.get_by_id(PlatformJobStep, step_id)
    await mock_store.delete_by_id(PlatformJob, job_id)

    with pytest.raises(JobStatusUpdateSkippedError):
        await mock_dispatcher.update_job_status_from_step(step, PlatformJobStatus.COMPLETED)


@pytest.mark.asyncio
async def test_cancel_job_multiple_steps_only_cancels_active(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test cancel job only cancels the active/non-terminal step."""
    # Modify the job to have multiple steps
    multi_step_spec = sample_platform_job_request.platform_spec.model_copy(deep=True)
    multi_step_spec.steps.append(
        PlatformJobStepSpec(
            name="second-step", executor=sample_platform_job_request.platform_spec.steps[0].executor, config={}
        )
    )
    sample_platform_job_request.platform_spec = multi_step_spec

    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the first step and set it to completed
    first_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert first_step is not None
    first_step.status = PlatformJobStatus.COMPLETED
    await mock_store.update(first_step)

    # Create and save a second step as active (parent-scoped: unique per attempt)
    second_step = PlatformJobStep(
        attempt_id=job.attempt_id,
        name="second-step",  # Simple name, unique within this attempt
        workspace=DEFAULT_WORKSPACE,
        config={},
        status=PlatformJobStatus.ACTIVE,
    )
    second_step = await mock_store.add(second_step)

    # Cancel the job
    await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)

    # Verify only the active step was cancelled
    updated_first_step = await mock_store.get_by_id(PlatformJobStep, first_step.id)
    assert updated_first_step is not None
    assert updated_first_step.status == PlatformJobStatus.COMPLETED  # Should remain completed

    updated_second_step = await mock_store.get_by_id(PlatformJobStep, second_step.id)
    assert updated_second_step is not None
    assert updated_second_step.status == PlatformJobStatus.CANCELLING  # Should be cancelling


@pytest.mark.asyncio
async def test_cancel_job_no_active_step(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test cancel job when no active step exists returns job unchanged."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to completed
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = PlatformJobStatus.COMPLETED
    await mock_store.update(current_step)

    # Cancel the job
    await mock_dispatcher.cancel_job(job.name, DEFAULT_WORKSPACE)

    # Verify the step status didn't change
    updated_step = await mock_store.get_by_id(PlatformJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == PlatformJobStatus.COMPLETED


@pytest.mark.asyncio
async def test_resume_job_success(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test successful job resume sets paused step to RESUMING status."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to paused
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = PlatformJobStatus.PAUSED
    await mock_store.update(current_step)

    # Resume the job
    await mock_dispatcher.resume_job(job.name, DEFAULT_WORKSPACE)

    # Verify the step was set to resuming
    updated_step = await mock_store.get_by_id(PlatformJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == PlatformJobStatus.RESUMING


@pytest.mark.asyncio
async def test_resume_job_no_paused_step(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test resume job when no paused step exists returns job unchanged."""
    # Create a job through the dispatcher
    job = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

    # Get the current step and set it to active
    current_step = await mock_dispatcher.get_current_job_step_by_name(job.name, "basic", DEFAULT_WORKSPACE)
    assert current_step is not None
    current_step.status = PlatformJobStatus.ACTIVE
    await mock_store.update(current_step)

    # Resume the job
    await mock_dispatcher.resume_job(job.name, DEFAULT_WORKSPACE)

    # Verify the step status didn't change
    updated_step = await mock_store.get_by_id(PlatformJobStep, current_step.id)
    assert updated_step is not None
    assert updated_step.status == PlatformJobStatus.ACTIVE


@pytest.mark.asyncio
async def test_get_job_status_returns_steps_in_spec_order(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """Test that get_job_status returns steps in the same order as defined in platform_job_spec."""
    # Create a multi-step job with steps in a specific order
    multi_step_spec = TestConstants.PLATFORM_SPEC.model_copy(deep=True)
    multi_step_spec.steps = [
        PlatformJobStepSpec(
            name="first-step",
            executor=TestConstants.PLATFORM_SPEC.steps[0].executor,
            config={"order": 1},
        ),
        PlatformJobStepSpec(
            name="second-step",
            executor=TestConstants.PLATFORM_SPEC.steps[0].executor,
            config={"order": 2},
        ),
        PlatformJobStepSpec(
            name="third-step",
            executor=TestConstants.PLATFORM_SPEC.steps[0].executor,
            config={"order": 3},
        ),
    ]

    job_request = CreatePlatformJobRequest(
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
    first_step.status = PlatformJobStatus.COMPLETED
    await mock_store.update(first_step)

    # Create second step
    second_step = PlatformJobStep(
        attempt_id=job.attempt_id,
        name="second-step",
        workspace=DEFAULT_WORKSPACE,
        config={"order": 2},
        status=PlatformJobStatus.COMPLETED,
    )
    second_step = await mock_store.add(second_step)

    # Create third step
    third_step = PlatformJobStep(
        attempt_id=job.attempt_id,
        name="third-step",
        workspace=DEFAULT_WORKSPACE,
        config={"order": 3},
        status=PlatformJobStatus.ACTIVE,
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
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test list_steps respects workspace filtering."""
    from unittest.mock import AsyncMock, MagicMock

    from nmp.common.entities.client import EntityClient
    from nmp.core.jobs.api.v2.jobs.schemas import PlatformJobSortField, PlatformJobStepsListFilter
    from nmp.testing import create_test_client

    # Create entity store with multiple workspaces and projects
    projects = ["default/test-project", "other-workspace/test-project"]
    with create_test_client(client_type=EntityClient, projects=projects) as mock_store:
        # Create mock SDK with patched files client
        mock_nmp_client = MagicMock()
        mock_files = AsyncMock()
        mock_fileset_obj = MagicMock()
        mock_fileset_obj.name = "test-fileset-id"
        mock_resp = MagicMock()
        mock_resp.data.return_value = mock_fileset_obj
        mock_files.create_fileset.return_value = mock_resp

        with patch("nmp.core.jobs.app.dispatcher.client_from_platform", return_value=mock_files):
            # Create dispatcher with the multi-workspace store
            mock_dispatcher = JobDispatcher(store=mock_store, sdk=mock_nmp_client)

            # Create jobs in workspace "default"
            job1 = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

            job2_request = CreatePlatformJobRequest(
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
            job3_request = CreatePlatformJobRequest(
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

            job4_request = CreatePlatformJobRequest(
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
        step_filter = PlatformJobStepsListFilter()
        steps_default, count_default = await mock_dispatcher.list_steps(
            filter=step_filter,
            sort=PlatformJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace=DEFAULT_WORKSPACE,
        )

        # Should return only steps from "default" workspace
        assert len(steps_default) == 2, f"Expected 2 steps in 'default' workspace, but got {len(steps_default)}"
        assert count_default == 2

        # Verify the steps belong to jobs in default workspace
        job_names_default = {step.job for step in steps_default}
        assert job1.name in job_names_default
        assert job2.name in job_names_default
        assert job3.name not in job_names_default
        assert job4.name not in job_names_default

        # List steps in "other-workspace"
        steps_other, count_other = await mock_dispatcher.list_steps(
            filter=step_filter,
            sort=PlatformJobSortField.CREATED_AT_ASC,
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
            sort=PlatformJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace=ALL_WORKSPACES,
        )

        # Should return all 4 steps
        assert len(steps_all) == 4, f"Expected 4 steps across all workspaces, but got {len(steps_all)}"
        assert count_all == 4


@pytest.mark.asyncio
async def test_list_steps_with_status_filter(
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """list_steps with filter.status set forwards a filter_str to the entity store.

    Regression: this branch used to pass ``search=`` to ``EntityClient.list``;
    after the kwarg was renamed to ``filter_str``, an unguarded call would
    ``TypeError`` at runtime. Existing tests left ``filter.status`` empty so
    the filter_str branch was never exercised.
    """
    from unittest.mock import AsyncMock, MagicMock

    from nmp.common.entities.client import EntityClient
    from nmp.core.jobs.api.v2.jobs.schemas import (
        PlatformJobSortField,
        PlatformJobStatus,
        PlatformJobStepsListFilter,
    )
    from nmp.testing import create_test_client

    with create_test_client(client_type=EntityClient) as mock_store:
        mock_nmp_client = MagicMock()
        mock_files = AsyncMock()
        mock_fileset_obj = MagicMock()
        mock_fileset_obj.name = "test-fileset-id"
        mock_resp = MagicMock()
        mock_resp.data.return_value = mock_fileset_obj
        mock_files.create_fileset.return_value = mock_resp

        with patch("nmp.core.jobs.app.dispatcher.client_from_platform", return_value=mock_files):
            mock_dispatcher = JobDispatcher(store=mock_store, sdk=mock_nmp_client)
            await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

        # Filtering by a status that no step has should return nothing — the
        # important assertion is that the call doesn't raise.
        step_filter = PlatformJobStepsListFilter(status=[PlatformJobStatus.ACTIVE])
        steps, count = await mock_dispatcher.list_steps(
            filter=step_filter,
            sort=PlatformJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace=DEFAULT_WORKSPACE,
        )
        assert steps == []
        assert count == 0


@pytest.mark.asyncio
async def test_list_jobs_across_multiple_workspaces(
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test list_jobs respects workspace filtering."""
    from unittest.mock import MagicMock

    from nmp.common.entities.client import EntityClient
    from nmp.core.jobs.api.v2.jobs.schemas import PlatformJobListSortField
    from nmp.testing import create_test_client

    # Create entity store with multiple workspaces and projects
    projects = ["default/test-project", "other-workspace/test-project"]
    with create_test_client(client_type=EntityClient, projects=projects) as mock_store:
        # Create mock SDK with patched files client
        mock_nmp_client = MagicMock()
        mock_files = AsyncMock()
        mock_fileset_obj = MagicMock()
        mock_fileset_obj.name = "test-fileset-id"
        mock_resp = MagicMock()
        mock_resp.data.return_value = mock_fileset_obj
        mock_files.create_fileset.return_value = mock_resp

        with patch("nmp.core.jobs.app.dispatcher.client_from_platform", return_value=mock_files):
            # Create dispatcher with the multi-workspace store
            mock_dispatcher = JobDispatcher(store=mock_store, sdk=mock_nmp_client)

            # Create jobs in workspace "default"
            job1 = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

            job2_request = CreatePlatformJobRequest(
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
            job3_request = CreatePlatformJobRequest(
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

            job4_request = CreatePlatformJobRequest(
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
            sort=PlatformJobListSortField.CREATED_AT_ASC,
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
            sort=PlatformJobListSortField.CREATED_AT_ASC,
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
            sort=PlatformJobListSortField.CREATED_AT_ASC,
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
    from nmp.core.jobs.api.v2.jobs.schemas import PlatformJobListSortField

    for source in ("zebra-source", "alpha-source", "middle-source"):
        await mock_dispatcher.create_job(
            CreatePlatformJobRequest(
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
        sort=PlatformJobListSortField.SOURCE_ASC,
        workspace=DEFAULT_WORKSPACE,
    )
    assert [j.source for j in jobs_asc] == ["alpha-source", "middle-source", "zebra-source"]

    jobs_desc, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(operation=None),
        sort=PlatformJobListSortField.SOURCE_DESC,
        workspace=DEFAULT_WORKSPACE,
    )
    assert [j.source for j in jobs_desc] == ["zebra-source", "middle-source", "alpha-source"]


# =============================================================================
# list_jobs: status in-memory filtering via ParsedFilter
# =============================================================================


async def _set_attempt_status(store: EntityClient, attempt_id: str, status: PlatformJobStatus) -> None:
    """Update the status of an attempt directly in the store."""
    attempt = await store.get_by_id(PlatformJobAttempt, attempt_id)
    attempt.status = status
    await store.update(attempt)


async def _make_job(
    dispatcher: JobDispatcher,
    store: EntityClient,
    name: str,
    status: PlatformJobStatus,
) -> PlatformJobResponse:
    """Create a job via the dispatcher and set its attempt to the given status."""
    request = CreatePlatformJobRequest(
        name=name,
        source=TestConstants.SOURCE,
        project=TestConstants.PROJECT,
        spec=TestConstants.SPEC_BASIC,
        platform_spec=TestConstants.PLATFORM_SPEC,
    )
    job = await dispatcher.create_job(request, DEFAULT_WORKSPACE)
    await _set_attempt_status(store, job.attempt_id, status)
    return job


@pytest.mark.asyncio
async def test_list_jobs_filter_status_single_excludes_non_matching(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """filter status with a single value returns only jobs whose attempt matches."""
    active_job = await _make_job(mock_dispatcher, mock_store, "job-active", PlatformJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "job-completed", PlatformJobStatus.COMPLETED)

    jobs, _ = await mock_dispatcher.list_jobs(
        parsed=ParsedFilter(
            operation=ComparisonOperation(field="data.status", operator=FilterOperator.EQ, value="active"),
        ),
        workspace=DEFAULT_WORKSPACE,
    )

    assert len(jobs) == 1
    assert jobs[0].id == active_job.id


@pytest.mark.asyncio
async def test_list_jobs_filter_status_list_is_or(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """filter status with multiple values matches any of the given statuses (OR logic)."""
    active_job = await _make_job(mock_dispatcher, mock_store, "job-active", PlatformJobStatus.ACTIVE)
    completed_job = await _make_job(mock_dispatcher, mock_store, "job-completed", PlatformJobStatus.COMPLETED)
    await _make_job(mock_dispatcher, mock_store, "job-error", PlatformJobStatus.ERROR)

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
    """$not/$eq on status (AIRCORE-324) returns jobs whose status is NOT the value."""
    await _make_job(mock_dispatcher, mock_store, "job-active", PlatformJobStatus.ACTIVE)
    completed = await _make_job(mock_dispatcher, mock_store, "job-completed", PlatformJobStatus.COMPLETED)
    error = await _make_job(mock_dispatcher, mock_store, "job-error", PlatformJobStatus.ERROR)

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

    active = await _make_job(mock_dispatcher, mock_store, "job-active", PlatformJobStatus.ACTIVE)
    completed = await _make_job(mock_dispatcher, mock_store, "job-completed", PlatformJobStatus.COMPLETED)
    await _make_job(mock_dispatcher, mock_store, "job-error", PlatformJobStatus.ERROR)

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
    """$nin on status (AIRCORE-324) returns jobs whose status is none of the values."""
    await _make_job(mock_dispatcher, mock_store, "job-active", PlatformJobStatus.ACTIVE)
    completed = await _make_job(mock_dispatcher, mock_store, "job-completed", PlatformJobStatus.COMPLETED)
    error = await _make_job(mock_dispatcher, mock_store, "job-error", PlatformJobStatus.ERROR)

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
    """$or mixing status with a non-status field (AIRCORE-324) returns the union.

    Matches jobs that are ACTIVE *or* whose name contains "special", regardless
    of the other condition.
    """
    active = await _make_job(mock_dispatcher, mock_store, "ordinary-active", PlatformJobStatus.ACTIVE)
    special_completed = await _make_job(mock_dispatcher, mock_store, "special-completed", PlatformJobStatus.COMPLETED)
    await _make_job(mock_dispatcher, mock_store, "ordinary-completed", PlatformJobStatus.COMPLETED)

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
    """$not wrapping a status+name subtree (AIRCORE-324) returns the negation.

    NOT (status == active AND name ~ "eval") keeps every job except the one that
    is both ACTIVE and name-matches "eval".
    """
    await _make_job(mock_dispatcher, mock_store, "eval-active", PlatformJobStatus.ACTIVE)
    eval_completed = await _make_job(mock_dispatcher, mock_store, "eval-completed", PlatformJobStatus.COMPLETED)
    other_active = await _make_job(mock_dispatcher, mock_store, "train-active", PlatformJobStatus.ACTIVE)

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
    """$or where each branch mixes status with a name term (AIRCORE-324).

    (active AND name~foo) OR (completed AND name~bar) returns exactly the jobs
    matching either full branch.
    """
    foo_active = await _make_job(mock_dispatcher, mock_store, "foo-job", PlatformJobStatus.ACTIVE)
    bar_completed = await _make_job(mock_dispatcher, mock_store, "bar-job", PlatformJobStatus.COMPLETED)
    # foo but wrong status; bar but wrong status — both excluded.
    await _make_job(mock_dispatcher, mock_store, "foo-completed", PlatformJobStatus.COMPLETED)
    await _make_job(mock_dispatcher, mock_store, "bar-active", PlatformJobStatus.ACTIVE)

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
async def test_list_jobs_status_not_forwarded_to_entity_store(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """status must not reach the entity store (lives on PlatformJobAttempt).

    name and project ARE forwarded to the entity store. Only status is
    filtered in-memory after the join.
    """

    await _make_job(mock_dispatcher, mock_store, "job-active", PlatformJobStatus.ACTIVE)

    captured_operations: list = []
    original_list = mock_store.list

    async def capturing_list(entity_type, *, filter_operation=None, **kwargs):
        if entity_type is PlatformJob:
            op_dict = filter_operation.to_dict() if filter_operation else {}
            captured_operations.append(op_dict)
        return await original_list(entity_type, filter_operation=filter_operation, **kwargs)

    with patch.object(mock_store, "list", side_effect=capturing_list):
        await mock_dispatcher.list_jobs(
            parsed=ParsedFilter(
                operation=LogicalOperation(
                    operator=FilterOperator.AND,
                    operations=[
                        ComparisonOperation(field="name", operator=FilterOperator.LIKE, value="job"),
                        ComparisonOperation(field="data.status", operator=FilterOperator.EQ, value="active"),
                    ],
                ),
            ),
            workspace=DEFAULT_WORKSPACE,
        )

    assert captured_operations, "Expected at least one entity store query for PlatformJob"
    for op_dict in captured_operations:
        # status lives on PlatformJobAttempt — must never appear in entity store query
        assert "data.status" not in json.dumps(op_dict), "data.status must not be forwarded to entity store"


@pytest.mark.asyncio
async def test_list_jobs_filter_name_like(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
):
    """filter name with $like applies substring matching."""
    match1 = await _make_job(mock_dispatcher, mock_store, "eval-training-run", PlatformJobStatus.ACTIVE)
    match2 = await _make_job(mock_dispatcher, mock_store, "training-eval-v2", PlatformJobStatus.COMPLETED)
    await _make_job(mock_dispatcher, mock_store, "finetune-run", PlatformJobStatus.ACTIVE)

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
    eval_job = await _make_job(mock_dispatcher, mock_store, "eval-run", PlatformJobStatus.ACTIVE)
    train_job = await _make_job(mock_dispatcher, mock_store, "training-run", PlatformJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "finetune-run", PlatformJobStatus.ACTIVE)

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
    await _make_job(mock_dispatcher, mock_store, "target-job", PlatformJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "other-job", PlatformJobStatus.ACTIVE)

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
    target = await _make_job(mock_dispatcher, mock_store, "exact-match-job", PlatformJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "exact-match-job-v2", PlatformJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "unrelated-job", PlatformJobStatus.ACTIVE)

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

    await _make_job(mock_dispatcher, mock_store, "excluded-job", PlatformJobStatus.ACTIVE)
    included = await _make_job(mock_dispatcher, mock_store, "included-job", PlatformJobStatus.ACTIVE)
    await _make_job(mock_dispatcher, mock_store, "included-but-wrong-status", PlatformJobStatus.COMPLETED)

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
    await _make_job(mock_dispatcher, mock_store, "excluded-job", PlatformJobStatus.ACTIVE)
    # This job passes name filter and has matching project
    await _make_job(mock_dispatcher, mock_store, "included-job", PlatformJobStatus.ACTIVE)
    # This job passes name filter but has no project, so project $like won't match
    no_project = CreatePlatformJobRequest(
        name="no-project-job",
        source=TestConstants.SOURCE,
        project=None,
        spec=TestConstants.SPEC_BASIC,
        platform_spec=TestConstants.PLATFORM_SPEC,
    )
    no_proj_job = await mock_dispatcher.create_job(no_project, DEFAULT_WORKSPACE)
    await _set_attempt_status(mock_store, no_proj_job.attempt_id, PlatformJobStatus.ACTIVE)

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
