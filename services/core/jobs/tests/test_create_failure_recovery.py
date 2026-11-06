# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure-injection regressions and baseline characterization for P-123."""

from typing import TypeVar
from unittest.mock import AsyncMock, patch

import pytest
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nhx.common.entities import DEFAULT_WORKSPACE, EntityClient
from nhx.common.entities.client import EntityBase
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import CreateHelixJobRequest
from nhx.core.jobs.app.constants import JOB_CREATION_PENDING_KEY
from nhx.core.jobs.app.dispatcher import JobDeletionConflictError, JobDispatcher
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt, HelixJobStep

# The shared EntityClient fixture runs the Entities service with SQLite.
pytestmark = pytest.mark.integration

EntityT = TypeVar("EntityT", bound=EntityBase)


@pytest.mark.asyncio
async def test_job_write_failure_does_not_allocate_a_fileset(
    mock_dispatcher: JobDispatcher,
    sample_platform_job_request: CreateHelixJobRequest,
    mock_files_client: AsyncMock,
) -> None:
    failure = RuntimeError("job write rejected before persistence")
    with patch.object(mock_dispatcher.store, "create", side_effect=failure):
        with pytest.raises(RuntimeError) as caught:
            await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert caught.value is failure
    mock_files_client.create_fileset.assert_not_awaited()
    mock_files_client.delete_fileset.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_create_preserves_caller_owned_fileset_and_original_error(
    mock_dispatcher: JobDispatcher,
    sample_platform_job_request: CreateHelixJobRequest,
    mock_files_client: AsyncMock,
) -> None:
    request = sample_platform_job_request.model_copy(update={"output_location": "caller-owned-output"})
    failure = RuntimeError("job write rejected before persistence")
    with patch.object(mock_dispatcher.store, "create", side_effect=failure):
        with pytest.raises(RuntimeError) as caught:
            await mock_dispatcher.create_job(request, DEFAULT_WORKSPACE)
    assert caught.value is failure
    mock_files_client.create_fileset.assert_not_awaited()
    mock_files_client.delete_fileset.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_failure_retains_an_unpublished_step(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
    mock_files_client: AsyncMock,
) -> None:
    """A persisted first step remains fenced when its creation call raises."""
    start_attempt = mock_dispatcher._start_attempt
    failure = RuntimeError("first-step write response lost")

    async def persist_then_raise(job: HelixJob, attempt: HelixJobAttempt) -> None:
        await start_attempt(job, attempt)
        raise failure

    with patch.object(mock_dispatcher, "_start_attempt", side_effect=persist_then_raise):
        with pytest.raises(RuntimeError) as caught:
            await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert caught.value is failure
    jobs = await mock_store.list(HelixJob, workspace=DEFAULT_WORKSPACE)
    assert len(jobs.data) == 1
    job = jobs.data[0]
    steps = await mock_store.list(
        HelixJobStep,
        workspace=DEFAULT_WORKSPACE,
        filter_obj={"attempt_id": job.current_attempt_id},
    )
    assert len(steps.data) == 1
    assert steps.data[0].status == HelixJobStatus.CREATED
    assert steps.data[0].status_details[JOB_CREATION_PENDING_KEY]
    mock_files_client.delete_fileset.assert_not_awaited()


@pytest.mark.asyncio
async def test_fileset_failure_retains_job_without_runnable_work(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
    mock_files_client: AsyncMock,
) -> None:
    failure = RuntimeError("fileset allocation failed")
    mock_files_client.create_fileset.side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert caught.value is failure
    jobs = await mock_store.list(HelixJob, workspace=DEFAULT_WORKSPACE)
    assert len(jobs.data) == 1
    job = jobs.data[0]
    assert job.fileset_creation_pending
    assert job.current_attempt_id is None
    assert not (await mock_store.list(HelixJobAttempt, workspace=DEFAULT_WORKSPACE)).data
    assert not (await mock_store.list(HelixJobStep, workspace=DEFAULT_WORKSPACE)).data
    mock_files_client.delete_fileset.assert_not_awaited()


@pytest.mark.asyncio
async def test_unresolved_fileset_allocation_cannot_delete_an_unrelated_fileset(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
    mock_files_client: AsyncMock,
) -> None:
    failure = RuntimeError("fileset create conflict or ambiguous response")
    mock_files_client.create_fileset.side_effect = failure
    with pytest.raises(RuntimeError):
        await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    jobs = await mock_store.list(HelixJob, workspace=DEFAULT_WORKSPACE)
    job = jobs.data[0]
    with pytest.raises(JobDeletionConflictError, match="awaiting controller cleanup"):
        await mock_dispatcher.delete_job(job.id, DEFAULT_WORKSPACE)
    retained = await mock_store.get_by_id(HelixJob, job.id)
    assert retained.fileset_creation_pending
    mock_files_client.delete_fileset.assert_not_awaited()


@pytest.mark.asyncio
async def test_fileset_allocation_has_a_persisted_owner(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
    mock_files_client: AsyncMock,
) -> None:
    fileset_name = "job-fileset-test-job"
    response = mock_files_client.create_fileset.return_value
    response.data.return_value.name = fileset_name
    allocation_entered = False

    async def check_owner(*, body: CreateFilesetRequest, workspace: str):
        nonlocal allocation_entered
        jobs = await mock_store.list(HelixJob, workspace=workspace)
        assert len(jobs.data) == 1
        owner = jobs.data[0]
        assert owner.fileset == body.name == fileset_name
        assert owner.fileset_creation_pending
        assert body.model_dump(exclude_unset=True)["custom_fields"]["job_id"] == owner.id
        assert owner.current_attempt_id is None
        assert not (await mock_store.list(HelixJobStep, workspace=workspace)).data
        allocation_entered = True
        return response

    mock_files_client.create_fileset.side_effect = check_owner
    result = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert allocation_entered
    stored = await mock_store.get_by_id(HelixJob, result.id)
    assert not stored.fileset_creation_pending
    assert stored.fileset == fileset_name
    assert stored.current_attempt_id == result.attempt_id


@pytest.mark.asyncio
async def test_lost_job_create_response_does_not_allocate_a_fileset(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
    mock_files_client: AsyncMock,
) -> None:
    create_entity = mock_store.create
    failure = RuntimeError("Job committed but its create response was lost")

    async def persist_then_raise(entity: EntityT) -> EntityT:
        await create_entity(entity)
        raise failure

    with patch.object(mock_store, "create", side_effect=persist_then_raise):
        with pytest.raises(RuntimeError) as caught:
            await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert caught.value is failure
    jobs = await mock_store.list(HelixJob, workspace=DEFAULT_WORKSPACE)
    assert len(jobs.data) == 1
    assert jobs.data[0].fileset_creation_pending
    mock_files_client.create_fileset.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_allocation_confirmation_retains_the_ownership_reference(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    sample_platform_job_request: CreateHelixJobRequest,
    mock_files_client: AsyncMock,
) -> None:
    failure = RuntimeError("allocation receipt update rejected")
    update = mock_store.update

    async def fail_confirmation(entity: EntityT) -> EntityT:
        if isinstance(entity, HelixJob) and not entity.fileset_creation_pending:
            raise failure
        return await update(entity)

    with patch.object(mock_store, "update", side_effect=fail_confirmation):
        with pytest.raises(RuntimeError) as caught:
            await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert caught.value is failure
    jobs = await mock_store.list(HelixJob, workspace=DEFAULT_WORKSPACE)
    assert len(jobs.data) == 1
    owner = jobs.data[0]
    assert owner.fileset_creation_pending
    assert owner.fileset == "job-fileset-test-job"
    assert not (await mock_store.list(HelixJobAttempt, workspace=DEFAULT_WORKSPACE)).data
    request = mock_files_client.create_fileset.call_args.kwargs["body"]
    assert request.custom_fields["job_id"] == owner.id
    mock_files_client.delete_fileset.assert_not_awaited()
