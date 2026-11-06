# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jobs creation cleanup through actual Entities, Files and SQLite boundaries."""

import asyncio
import os
import threading
from collections.abc import Iterator
from typing import TypeVar
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from nemo_helix_plugin.client.errors import ConflictError, NemoTransportError
from nemo_helix_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, UpdateFilesetRequest
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.common.auth.models import Principal
from nhx.common.config.base import DatabaseConfig
from nhx.common.entities import (
    DEFAULT_WORKSPACE,
    SYSTEM_WORKSPACE,
    EntityConflictError,
    EntityNotFoundError,
    SyncEntityClient,
)
from nhx.common.entities.client import EntityBase
from nhx.common.observability import MARK_INTERNAL_REQUEST_HEADERS
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.service import EntitiesService
from nhx.core.files.app.backends.local import LocalStorageImpl
from nhx.core.files.service import FilesService
from nhx.core.jobs.api.v2.jobs.schemas import CreateHelixJobRequest, HelixJobSortField, HelixJobStepsListFilter
from nhx.core.jobs.app.constants import JOB_CREATION_PENDING_KEY
from nhx.core.jobs.app.dispatcher import JobDeletionConflictError, JobDispatcher, JobStorageReclaimPendingError
from nhx.core.jobs.app.schemas import HelixJobEnvironmentVariable
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.reconciler import JobReconciler
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt, HelixJobStep, HelixJobStorageReclaim
from nhx.testing.client import ClientContext, create_test_client
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

pytestmark = pytest.mark.integration
EntityT = TypeVar("EntityT", bound=EntityBase)


@pytest.fixture
def cleanup_platform() -> Iterator[ClientContext]:
    with create_test_client(FilesService, client_type=ClientContext, projects=["default/test-project"]) as context:
        yield context


def dispatcher_for(context: ClientContext) -> JobDispatcher:
    return JobDispatcher(
        store=context.entity_client,
        system_store=context.entity_client.as_service("jobs", internal=True),
        files=AsyncFilesClient.from_client(context.async_client),
        secrets=AsyncSecretsClient.from_client(context.async_client),
    )


def reconciler_for(context: ClientContext) -> JobReconciler:
    return JobReconciler(
        backend_registry=MagicMock(spec=BackendRegistry),
        nemo_client=context.client,
        system_entity_client=SyncEntityClient(EntitiesClient.from_client(context.client)).as_service(
            "jobs", internal=True
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["attempt", "pointer", "step", "release_before", "release_after"])
async def test_creation_cleanup_across_real_store_and_files(
    cleanup_platform: ClientContext,
    sample_platform_job_request: CreateHelixJobRequest,
    boundary: str,
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    create = dispatcher.store.create
    update = dispatcher.store.update
    failure = RuntimeError(f"creation failure at {boundary}")

    async def create_with_failure(entity: EntityT) -> EntityT:
        if boundary == "attempt" and isinstance(entity, HelixJobAttempt):
            raise failure
        result = await create(entity)
        if boundary == "step" and isinstance(entity, HelixJobStep):
            raise failure
        return result

    async def update_with_failure(entity: EntityT) -> EntityT:
        if boundary == "pointer" and isinstance(entity, HelixJob) and entity.current_attempt_id is not None:
            raise failure
        releasing = isinstance(entity, HelixJobStep) and not entity.status_details.get(JOB_CREATION_PENDING_KEY)
        if releasing and boundary == "release_before":
            raise failure
        result = await update(entity)
        if releasing and boundary == "release_after":
            raise failure
        return result

    with (
        patch.object(dispatcher.store, "create", side_effect=create_with_failure),
        patch.object(dispatcher.store, "update", side_effect=update_with_failure),
    ):
        with pytest.raises(RuntimeError) as caught:
            await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert caught.value is failure
    jobs = await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)
    assert len(jobs.data) == 1
    owner = jobs.data[0]
    fileset = (await dispatcher.files.get_fileset(name=owner.fileset, workspace=owner.workspace)).data()
    assert fileset.custom_fields["job_id"] == owner.id
    with pytest.raises(JobDeletionConflictError, match="awaiting controller cleanup"):
        await dispatcher.delete_job(owner.id, owner.workspace)
    assert (await context.entity_client.get_by_id(HelixJob, owner.id)).id == owner.id
    visible, total = await dispatcher.list_steps(
        filter=HelixJobStepsListFilter(),
        sort=HelixJobSortField.CREATED_AT_ASC,
        limit=10,
        offset=0,
        workspace=DEFAULT_WORKSPACE,
    )
    assert len(visible) == total == (1 if boundary == "release_after" else 0)
    worker = reconciler_for(context)
    await asyncio.to_thread(worker._reclaim_queued_storage)
    queue = await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)
    assert not queue.data
    if boundary == "release_after":
        retained = await context.entity_client.get_by_id(HelixJob, owner.id)
        assert not retained.creation_failed
        assert not retained.creation_pending
        await dispatcher.files.get_fileset(name=fileset.name, workspace=owner.workspace)
    else:
        with pytest.raises(EntityNotFoundError):
            await context.entity_client.get_by_id(HelixJob, owner.id)
        assert not (await context.entity_client.list(HelixJobAttempt, workspace=DEFAULT_WORKSPACE)).data
        assert not (await context.entity_client.list(HelixJobStep, workspace=DEFAULT_WORKSPACE)).data
        with pytest.raises(ClientNotFoundError):
            await dispatcher.files.get_fileset(name=fileset.name, workspace=owner.workspace)
        retry = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
        assert retry.id != owner.id


@pytest.mark.asyncio
async def test_conflicting_fileset_survives_failed_creation_cleanup(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    original = (
        await dispatcher.files.create_fileset(
            workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name="job-fileset-test-job")
        )
    ).data()
    with pytest.raises(ConflictError):
        await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    assert not (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data
    retained = (await dispatcher.files.get_fileset(name=original.name, workspace=DEFAULT_WORKSPACE)).data()
    assert retained.id == original.id


@pytest.mark.asyncio
async def test_committed_files_allocation_with_lost_response_is_reconciled(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    allocate = dispatcher.files.create_fileset
    failure = NemoTransportError(httpx.ReadTimeout("Files response lost"))

    async def allocate_then_raise(*, body: CreateFilesetRequest, workspace: str):
        await allocate(body=body, workspace=workspace)
        raise failure

    with patch.object(dispatcher.files, "create_fileset", side_effect=allocate_then_raise):
        with pytest.raises(NemoTransportError) as caught:
            await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert caught.value is failure
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    assert not (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data
    with pytest.raises(ClientNotFoundError):
        await dispatcher.files.get_fileset(name="job-fileset-test-job", workspace=DEFAULT_WORKSPACE)


@pytest.mark.asyncio
async def test_missing_ambiguous_files_allocation_retains_ownership(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    failure = NemoTransportError(httpx.ReadTimeout("Files outcome unknown"))
    with patch.object(dispatcher.files, "create_fileset", side_effect=failure):
        with pytest.raises(NemoTransportError):
            await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    jobs = await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)
    queue = await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)
    assert len(jobs.data) == len(queue.data) == 1
    assert jobs.data[0].fileset_allocation_uncertain


@pytest.mark.asyncio
async def test_cleanup_rediscovers_failed_job_when_queue_write_fails(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    create = dispatcher.store.create
    failure = RuntimeError("attempt failed")

    async def fail_attempt(entity: EntityT) -> EntityT:
        if isinstance(entity, HelixJobAttempt):
            raise failure
        return await create(entity)

    with (
        patch.object(dispatcher.store, "create", side_effect=fail_attempt),
        patch.object(dispatcher.system_store, "create", side_effect=RuntimeError("queue unavailable")),
    ):
        with pytest.raises(RuntimeError) as caught:
            await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    assert caught.value is failure
    assert not (await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
    worker = reconciler_for(context)
    await asyncio.to_thread(worker._discover_failed_creations)
    await asyncio.to_thread(worker._reclaim_queued_storage)
    assert not (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data


@pytest.mark.asyncio
async def test_artifact_delete_failure_survives_controller_restart(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    created = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    await dispatcher.cancel_job(created.id, DEFAULT_WORKSPACE)
    await dispatcher.delete_job(created.id, DEFAULT_WORKSPACE)
    worker = reconciler_for(context)
    with patch.object(
        worker._files, "delete_fileset_by_id", side_effect=NemoTransportError(httpx.ReadTimeout("delete failed"))
    ):
        await asyncio.to_thread(worker._reclaim_queued_storage)
    queue = await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)
    assert len(queue.data) == 1
    assert queue.data[0].fileset_name == created.fileset
    assert queue.data[0].fileset_id is not None
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    assert not (await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
    with pytest.raises(ClientNotFoundError):
        await dispatcher.files.get_fileset(name=created.fileset, workspace=DEFAULT_WORKSPACE)


@pytest.mark.asyncio
async def test_cleanup_claim_fences_a_late_scheduling_release(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    start = dispatcher._start_attempt

    async def persist_step_then_fail(job: HelixJob, attempt: HelixJobAttempt) -> None:
        await start(job, attempt)
        raise RuntimeError("start response lost")

    with patch.object(dispatcher, "_start_attempt", side_effect=persist_step_then_fail):
        with pytest.raises(RuntimeError):
            await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    step = (await context.entity_client.list(HelixJobStep, workspace=DEFAULT_WORKSPACE)).data[0]
    stale_release = step.model_copy(update={"status_details": {}})
    worker = reconciler_for(context)
    delete = worker._system_entities.delete_by_id
    claimed = threading.Event()
    proceed = threading.Event()

    def pause_delete(entity_type: type[HelixJob], entity_id: str, *, expected_db_version: int):
        claimed.set()
        if not proceed.wait(timeout=10):
            raise TimeoutError("cleanup deletion checkpoint not released")
        return delete(entity_type, entity_id, expected_db_version=expected_db_version)

    with patch.object(worker._system_entities, "delete_by_id", side_effect=pause_delete):
        cleanup = asyncio.create_task(asyncio.to_thread(worker._reclaim_queued_storage))
        try:
            assert await asyncio.to_thread(claimed.wait, 10)
            with pytest.raises(EntityConflictError):
                await context.entity_client.update(stale_release)
        finally:
            proceed.set()
            await cleanup
    assert not (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data


@pytest.mark.asyncio
async def test_release_winning_the_cleanup_claim_preserves_the_job(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    start = dispatcher._start_attempt

    async def persist_step_then_fail(job: HelixJob, attempt: HelixJobAttempt) -> None:
        await start(job, attempt)
        raise RuntimeError("start response lost")

    with patch.object(dispatcher, "_start_attempt", side_effect=persist_step_then_fail):
        with pytest.raises(RuntimeError):
            await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    worker = reconciler_for(context)
    update = worker._system_entities.update
    released = False

    def release_before_claim(entity: EntityT) -> EntityT:
        nonlocal released
        if isinstance(entity, HelixJobStep) and not released:
            released = True
            update(entity.model_copy(update={"status_details": {}}))
        return update(entity)

    with patch.object(worker._system_entities, "update", side_effect=release_before_claim):
        await asyncio.to_thread(worker._reclaim_queued_storage)
    assert released
    assert len((await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data) == 1
    assert len((await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data) == 1
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    retained = (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data[0]
    assert not retained.creation_failed
    assert not (await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
    await dispatcher.files.get_fileset(name=retained.fileset, workspace=DEFAULT_WORKSPACE)


@pytest.mark.asyncio
@pytest.mark.parametrize("expire", [False, True])
async def test_setup_timeout_claims_creation_without_racing_its_owner(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest, expire: bool
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    allocate = dispatcher.files.create_fileset
    allocated = asyncio.Event()
    finish = asyncio.Event()

    async def pause_after_allocation(*, body: CreateFilesetRequest, workspace: str):
        response = await allocate(body=body, workspace=workspace)
        allocated.set()
        await finish.wait()
        return response

    with patch.object(dispatcher.files, "create_fileset", side_effect=pause_after_allocation):
        submission = asyncio.create_task(dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE))
        worker = reconciler_for(context)
        try:
            async with asyncio.timeout(10):
                await allocated.wait()
            if expire:
                with patch("nhx.core.jobs.controllers.reconciler.storage_window_expired", return_value=True):
                    await asyncio.to_thread(worker._discover_failed_creations)
                await asyncio.to_thread(worker._reclaim_queued_storage)
                assert not (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data
            else:
                await asyncio.to_thread(worker._discover_failed_creations)
                assert not (await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
            finish.set()
            if expire:
                with pytest.raises(EntityNotFoundError):
                    await submission
                # The fenced creator's failed final write may requeue its original
                # receipt. A fresh pass must also finish that idempotent handoff.
                await asyncio.to_thread(worker._reclaim_queued_storage)
                assert not (await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
            else:
                result = await submission
                current = await context.entity_client.get_by_id(HelixJob, result.id)
                assert not current.creation_failed
                assert not current.creation_pending
        finally:
            finish.set()
            await asyncio.gather(submission, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_creation_keeps_a_caller_owned_output_location(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    output = (
        await dispatcher.files.create_fileset(
            workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name="job-fileset-test-job")
        )
    ).data()
    request = sample_platform_job_request.model_copy(update={"output_location": output.name})
    create = dispatcher.store.create

    async def fail_attempt(entity: EntityT) -> EntityT:
        if isinstance(entity, HelixJobAttempt):
            raise RuntimeError("attempt failed")
        return await create(entity)

    with patch.object(dispatcher.store, "create", side_effect=fail_attempt):
        with pytest.raises(RuntimeError):
            await dispatcher.create_job(request, DEFAULT_WORKSPACE)
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    retained = (await dispatcher.files.get_fileset(name=output.name, workspace=DEFAULT_WORKSPACE)).data()
    assert retained.id == output.id
    assert not (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data


@pytest.mark.asyncio
async def test_file_replacement_is_not_deleted_by_a_retained_receipt(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    created = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    await dispatcher.cancel_job(created.id, DEFAULT_WORKSPACE)
    await dispatcher.delete_job(created.id, DEFAULT_WORKSPACE)
    worker = reconciler_for(context)
    with patch.object(
        worker._files, "delete_fileset_by_id", side_effect=NemoTransportError(httpx.ReadTimeout("delete failed"))
    ):
        await asyncio.to_thread(worker._reclaim_queued_storage)
    await dispatcher.files.delete_fileset(name=created.fileset, workspace=DEFAULT_WORKSPACE)
    replacement = (
        await dispatcher.files.create_fileset(
            workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name=created.fileset)
        )
    ).data()
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    retained = (await dispatcher.files.get_fileset(name=replacement.name, workspace=DEFAULT_WORKSPACE)).data()
    assert retained.id == replacement.id
    assert not (await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data


@pytest.mark.asyncio
async def test_deletion_response_loss_retries_an_already_removed_fileset(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    created = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    await dispatcher.cancel_job(created.id, DEFAULT_WORKSPACE)
    await dispatcher.delete_job(created.id, DEFAULT_WORKSPACE)
    worker = reconciler_for(context)
    delete_file = worker._files.delete_fileset_by_id

    def delete_then_raise(*, fileset_id: str, workspace: str):
        delete_file(fileset_id=fileset_id, workspace=workspace)
        raise NemoTransportError(httpx.ReadTimeout("delete response lost"))

    with patch.object(worker._files, "delete_fileset_by_id", side_effect=delete_then_raise):
        await asyncio.to_thread(worker._reclaim_queued_storage)
    assert len((await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data) == 1
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    assert not (await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_release_record", [False, True])
async def test_job_deletion_cannot_lose_an_artifact_handoff_during_queue_removal(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest, failed_release_record: bool
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    original = sample_platform_job_request.platform_spec
    step = original.steps[0].model_copy(
        update={"environment": [HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/storage")]}
    )
    request = sample_platform_job_request.model_copy(
        update={"platform_spec": original.model_copy(update={"steps": [step]})}
    )
    created = await dispatcher.create_job(request, DEFAULT_WORKSPACE)
    await dispatcher.cancel_job(created.id, DEFAULT_WORKSPACE)
    owner = await context.entity_client.get_by_id(HelixJob, created.id)
    record = (
        HelixJobStorageReclaim.for_failed_creation(owner)
        if failed_release_record
        else HelixJobStorageReclaim(
            name=HelixJobStorageReclaim.name_for_job(created.id),
            workspace=SYSTEM_WORKSPACE,
            job_id=created.id,
            job_workspace=DEFAULT_WORKSPACE,
            job_name=created.name,
        )
    )
    await dispatcher.system_store.create(record)
    worker = reconciler_for(context)
    delete_record = worker._system_entities.delete_by_id
    removing = threading.Event()
    proceed = threading.Event()

    def pause_removal(entity_type, entity_id: str, *, expected_db_version: int):
        if entity_type is HelixJobStorageReclaim:
            removing.set()
            if not proceed.wait(timeout=10):
                raise TimeoutError("queue-removal checkpoint not released")
        return delete_record(entity_type, entity_id, expected_db_version=expected_db_version)

    with patch.object(worker._system_entities, "delete_by_id", side_effect=pause_removal):
        cleanup = asyncio.create_task(asyncio.to_thread(worker._reclaim_queued_storage))
        try:
            assert await asyncio.to_thread(removing.wait, 10)
            await dispatcher.delete_job(created.id, DEFAULT_WORKSPACE)
        finally:
            proceed.set()
            await cleanup
    queue = await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)
    assert len(queue.data) == 1
    assert queue.data[0].fileset_name == created.fileset
    assert bool(queue.data[0].targets) == failed_release_record
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    assert not (await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
    with pytest.raises(ClientNotFoundError):
        await dispatcher.files.get_fileset(name=created.fileset, workspace=DEFAULT_WORKSPACE)


@pytest.mark.asyncio
async def test_files_service_rejects_a_stale_fileset_generation(
    cleanup_platform: ClientContext,
) -> None:
    files = AsyncFilesClient.from_client(cleanup_platform.async_client)
    old = (await files.create_fileset(workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name="generation"))).data()
    await files.delete_fileset_by_id(fileset_id=old.id, workspace=DEFAULT_WORKSPACE)
    new = (await files.create_fileset(workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name=old.name))).data()
    with pytest.raises(ClientNotFoundError):
        await files.delete_fileset_by_id(fileset_id=old.id, workspace=DEFAULT_WORKSPACE)
    current = (await files.get_fileset(name=new.name, workspace=DEFAULT_WORKSPACE)).data()
    assert current.id == new.id


@pytest.mark.asyncio
async def test_files_service_serializes_concurrent_physical_deletion(
    cleanup_platform: ClientContext,
) -> None:
    files = AsyncFilesClient.from_client(cleanup_platform.async_client)
    owned = (
        await files.create_fileset(workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name="serialized"))
    ).data()
    entered = threading.Event()
    finish = threading.Event()
    delete_all = LocalStorageImpl.delete_all

    async def paused_delete(storage: LocalStorageImpl) -> None:
        entered.set()
        if not await asyncio.to_thread(finish.wait, 10):
            raise TimeoutError("Files deletion checkpoint not released")
        await delete_all(storage)

    async def delete_original():
        return await files.delete_fileset_by_id(fileset_id=owned.id, workspace=DEFAULT_WORKSPACE)

    with patch.object(LocalStorageImpl, "delete_all", paused_delete):
        first = asyncio.create_task(delete_original())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            with pytest.raises(ConflictError):
                await files.delete_fileset_by_id(fileset_id=owned.id, workspace=DEFAULT_WORKSPACE)
        finally:
            finish.set()
            await first
    with pytest.raises(ClientNotFoundError):
        await files.get_fileset(name=owned.name, workspace=DEFAULT_WORKSPACE)


@pytest.fixture
def authenticated_cleanup_platform() -> Iterator[ClientContext]:
    with create_test_client(
        FilesService, client_type=ClientContext, projects=["default/test-project"], auth_enabled=True
    ) as context:
        yield context


@pytest.mark.asyncio
async def test_jobs_service_principal_can_finish_artifact_cleanup_with_auth_enabled(
    authenticated_cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = authenticated_cleanup_platform
    dispatcher = dispatcher_for(context)
    dispatcher.store = context.entity_client.as_service("jobs", internal=True)
    service_headers = {
        **Principal(id="service:jobs", authz_aliases=["service:jobs"]).get_headers(),
        **MARK_INTERNAL_REQUEST_HEADERS,
    }
    dispatcher.files = AsyncFilesClient.from_client(context.async_client.with_headers(service_headers))
    created = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    await dispatcher.cancel_job(created.id, DEFAULT_WORKSPACE)
    await dispatcher.delete_job(created.id, DEFAULT_WORKSPACE)
    service_client = context.client.with_headers(service_headers)
    worker = JobReconciler(
        backend_registry=MagicMock(spec=BackendRegistry),
        nemo_client=service_client,
    )
    await asyncio.to_thread(worker._reclaim_queued_storage)
    assert not (await dispatcher.system_store.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
    with pytest.raises(ClientNotFoundError):
        await dispatcher.files.get_fileset(name=created.fileset, workspace=DEFAULT_WORKSPACE)


@pytest.mark.asyncio
@pytest.mark.parametrize("uncertain", [False, True])
async def test_non_idempotent_allocation_is_not_replayed(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest, uncertain: bool
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    requests = 0

    def fail_transport(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if uncertain:
            raise httpx.ReadTimeout("allocation outcome unknown", request=request)
        raise httpx.ConnectError("connection not established", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail_transport)) as transport:
        dispatcher.files = AsyncFilesClient(
            base_url="http://files.test", http_client=transport, retry=RetryPolicy(max_retries=3)
        )
        with pytest.raises(NemoTransportError):
            await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
        assert requests == 1
        assert dispatcher.files.retry is not None
        assert dispatcher.files.retry.max_retries == 3
    owner = (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data[0]
    assert owner.fileset_allocation_uncertain == uncertain
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    if uncertain:
        with pytest.raises(JobStorageReclaimPendingError, match="awaiting cleanup"):
            await dispatcher_for(context).create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    else:
        assert not (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data
        await dispatcher_for(context).create_job(sample_platform_job_request, DEFAULT_WORKSPACE)


@pytest.mark.asyncio
async def test_abandoned_job_before_allocation_can_release_its_name(
    cleanup_platform: ClientContext, sample_platform_job_request: CreateHelixJobRequest
) -> None:
    context = cleanup_platform
    await context.entity_client.create(
        HelixJob(
            name="test-job",
            workspace=DEFAULT_WORKSPACE,
            source=sample_platform_job_request.source,
            spec=sample_platform_job_request.spec,
            platform_spec=sample_platform_job_request.platform_spec,
            fileset="job-fileset-test-job",
            creation_token="lost-job-response",
            creation_pending=True,
            fileset_creation_pending=True,
            fileset_creation_started=False,
        )
    )
    worker = reconciler_for(context)
    with patch("nhx.core.jobs.controllers.reconciler.storage_window_expired", return_value=True):
        await asyncio.to_thread(worker._discover_failed_creations)
    await asyncio.to_thread(worker._reclaim_queued_storage)
    assert not (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data
    assert not (await context.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
    await dispatcher_for(context).create_job(sample_platform_job_request, DEFAULT_WORKSPACE)


@pytest.mark.asyncio
async def test_expired_delete_cannot_remove_recreated_fileset_data(cleanup_platform: ClientContext) -> None:
    from datetime import UTC, datetime, timedelta
    from pathlib import Path

    from nhx.common.files.storage_config import LocalStorageConfig
    from nhx.core.files.entities import FileLock

    files = AsyncFilesClient.from_client(cleanup_platform.async_client)
    owned = (
        await files.create_fileset(workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name="reuse-data"))
    ).data()
    assert isinstance(owned.storage, LocalStorageConfig)
    old_root = Path(owned.storage.path)
    old_root.mkdir(parents=True, exist_ok=True)
    (old_root / "old.txt").write_text("old data")
    entered, finish = threading.Event(), threading.Event()
    delete_all = LocalStorageImpl.delete_all
    calls = 0

    async def pause_first(storage: LocalStorageImpl) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            if not await asyncio.to_thread(finish.wait, 10):
                raise TimeoutError("old deletion was not released")
        await delete_all(storage)

    async def delete_first():
        return await files.delete_fileset_by_id(fileset_id=owned.id, workspace=DEFAULT_WORKSPACE)

    with patch.object(LocalStorageImpl, "delete_all", pause_first):
        first = asyncio.create_task(delete_first())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            store = cleanup_platform.entity_client.as_service("files", internal=True)
            locks = (await store.list(FileLock, workspace=SYSTEM_WORKSPACE)).data
            held = next(lock for lock in locks if lock.path == f"fileset-delete:{DEFAULT_WORKSPACE}:{owned.id}")
            held.acquired_at = datetime.now(UTC) - timedelta(hours=1)
            await store.update(held)
            await files.delete_fileset_by_id(fileset_id=owned.id, workspace=DEFAULT_WORKSPACE)
            replacement = (
                await files.create_fileset(workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name=owned.name))
            ).data()
            assert isinstance(replacement.storage, LocalStorageConfig)
            new_root = Path(replacement.storage.path)
            assert new_root != old_root
            assert not new_root.is_relative_to(old_root)
            new_root.mkdir(parents=True, exist_ok=True)
            marker = new_root / "new.txt"
            marker.write_text("replacement data")
        finally:
            finish.set()
            await first
    assert marker.read_text() == "replacement data"
    assert (
        await files.get_fileset_by_id(fileset_id=replacement.id, workspace=DEFAULT_WORKSPACE)
    ).data().id == replacement.id


@pytest.mark.asyncio
async def test_legacy_name_without_ownership_proof_is_retained(cleanup_platform: ClientContext) -> None:
    files = AsyncFilesClient.from_client(cleanup_platform.async_client)
    existing = (
        await files.create_fileset(workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name="legacy-unproven"))
    ).data()
    record = HelixJobStorageReclaim(
        name=HelixJobStorageReclaim.name_for_job("old-job"),
        workspace=SYSTEM_WORKSPACE,
        job_id="old-job",
        job_workspace=DEFAULT_WORKSPACE,
        job_name="old-job",
        fileset_name=existing.name,
    )
    await cleanup_platform.entity_client.create(record)
    await asyncio.to_thread(reconciler_for(cleanup_platform)._reclaim_queued_storage)
    assert (await files.get_fileset_by_id(fileset_id=existing.id, workspace=DEFAULT_WORKSPACE)).data().id == existing.id
    retained = (await cleanup_platform.entity_client.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
    assert len(retained) == 1
    assert retained[0].fileset_name == existing.name


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_creation", [False, True])
async def test_cleanup_receipt_survives_editable_owner_metadata(
    cleanup_platform: ClientContext,
    sample_platform_job_request: CreateHelixJobRequest,
    failed_creation: bool,
) -> None:
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    if failed_creation:
        with patch.object(dispatcher, "_start_attempt", side_effect=RuntimeError("setup interrupted")):
            with pytest.raises(RuntimeError, match="setup interrupted"):
                await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
        owner = (await context.entity_client.list(HelixJob, workspace=DEFAULT_WORKSPACE)).data[0]
    else:
        created = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
        owner = await context.entity_client.get_by_id(HelixJob, created.id)
        await dispatcher.cancel_job(created.id, DEFAULT_WORKSPACE)
        await dispatcher.delete_job(created.id, DEFAULT_WORKSPACE)
    assert owner.fileset_id is not None
    receipt = (await dispatcher.system_store.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data[0]
    assert receipt.fileset_id == owner.fileset_id
    edited = (
        await dispatcher.files.update_fileset(
            workspace=DEFAULT_WORKSPACE,
            name=owner.fileset,
            body=UpdateFilesetRequest(custom_fields={"user_note": "edited"}),
        )
    ).data()
    assert "job_id" not in edited.custom_fields
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    with pytest.raises(ClientNotFoundError):
        await dispatcher.files.get_fileset_by_id(fileset_id=owner.fileset_id, workspace=DEFAULT_WORKSPACE)
    assert not (await dispatcher.system_store.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data


@pytest.fixture(params=["sqlite", "postgresql"])
def polling_platform(request: pytest.FixtureRequest) -> Iterator[ClientContext]:
    """Use a fresh database per SQL dialect without adding infrastructure to unit tests."""
    if request.param == "sqlite":
        with create_test_client(FilesService, client_type=ClientContext, projects=["default/test-project"]) as context:
            yield context
        return
    database_url = os.environ.get("P123_TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("Set P123_TEST_POSTGRES_URL for the isolated PostgreSQL polling regression")
    admin_url = make_url(database_url).set(drivername="postgresql")
    database = f"step_polling_{uuid4().hex}"
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql(f'CREATE DATABASE "{database}"')
        isolated_url = admin_url.set(database=database).render_as_string(hide_password=False)
        config = EntitiesConfig(database_config=DatabaseConfig(url=isolated_url))
        with create_test_client(
            FilesService,
            client_type=ClientContext,
            projects=["default/test-project"],
            service_configs={EntitiesService: config},
        ) as context:
            yield context
    finally:
        with engine.connect() as connection:
            connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("details", "expected_count"),
    [
        ({}, 1),
        ({JOB_CREATION_PENDING_KEY: None}, 1),
        ({JOB_CREATION_PENDING_KEY: False}, 1),
        ({JOB_CREATION_PENDING_KEY: True}, 0),
    ],
)
async def test_step_polling_includes_released_and_legacy_markers(
    polling_platform: ClientContext,
    sample_platform_job_request: CreateHelixJobRequest,
    details: dict[str, bool | None],
    expected_count: int,
) -> None:
    """Poll released, pending and legacy markers through actual database filters."""
    dispatcher = dispatcher_for(polling_platform)
    created = await dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)
    steps = (
        await dispatcher.store.list(
            HelixJobStep, workspace=DEFAULT_WORKSPACE, filter_obj={"attempt_id": created.attempt_id}
        )
    ).data
    assert len(steps) == 1
    step = steps[0]
    step.status_details = details
    await dispatcher.store.update(step)
    visible, total = await dispatcher.list_steps(
        filter=HelixJobStepsListFilter(),
        sort=HelixJobSortField.CREATED_AT_ASC,
        limit=1,
        offset=0,
        workspace=DEFAULT_WORKSPACE,
    )
    assert len(visible) == total == expected_count
    if visible:
        assert visible[0].id == step.id


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_receipt_name", [False, True])
async def test_delete_retry_keeps_queued_identity_after_job_output_changes(
    cleanup_platform: ClientContext,
    sample_platform_job_request: CreateHelixJobRequest,
    missing_receipt_name: bool,
) -> None:
    """Retain queued allocation authority through edited outputs and Files retries."""
    context = cleanup_platform
    dispatcher = dispatcher_for(context)
    original = sample_platform_job_request.platform_spec
    step = original.steps[0].model_copy(
        update={"environment": [HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/storage")]}
    )
    request = sample_platform_job_request.model_copy(
        update={"platform_spec": original.model_copy(update={"steps": [step]})}
    )
    created = await dispatcher.create_job(request, DEFAULT_WORKSPACE)
    await dispatcher.cancel_job(created.id, DEFAULT_WORKSPACE)
    owner = await dispatcher.store.get_by_id(HelixJob, created.id)
    owned_fileset_id = owner.fileset_id
    assert owned_fileset_id is not None
    with patch.object(dispatcher.store, "delete_by_id", side_effect=RuntimeError("record deletion interrupted")):
        with pytest.raises(RuntimeError, match="record deletion interrupted"):
            await dispatcher.delete_job(created.id, DEFAULT_WORKSPACE)
    caller_fileset = (
        await dispatcher.files.create_fileset(
            workspace=DEFAULT_WORKSPACE, body=CreateFilesetRequest(name="caller-output")
        )
    ).data()
    owner.fileset = caller_fileset.name
    owner.fileset_id = None
    owner.output_location = caller_fileset.name
    await dispatcher.store.update(owner)
    await dispatcher.delete_job(created.id, DEFAULT_WORKSPACE)
    queue = (await dispatcher.system_store.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
    assert len(queue) == 1
    receipt = queue[0]
    assert receipt.fileset_id == owned_fileset_id
    assert receipt.fileset_name == created.fileset
    if missing_receipt_name:
        receipt.fileset_name = None
        await dispatcher.system_store.update(receipt)
    worker = reconciler_for(context)
    with patch.object(
        worker._files, "delete_fileset_by_id", side_effect=NemoTransportError(httpx.ReadTimeout("retry Files"))
    ):
        await asyncio.to_thread(worker._reclaim_queued_storage)
    retained = (await dispatcher.system_store.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
    assert len(retained) == 1
    assert retained[0].fileset_id == owned_fileset_id
    await asyncio.to_thread(reconciler_for(context)._reclaim_queued_storage)
    assert not (await dispatcher.system_store.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE)).data
    with pytest.raises(ClientNotFoundError):
        await dispatcher.files.get_fileset_by_id(fileset_id=owned_fileset_id, workspace=DEFAULT_WORKSPACE)
    assert (
        await dispatcher.files.get_fileset_by_id(fileset_id=caller_fileset.id, workspace=DEFAULT_WORKSPACE)
    ).data().id == caller_fileset.id
