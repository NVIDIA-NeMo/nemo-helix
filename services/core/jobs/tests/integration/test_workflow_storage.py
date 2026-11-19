# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow storage ownership during normal lifecycle requests."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.workflows import WorkflowNodeState, WorkflowSpec, WorkflowState
from nhx.common.entities.client import EntityBase, EntityClient
from nhx.core.jobs.app import dispatcher as dispatcher_module
from nhx.core.jobs.app.constants import STORAGE_RECLAIMED_AT
from nhx.core.jobs.app.dispatcher import JobDispatcher, JobOperationConflictError
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt, HelixJobStorageReclaim

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def stopped_workflow(
    mock_store: EntityClient, monkeypatch: pytest.MonkeyPatch
) -> tuple[HelixJob, HelixJobAttempt]:
    monkeypatch.setattr(dispatcher_module.jobs_config.storage, "failed_storage_ttl_seconds", 0)
    spec = HelixJobSpec(
        workflow=WorkflowSpec(mode="continuous"),
        steps=[
            HelixJobStepSpec(
                name="work",
                depends_on=[],
                executor=TestConstants.TEST_EXECUTOR,
                environment=[HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/storage")],
            )
        ],
    )
    job = await mock_store.create(
        HelixJob(name="workflow-storage", workspace="default", source="test", fileset="artifacts", platform_spec=spec)
    )
    attempt = await mock_store.create(
        HelixJobAttempt(
            name="attempt-0",
            workspace="default",
            job=job.id,
            platform_spec=spec,
            status=HelixJobStatus.ERROR,
            workflow_state=WorkflowState(
                nodes={"work": WorkflowNodeState(last_outcome="error")},
                stopped_at=datetime.now(timezone.utc) - timedelta(hours=1),
            ),
        )
    )
    job.current_attempt_id = attempt.id
    job.status = attempt.status
    return await mock_store.update(job), attempt


async def test_rerun_can_publish_during_reconciliation_without_losing_storage(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    stopped_workflow: tuple[HelixJob, HelixJobAttempt],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, attempt = stopped_workflow
    entered = asyncio.Event()
    release = asyncio.Event()
    original = dispatcher_module.reconcile_workflow

    async def reconcile(
        store: EntityClient, files: AsyncFilesClient, job: HelixJob, attempt: HelixJobAttempt
    ) -> HelixJobAttempt:
        entered.set()
        await release.wait()
        return await original(store, files, job, attempt)

    monkeypatch.setattr(dispatcher_module, "reconcile_workflow", reconcile)
    controller = asyncio.create_task(
        mock_dispatcher.update_attempt_status_details(job.id, attempt.id, "default", {"workflow_reconcile": True})
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=30)
        replacement = await asyncio.wait_for(mock_dispatcher.rerun_job(job.id, "default"), timeout=30)
        assert replacement is not None and replacement.attempt_id != attempt.id
    finally:
        release.set()
        await controller

    owner = await mock_store.get_by_id(HelixJob, job.id)
    previous = await mock_store.get_by_id(HelixJobAttempt, attempt.id)
    assert owner.current_attempt_id == replacement.attempt_id
    assert STORAGE_RECLAIMED_AT not in previous.status_details
    assert (await mock_store.list(HelixJobStorageReclaim, workspace="system")).data == []


async def test_cleanup_waits_for_rerun_publication_before_checking_ownership(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    stopped_workflow: tuple[HelixJob, HelixJobAttempt],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, attempt = stopped_workflow
    publication_entered = asyncio.Event()
    release_publication = asyncio.Event()
    cleanup_started = asyncio.Event()
    original_update = EntityClient.update

    async def update[EntityT: EntityBase](
        self: EntityClient, entity: EntityT, *, original_name: str | None = None
    ) -> EntityT:
        if isinstance(entity, HelixJob) and entity.id == job.id and entity.current_attempt_id != attempt.id:
            publication_entered.set()
            await release_publication.wait()
        return await original_update(self, entity, original_name=original_name)

    monkeypatch.setattr(EntityClient, "update", update)
    rerun = asyncio.create_task(mock_dispatcher.rerun_job(job.id, "default"))

    async def cleanup() -> HelixJobAttempt:
        cleanup_started.set()
        return await mock_dispatcher._retain_workflow_storage(mock_store, job, attempt)

    try:
        await asyncio.wait_for(publication_entered.wait(), timeout=30)
        retention = asyncio.create_task(cleanup())
        await asyncio.wait_for(cleanup_started.wait(), timeout=30)
    finally:
        release_publication.set()
    replacement = await asyncio.wait_for(rerun, timeout=30)
    await asyncio.wait_for(retention, timeout=30)

    assert replacement is not None
    assert (await mock_store.get_by_id(HelixJob, job.id)).current_attempt_id == replacement.attempt_id
    assert STORAGE_RECLAIMED_AT not in (await mock_store.get_by_id(HelixJobAttempt, attempt.id)).status_details
    assert (await mock_store.list(HelixJobStorageReclaim, workspace="system")).data == []


async def test_cleanup_committed_first_prevents_rerun(
    mock_dispatcher: JobDispatcher,
    mock_store: EntityClient,
    stopped_workflow: tuple[HelixJob, HelixJobAttempt],
) -> None:
    job, attempt = stopped_workflow
    retained = await mock_dispatcher._retain_workflow_storage(mock_store, job, attempt)
    assert retained.status_details[STORAGE_RECLAIMED_AT]
    records = (await mock_store.list(HelixJobStorageReclaim, workspace="system")).data
    assert len(records) == 1 and records[0].job_id == job.id

    with pytest.raises(JobOperationConflictError, match="storage has been reclaimed"):
        await mock_dispatcher.rerun_job(job.id, "default")
    assert (await mock_store.get_by_id(HelixJob, job.id)).current_attempt_id == attempt.id
    assert len((await mock_store.list(HelixJobAttempt, workspace="default")).data) == 1
