# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Jobs-owned events through the ordinary Entity Store HTTP API."""

import asyncio
import os
import threading
import time
from collections.abc import Generator
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pytest
from fastapi import FastAPI
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.events import (
    ExpiredJobEventCursor,
    JobEventCursor,
    JobEventFilters,
    JobEventPage,
    JobEventRequest,
    JobEventType,
)
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.workspaces.client import AsyncWorkspacesClient, WorkspacesClient
from nhx.common.config import DatabaseConfig, ServiceConfig
from nhx.common.entities import SYSTEM_WORKSPACE, SyncEntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.service import EntitiesService
from nhx.core.jobs.api.dependencies import dep_system_entity_client
from nhx.core.jobs.api.v2.jobs.endpoints import router
from nhx.core.jobs.app.dispatcher import JobDispatcher
from nhx.core.jobs.app.event_capture import (
    JobEventRecorder,
    LifecycleImage,
    LifecycleResource,
    SyncJobEventRecorder,
    image,
)
from nhx.core.jobs.app.event_stream import (
    RETENTION_SECONDS,
    JobEventStream,
    LifecycleChange,
    StoredJobEvent,
    SyncJobEventStream,
    reservation,
)
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt, HelixJobStep, HelixJobTask
from nhx.testing import ClientContext, create_test_client
from psycopg2 import connect, sql

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(params=["sqlite", "postgresql"])
def event_context(request: pytest.FixtureRequest) -> Generator[ClientContext]:
    configs: dict[type[object], ServiceConfig] = {}
    database = "events_" + uuid4().hex
    admin = None
    if request.param == "postgresql":
        # Accept PostgreSQL authority URLs and Unix-socket host/port query parameters.
        configured = os.environ.get("NHX_TEST_JOB_EVENTS_POSTGRES_URL")
        if configured is None:
            pytest.skip("PostgreSQL integration URL is not configured")
        url = urlsplit(configured)
        admin = connect(urlunsplit(url._replace(scheme="postgresql")))
        admin.autocommit = True
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        configs[EntitiesService] = EntitiesConfig(
            database_config=DatabaseConfig(
                url=urlunsplit(url._replace(scheme="postgresql+asyncpg", path="/" + database))
            )
        )
    try:
        with create_test_client(
            client_type=ClientContext, workspaces=["default", "other"], service_configs=configs
        ) as ctx:
            app = ctx.test_client.app
            assert isinstance(app, FastAPI)
            app.include_router(router, prefix="/apis/jobs")
            app.dependency_overrides[dep_system_entity_client] = lambda: ctx.entity_client
            yield ctx
    finally:
        if admin is not None:
            with admin.cursor() as cursor:
                cursor.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))
            admin.close()


def change(
    index: int = 0, workspace: str = "workspace-id", kind: JobEventType = JobEventType.JOB_CREATED
) -> LifecycleChange:
    return LifecycleChange(
        event_type=kind,
        workspace_id=workspace,
        job_id="job-" + str(index),
        attempt_id=None,
        step_id=None,
        previous_state=None,
        new_state="created",
        occurred_at=datetime.now(timezone.utc),
        snapshot={"name": "test"},
    )


async def test_generic_entities_page_filters_and_reverse(event_context: ClientContext):
    stream = JobEventStream(event_context.entity_client)
    for index in range(5):
        await stream.append(change(index, workspace="left" if index % 2 == 0 else "right"))
    page = await stream.read(JobEventRequest(limit=2))
    assert [event.id for event in page.data] == ["1", "2"]
    second = await stream.read(JobEventRequest(cursor=page.next_cursor, limit=2))
    assert [event.id for event in second.data] == ["3", "4"]
    assert second.prev_cursor is not None
    reverse = await stream.read(JobEventRequest(cursor=second.prev_cursor, limit=2))
    assert [event.id for event in reverse.data] == ["2", "1"]
    filtered = await stream.read(JobEventRequest(filters=JobEventFilters(workspace_id="left")))
    assert [event.id for event in filtered.data] == ["1", "3", "5"]
    replaced = await stream.read(
        JobEventRequest(cursor=page.next_cursor, filters=JobEventFilters(workspace_id="right"))
    )
    assert [event.id for event in replaced.data] == ["4"]
    assert JobEventCursor.decode(replaced.next_cursor).filters.workspace_id == "right"
    empty = await stream.read(JobEventRequest(filters=JobEventFilters(event_types=[JobEventType.JOB_DELETED])))
    assert not empty.data and JobEventCursor.decode(empty.next_cursor).position == "5"
    assert JobEventCursor.decode(empty.next_cursor).model_dump().keys() == {"v", "position", "direction", "filters"}
    reverse_default = await stream.read(JobEventRequest(direction="backward"))
    assert [event.id for event in reverse_default.data] == ["5", "4", "3", "2", "1"]


async def test_replica_append_order_and_abandoned_slot_fences_late_writer(event_context: ClientContext):
    store = event_context.entity_client
    first, second = JobEventStream(store), JobEventStream(store)
    result = await asyncio.gather(first.append(change(1)), second.append(change(2)))
    assert all(event is not None for event in result)
    assert [event.id for event in (await first.read(JobEventRequest())).data] == ["1", "2"]
    stale = await store.update(reservation(await first.state()))
    await store.update(stale.model_copy(update={"pending_until": 0}))
    recovered = await second.append(change(4))
    assert recovered is not None and recovered.id == "4"
    late = await first.complete(stale, change(3).stored(stale))
    assert late.event is None
    assert [event.id for event in (await second.read(JobEventRequest())).data] == ["1", "2", "4"]


async def test_id_allocation_failure_does_not_publish_missing_lower_id(event_context: ClientContext):
    store = event_context.entity_client
    stream = JobEventStream(store)
    claimed = await store.update(reservation(await stream.state()))
    assert not (await stream.read(JobEventRequest())).data
    await store.update(claimed.model_copy(update={"pending_until": 0}))
    assert not (await stream.read(JobEventRequest())).data
    assert JobEventCursor.decode((await stream.read(JobEventRequest())).next_cursor).position == "1"
    saved = await stream.append(change())
    assert saved is not None and saved.id == "2"


async def test_retention_cleanup_restart_and_source_independence(event_context: ClientContext):
    stream = JobEventStream(event_context.entity_client)
    with patch("nhx.core.jobs.app.event_stream.wall_time", return_value=time.time() - RETENTION_SECONDS - 10):
        event = await stream.append(change())
    assert event is not None
    latest = (await stream.read(JobEventRequest(start="latest"))).next_cursor
    with pytest.raises(ExpiredJobEventCursor):
        await stream.read(JobEventRequest(cursor=JobEventCursor(position="0").encode()))
    assert not (await stream.read(JobEventRequest())).data
    sync_store = SyncEntityClient(EntitiesClient.from_client(event_context.client))
    progress = Mock()
    assert SyncJobEventStream(sync_store).cleanup(on_progress=progress) == 1
    assert progress.call_count == 2
    restarted = JobEventStream(event_context.entity_client)
    fresh = await restarted.append(change(2))
    assert fresh is not None and fresh.id == "2"
    assert [event.id for event in (await restarted.read(JobEventRequest(cursor=latest))).data] == ["2"]
    assert (await event_context.entity_client.list(StoredJobEvent, workspace=SYSTEM_WORKSPACE)).data[0]._parent is None


async def test_jobs_capture_uses_workspace_ids_and_ignores_noops(event_context: ClientContext):
    store = event_context.entity_client
    recorder = JobEventRecorder(store, AsyncWorkspacesClient.from_client(event_context.async_client))
    job = await store.create(
        HelixJob(
            name="capture",
            workspace="default",
            source="test",
            fileset="files",
            platform_spec=TestConstants.PLATFORM_SPEC,
            status="created",
        )
    )
    await recorder.created(job)
    before = await recorder.before(job)
    saved = await store.update(job.model_copy(update={"description": "metadata"}))
    await recorder.updated(before, saved)
    before = await recorder.before(saved)
    saved = await store.update(saved.model_copy(update={"status": "active"}))
    await recorder.updated(before, saved)
    workspace = WorkspacesClient.from_client(event_context.client).get_workspace(name="default").data()
    events = (await recorder.stream.read(JobEventRequest())).data
    assert [event.event_type for event in events] == [JobEventType.JOB_CREATED, JobEventType.JOB_STATE_CHANGED]
    assert all(event.workspace_id == workspace.id for event in events)
    assert events[-1].previous_state == "created" and events[-1].new_state == "active"


async def test_capture_failure_is_best_effort_and_does_not_undo_source(event_context: ClientContext):
    store = event_context.entity_client
    recorder = JobEventRecorder(store, AsyncWorkspacesClient.from_client(event_context.async_client))
    job = await store.create(
        HelixJob(
            name="drop", workspace="default", source="test", fileset="files", platform_spec=TestConstants.PLATFORM_SPEC
        )
    )
    with patch.object(recorder.stream, "append", side_effect=RuntimeError("unavailable")):
        await recorder.created(job)
    assert (await store.get_by_id(HelixJob, job.id)).id == job.id
    assert not (await recorder.stream.read(JobEventRequest())).data


async def test_controller_capture_and_deletion_survive_source_removal(event_context: ClientContext):
    store = event_context.entity_client
    job = await store.create(
        HelixJob(
            name="delete",
            workspace="default",
            source="test",
            fileset="files",
            platform_spec=TestConstants.PLATFORM_SPEC,
        )
    )
    attempt = await store.create(
        HelixJobAttempt(name="attempt", workspace="default", job=job.id, platform_spec=TestConstants.PLATFORM_SPEC)
    )
    step = await store.create(HelixJobStep(name="step", workspace="default", attempt_id=attempt.id))
    sync_store = SyncEntityClient(EntitiesClient.from_client(event_context.client))
    recorder = SyncJobEventRecorder(sync_store, WorkspacesClient.from_client(event_context.client))
    recorder.created(step)
    images = recorder.deletion_images(job, None)
    await store.delete_by_id(HelixJob, job.id)
    recorder.deleted(images)
    events = (await JobEventStream(store).read(JobEventRequest())).data
    assert [event.event_type for event in events] == [
        JobEventType.STEP_CREATED,
        JobEventType.STEP_DELETED,
        JobEventType.ATTEMPT_DELETED,
        JobEventType.JOB_DELETED,
    ]
    assert events[-1].snapshot["source"] == "test"


async def test_post_only_typed_api_cursor_validation_and_polling(event_context: ClientContext):
    jobs = AsyncJobsClient.from_client(event_context.async_client)
    stream = JobEventStream(event_context.entity_client)
    await stream.append(change())
    page = (await jobs.read_job_events(body=JobEventRequest())).data()
    assert [event.id for event in page.data] == ["1"]
    assert not (await jobs.read_job_events(body=JobEventRequest(cursor=page.next_cursor))).data().data
    assert event_context.test_client.get("/apis/jobs/v2/jobs/events").status_code == 405
    assert event_context.test_client.post("/apis/jobs/v2/jobs/events", json={"cursor": "broken"}).status_code == 400
    first_empty = threading.Event()
    read = JobEventStream.read

    async def observe_read(self: JobEventStream, request: JobEventRequest) -> JobEventPage:
        result = await read(self, request)
        if not result.data:
            first_empty.set()
        return result

    with patch.object(JobEventStream, "read", observe_read):
        waiting = asyncio.ensure_future(
            jobs.read_job_events(body=JobEventRequest(cursor=page.next_cursor, wait_seconds=5))
        )
        try:
            assert await asyncio.to_thread(first_empty.wait, 3)
            assert not waiting.done()
            await stream.append(change(2))
            assert [event.id for event in (await asyncio.wait_for(waiting, 5)).data().data] == ["2"]
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)


async def test_large_ids_remain_decimal_strings(event_context: ClientContext):
    stream = JobEventStream(event_context.entity_client)
    state = await stream.state()
    await event_context.entity_client.update(state.model_copy(update={"position": 9007199254740992}))
    event = await stream.append(change())
    assert event is not None and event.id == "9007199254740993"
    assert (await stream.read(JobEventRequest())).data[0].id == event.id


async def test_dispatcher_records_transitions_and_tolerates_dropped_events(event_context: ClientContext):
    store = event_context.entity_client
    job = await store.create(
        HelixJob(
            name="dispatch",
            workspace="default",
            source="test",
            fileset="files",
            platform_spec=TestConstants.PLATFORM_SPEC,
            status=HelixJobStatus.CREATED,
        )
    )
    attempt = await store.create(
        HelixJobAttempt(name="attempt", workspace="default", job=job.id, platform_spec=TestConstants.PLATFORM_SPEC)
    )
    step = await store.create(HelixJobStep(name="basic", workspace="default", attempt_id=attempt.id))
    await store.update(job.model_copy(update={"current_attempt_id": attempt.id}))
    recorder = JobEventRecorder(store, AsyncWorkspacesClient.from_client(event_context.async_client))
    dispatcher = JobDispatcher(
        store=store,
        system_store=store,
        events=recorder,
        files=AsyncFilesClient.from_client(event_context.async_client),
        secrets=AsyncSecretsClient.from_client(event_context.async_client),
    )
    assert not (await recorder.stream.read(JobEventRequest())).data
    saved, _ = await dispatcher.update_job_status_from_step(step, HelixJobStatus.PENDING)
    events = (await recorder.stream.read(JobEventRequest())).data
    assert {event.event_type for event in events} == {
        JobEventType.STEP_STATE_CHANGED,
        JobEventType.ATTEMPT_STATE_CHANGED,
        JobEventType.JOB_STATE_CHANGED,
    }
    assert [event.id for event in events] == ["1", "2", "3"]
    task = await store.create(HelixJobTask(name="task", workspace="default", step_id=step.id))
    await recorder.created(task)
    with patch.object(recorder.stream, "append", side_effect=RuntimeError("unavailable")):
        await dispatcher.update_job_status_from_step(saved, HelixJobStatus.ACTIVE)
    assert (await store.get_by_id(HelixJobStep, step.id)).status == HelixJobStatus.ACTIVE
    assert len((await recorder.stream.read(JobEventRequest())).data) == 3


@pytest.mark.parametrize("pending", [True, False])
async def test_noncurrent_and_pending_job_projection_do_not_emit_job_changes(
    event_context: ClientContext, pending: bool
):
    store = event_context.entity_client
    job = await store.create(
        HelixJob(
            name="projection",
            workspace="default",
            source="test",
            fileset="files",
            platform_spec=TestConstants.PLATFORM_SPEC,
            status=HelixJobStatus.DELETION_PENDING if pending else HelixJobStatus.CREATED,
        )
    )
    current = await store.create(
        HelixJobAttempt(name="current", workspace="default", job=job.id, platform_spec=TestConstants.PLATFORM_SPEC)
    )
    other = await store.create(
        HelixJobAttempt(name="other", workspace="default", job=job.id, platform_spec=TestConstants.PLATFORM_SPEC)
    )
    await store.update(job.model_copy(update={"current_attempt_id": current.id}))
    target = current if pending else other
    recorder = JobEventRecorder(store, AsyncWorkspacesClient.from_client(event_context.async_client))
    before = await recorder.before(target)
    if not pending:
        await store.update(current.model_copy(update={"status": HelixJobStatus.PENDING}))
    saved = await store.update(target.model_copy(update={"status": HelixJobStatus.ACTIVE}))
    await recorder.updated(before, saved)
    assert [event.event_type for event in (await recorder.stream.read(JobEventRequest())).data] == [
        JobEventType.ATTEMPT_STATE_CHANGED
    ]


@pytest.mark.parametrize("recoverer", ["reader", "async_writer", "sync_writer"])
async def test_persisted_slot_is_published_before_lease_expiry(event_context: ClientContext, recoverer: str):
    store = event_context.entity_client
    stream = JobEventStream(store)
    claimed = await store.update(reservation(await stream.state()))
    await store.create(change(1).stored(claimed))
    if recoverer == "reader":
        assert [event.id for event in (await stream.read(JobEventRequest())).data] == ["1"]
    elif recoverer == "async_writer":
        saved = await stream.append(change(2))
        assert saved is not None and saved.id == "2"
    else:
        sync_store = SyncEntityClient(EntitiesClient.from_client(event_context.client))
        saved = SyncJobEventStream(sync_store).append(change(2))
        assert saved is not None and saved.id == "2"
    state = await stream.state()
    assert state.pending_until is None
    assert state.position == (1 if recoverer == "reader" else 2)
    assert [event.id for event in (await stream.read(JobEventRequest())).data] == (
        ["1"] if recoverer == "reader" else ["1", "2"]
    )


async def test_current_attempt_projection_uses_the_saved_transition(event_context: ClientContext):
    store = event_context.entity_client
    job = await store.create(
        HelixJob(
            name="projection-race",
            workspace="default",
            source="test",
            fileset="files",
            platform_spec=TestConstants.PLATFORM_SPEC,
        )
    )
    attempt = await store.create(
        HelixJobAttempt(
            name="attempt",
            workspace="default",
            job=job.id,
            platform_spec=TestConstants.PLATFORM_SPEC,
        )
    )
    await store.update(job.model_copy(update={"current_attempt_id": attempt.id}))
    recorder = JobEventRecorder(store, AsyncWorkspacesClient.from_client(event_context.async_client))
    before = await recorder.before(attempt)
    saved = await store.update(attempt.model_copy(update={"status": HelixJobStatus.PENDING}))
    await store.update(saved.model_copy(update={"status": HelixJobStatus.ACTIVE}))
    await recorder.updated(before, saved)
    events = (await recorder.stream.read(JobEventRequest())).data
    assert [event.event_type for event in events] == [
        JobEventType.ATTEMPT_STATE_CHANGED,
        JobEventType.JOB_STATE_CHANGED,
    ]
    assert all(event.previous_state == "created" and event.new_state == "pending" for event in events)
    assert events[0].occurred_at == events[1].occurred_at


@pytest.mark.parametrize("interruption", ["deadline", "shutdown", "already_stopped"])
async def test_deletion_capture_stops_and_preserves_partial_images(event_context: ClientContext, interruption: str):
    store = event_context.entity_client
    job = await store.create(
        HelixJob(
            name="bounded-delete",
            workspace="default",
            source="test",
            fileset="files",
            status=HelixJobStatus.DELETION_PENDING,
            platform_spec=TestConstants.PLATFORM_SPEC,
        )
    )
    attempt = await store.create(
        HelixJobAttempt(
            name="attempt",
            workspace="default",
            job=job.id,
            platform_spec=TestConstants.PLATFORM_SPEC,
        )
    )
    for name in ["first", "second"]:
        await store.create(HelixJobStep(name=name, workspace="default", attempt_id=attempt.id))
    sync_store = SyncEntityClient(EntitiesClient.from_client(event_context.client))
    recorder = SyncJobEventRecorder(sync_store, WorkspacesClient.from_client(event_context.client))
    stop = threading.Event()
    clock = [0.0]
    if interruption == "already_stopped":
        stop.set()

    def interrupt(
        resource: LifecycleResource, job: HelixJob, workspace_id: str, job_state: HelixJobStatus
    ) -> LifecycleImage:
        captured = image(resource, job, workspace_id, job_state)
        if isinstance(resource, HelixJobStep):
            if interruption == "deadline":
                clock[0] = 3.0
            else:
                stop.set()
        return captured

    with patch("nhx.core.jobs.app.event_capture.monotonic", side_effect=lambda: clock[0]):
        with patch("nhx.core.jobs.app.event_capture.image", side_effect=interrupt):
            captured = recorder.deletion_images(job, stop)
    assert len(captured) == (0 if interruption == "already_stopped" else 1)
    assert all(item.kind == "step" for item in captured)
    await store.delete_by_id(HelixJob, job.id)
    recorder.deleted(captured)
    assert len((await JobEventStream(store).read(JobEventRequest())).data) == len(captured)
