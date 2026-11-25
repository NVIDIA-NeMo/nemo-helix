# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-only lifecycle publication, terminal retries, and stream traversal."""

import asyncio
import os
import threading
import time
from collections.abc import Generator
from datetime import datetime, timezone
from itertools import count
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
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nhx.common.api.parsed_filter import ParsedFilter
from nhx.common.config import DatabaseConfig, ServiceConfig
from nhx.common.entities import SYSTEM_WORKSPACE, EntityNotFoundError, SyncEntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.service import EntitiesService
from nhx.core.jobs.api.dependencies import dep_system_entity_client
from nhx.core.jobs.api.v2.jobs.endpoints import router
from nhx.core.jobs.api.v2.jobs.schemas import CreateHelixJobRequest
from nhx.core.jobs.app.dispatcher import JobDispatcher, JobPublicationTimeoutError
from nhx.core.jobs.app.event_publisher import JobEventPublisher
from nhx.core.jobs.app.event_stream import (
    RETENTION_SECONDS,
    JobEventSequence,
    JobEventStream,
    LifecycleChange,
    StoredJobEvent,
    SyncJobEventStream,
)
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.reconciler import JobReconciler
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt, HelixJobStep
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


def writer(ctx: ClientContext) -> SyncJobEventStream:
    return SyncJobEventStream(SyncEntityClient(EntitiesClient.from_client(ctx.client)))


def publisher(ctx: ClientContext) -> JobEventPublisher:
    return JobEventPublisher(
        SyncEntityClient(EntitiesClient.from_client(ctx.client)), WorkspacesClient.from_client(ctx.client)
    )


def dispatcher(ctx: ClientContext) -> JobDispatcher:
    return JobDispatcher(
        store=ctx.entity_client,
        system_store=ctx.entity_client,
        files=AsyncFilesClient.from_client(ctx.async_client),
        secrets=AsyncSecretsClient.from_client(ctx.async_client),
    )


async def graph(
    ctx: ClientContext, state: HelixJobStatus = HelixJobStatus.ACTIVE
) -> tuple[HelixJob, HelixJobAttempt, HelixJobStep]:
    store = ctx.entity_client
    job = await store.create(
        HelixJob(
            name="job",
            workspace="default",
            source="test",
            fileset="files",
            platform_spec=TestConstants.PLATFORM_SPEC,
            status=state,
        )
    )
    attempt = await store.create(
        HelixJobAttempt(
            name="attempt",
            workspace="default",
            job=job.id,
            status=state,
            platform_spec=job.platform_spec,
        )
    )
    step = await store.create(HelixJobStep(name="basic", workspace="default", attempt_id=attempt.id, status=state))
    job = await store.update(job.model_copy(update={"current_attempt_id": attempt.id}))
    return job, attempt, step


def cleanup_controller(ctx: ClientContext, events: JobEventPublisher) -> JobReconciler:
    return JobReconciler(BackendRegistry({}), ctx.client, events=events, system_entity_client=events.store)


class SimulatedCrash(BaseException):
    pass


async def test_paging_filters_reverse_and_backward_default(event_context: ClientContext):
    output = writer(event_context)
    stream = JobEventStream(event_context.entity_client)
    for index in range(5):
        output.append(change(index, workspace="left" if index % 2 == 0 else "right"))
    first = await stream.read(JobEventRequest(limit=2))
    assert [event.id for event in first.data] == ["1", "2"]
    second = await stream.read(JobEventRequest(cursor=first.next_cursor, limit=2))
    assert [event.id for event in second.data] == ["3", "4"]
    assert second.prev_cursor is not None
    reverse = await stream.read(JobEventRequest(cursor=second.prev_cursor, limit=2))
    assert [event.id for event in reverse.data] == ["2", "1"]
    assert [event.id for event in (await stream.read(JobEventRequest(direction="backward"))).data] == [
        "5",
        "4",
        "3",
        "2",
        "1",
    ]
    replacement = await stream.read(
        JobEventRequest(cursor=first.next_cursor, filters=JobEventFilters(workspace_id="right"))
    )
    assert [event.id for event in replacement.data] == ["4"]
    empty = await stream.read(JobEventRequest(filters=JobEventFilters(event_types=[JobEventType.JOB_DELETED])))
    assert not empty.data and JobEventCursor.decode(empty.next_cursor).position == "5"
    assert JobEventCursor.decode(empty.next_cursor).model_dump().keys() == {"v", "position", "direction", "filters"}


async def test_reads_do_not_create_or_advance_stream_state(event_context: ClientContext):
    stream = JobEventStream(event_context.entity_client)
    assert not (await stream.read(JobEventRequest())).data
    assert not (await event_context.entity_client.list(JobEventSequence, workspace=SYSTEM_WORKSPACE)).data
    output = writer(event_context)
    output.state()
    await event_context.entity_client.create(change().stored(1, time.time()))
    assert not (await stream.read(JobEventRequest())).data
    output.recover()
    assert [event.id for event in (await stream.read(JobEventRequest())).data] == ["1"]


async def test_counter_recovers_after_ambiguous_append_and_empty_retention(event_context: ClientContext):
    output = writer(event_context)
    output.state()
    with patch.object(output.store, "update", side_effect=SimulatedCrash):
        with pytest.raises(SimulatedCrash):
            output.append(change())
    assert not (await JobEventStream(event_context.entity_client).read(JobEventRequest())).data
    restarted = writer(event_context)
    assert restarted.append(change()).id == "2"
    assert [row.id for row in (await JobEventStream(event_context.entity_client).read(JobEventRequest())).data] == [
        "1",
        "2",
    ]
    for row in (await event_context.entity_client.list(StoredJobEvent, workspace=SYSTEM_WORKSPACE)).data:
        await event_context.entity_client.delete(StoredJobEvent, row.name, workspace=SYSTEM_WORKSPACE)
    assert writer(event_context).append(change()).id == "3"


async def test_retention_cleanup_and_expired_cursor(event_context: ClientContext):
    output = writer(event_context)
    with patch("nhx.core.jobs.app.event_stream.wall_time", return_value=time.time() - RETENTION_SECONDS - 10):
        output.append(change())
    stream = JobEventStream(event_context.entity_client)
    assert not (await stream.read(JobEventRequest())).data
    with pytest.raises(ExpiredJobEventCursor):
        await stream.read(JobEventRequest(cursor=JobEventCursor(position="0").encode()))
    progress = Mock()
    assert output.cleanup(on_progress=progress) == 1
    assert progress.call_count == 2
    assert writer(event_context).append(change()).id == "2"


@pytest.mark.parametrize("position", [10**19 - 2, 2**100])
async def test_large_sequence_keys_and_cursors_remain_ordered(event_context: ClientContext, position: int):
    output = writer(event_context)
    state = output.state()
    output.store.update(state.model_copy(update={"position": position}))
    a, b = output.append(change()), output.append(change())
    assert (a.id, b.id) == (str(position + 1), str(position + 2))
    stream = JobEventStream(event_context.entity_client)
    page = await stream.read(JobEventRequest())
    assert [event.id for event in page.data] == [a.id, b.id]
    assert JobEventCursor.decode(page.next_cursor).position == b.id


@pytest.mark.parametrize("terminal", HelixJobStatus.terminals())
async def test_terminal_failures_retry_without_repeating_action(event_context: ClientContext, terminal: HelixJobStatus):
    ctx = event_context
    _, _, step = await graph(ctx)
    await dispatcher(ctx).update_job_status_from_step(step, terminal)
    assert not (await JobEventStream(ctx.entity_client).read(JobEventRequest())).data
    events = publisher(ctx)
    with patch.object(events.stream, "append", side_effect=RuntimeError("unavailable")):
        events.publish_observations(lambda: None, None)
    assert not events.is_healthy
    assert (await ctx.entity_client.get_by_id(HelixJobStep, step.id)).status == terminal
    publisher(ctx).publish_observations(lambda: None, None)
    result = (await JobEventStream(ctx.entity_client).read(JobEventRequest())).data
    assert {row.event_type for row in result} == {
        JobEventType.STEP_STATE_CHANGED,
        JobEventType.ATTEMPT_STATE_CHANGED,
        JobEventType.JOB_STATE_CHANGED,
    }
    assert all(row.new_state == terminal.value for row in result)
    assert [row.id for row in result] == ["1", "2", "3"]


async def test_restart_rebuilds_published_outcomes_from_stream(event_context: ClientContext):
    await graph(event_context, HelixJobStatus.COMPLETED)
    first = publisher(event_context)
    first.publish_observations(lambda: None, None)
    first.publish_observations(lambda: None, None)
    stream = JobEventStream(event_context.entity_client)
    assert len((await stream.read(JobEventRequest())).data) == 3
    publisher(event_context).publish_observations(lambda: None, None)
    rows = (await stream.read(JobEventRequest())).data
    assert [row.id for row in rows] == ["1", "2", "3"]
    assert all(row.new_state == "completed" for row in rows)


async def test_terminal_job_outcome_survives_a_quick_rerun(event_context: ClientContext):
    job, attempt, step = await graph(event_context)
    commands = dispatcher(event_context)
    publisher(event_context).publish_observations(lambda: None, None)
    await commands.update_job_status_from_step(step, HelixJobStatus.ERROR)
    rerun = await commands.rerun_job(job.id, "default")
    assert rerun is not None and rerun.attempt_id != attempt.id
    publisher(event_context).publish_observations(lambda: None, None)
    rows = (await JobEventStream(event_context.entity_client).read(JobEventRequest())).data
    final = [row for row in rows if row.new_state == "error"]
    assert {row.event_type for row in final} == {
        JobEventType.STEP_STATE_CHANGED,
        JobEventType.ATTEMPT_STATE_CHANGED,
        JobEventType.JOB_STATE_CHANGED,
    }
    assert all(row.attempt_id == attempt.id for row in final)


async def test_noncurrent_orphan_attempt_does_not_report_job_failure(event_context: ClientContext):
    job, _, _ = await graph(event_context, HelixJobStatus.CREATED)
    await event_context.entity_client.create(
        HelixJobAttempt(
            name="orphan", workspace="default", job=job.id, status=HelixJobStatus.ERROR, platform_spec=job.platform_spec
        )
    )
    publisher(event_context).publish_observations(lambda: None, None)
    final = [
        row
        for row in (await JobEventStream(event_context.entity_client).read(JobEventRequest())).data
        if row.new_state == "error"
    ]
    assert [row.event_type for row in final] == [JobEventType.ATTEMPT_STATE_CHANGED]


@pytest.mark.parametrize("failure", ["append", "crash_before_delete"])
async def test_cleanup_republishes_then_deletes_with_restart_recovery(event_context: ClientContext, failure: str):
    ctx = event_context
    job, attempt, step = await graph(ctx, HelixJobStatus.COMPLETED)
    await ctx.entity_client.update(job.model_copy(update={"status": HelixJobStatus.DELETION_PENDING}))
    output = publisher(ctx)
    output.publish_observations(lambda: None, None)
    cleaner = cleanup_controller(ctx, output)
    with patch.object(cleaner, "_settle_pending_storage", return_value=True):
        with patch.object(cleaner, "_settle_pending_artifacts", side_effect=lambda job: job):
            if failure == "append":
                with patch.object(output.stream, "append", side_effect=RuntimeError("unavailable")):
                    cleaner._cleanup_pending_jobs()
            else:
                with patch.object(output.store, "delete_by_id", side_effect=SimulatedCrash):
                    with pytest.raises(SimulatedCrash):
                        cleaner._cleanup_pending_jobs()
    assert (await ctx.entity_client.get_by_id(HelixJob, job.id)).status == HelixJobStatus.DELETION_PENDING
    restarted = cleanup_controller(ctx, publisher(ctx))
    with patch.object(restarted, "_settle_pending_storage", return_value=True):
        with patch.object(restarted, "_settle_pending_artifacts", side_effect=lambda job: job):
            for _ in range(10):
                restarted._cleanup_pending_jobs()
    for entity_type, identity in [(HelixJob, job.id), (HelixJobAttempt, attempt.id), (HelixJobStep, step.id)]:
        with pytest.raises(EntityNotFoundError):
            await ctx.entity_client.get_by_id(entity_type, identity)
    rows = (await JobEventStream(ctx.entity_client).read(JobEventRequest())).data
    assert len([row for row in rows if row.new_state == "completed"]) >= 6


async def test_bounded_cleanup_scan_covers_all_terminal_children(event_context: ClientContext):
    ctx = event_context
    job, attempt, _ = await graph(ctx, HelixJobStatus.CANCELLED)
    for n in range(104):
        await ctx.entity_client.create(
            HelixJobStep(
                name="child-" + str(n), workspace="default", attempt_id=attempt.id, status=HelixJobStatus.CANCELLED
            )
        )
    events = publisher(ctx)
    calls = 0
    while not events.flush_job(job, lambda: None, None):
        calls += 1
        assert calls < 100
    assert calls >= 1
    rows = (await JobEventStream(ctx.entity_client).read(JobEventRequest(limit=1000))).data
    assert len([row for row in rows if row.event_type == JobEventType.STEP_STATE_CHANGED]) == 105
    assert all(row.new_state == "cancelled" for row in rows)


async def test_post_only_typed_client_and_arrival_during_poll(event_context: ClientContext):
    ctx = event_context
    output = writer(ctx)
    output.append(change())
    jobs = AsyncJobsClient.from_client(ctx.async_client)
    page = (await jobs.read_job_events(body=JobEventRequest())).data()
    assert [row.id for row in page.data] == ["1"]
    assert ctx.test_client.get("/apis/jobs/v2/jobs/events").status_code == 405
    assert ctx.test_client.post("/apis/jobs/v2/jobs/events", json={"cursor": "broken"}).status_code == 400
    empty = threading.Event()
    read = JobEventStream.read

    async def observed_read(self: JobEventStream, request: JobEventRequest) -> JobEventPage:
        result = await read(self, request)
        if not result.data:
            empty.set()
        return result

    with patch.object(JobEventStream, "read", observed_read):
        waiting = asyncio.ensure_future(
            jobs.read_job_events(body=JobEventRequest(cursor=page.next_cursor, wait_seconds=5))
        )
        try:
            assert await asyncio.to_thread(empty.wait, 3)
            assert not waiting.done()
            output.append(change(2))
            assert [row.id for row in (await asyncio.wait_for(waiting, 5)).data().data] == ["2"]
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.parametrize("failure", ["first_event", "last_event", "expose"])
async def test_creation_stays_hidden_until_controller_publication(event_context: ClientContext, failure: str):
    ctx = event_context
    job, attempt, step = await graph(ctx, HelixJobStatus.CREATED)
    job.status = HelixJobStatus.CREATION_PENDING
    job.fileset_id = "fileset-ready"
    await ctx.entity_client.update(job)
    commands = dispatcher(ctx)
    events = publisher(ctx)
    assert await commands.get_job(job.id, "default") is None
    assert await commands.get_job(job.name, "default") is None
    assert await commands.resolve_attempt_by_id(job.id, attempt.id, "default") is None
    assert await commands.cancel_job(job.id, "default") is None
    assert await commands.list_jobs(ParsedFilter(operation=None), "default") == ([], 0)
    if failure == "expose":
        update = events.store.update

        def interrupted_exposure(resource):
            if isinstance(resource, HelixJob):
                raise SimulatedCrash()
            return update(resource)

        with patch.object(events.store, "update", side_effect=interrupted_exposure):
            with pytest.raises(SimulatedCrash):
                events.publish_observations(lambda: None, None)
    else:
        append = events.stream.append
        counter = 0

        def interrupted(change: LifecycleChange, on_progress=None):
            nonlocal counter
            counter += 1
            if counter == (1 if failure == "first_event" else 3):
                raise RuntimeError("publication unavailable")
            return append(change, on_progress)

        with patch.object(events.stream, "append", side_effect=interrupted):
            events.publish_observations(lambda: None, None)
    assert await commands.get_job(job.id, "default") is None
    publisher(ctx).publish_observations(lambda: None, None)
    visible = await commands.get_job(job.id, "default")
    assert visible is not None and visible.status == HelixJobStatus.CREATED
    rows = (await JobEventStream(ctx.entity_client).read(JobEventRequest())).data
    assert [row.event_type for row in rows] == [
        JobEventType.JOB_CREATED,
        JobEventType.ATTEMPT_CREATED,
        JobEventType.STEP_CREATED,
    ]
    assert rows[0].job_id == job.id and rows[1].attempt_id == attempt.id and rows[2].step_id == step.id
    assert (
        "terminal_event_published" not in (await ctx.entity_client.get_by_id(HelixJobAttempt, attempt.id)).model_dump()
    )


async def test_create_request_waits_for_controller(
    event_context: ClientContext, mock_files_client, mock_secrets_client
):
    ctx = event_context
    commands = JobDispatcher(ctx.entity_client, mock_files_client, mock_secrets_client)
    task = asyncio.create_task(
        commands.create_job(
            CreateHelixJobRequest(name="prepared", source="test", spec={}, platform_spec=TestConstants.PLATFORM_SPEC),
            "default",
        )
    )
    try:
        async with asyncio.timeout(5):
            while True:
                jobs = await ctx.entity_client.list(HelixJob, workspace="default")
                if jobs.data and jobs.data[0].current_attempt_id:
                    break
                await asyncio.sleep(0)
        assert not task.done()
        assert await commands.get_job("prepared", "default") is None
        assert not (await JobEventStream(ctx.entity_client).read(JobEventRequest())).data
        publisher(ctx).publish_observations(lambda: None, None)
        result = await asyncio.wait_for(task, timeout=5)
        assert result.status == HelixJobStatus.CREATED
        assert len((await JobEventStream(ctx.entity_client).read(JobEventRequest())).data) == 3
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


async def test_creation_wait_timeout_keeps_cleanup_source(event_context: ClientContext):
    job, _, _ = await graph(event_context)
    job.status = HelixJobStatus.CREATION_PENDING
    await event_context.entity_client.update(job)
    with patch("nhx.core.jobs.app.dispatcher.timeout", return_value=asyncio.timeout(0)):
        with pytest.raises(JobPublicationTimeoutError):
            await dispatcher(event_context)._wait_for_publication(job)
    assert (await event_context.entity_client.get_by_id(HelixJob, job.id)).status == HelixJobStatus.CREATION_PENDING


async def test_completed_step_retries_interrupted_parent_finalization(event_context: ClientContext):
    ctx = event_context
    job, attempt, step = await graph(ctx)
    events = publisher(ctx)
    events.publish_observations(lambda: None, None)
    commands = dispatcher(ctx)
    with patch.object(commands, "_save_attempt_status", side_effect=SimulatedCrash):
        with pytest.raises(SimulatedCrash):
            await commands.update_job_status_from_step(step, HelixJobStatus.COMPLETED)
    saved = await ctx.entity_client.get_by_id(HelixJobStep, step.id)
    assert saved.status == HelixJobStatus.COMPLETED and saved.status_details["task_cleanup_pending"]
    assert (await ctx.entity_client.get_by_id(HelixJobAttempt, attempt.id)).status == HelixJobStatus.ACTIVE
    await dispatcher(ctx).update_job_status_from_step(saved, HelixJobStatus.COMPLETED)
    assert (await ctx.entity_client.get_by_id(HelixJob, job.id)).status == HelixJobStatus.COMPLETED
    publisher(ctx).publish_observations(lambda: None, None)
    rows = (await JobEventStream(ctx.entity_client).read(JobEventRequest())).data
    assert {row.event_type for row in rows if row.new_state == "completed"} == {
        JobEventType.JOB_STATE_CHANGED,
        JobEventType.ATTEMPT_STATE_CHANGED,
        JobEventType.STEP_STATE_CHANGED,
    }


async def test_creation_waits_for_artifact_readiness(event_context: ClientContext):
    ctx = event_context
    job, _, _ = await graph(ctx, HelixJobStatus.CREATED)
    job.status = HelixJobStatus.CREATION_PENDING
    job = await ctx.entity_client.update(job)
    events = publisher(ctx)
    events.publish_observations(lambda: None, None)
    assert await dispatcher(ctx).get_job(job.id, "default") is None
    assert not (await JobEventStream(ctx.entity_client).read(JobEventRequest())).data
    job.fileset_id = "allocated"
    await ctx.entity_client.update(job)
    events.publish_observations(lambda: None, None)
    visible = await dispatcher(ctx).get_job(job.id, "default")
    assert visible is not None and visible.status == HelixJobStatus.CREATED
    assert [row.event_type for row in (await JobEventStream(ctx.entity_client).read(JobEventRequest())).data] == [
        JobEventType.JOB_CREATED,
        JobEventType.ATTEMPT_CREATED,
        JobEventType.STEP_CREATED,
    ]


async def test_terminal_scan_resumes_after_its_time_budget(event_context: ClientContext):
    ctx = event_context
    job, _, _ = await graph(ctx)
    events = publisher(ctx)
    events.publish_observations(lambda: None, None)
    identities = set()
    for index in range(3):
        attempt = await ctx.entity_client.create(
            HelixJobAttempt(
                name="orphan-" + str(index),
                workspace="default",
                job=job.id,
                status=HelixJobStatus.ERROR,
                platform_spec=job.platform_spec,
            )
        )
        identities.add(attempt.id)
    clock = count()
    with patch("nhx.core.jobs.app.event_publisher.monotonic", side_effect=lambda: next(clock)):
        for _ in range(5):
            events.publish_observations(lambda: None, None)
    rows = (await JobEventStream(ctx.entity_client).read(JobEventRequest())).data
    assert {row.attempt_id for row in rows if row.new_state == "error"} == identities
