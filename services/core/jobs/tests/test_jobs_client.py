# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests that drive the typed ``JobsClient`` against the real Jobs
service routes (in-memory ASGI app).

Unlike ``tests/jobs/test_endpoints.py`` (which only asserts ``PreparedRequest``
shape) these exercise ``send()`` all the way through path resolution, HTTP,
and response parsing — the layer where response-type bugs actually surface.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, call, patch

import pytest
from httpx import AsyncClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.schemas import (
    FileStorageType,
    PlatformJobLog,
    PlatformJobLogPage,
    PlatformJobResultCreateRequest,
    PlatformJobStatus,
)
from nemo_helix_plugin.jobs.types import (
    CreatePlatformJobRequest,
    JobStatusDetailsUpdate,
    PlatformJobStatusUpdateRequest,
    PlatformJobTaskUpdate,
)
from nhx.common.jobs.file_manager import TmpDirPath
from nhx.common.jobs.log_client import dep_job_logs_client


@pytest.fixture
def jobs_client(test_client: AsyncClient) -> AsyncJobsClient:
    """A typed AsyncJobsClient bound to the in-memory Jobs app.

    Mirrors how ``test_sdk`` builds the Stainless SDK, but returns the new
    typed client so responses flow through ``NemoClient.send()``.
    """
    return AsyncJobsClient(base_url=str(test_client.base_url), http_client=test_client)


async def _create_job(
    jobs_client: AsyncJobsClient,
    request: CreatePlatformJobRequest,
    name: str,
):
    body = request.model_copy(update={"name": name})
    return (await jobs_client.create_job(workspace="default", body=body)).data()


@pytest.mark.asyncio
async def test_get_execution_profiles_parses_response(jobs_client: AsyncJobsClient, test_client: AsyncClient):
    """``get_execution_profiles`` parses the route's JSON array response."""
    # Sanity: the raw route really does return a JSON list (server side is fine).
    raw = await test_client.get("/apis/jobs/v2/execution-profiles")
    assert raw.status_code == 200
    assert isinstance(raw.json(), list)

    # The actual regression: the typed client must not crash parsing it.
    resp = await jobs_client.get_execution_profiles()
    profiles = resp.data()
    assert isinstance(profiles, list)


@pytest.mark.asyncio
async def test_get_execution_profiles_available_without_controller_ready_gate(
    test_client: AsyncClient,
):
    """API must advertise merge-filtered profiles without a controller ready flag.

    Split topologies (API pod without controllers) previously 503'd forever when
    readiness lived in controller-only process memory (AIRCORE-971).
    """
    raw = await test_client.get("/apis/jobs/v2/execution-profiles")
    assert raw.status_code == 200
    assert isinstance(raw.json(), list)


async def _create_hello_world_job(test_client: AsyncClient, name: str = "e2e-client-job") -> None:
    """Create a job via the hello-world factory route (service-specific body)."""
    resp = await test_client.post(
        "/apis/jobs/v2/workspaces/default/hello-world/jobs",
        json={
            "name": name,
            "description": "typed-client e2e",
            "spec": {"config": {"key": "Value"}, "target": "str"},
            "ownership": {"user": "u", "service": "s"},
        },
    )
    assert resp.status_code == 201, f"create failed: {resp.status_code} {resp.text}"


@pytest.mark.asyncio
async def test_list_jobs_round_trips_through_client(jobs_client: AsyncJobsClient, test_client: AsyncClient):
    """``list_jobs`` must page + parse real ``PlatformJobResponse`` items."""
    await _create_hello_world_job(test_client, name="list-me")

    page = (await jobs_client.list_jobs(workspace="default")).page()
    assert page.metadata["total_results"] is not None and page.metadata["total_results"] >= 1
    names = [j.name for j in page.items]
    assert "list-me" in names
    # items are the plugin DTO, fully parsed
    job = next(j for j in page.items if j.name == "list-me")
    assert job.workspace == "default"
    assert job.status is not None


@pytest.mark.asyncio
async def test_get_job_and_status_round_trip(jobs_client: AsyncJobsClient, test_client: AsyncClient):
    """``get_job`` and ``get_job_status`` must parse their real responses."""
    await _create_hello_world_job(test_client, name="get-me")

    job = (await jobs_client.get_job(name="get-me", workspace="default")).data()
    assert job.name == "get-me"
    assert job.fileset  # non-empty

    status = (await jobs_client.get_job_status(name="get-me", workspace="default")).data()
    assert status.status is not None


@pytest.mark.asyncio
async def test_job_lifecycle_methods_round_trip(
    jobs_client: AsyncJobsClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    paused_job = await _create_job(jobs_client, sample_platform_job_request, "typed-lifecycle")
    active_step = (
        await jobs_client.update_job_step_status(
            workspace="default",
            job=paused_job.name,
            name="basic",
            body=PlatformJobStatusUpdateRequest(status=PlatformJobStatus.ACTIVE),
        )
    ).data()
    assert active_step.status == PlatformJobStatus.ACTIVE

    pausing = (await jobs_client.pause_job(workspace="default", name=paused_job.name)).data()
    assert pausing.status == PlatformJobStatus.PAUSING
    await jobs_client.update_job_step_status(
        workspace="default",
        job=paused_job.name,
        name="basic",
        body=PlatformJobStatusUpdateRequest(status=PlatformJobStatus.PAUSED),
    )
    resuming = (await jobs_client.resume_job(workspace="default", name=paused_job.name)).data()
    assert resuming.status == PlatformJobStatus.RESUMING

    cancelled_job = await _create_job(jobs_client, sample_platform_job_request, "typed-cancel")
    cancelled = (await jobs_client.cancel_job(workspace="default", name=cancelled_job.name)).data()
    assert cancelled.status == PlatformJobStatus.CANCELLED

    deleted_job = await _create_job(jobs_client, sample_platform_job_request, "typed-delete")
    delete_ready = (await jobs_client.cancel_job(workspace="default", name=deleted_job.name)).data()
    assert delete_ready.status == PlatformJobStatus.CANCELLED
    deleted = await jobs_client.delete_job(workspace="default", name=deleted_job.name)
    assert deleted.http_response.status_code == 204


@pytest.mark.asyncio
async def test_status_steps_and_tasks_round_trip(
    jobs_client: AsyncJobsClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    job = await _create_job(jobs_client, sample_platform_job_request, "typed-state")
    status_update = await jobs_client.update_job_status_details(
        workspace="default",
        name=job.name,
        body=JobStatusDetailsUpdate(root={"progress": 25}),
    )
    assert status_update.http_response.status_code == 200
    updated_job = (await jobs_client.get_job(workspace="default", name=job.name)).data()
    assert updated_job.status_details == {"progress": 25}

    step_page = (await jobs_client.list_steps(workspace="default", name=job.name)).page()
    assert [step.name for step in step_page.items] == ["basic"]
    step = (await jobs_client.get_job_step(workspace="default", job=job.name, name="basic")).data()
    assert step.name == "basic"

    task = (
        await jobs_client.update_job_step_task(
            workspace="default",
            job=job.name,
            step="basic",
            name="task-1",
            body=PlatformJobTaskUpdate(
                status=PlatformJobStatus.ACTIVE,
                status_details={"message": "running"},
            ),
        )
    ).data()
    assert task.status == PlatformJobStatus.ACTIVE
    tasks = (await jobs_client.list_job_step_tasks(workspace="default", job=job.name, name="basic")).data()
    assert [item.name for item in tasks.data] == ["task-1"]
    fetched_task = (
        await jobs_client.get_job_step_task(
            workspace="default",
            job=job.name,
            step="basic",
            name="task-1",
        )
    ).data()
    assert fetched_task.status_details == {"message": "running"}


@pytest.mark.asyncio
async def test_logs_round_trip(
    jobs_client: AsyncJobsClient,
    test_client: AsyncClient,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    job = await _create_job(jobs_client, sample_platform_job_request, "typed-logs")
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    logs_client = AsyncMock()
    logs_client.query_logs.side_effect = [
        PlatformJobLogPage(
            data=[
                PlatformJobLog(
                    timestamp=timestamp,
                    job=job.name,
                    job_step="basic",
                    job_task="task-1",
                    message="hello",
                )
            ],
            total=2,
            next_page="cursor-2",
            prev_page=None,
        ),
        PlatformJobLogPage(
            data=[
                PlatformJobLog(
                    timestamp=timestamp,
                    job=job.name,
                    job_step="basic",
                    job_task="task-1",
                    message="world",
                )
            ],
            total=2,
            next_page=None,
            prev_page="cursor-1",
        ),
    ]
    app = test_client._transport.app  # type: ignore[attr-defined]
    app.dependency_overrides[dep_job_logs_client] = lambda: logs_client
    try:
        response = await jobs_client.list_job_logs(
            workspace="default",
            name=job.name,
            query_params={"limit": 5, "step_id": "basic", "task_id": "task-1"},
        )
        page = response.page()
        logs = [log async for log in response.items()]
    finally:
        app.dependency_overrides.pop(dep_job_logs_client)

    assert [log.message for log in page.items] == ["hello"]
    assert page.metadata == {"total": 2, "next_page": "cursor-2", "prev_page": None}
    assert [log.message for log in logs] == ["hello", "world"]
    filters = {
        "job": job.name,
        "job_attempt": job.attempt_id,
        "job_step": "basic",
        "job_task": "task-1",
    }
    assert logs_client.query_logs.await_args_list == [
        call(
            job.fileset,
            workspace="default",
            filters=filters,
            page_size=5,
            page_cursor=None,
            tail=None,
            artifact_base_path=None,
        ),
        call(
            job.fileset,
            workspace="default",
            filters=filters,
            page_size=5,
            page_cursor="cursor-2",
            tail=None,
            artifact_base_path=None,
        ),
    ]


@pytest.mark.asyncio
async def test_result_methods_round_trip(
    jobs_client: AsyncJobsClient,
    sample_platform_job_request: CreatePlatformJobRequest,
    tmp_path,
):
    job = await _create_job(jobs_client, sample_platform_job_request, "typed-results")
    result = (
        await jobs_client.create_job_result(
            workspace="default",
            job=job.name,
            name="output",
            body=PlatformJobResultCreateRequest(
                artifact_url="default/test-fileset#output.txt",
                artifact_storage_type=FileStorageType.FILESET,
            ),
        )
    ).data()
    assert result.name == "output"

    listed = (await jobs_client.list_job_results(workspace="default", name=job.name)).data()
    assert [item.name for item in listed.data] == ["output"]
    fetched = (await jobs_client.get_job_result(workspace="default", job=job.name, name="output")).data()
    assert fetched.artifact_url == "default/test-fileset#output.txt"

    result_dir = tmp_path / "typed-result"
    result_dir.mkdir()
    result_path = result_dir / "output.txt"
    result_path.write_bytes(b"typed result")
    downloaded = TmpDirPath(path=result_path, tmp_dir=result_dir)
    with patch(
        "nhx.core.jobs.api.v2.jobs.endpoints.download_from_result_info",
        new=AsyncMock(return_value=("output.txt", downloaded)),
    ):
        content = await (await jobs_client.download_job_result(workspace="default", job=job.name, name="output")).read()

    assert content == b"typed result"
