# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import get_origin

import httpx
from nemo_platform_plugin.client.types import Paginated, PreparedRequest
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus, PlatformJobStatusResponse
from pydantic import BaseModel


class _Spec(BaseModel):
    model: str


def _status_response(status: PlatformJobStatus = PlatformJobStatus.COMPLETED) -> PlatformJobStatusResponse:
    now = datetime.now(timezone.utc)
    return PlatformJobStatusResponse(
        id="job-id",
        name="job-a",
        status=status,
        status_details={},
        error_details=None,
        steps=[],
        created_at=now,
        updated_at=now,
    )


def test_create_customization_job_endpoint_shape() -> None:
    from nmp.customization_common.sdk import endpoints
    from nmp.customization_common.sdk.types import CustomizationJob, CustomizationJobCreateRequest

    body = CustomizationJobCreateRequest(
        name="job-a",
        spec={"model": "default/qwen"},
        profile="gpu",
        options={"kubernetes": {"priority": "normal"}},
    )

    prepared = endpoints.create_customization_job(
        workspace="team-a",
        backend="automodel",
        body=body,
    )

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/customization/v2/workspaces/{workspace}/{backend}/jobs"
    assert prepared.path_params == {"workspace": "team-a", "backend": "automodel"}
    assert isinstance(prepared.content, bytes)
    assert json.loads(prepared.content) == {
        "name": "job-a",
        "spec": {"model": "default/qwen"},
        "profile": "gpu",
        "options": {"kubernetes": {"priority": "normal"}},
    }
    assert prepared.response_type is CustomizationJob


def test_customization_job_preserves_legacy_job_alias() -> None:
    from nmp.customization_common.sdk.types import CustomizationJob

    job = CustomizationJob(name="job-a", spec={"model": "default/qwen"})

    assert job.job is job


def test_list_customization_jobs_endpoint_shape() -> None:
    from nmp.customization_common.sdk import endpoints

    prepared = endpoints.list_customization_jobs(
        workspace="team-a",
        backend="unsloth",
        query_params={"page": 2, "page_size": 25, "sort": "-created_at", "filter": '{"status":"completed"}'},
    )

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/customization/v2/workspaces/{workspace}/{backend}/jobs"
    assert prepared.path_params == {"workspace": "team-a", "backend": "unsloth"}
    assert prepared.query_params == {
        "page": 2,
        "page_size": 25,
        "sort": "-created_at",
        "filter": '{"status":"completed"}',
    }
    assert get_origin(prepared.response_type) is Paginated


def test_get_customization_job_endpoint_shape() -> None:
    from nmp.customization_common.sdk import endpoints
    from nmp.customization_common.sdk.types import CustomizationJob

    prepared = endpoints.get_customization_job(
        workspace="team-a",
        backend="rl",
        name="job-a",
    )

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/customization/v2/workspaces/{workspace}/{backend}/jobs/{name}"
    assert prepared.path_params == {
        "workspace": "team-a",
        "backend": "rl",
        "name": "job-a",
    }
    assert prepared.response_type is CustomizationJob


def test_jobs_resource_create_delegates_to_typed_customization_client() -> None:
    from nemo_platform_plugin.client.client import NemoClient
    from nmp.customization_common.sdk.client import make_customization_sdk, make_customization_sdk_context
    from nmp.customization_common.sdk.types import CustomizationJob

    captured: list[httpx.Request] = []
    returned = CustomizationJob(
        id="job-id",
        name="job-a",
        workspace="team-a",
        spec={"model": "default/qwen"},
        status=PlatformJobStatus.CREATED,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(201, json=returned.model_dump(mode="json"))

    owner = NemoClient(
        base_url="http://nmp.test",
        workspace="team-a",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    Customization, _AsyncCustomization = make_customization_sdk("automodel")
    resource = Customization(make_customization_sdk_context(owner))
    response = resource.jobs.create(
        spec=_Spec(model="default/qwen"),
        workspace="team-a",
        name="job-a",
        profile="gpu",
        options={"docker": {"network": "bridge"}},
    )

    assert response.data() == returned
    assert len(captured) == 1
    assert captured[0].method == "POST"
    assert str(captured[0].url) == "http://nmp.test/apis/customization/v2/workspaces/team-a/automodel/jobs"
    assert json.loads(captured[0].content) == {
        "name": "job-a",
        "spec": {"model": "default/qwen"},
        "profile": "gpu",
        "options": {"docker": {"network": "bridge"}},
    }


def test_jobs_resource_get_status_uses_core_jobs_client() -> None:
    from nemo_platform_plugin.client.client import NemoClient
    from nmp.customization_common.sdk.client import make_customization_sdk, make_customization_sdk_context

    status = _status_response()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=status.model_dump(mode="json"))

    owner = NemoClient(
        base_url="http://nmp.test",
        workspace="team-a",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    Customization, _AsyncCustomization = make_customization_sdk("automodel")
    resource = Customization(make_customization_sdk_context(owner))

    assert resource.jobs.get_status("job-a", workspace="team-a").data() == status
    assert len(captured) == 1
    assert str(captured[0].url) == "http://nmp.test/apis/jobs/v2/workspaces/team-a/jobs/job-a/status"


def test_jobs_resource_get_logs_uses_core_jobs_client() -> None:
    from nemo_platform_plugin.client.client import NemoClient
    from nmp.customization_common.sdk.client import make_customization_sdk, make_customization_sdk_context

    now = datetime.now(timezone.utc)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "timestamp": now.isoformat(),
                        "job": "job-a",
                        "job_step": "train",
                        "job_task": "task-1",
                        "message": "hello",
                    }
                ],
                "total": 1,
                "next_page": None,
                "prev_page": None,
            },
        )

    owner = NemoClient(
        base_url="http://nmp.test",
        workspace="team-a",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    Customization, _AsyncCustomization = make_customization_sdk("automodel")
    resource = Customization(make_customization_sdk_context(owner))

    page = resource.jobs.get_logs(
        "job-a",
        workspace="team-a",
        limit=50,
        page_cursor="cursor-1",
        attempt_id=3,
        step_id="train",
        tail=20,
        task_id="task-1",
    ).page()

    assert page.items[0].message == "hello"
    assert page.metadata == {"total": 1, "next_page": None, "prev_page": None}
    assert len(captured) == 1
    assert str(captured[0].url) == (
        "http://nmp.test/apis/jobs/v2/workspaces/team-a/jobs/job-a/logs"
        "?limit=50&page_cursor=cursor-1&attempt_id=3&step_id=train&tail=20&task_id=task-1"
    )


def test_sync_customization_resource_accepts_typed_nemo_client_owner() -> None:
    from nemo_platform_plugin.client.client import NemoClient
    from nmp.customization_common.sdk.client import (
        CustomizationClient,
        make_customization_sdk,
        make_customization_sdk_context,
    )
    from nmp.customization_common.sdk.types import CustomizationJob

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            201,
            json={
                "id": "job-id",
                "name": "job-a",
                "workspace": "team-a",
                "spec": {"model": "default/qwen"},
                "status": "created",
            },
        )

    owner = NemoClient(
        base_url="http://nmp.test",
        workspace="team-a",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    Customization, _AsyncCustomization = make_customization_sdk("automodel")

    response = Customization(make_customization_sdk_context(owner)).jobs.create(
        spec={"model": "default/qwen"},
        name="job-a",
    )

    assert isinstance(response.data(), CustomizationJob)
    assert str(captured[0].url) == "http://nmp.test/apis/customization/v2/workspaces/team-a/automodel/jobs"
    assert isinstance(CustomizationClient.from_client(owner), CustomizationClient)


async def test_async_customization_resource_accepts_typed_nemo_client_owner() -> None:
    from nemo_platform_plugin.client.client import AsyncNemoClient
    from nemo_platform_plugin.client.response import AsyncNemoPaginatedResponse, NemoResponse
    from nmp.customization_common.sdk.client import (
        AsyncCustomizationClient,
        make_async_customization_sdk_context,
        make_customization_sdk,
    )
    from nmp.customization_common.sdk.types import CustomizationJob

    captured: list[httpx.Request] = []
    returned = CustomizationJob(
        id="job-id",
        name="job-a",
        workspace="team-a",
        spec={"model": "default/qwen"},
        status=PlatformJobStatus.CREATED,
    )
    returned_payload = returned.model_dump(mode="json")
    status = _status_response()

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/apis/customization/v2/workspaces/team-a/automodel/jobs":
            return httpx.Response(201, json=returned_payload)
        if request.method == "GET" and path == "/apis/customization/v2/workspaces/team-a/automodel/jobs":
            return httpx.Response(
                200,
                json={
                    "data": [returned_payload],
                    "pagination": {
                        "page": 1,
                        "page_size": 100,
                        "current_page_size": 1,
                        "total_pages": 1,
                        "total_results": 1,
                    },
                },
            )
        if request.method == "GET" and path == "/apis/customization/v2/workspaces/team-a/automodel/jobs/job-a":
            return httpx.Response(200, json=returned_payload)
        if request.method == "GET" and path == "/apis/jobs/v2/workspaces/team-a/jobs/job-a/status":
            return httpx.Response(200, json=status.model_dump(mode="json"))
        return httpx.Response(404, request=request, json={"detail": "unexpected request"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        owner = AsyncNemoClient(
            base_url="http://nmp.test",
            workspace="team-a",
            http_client=http_client,
        )
        _Customization, AsyncCustomization = make_customization_sdk("automodel")
        resource = AsyncCustomization(make_async_customization_sdk_context(owner))

        create_response = await resource.jobs.create(
            spec={"model": "default/qwen"},
            name="job-a",
        )
        list_response = await resource.jobs.list(workspace="team-a")
        retrieve_response = await resource.jobs.retrieve("job-a", workspace="team-a")
        status_response = await resource.jobs.get_status("job-a", workspace="team-a")

        assert isinstance(create_response, NemoResponse)
        assert create_response.data() == returned
        assert isinstance(list_response, AsyncNemoPaginatedResponse)
        assert list_response.page().items[0] == returned
        assert isinstance(retrieve_response, NemoResponse)
        assert retrieve_response.data() == returned
        assert isinstance(status_response, NemoResponse)
        assert status_response.data() == status
        assert isinstance(AsyncCustomizationClient.from_client(owner), AsyncCustomizationClient)
    assert [str(request.url) for request in captured] == [
        "http://nmp.test/apis/customization/v2/workspaces/team-a/automodel/jobs",
        "http://nmp.test/apis/customization/v2/workspaces/team-a/automodel/jobs",
        "http://nmp.test/apis/customization/v2/workspaces/team-a/automodel/jobs/job-a",
        "http://nmp.test/apis/jobs/v2/workspaces/team-a/jobs/job-a/status",
    ]
