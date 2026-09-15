# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pytest
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.jobs.schemas import PlatformJobStatusResponse
from nmp.customization_common.sdk.client import (
    make_async_customization_sdk_context,
    make_customization_sdk,
    make_customization_sdk_context,
)

BASE = "http://test:8000"


def _status_payload(name: str = "customizer-job", status: str = "active") -> dict[str, object]:
    return {
        "id": f"job-{name}",
        "name": name,
        "status": status,
        "status_details": {},
        "error_details": None,
        "steps": [],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:01:00Z",
    }


def test_jobs_resource_fetches_status_from_core_jobs_route() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, request=request, json=_status_payload(status="completed"))

    owner = NemoClient(
        base_url=BASE,
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    Customization, _AsyncCustomization = make_customization_sdk("automodel")
    resource = Customization(make_customization_sdk_context(owner))

    status = resource.jobs.get_status("customizer-job").data()

    assert isinstance(status, PlatformJobStatusResponse)
    assert status.name == "customizer-job"
    assert status.status == "completed"
    assert [request.url.path for request in seen] == [
        "/apis/jobs/v2/workspaces/default/jobs/customizer-job/status",
    ]


@pytest.mark.asyncio
async def test_async_jobs_resource_fetches_status_from_core_jobs_route() -> None:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, request=request, json=_status_payload(status="completed"))

    owner = AsyncNemoClient(
        base_url=BASE,
        workspace="default",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    _Customization, AsyncCustomization = make_customization_sdk("automodel")
    resource = AsyncCustomization(make_async_customization_sdk_context(owner))

    response = await resource.jobs.get_status("customizer-job", workspace="default")
    status = response.data()

    assert status.status == "completed"
    assert [request.url.path for request in seen] == [
        "/apis/jobs/v2/workspaces/default/jobs/customizer-job/status",
    ]
