# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JobsClient / AsyncJobsClient via a mocked httpx transport.

Drives ``send()`` end-to-end (path resolution, body serialization, response
unwrapping, pagination, binary, error mapping) without a network."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_platform_plugin.jobs.spec import PlatformJobSpec
from nemo_platform_plugin.jobs.types import CreatePlatformJobRequest

BASE = "http://test:8000"

_JOB_JSON = {
    "id": "job-1",
    "attempt_id": "att-1",
    "name": "my-job",
    "workspace": "default",
    "source": "test",
    "spec": {},
    "platform_spec": {"steps": [{"name": "step-one", "executor": {"provider": "cpu", "container": {"image": "x"}}}]},
    "fileset": "fs-1",
    "status": "created",
}

_STATUS_JSON = {
    "id": "job-1",
    "name": "my-job",
    "status": "created",
    "status_details": {},
    "error_details": None,
    "steps": [],
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}


def _mock_http(response: httpx.Response) -> MagicMock:
    mock = MagicMock(spec=httpx.Client)
    mock.request.return_value = response
    return mock


def test_create_job_serializes_body_and_unwraps() -> None:
    mock_http = _mock_http(
        httpx.Response(
            201,
            request=httpx.Request("POST", f"{BASE}/apis/jobs/v2/workspaces/default/jobs"),
            json=_JOB_JSON,
        )
    )
    client = JobsClient(base_url=BASE, workspace="default", http_client=mock_http)
    platform_spec = PlatformJobSpec.model_validate(
        {"steps": [{"name": "step-one", "executor": {"provider": "cpu", "container": {"image": "x"}}}]}
    )
    body = CreatePlatformJobRequest(
        spec={},
        source="test",
        platform_spec=platform_spec,
    )
    resp = client.create_job(body=body)

    assert resp.data().name == "my-job"
    _, kwargs = mock_http.request.call_args
    assert b'"source":"test"' in kwargs["content"]


def test_get_job_resolves_path() -> None:
    mock_http = _mock_http(
        httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs/my-job"),
            json=_JOB_JSON,
        )
    )
    client = JobsClient(base_url=BASE, workspace="default", http_client=mock_http)
    job = client.get_job(name="my-job").data()

    assert job.id == "job-1"
    args, _ = mock_http.request.call_args
    assert "/apis/jobs/v2/workspaces/default/jobs/my-job" in str(mock_http.request.call_args)


def test_list_jobs_paginated_items() -> None:
    mock_http = _mock_http(
        httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs"),
            json={
                "data": [_JOB_JSON],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        )
    )
    client = JobsClient(base_url=BASE, workspace="default", http_client=mock_http)
    jobs = list(client.list_jobs().items())

    assert len(jobs) == 1
    assert jobs[0].name == "my-job"


def test_legacy_list_alias_exposes_data_and_iteration() -> None:
    mock_http = _mock_http(
        httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs"),
            json={
                "data": [_JOB_JSON],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        )
    )
    client = JobsClient(base_url=BASE, workspace="default", http_client=mock_http)

    jobs = client.list(filter={"source": "customization"}, page=1, page_size=10)

    assert jobs.data[0].name == "my-job"
    assert [job.name for job in jobs] == ["my-job"]
    _, kwargs = mock_http.request.call_args
    assert json.loads(kwargs["params"]["filter"]) == {"source": "customization"}


def test_legacy_list_alias_exposes_stainless_page_helpers_lazily() -> None:
    next_job_json = {**_JOB_JSON, "id": "job-2", "name": "next-job"}
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = [
        httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs"),
            json={
                "data": [_JOB_JSON],
                "pagination": {
                    "page": 1,
                    "page_size": 1,
                    "current_page_size": 1,
                    "total_pages": 2,
                    "total_results": 2,
                },
            },
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs"),
            json={
                "data": [next_job_json],
                "pagination": {
                    "page": 2,
                    "page_size": 1,
                    "current_page_size": 1,
                    "total_pages": 2,
                    "total_results": 2,
                },
            },
        ),
    ]
    client = JobsClient(base_url=BASE, workspace="default", http_client=mock_http)

    first_page = next(client.list(page_size=1).iter_pages())

    assert first_page.has_next_page()
    page_info = first_page.next_page_info()
    assert page_info is not None
    assert page_info.params == {"page": 2}
    assert mock_http.request.call_count == 1
    next_page = first_page.get_next_page()
    assert next_page.data[0].name == "next-job"
    assert mock_http.request.call_count == 2


def test_legacy_get_status_alias_unwraps_response() -> None:
    mock_http = _mock_http(
        httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs/my-job/status"),
            json=_STATUS_JSON,
        )
    )
    client = JobsClient(base_url=BASE, workspace="default", http_client=mock_http)

    status = client.get_status("my-job")

    assert status.name == "my-job"
    assert status.status == "created"


def test_legacy_jobs_type_import_paths_resolve_to_source_owned_models() -> None:
    from nemo_platform.types import PlatformJobStatusResponse as TopLevelStatusResponse
    from nemo_platform.types.jobs import PlatformJobStep
    from nemo_platform.types.shared import PlatformJobStatusResponse as SharedStatusResponse
    from nemo_platform_plugin.jobs.schemas import PlatformJobStatusResponse
    from nemo_platform_plugin.jobs.types import PlatformJobStepResponse

    assert PlatformJobStep is PlatformJobStepResponse
    assert SharedStatusResponse is PlatformJobStatusResponse
    assert TopLevelStatusResponse is PlatformJobStatusResponse


def test_delete_job_returns_none() -> None:
    mock_http = _mock_http(
        httpx.Response(
            204,
            request=httpx.Request("DELETE", f"{BASE}/apis/jobs/v2/workspaces/default/jobs/my-job"),
        )
    )
    client = JobsClient(base_url=BASE, workspace="default", http_client=mock_http)
    assert client.delete_job(name="my-job").data() is None


def test_download_job_result_reads_bytes() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    stream_ctx = MagicMock()
    raw = httpx.Response(
        200,
        request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs/j/results/out/download"),
        content=b"artifact-bytes",
    )
    stream_ctx.__enter__.return_value = raw
    stream_ctx.__exit__.return_value = False
    mock_http.stream.return_value = stream_ctx

    client = JobsClient(base_url=BASE, workspace="default", http_client=mock_http)
    data = client.download_job_result(job="j", name="out").read()

    assert data == b"artifact-bytes"


def test_get_job_not_found_maps_error() -> None:
    mock_http = _mock_http(
        httpx.Response(
            404,
            request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs/missing"),
            json={"detail": "Job not found"},
        )
    )
    client = JobsClient(base_url=BASE, workspace="default", http_client=mock_http)
    with pytest.raises(NotFoundError) as exc:
        client.get_job(name="missing")
    assert exc.value.status_code == 404


# A JSON array response (not an object) — the shape that broke ``send()`` when
# the endpoint's return annotation is a bare ``list[...]`` generic.
_PROFILES_JSON = [
    {"backend": "subprocess", "provider": "subprocess", "profile": "default", "config": {}},
    {"backend": "e2e", "provider": "cpu", "profile": "default", "config": {}},
]


def test_get_execution_profiles_parses_list_response() -> None:
    mock_http = _mock_http(
        httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/execution-profiles"),
            json=_PROFILES_JSON,
        )
    )
    client = JobsClient(base_url=BASE, workspace="default", http_client=mock_http)

    profiles = client.get_execution_profiles().data()

    assert isinstance(profiles, list)
    assert len(profiles) == 2
    assert {p.backend for p in profiles} == {"subprocess", "e2e"}


@pytest.mark.asyncio
async def test_async_get_execution_profiles_parses_list_response() -> None:
    mock_http = MagicMock(spec=httpx.AsyncClient)
    mock_http.request = AsyncMock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/execution-profiles"),
            json=_PROFILES_JSON,
        )
    )
    client = AsyncJobsClient(base_url=BASE, workspace="default", http_client=mock_http)

    profiles = (await client.get_execution_profiles()).data()

    assert isinstance(profiles, list)
    assert {p.backend for p in profiles} == {"subprocess", "e2e"}


@pytest.mark.asyncio
async def test_async_legacy_list_alias_exposes_awaitable_page_and_iteration() -> None:
    mock_http = MagicMock(spec=httpx.AsyncClient)
    mock_http.request = AsyncMock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs"),
            json={
                "data": [_JOB_JSON],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        )
    )
    client = AsyncJobsClient(base_url=BASE, workspace="default", http_client=mock_http)

    page = await client.list(filter={"source": "customization"})

    assert page.data[0].name == "my-job"
    assert [job.name async for job in client.list(filter={"source": "customization"})] == ["my-job"]


@pytest.mark.asyncio
async def test_async_legacy_list_alias_exposes_stainless_page_helpers_lazily() -> None:
    next_job_json = {**_JOB_JSON, "id": "job-2", "name": "next-job"}
    mock_http = MagicMock(spec=httpx.AsyncClient)
    mock_http.request = AsyncMock(
        side_effect=[
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs"),
                json={
                    "data": [_JOB_JSON],
                    "pagination": {
                        "page": 1,
                        "page_size": 1,
                        "current_page_size": 1,
                        "total_pages": 2,
                        "total_results": 2,
                    },
                },
            ),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/jobs/v2/workspaces/default/jobs"),
                json={
                    "data": [next_job_json],
                    "pagination": {
                        "page": 2,
                        "page_size": 1,
                        "current_page_size": 1,
                        "total_pages": 2,
                        "total_results": 2,
                    },
                },
            ),
        ]
    )
    client = AsyncJobsClient(base_url=BASE, workspace="default", http_client=mock_http)

    first_page = await client.list(page_size=1)

    assert first_page.has_next_page()
    page_info = first_page.next_page_info()
    assert page_info is not None
    assert page_info.params == {"page": 2}
    assert mock_http.request.call_count == 1
    next_page = await first_page.get_next_page()
    assert next_page.data[0].name == "next-job"
    assert mock_http.request.call_count == 2
