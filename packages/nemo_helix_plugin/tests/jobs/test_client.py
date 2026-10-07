# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JobsClient / AsyncJobsClient via a mocked httpx transport.

Drives ``send()`` end-to-end (path resolution, body serialization, response
unwrapping, pagination, binary, error mapping) without a network."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

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
    platform_spec = HelixJobSpec.model_validate(
        {"steps": [{"name": "step-one", "executor": {"provider": "cpu", "container": {"image": "x"}}}]}
    )
    body = CreateHelixJobRequest(
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
