# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DataDesignerClient / AsyncDataDesignerClient with mocked httpx transports."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from nemo_platform_plugin.client.errors import (
    ConflictError,
    InternalServerError,
    NotFoundError,
    UnprocessableEntityError,
)
from nemo_platform_plugin.data_designer.client import AsyncDataDesignerClient, DataDesignerClient
from nemo_platform_plugin.data_designer.types import DataDesignerJobRequest, JsonMap, PreviewRequest
from pydantic import JsonValue

BASE = "http://test:8000"

_JOB_JSON: JsonMap = {
    "id": "job-id",
    "name": "job-a",
    "workspace": "default",
    "spec": {"config": {"columns": []}, "num_records": 5},
    "status": "created",
    "status_details": {},
    "error_details": None,
}

_STATUS_JSON: JsonMap = {
    "id": "job-id",
    "name": "job-a",
    "status": "completed",
    "status_details": {},
    "error_details": None,
    "steps": [],
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}

_RESULT_JSON: JsonMap = {
    "name": "artifacts",
    "job": "job-a",
    "workspace": "default",
    "artifact_url": "fileset://default/job-a#artifacts",
    "artifact_storage_type": "fileset",
}


ResponseHandler = Callable[[httpx.Request], httpx.Response]


def _json_handler(status_code: int, payload: JsonValue) -> ResponseHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request, json=payload)

    return handler


def _bytes_handler(status_code: int, content: bytes) -> ResponseHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request, content=content)

    return handler


def _sequence_handler(handlers: list[ResponseHandler]) -> ResponseHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        return handlers.pop(0)(request)

    return handler


def _sync_http(handler: ResponseHandler) -> tuple[httpx.Client, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(capture)), requests


def _async_http(handler: ResponseHandler) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    async def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(capture)), requests


def test_create_job_serializes_body_and_unwraps_response() -> None:
    http_client, requests = _sync_http(_json_handler(201, _JOB_JSON))
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)
    body = DataDesignerJobRequest(
        name="job-a",
        spec={"config": {"columns": []}, "num_records": 5},
        profile="gpu-large",
        options={"priority": "high"},
    )

    job = client.create_job(body=body).data()

    assert job.name == "job-a"
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/apis/data-designer/v2/workspaces/default/jobs/create"
    assert json.loads(request.content) == {
        "name": "job-a",
        "spec": {"config": {"columns": []}, "num_records": 5},
        "profile": "gpu-large",
        "options": {"priority": "high"},
    }


def test_create_retrieval_job_resolves_collection_path() -> None:
    http_client, requests = _sync_http(_json_handler(201, {**_JOB_JSON, "name": "retrieval-job"}))
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    client.create_job(job_collection="retrieval-generate", body=DataDesignerJobRequest(spec={"corpus": "system/c"}))

    assert requests[0].url.path == "/apis/data-designer/v2/workspaces/default/jobs/retrieval-generate"


def test_list_jobs_paginated_items() -> None:
    http_client, _ = _sync_http(
        _json_handler(
            200,
            {
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
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    jobs = list(client.list_jobs().items())

    assert [job.name for job in jobs] == ["job-a"]


def test_get_job_status_unwraps_response() -> None:
    http_client, _ = _sync_http(_json_handler(200, _STATUS_JSON))
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    status = client.get_job_status(name="job-a").data()

    assert status.name == "job-a"
    assert status.status == "completed"


def test_get_job_logs_unwraps_log_page() -> None:
    http_client, _ = _sync_http(
        _json_handler(
            200,
            {
                "data": [
                    {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "job": "job-a",
                        "job_step": "step",
                        "job_task": "task",
                        "message": '{"message":"hello","levelname":"INFO","name":"data_designer"}',
                    }
                ],
                "total": 1,
                "next_page": None,
                "prev_page": None,
            },
        )
    )
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    logs = client.get_job_logs(name="job-a").data()

    assert logs.data[0].job == "job-a"
    assert logs.next_page is None


def test_list_and_get_job_results_unwrap_models() -> None:
    http_client, requests = _sync_http(
        _sequence_handler([_json_handler(200, {"data": [_RESULT_JSON]}), _json_handler(200, _RESULT_JSON)])
    )
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    results = client.list_job_results(name="job-a").data()
    result = client.get_job_result(job="job-a", name="artifacts").data()

    assert requests[0].url.path == "/apis/data-designer/v2/workspaces/default/jobs/create/job-a/results"
    assert requests[1].url.path == "/apis/data-designer/v2/workspaces/default/jobs/create/job-a/results/artifacts"
    assert results.data[0].name == "artifacts"
    assert result.artifact_url == "fileset://default/job-a#artifacts"


def test_download_job_result_reads_bytes() -> None:
    http_client, requests = _sync_http(_bytes_handler(200, b"artifact-bytes"))
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    data = client.download_job_result(job="job-a", name="artifacts").read()

    assert (
        requests[0].url.path == "/apis/data-designer/v2/workspaces/default/jobs/create/job-a/results/artifacts/download"
    )
    assert data == b"artifact-bytes"


def test_preview_stream_parses_ndjson_frames() -> None:
    http_client, requests = _sync_http(_bytes_handler(200, b'{"kind":"heartbeat"}\n{"kind":"done"}\n'))
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    response = client.preview(body=PreviewRequest(config={"columns": []}, num_records=1))
    with response.stream() as stream:
        frames = list(stream)

    assert requests[0].url.path == "/apis/data-designer/v2/workspaces/default/preview"
    assert [frame.kind for frame in frames] == ["heartbeat", "done"]


def test_not_found_maps_to_typed_error() -> None:
    http_client, _ = _sync_http(_json_handler(404, {"detail": "not found"}))
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    with pytest.raises(NotFoundError, match="not found"):
        client.get_job(name="missing")


def test_validation_error_maps_to_typed_error() -> None:
    http_client, _ = _sync_http(_json_handler(422, {"detail": "invalid config"}))
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    with pytest.raises(UnprocessableEntityError, match="invalid config"):
        client.create_job(body=DataDesignerJobRequest(spec={}))


@pytest.mark.parametrize(
    ("status_code", "error_cls", "detail"),
    [
        (409, ConflictError, "job already exists"),
        (500, InternalServerError, "server exploded"),
    ],
)
def test_additional_http_errors_map_to_typed_errors(status_code: int, error_cls: type[Exception], detail: str) -> None:
    http_client, _ = _sync_http(_json_handler(status_code, {"detail": detail}))
    client = DataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    with pytest.raises(error_cls, match=detail):
        client.create_job(body=DataDesignerJobRequest(spec={}))


@pytest.mark.asyncio
async def test_async_get_job_unwraps_response() -> None:
    http_client, requests = _async_http(_json_handler(200, _JOB_JSON))
    client = AsyncDataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    response = await client.get_job(name="job-a")

    assert requests[0].url.path == "/apis/data-designer/v2/workspaces/default/jobs/create/job-a"
    assert response.data().name == "job-a"


@pytest.mark.asyncio
async def test_async_download_job_result_reads_bytes() -> None:
    http_client, requests = _async_http(_bytes_handler(200, b"artifact-bytes"))
    client = AsyncDataDesignerClient(base_url=BASE, workspace="default", http_client=http_client)

    data = await (await client.download_job_result(job="job-a", name="artifacts")).read()

    assert (
        requests[0].url.path == "/apis/data-designer/v2/workspaces/default/jobs/create/job-a/results/artifacts/download"
    )
    assert data == b"artifact-bytes"
