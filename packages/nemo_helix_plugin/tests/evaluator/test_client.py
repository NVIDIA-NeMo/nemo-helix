# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import httpx
import pytest
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient
from nemo_helix_plugin.evaluator.types import (
    BundledMetricOutputSpec,
    CreateMetricRequest,
    InlineMetricPayload,
    SubmitEvaluateJobRequest,
)
from pydantic import JsonValue

BASE = "http://test:8000"


def _metric_create_request() -> CreateMetricRequest:
    value_json_schema: dict[str, JsonValue] = {"type": "number"}
    metric: dict[str, JsonValue] = {
        "type": "exact-match",
        "reference": "{{item.expected}}",
        "candidate": "{{item.output}}",
    }
    return CreateMetricRequest(
        metric_type="exact-match",
        outputs=[BundledMetricOutputSpec(name="score", value_json_schema=value_json_schema)],
        payload=InlineMetricPayload(kind="inline", metric=metric),
    )


def test_sync_submit_evaluate_job_uses_typed_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/apis/evaluator/v2/workspaces/default/evaluate/jobs"
        assert request.read() == b'{"spec":{"metrics":[],"dataset":[]}}'
        return httpx.Response(201, request=request, json={"name": "job-1", "status": "created"})

    client = EvaluatorClient(
        base_url=BASE,
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.submit_evaluate_job(body=SubmitEvaluateJobRequest(spec={"metrics": [], "dataset": []}))

    assert response.data().name == "job-1"


@pytest.mark.asyncio
async def test_async_health_uses_typed_transport() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/apis/evaluator/v1/healthz"
        return httpx.Response(200, request=request, json={"plugin": "evaluator", "status": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncEvaluatorClient(base_url=BASE, http_client=http_client)

        response = await client.get_health()

    assert response.data().status == "ok"


def test_binary_download_returns_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert (
            request.url.path == "/apis/evaluator/v2/workspaces/default/evaluate/jobs/job-1/results/artifacts/download"
        )
        return httpx.Response(200, request=request, content=b"tar-bytes")

    client = EvaluatorClient(
        base_url=BASE,
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.download_evaluate_job_artifacts(name="job-1").read() == b"tar-bytes"


def test_create_metric_sends_name_path_body_and_project_query() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/apis/evaluator/v2/workspaces/default/metrics/accuracy"
        assert request.url.params["project"] == "proj-a"
        assert json.loads(request.read()) == {
            "metric_type": "exact-match",
            "outputs": [{"name": "score", "value_json_schema": {"type": "number"}}],
            "payload": {
                "kind": "inline",
                "metric": {
                    "type": "exact-match",
                    "reference": "{{item.expected}}",
                    "candidate": "{{item.output}}",
                },
            },
        }
        return httpx.Response(201, request=request, json={"name": "accuracy", "workspace": "default"})

    client = EvaluatorClient(
        base_url=BASE,
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.create_metric(
        name="accuracy",
        body=_metric_create_request(),
        query_params={"project": "proj-a"},
    )

    assert response.data().name == "accuracy"


def test_create_metric_exist_ok_returns_existing_on_conflict() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            assert request.url.path == "/apis/evaluator/v2/workspaces/default/metrics/accuracy"
            return httpx.Response(409, request=request, json={"detail": "Metric already exists"})
        assert request.method == "GET"
        assert request.url.path == "/apis/evaluator/v2/workspaces/default/metrics/accuracy"
        return httpx.Response(200, request=request, json={"name": "accuracy", "workspace": "default"})

    client = EvaluatorClient(
        base_url=BASE,
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.create_metric(
        name="accuracy",
        body=_metric_create_request(),
        exist_ok=True,
    )

    assert response.data().name == "accuracy"
    assert [request.method for request in calls] == ["POST", "GET"]


@pytest.mark.asyncio
async def test_async_create_metric_exist_ok_returns_existing_on_conflict() -> None:
    calls: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            assert request.url.path == "/apis/evaluator/v2/workspaces/default/metrics/accuracy"
            return httpx.Response(409, request=request, json={"detail": "Metric already exists"})
        assert request.method == "GET"
        assert request.url.path == "/apis/evaluator/v2/workspaces/default/metrics/accuracy"
        return httpx.Response(200, request=request, json={"name": "accuracy", "workspace": "default"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncEvaluatorClient(base_url=BASE, workspace="default", http_client=http_client)

        response = await client.create_metric(
            name="accuracy",
            body=_metric_create_request(),
            exist_ok=True,
        )

    assert response.data().name == "accuracy"
    assert [request.method for request in calls] == ["POST", "GET"]


def test_list_metrics_returns_paginated_items() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/apis/evaluator/v2/workspaces/default/metrics"
        assert request.url.params["include_derived"] == "true"
        return httpx.Response(
            200,
            request=request,
            json={
                "data": [{"name": "accuracy", "workspace": "default"}],
                "pagination": {
                    "page": 1,
                    "page_size": 100,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        )

    client = EvaluatorClient(
        base_url=BASE,
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    metrics = list(client.list_metrics(query_params={"include_derived": True}).items())

    assert len(metrics) == 1
    assert metrics[0].name == "accuracy"
