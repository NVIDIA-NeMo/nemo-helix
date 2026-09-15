# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import httpx
import pytest
from nemo_platform_plugin.intake.client import AsyncIntakeClient, IntakeClient
from nemo_platform_plugin.intake.types import AtifCreateRequest, EvaluationPatchRequest, EvaluatorResultCreateRequest

BASE = "http://test:8000"


def _evaluator_result_json() -> dict[str, object]:
    return {
        "evaluator_result_id": "eval-result-1",
        "span_id": "span-1",
        "session_id": "session-1",
        "workspace": "default",
        "name": "accuracy.score",
        "value": 1.0,
        "data_type": "NUMERIC",
        "created_at": "2026-01-02T03:04:05Z",
        "ingested_at": "2026-01-02T03:04:06Z",
    }


def _evaluation_json() -> dict[str, object]:
    return {
        "id": "eval-1",
        "name": "eval-1",
        "workspace": "default",
        "experiment_ids": ["experiment-1"],
        "dataset_name": "dataset",
        "metadata": {"eval_config_fileset": "fs-1"},
    }


def test_sync_create_evaluator_result_uses_typed_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/apis/intake/v2/workspaces/default/evaluator-results"
        assert json.loads(request.read()) == {
            "span_id": "span-1",
            "session_id": "session-1",
            "name": "accuracy.score",
            "value": 1.0,
            "data_type": "NUMERIC",
        }
        return httpx.Response(201, request=request, json=_evaluator_result_json())

    client = IntakeClient(
        base_url=BASE,
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.create_evaluator_result(
        body=EvaluatorResultCreateRequest(
            span_id="span-1",
            session_id="session-1",
            name="accuracy.score",
            value=1.0,
            data_type="NUMERIC",
        )
    )

    assert response.data().evaluator_result_id == "eval-result-1"


def test_sync_patch_evaluation_uses_typed_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/apis/intake/v2/workspaces/default/evaluations/eval-1"
        assert json.loads(request.read()) == {"metadata": {"publish_duration_sec": "0.4"}}
        return httpx.Response(200, request=request, json=_evaluation_json())

    client = IntakeClient(
        base_url=BASE,
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.patch_evaluation(
        name="eval-1",
        body=EvaluationPatchRequest(metadata={"publish_duration_sec": "0.4"}),
    )

    assert response.data().name == "eval-1"


@pytest.mark.asyncio
async def test_async_create_atif_uses_typed_transport() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/apis/intake/v2/workspaces/default/ingest/atif"
        assert json.loads(request.content) == {
            "schema_version": "atif-0.1",
            "agent": {"name": "agent", "version": "1"},
            "session_id": "session-1",
        }
        return httpx.Response(204, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncIntakeClient(base_url=BASE, workspace="default", http_client=http_client)

        response = await client.create_atif(
            body=AtifCreateRequest(
                root={
                    "schema_version": "atif-0.1",
                    "agent": {"name": "agent", "version": "1"},
                    "session_id": "session-1",
                }
            )
        )

    assert response.data() is None


@pytest.mark.asyncio
async def test_async_create_otlp_traces_uses_typed_transport() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces"
        assert request.headers["Content-Type"] == "application/x-protobuf"
        assert request.content == b"trace-protobuf"
        return httpx.Response(200, request=request, json={"errors": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncIntakeClient(base_url=BASE, workspace="default", http_client=http_client)

        response = await client.create_otlp_traces(content=b"trace-protobuf")

    assert response.data().errors == []


@pytest.mark.asyncio
async def test_async_list_traces_returns_paginated_items_and_serializes_filter() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/apis/intake/v2/workspaces/default/traces"
        assert request.url.params["mode"] == "detailed"
        assert json.loads(request.url.params["filter"]) == {"session_id": "session-1"}
        return httpx.Response(
            200,
            request=request,
            json={
                "data": [
                    {
                        "id": "trace-1",
                        "root_span_id": "span-1",
                        "session_id": "session-1",
                        "workspace": "default",
                        "started_at": "2026-01-02T03:04:05Z",
                        "status": "OK",
                    }
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncIntakeClient(base_url=BASE, workspace="default", http_client=http_client)

        traces = [
            trace
            async for trace in (
                await client.list_traces(query_params={"mode": "detailed", "filter": {"session_id": "session-1"}})
            ).items()
        ]

    assert len(traces) == 1
    assert traces[0].root_span_id == "span-1"


@pytest.mark.asyncio
async def test_async_list_spans_accepts_deprecated_evaluation_context_aliases() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/apis/intake/v2/workspaces/default/spans"
        assert json.loads(request.url.params["filter"]) == {"evaluation_name": "eval-1"}
        return httpx.Response(
            200,
            request=request,
            json={
                "data": [
                    {
                        "span_id": "span-1",
                        "session_id": "session-1",
                        "workspace": "default",
                        "evaluation_context": {
                            "evaluation_name": "eval-1",
                            "test_case_name": "case-1",
                            "evaluation_id": "eval-1",
                            "test_case_id": "case-1",
                        },
                        "kind": "CHAIN",
                        "source": "otel",
                        "trace_id": "trace-1",
                        "started_at": "2026-01-02T03:04:05Z",
                        "status": "success",
                        "ingested_at": "2026-01-02T03:04:06Z",
                    }
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncIntakeClient(base_url=BASE, workspace="default", http_client=http_client)

        spans = [
            span
            async for span in (await client.list_spans(query_params={"filter": {"evaluation_name": "eval-1"}})).items()
        ]

    assert len(spans) == 1
    assert spans[0].evaluation_context is not None
    assert spans[0].evaluation_context.evaluation_name == "eval-1"
    assert spans[0].evaluation_context.evaluation_id == "eval-1"
