# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the client.evaluator.{agent_eval_results,eval_results} SDK resources.

Drives the resources through the typed evaluator client, asserting the route they target and that the
response JSON is deserialized into the public SDK DTO.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from nemo_evaluator.api.schemas import AgentEvalResult, EvaluateResult
from nemo_evaluator.sdk.result_resources import (
    AsyncEvaluatorEvalResultsResource,
    EvaluatorAgentEvalResultsResource,
    EvaluatorEvalResultsResource,
)
from nemo_evaluator_sdk.values.results import AggregatedMetricResult
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient

_BASE = "http://localhost:8080/apis/evaluator/v2/workspaces/default"


class _Recorder:
    def __init__(self, *payloads: dict[str, Any] | httpx.Response) -> None:
        self.payloads = list(payloads)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        payload = self.payloads.pop(0)
        if isinstance(payload, httpx.Response):
            return payload
        return httpx.Response(200, json=payload)

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        return self(request)


def _agent_payload(name: str, *, target_kind: str = "fabric", target_name: str = "openai/gpt-5.4") -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return AgentEvalResult(
        id=f"agent_eval_result-{name}",
        name=name,
        workspace="default",
        job_id=name,
        target_kind=target_kind,
        target_name=target_name,
        target_url=None,
        scores=AggregatedMetricResult(scores=[]),
        bundle_ref="fileset://default/agent-eval-results#b",
        created_at=now,
        updated_at=now,
    ).model_dump(mode="json")


def _eval_payload(name: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return EvaluateResult(
        id=f"evaluate_result-{name}",
        name=name,
        workspace="default",
        job_id=name,
        target_kind="model",
        target_name="m",
        target_url="https://m.test/v1/chat/completions",
        scores=AggregatedMetricResult(scores=[]),
        bundle_ref="fileset://default/eval-results#b",
        created_at=now,
        updated_at=now,
        dataset_ref="default/ds",
        metric_types=["exact_match"],
    ).model_dump(mode="json")


def _page(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": items,
        "pagination": {
            "page": 1,
            "page_size": 100,
            "current_page_size": len(items),
            "total_pages": 1,
            "total_results": len(items),
        },
    }


def _sync_agent_resource(
    *payloads: dict[str, Any] | httpx.Response,
) -> tuple[EvaluatorAgentEvalResultsResource, _Recorder]:
    recorder = _Recorder(*payloads)
    http_client = httpx.Client(transport=httpx.MockTransport(recorder))
    client = EvaluatorClient(base_url="http://localhost:8080", workspace="default", http_client=http_client)
    return EvaluatorAgentEvalResultsResource(client), recorder


def _sync_eval_resource(
    *payloads: dict[str, Any] | httpx.Response,
) -> tuple[EvaluatorEvalResultsResource, _Recorder]:
    recorder = _Recorder(*payloads)
    http_client = httpx.Client(transport=httpx.MockTransport(recorder))
    client = EvaluatorClient(base_url="http://localhost:8080", workspace="default", http_client=http_client)
    return EvaluatorEvalResultsResource(client), recorder


def _async_eval_resource(
    *payloads: dict[str, Any] | httpx.Response,
) -> tuple[AsyncEvaluatorEvalResultsResource, _Recorder]:
    recorder = _Recorder(*payloads)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(recorder.async_handler))
    client = AsyncEvaluatorClient(base_url="http://localhost:8080", workspace="default", http_client=http_client)
    return AsyncEvaluatorEvalResultsResource(client), recorder


def _request_url(request: httpx.Request) -> str:
    return str(request.url).split("?", 1)[0]


# ---- sync ------------------------------------------------------------------


def test_sync_retrieve_agent_eval_targets_item_url_and_parses_dto() -> None:
    resource, recorder = _sync_agent_resource(_agent_payload("job-1"))

    result = resource.retrieve("job-1")

    request = recorder.requests[0]
    assert isinstance(result, AgentEvalResult)
    assert result.job_id == "job-1"
    assert result.target_kind == "fabric"
    assert request.method == "GET"
    assert _request_url(request) == f"{_BASE}/agent-eval-results/job-1"


def test_sync_retrieve_agent_eval_parses_a_retired_runner_kind() -> None:
    # The wire DTO must keep reading rows written by a runner that has since been removed; nothing
    # in the response schema constrains `target_kind` to the runners that currently exist.
    resource, _ = _sync_agent_resource(_agent_payload("job-legacy", target_kind="codex", target_name="gpt-5.5"))

    result = resource.retrieve("job-legacy")

    assert isinstance(result, AgentEvalResult)
    assert result.target_kind == "codex"
    assert result.target_name == "gpt-5.5"


def test_sync_list_eval_results_parses_dtos_and_targets_collection() -> None:
    resource, recorder = _sync_eval_resource(_page([_eval_payload("a"), _eval_payload("b")]))

    page = resource.list(sort="-created_at")

    request = recorder.requests[0]
    assert {r.name for r in page.data} == {"a", "b"}
    assert all(isinstance(r, EvaluateResult) for r in page.data)
    assert page.data[0].dataset_ref == "default/ds"
    assert page.pagination is not None and page.pagination.total_results == 2
    assert request.method == "GET"
    assert _request_url(request) == f"{_BASE}/eval-results"
    assert request.url.params["sort"] == "-created_at"


def test_sync_list_encodes_trait_filters_as_bracket_params() -> None:
    # The route filters via filter[field]=value bracket params; the SDK must encode them so a
    # caller can narrow by job/target/dataset without hand-building query strings.
    resource, recorder = _sync_eval_resource(_page([]))

    resource.list(job_id="j1", target_kind="model", dataset_ref="ws/ds")

    params = recorder.requests[0].url.params
    assert params["filter[job_id]"] == "j1"
    assert params["filter[target_kind]"] == "model"
    assert params["filter[dataset_ref]"] == "ws/ds"
    # Unset filters are omitted entirely (no empty filter[...] keys).
    assert "filter[target_name]" not in params


def test_sync_list_parses_payload_with_none_fields_omitted() -> None:
    # Regression guard: the list route serializes with response_model_exclude_none, so an offline
    # result (no target / inline dataset) arrives with target_*/dataset_ref absent. The DTO must
    # still deserialize; a live round-trip caught this, so this locks it.
    item = _eval_payload("offline")
    for dropped in ("target_kind", "target_name", "target_url", "dataset_ref"):
        item.pop(dropped, None)
    resource, _ = _sync_eval_resource(_page([item]))

    (result,) = resource.list().data

    assert result.target_kind is None
    assert result.target_name is None
    assert result.target_url is None
    assert result.dataset_ref is None
    assert result.metric_types == ["exact_match"]


def test_sync_delete_issues_delete_request() -> None:
    resource, recorder = _sync_eval_resource(httpx.Response(204))

    resource.delete("job-1")

    request = recorder.requests[0]
    assert request.method == "DELETE"
    assert _request_url(request) == f"{_BASE}/eval-results/job-1"


# ---- async -----------------------------------------------------------------


async def test_async_retrieve_eval_result_parses_dto() -> None:
    resource, recorder = _async_eval_resource(_eval_payload("job-9"))

    result = await resource.retrieve("job-9")

    request = recorder.requests[0]
    assert isinstance(result, EvaluateResult)
    assert result.metric_types == ["exact_match"]
    assert request.method == "GET"
    assert _request_url(request) == f"{_BASE}/eval-results/job-9"
