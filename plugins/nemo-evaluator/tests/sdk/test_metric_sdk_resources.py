# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the client.evaluator.metrics SDK resources."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, cast

import httpx
import pytest
from nemo_evaluator.api.schemas import Metric
from nemo_evaluator.sdk.metric_resources import (
    AsyncEvaluatorMetricsResource,
    EvaluatorMetricsResource,
)
from nemo_evaluator.shared.metric_bundles.bundles import (
    MetricBundle,
    MetricBundlePackagerPolicyError,
    bundle_metric,
)
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.protocol import Metric as RuntimeMetric
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
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


def _bundle() -> MetricBundle:
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    return bundle_metric(metric, CloudpickleMetricBundlePackager())


def _metric_response(name: str, bundle: MetricBundle) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return Metric(
        id=f"metric_bundle-{name}",
        name=name,
        workspace="default",
        metric_type=bundle.metric_type,
        description=bundle.metadata.description,
        labels=bundle.metadata.labels,
        outputs=bundle.outputs,
        secrets=bundle.secrets,
        payload_kind=bundle.payload.kind,
        payload_digest=bundle.payload.digest,
        bundle_ref=f"default/metric-bundle.{name}#bundle.json",
        created_at=now,
        updated_at=now,
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


def _sync_resource(
    *payloads: dict[str, Any] | httpx.Response,
) -> tuple[EvaluatorMetricsResource, _Recorder]:
    recorder = _Recorder(*payloads)
    http_client = httpx.Client(transport=httpx.MockTransport(recorder))
    client = EvaluatorClient(base_url="http://localhost:8080", workspace="default", http_client=http_client)
    return EvaluatorMetricsResource(client), recorder


def _async_resource(
    *payloads: dict[str, Any] | httpx.Response,
) -> tuple[AsyncEvaluatorMetricsResource, _Recorder]:
    recorder = _Recorder(*payloads)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(recorder.async_handler))
    client = AsyncEvaluatorClient(base_url="http://localhost:8080", workspace="default", http_client=http_client)
    return AsyncEvaluatorMetricsResource(client), recorder


def _request_url(request: httpx.Request) -> str:
    return str(request.url).split("?", 1)[0]


def _request_body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content.decode())


class _CustomRuntimeMetric:
    """A protocol-satisfying metric that is not inline-bundleable."""

    type = "custom-score"
    description = "custom metric"
    labels: dict[str, str] = {}

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


# ---- sync ------------------------------------------------------------------


def test_sync_create_posts_bundle_and_returns_metric() -> None:
    bundle = _bundle()
    resource, recorder = _sync_resource(_metric_response("exact", bundle))

    result = resource.create("exact", metric=bundle)

    request = recorder.requests[0]
    assert result.name == "exact"
    assert request.method == "POST"
    assert _request_url(request) == f"{_BASE}/metrics/exact"
    body = _request_body(request)
    assert body["metric_type"] == bundle.metric_type
    assert body["payload"]["kind"] == "cloudpickle"


def test_sync_create_defaults_to_inline_for_builtin_metric() -> None:
    bundle = _bundle()
    resource, recorder = _sync_resource(_metric_response("exact", bundle))
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")

    resource.create("exact", metric=metric)

    body = _request_body(recorder.requests[0])
    assert body["payload"]["kind"] == "inline"


def test_sync_create_requires_explicit_packager_for_custom_metric() -> None:
    resource, _ = _sync_resource()

    with pytest.raises(MetricBundlePackagerPolicyError, match="CloudpickleMetricBundlePackager"):
        resource.create("custom", metric=cast(RuntimeMetric, _CustomRuntimeMetric()))


def test_sync_retrieve_targets_item_url() -> None:
    bundle = _bundle()
    resource, recorder = _sync_resource(_metric_response("exact", bundle))

    resource.retrieve("exact")

    request = recorder.requests[0]
    assert request.method == "GET"
    assert _request_url(request) == f"{_BASE}/metrics/exact"


def test_sync_list_returns_data_items() -> None:
    bundle = _bundle()
    resource, _ = _sync_resource(_page([_metric_response("a", bundle), _metric_response("b", bundle)]))

    result = resource.list()

    assert {m.name for m in result.data} == {"a", "b"}


def test_sync_list_encodes_metric_type_filter_and_sort() -> None:
    # metric_type is a custom (data.*) field; the SDK sends it as the route's filter[...] param so a
    # caller can narrow by type without hand-building query strings.
    resource, recorder = _sync_resource(_page([]))

    resource.list(metric_type="exact-match", sort="-created_at")

    params = recorder.requests[0].url.params
    assert params["filter[metric_type]"] == "exact-match"
    assert params["sort"] == "-created_at"


def test_sync_list_omits_include_derived_unless_requested() -> None:
    # Derived (task-internal) metrics are hidden by default: the param is only sent when explicitly set,
    # so the default listing matches the route's own default without a redundant query arg.
    resource, recorder = _sync_resource(_page([]), _page([]))

    resource.list()
    assert "include_derived" not in recorder.requests[0].url.params

    resource.list(include_derived=True)
    assert recorder.requests[1].url.params["include_derived"] == "true"


def test_sync_delete_issues_delete_request() -> None:
    resource, recorder = _sync_resource(httpx.Response(204))

    resource.delete("exact")

    request = recorder.requests[0]
    assert request.method == "DELETE"
    assert _request_url(request) == f"{_BASE}/metrics/exact"


# ---- async -----------------------------------------------------------------


async def test_async_create_posts_bundle_and_returns_metric() -> None:
    bundle = _bundle()
    resource, recorder = _async_resource(_metric_response("exact", bundle))

    result = await resource.create("exact", metric=bundle, workspace="ws1")

    assert result.name == "exact"
    request = recorder.requests[0]
    assert request.method == "POST"
    assert _request_url(request) == "http://localhost:8080/apis/evaluator/v2/workspaces/ws1/metrics/exact"
