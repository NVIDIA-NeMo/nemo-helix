# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace metrics API tests."""

from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.routing import APIRoute
from nhx.intake.spans.api import trace_metrics, traces
from nhx.intake.spans.api.trace_metrics import (
    _default_started_at_window,
    _validate_query_params,
    _validate_timezone,
)
from nhx.intake.spans.api.trace_metrics_schemas import (
    TraceMetricBucketParam,
    TraceMetricPointResponse,
    TraceMetrics,
)
from nhx.intake.spans.domain import (
    CostRollup,
    LatencyRollup,
    TokenRollup,
    TraceListFilter,
    TraceMetricPoint,
)
from starlette.routing import Match


def test_validate_timezone_accepts_iana_zones() -> None:
    _validate_timezone("UTC")
    _validate_timezone("America/Los_Angeles")


@pytest.mark.parametrize("bad", ["Not/AZone", "", "PST8PDT7"])
def test_validate_timezone_rejects_unknown_zones(bad: str) -> None:
    # ClickHouse would otherwise fail with an opaque server error.
    with pytest.raises(HTTPException) as exc:
        _validate_timezone(bad)
    assert exc.value.status_code == 400


def _metric_point(
    *,
    run_count: int = 0,
    failed_run_count: int = 0,
    bucket_start: datetime | None = None,
    input_tokens: TokenRollup | None = None,
    output_tokens: TokenRollup | None = None,
    cached_tokens: TokenRollup | None = None,
    total_tokens: TokenRollup | None = None,
    cost_usd: CostRollup | None = None,
    latency_ms: LatencyRollup | None = None,
) -> TraceMetricPoint:
    """Every rollup object is required; name only the ones a test cares about."""

    return TraceMetricPoint(
        bucket_start=bucket_start,
        run_count=run_count,
        failed_run_count=failed_run_count,
        input_tokens=input_tokens or TokenRollup(),
        output_tokens=output_tokens or TokenRollup(),
        cached_tokens=cached_tokens or TokenRollup(),
        total_tokens=total_tokens or TokenRollup(),
        cost_usd=cost_usd or CostRollup(),
        latency_ms=latency_ms or LatencyRollup(),
    )


def test_metric_point_response_round_trips_the_domain_model() -> None:
    point = _metric_point(
        bucket_start=datetime(2026, 8, 14, tzinfo=timezone.utc),
        run_count=3,
        failed_run_count=1,
        input_tokens=TokenRollup(sum=300, mean=100.0, p90=180.0, p99=190.0),
        latency_ms=LatencyRollup(mean=2100.0, p95=4300.0),
    )

    response = TraceMetricPointResponse.from_domain(point)

    assert response.bucket_start == point.bucket_start
    assert response.run_count == 3
    assert response.failed_run_count == 1
    assert response.input_tokens.sum == 300
    assert response.input_tokens.mean == 100.0
    assert response.latency_ms.p95 == 4300.0
    # Unset metrics stay null rather than defaulting to zero.
    assert response.cost_usd.sum is None
    assert response.latency_ms.p50 is None


def test_total_bucket_omits_bucket_start_from_the_payload() -> None:
    metrics = TraceMetrics(
        bucket=TraceMetricBucketParam.TOTAL,
        timezone="UTC",
        data=[TraceMetricPointResponse.from_domain(_metric_point(run_count=5))],
    )

    payload = metrics.model_dump(exclude_none=True)

    assert payload["bucket"] == "total"
    assert "bucket_start" not in payload["data"][0]
    assert payload["data"][0]["run_count"] == 5


def test_unspecified_range_defaults_to_the_last_seven_days() -> None:
    filters = _default_started_at_window(TraceListFilter(workspace="workspace-a"))

    assert filters.started_at_gte is not None
    window = datetime.now(timezone.utc) - filters.started_at_gte
    assert timedelta(days=7) <= window < timedelta(days=7, minutes=1)


def test_default_window_anchors_to_an_explicit_upper_bound() -> None:
    upper = datetime(2026, 8, 14, tzinfo=timezone.utc)

    filters = _default_started_at_window(TraceListFilter(workspace="workspace-a", started_at_lte=upper))

    assert filters.started_at_gte == upper - timedelta(days=7)


def test_default_window_leaves_an_explicit_lower_bound_alone() -> None:
    lower = datetime(2025, 1, 1, tzinfo=timezone.utc)

    filters = _default_started_at_window(TraceListFilter(workspace="workspace-a", started_at_gte=lower))

    assert filters.started_at_gte == lower


@pytest.mark.parametrize("param", ["page", "page_size", "sort"])
def test_rejects_pagination_params_the_endpoint_does_not_implement(param: str) -> None:
    # The shared list validator allows these; accepting them would imply the
    # response is paginated when it returns every bucket in the filtered range.
    request = Request({"type": "http", "query_string": f"{param}=2".encode(), "headers": []})

    with pytest.raises(HTTPException) as exc:
        _validate_query_params(request)
    assert exc.value.status_code == 400
    assert param in str(exc.value.detail)


@pytest.mark.parametrize("query", [b"", b"bucket=day", b"timezone=UTC", b"filter[agent_name]=x"])
def test_accepts_supported_params_and_deep_object_filters(query: bytes) -> None:
    request = Request({"type": "http", "query_string": query, "headers": []})

    _validate_query_params(request)


def test_metrics_route_is_registered_ahead_of_the_trace_id_route() -> None:
    """/traces/metrics must not be swallowed by /traces/{id}."""

    app = FastAPI()
    for router in (trace_metrics.router, traces.router):
        app.include_router(router)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v2/workspaces/workspace-a/traces/metrics",
        "path_params": {},
        "root_path": "",
        "headers": [],
    }

    matched = next(route for route in _api_routes(app.routes) if route.matches(scope)[0] is Match.FULL)

    assert matched.endpoint is trace_metrics.get_trace_metrics


def _api_routes(routes: Iterable[object]) -> Iterator[APIRoute]:
    """Flatten routes in match order; include_router wraps each router in a proxy."""

    for route in routes:
        if isinstance(route, APIRoute):
            yield route
            continue
        included = getattr(route, "original_router", None)
        if included is not None:
            yield from _api_routes(included.routes)


def test_metrics_route_documents_its_error_responses() -> None:
    """400 and 503 are both reachable, so generated clients must know about them."""

    route = next(
        route for route in _api_routes(trace_metrics.router.routes) if route.endpoint is trace_metrics.get_trace_metrics
    )

    assert set(route.responses) == {400, 503}
    for status_code in (400, 503):
        schema = route.responses[status_code]["content"]["application/json"]["schema"]
        assert schema["required"] == ["detail"]
