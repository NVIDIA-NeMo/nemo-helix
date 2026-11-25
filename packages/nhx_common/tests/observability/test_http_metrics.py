# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import httpx
import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Request
from nhx.common.observability import tracing
from nhx.common.observability.otel import settings
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import Histogram, InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from starlette.responses import Response, StreamingResponse


@pytest.mark.asyncio
async def test_otel_http_duration_uses_full_matched_route_for_each_outcome(monkeypatch, request: pytest.FixtureRequest):
    app = FastAPI()
    router = APIRouter()

    @app.middleware("http")
    async def reject_before_routing(request: Request, call_next):
        if request.headers.get("x-test-denied"):
            return Response(status_code=403)
        return await call_next(request)

    async def interrupted_stream():
        yield b"partial"
        raise RuntimeError("stream interrupted")

    @router.get("/deployments/{name}")
    async def deployment(name: str):
        if name == "denied":
            raise HTTPException(status_code=403)
        if name == "bad-upstream":
            raise HTTPException(status_code=502)
        if name == "crash":
            raise RuntimeError("unexpected error before response")
        if name == "stream":
            return StreamingResponse(interrupted_stream())
        return {"name": name}

    service = APIRouter()
    service.include_router(router, prefix="/v2/workspaces/{workspace}")
    app.include_router(service, prefix="/apis/agents")
    app.include_router(service, prefix="/apis/models")

    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    tracer_provider = TracerProvider(sampler=ALWAYS_OFF, shutdown_on_exit=False)
    request.addfinalizer(meter_provider.shutdown)
    request.addfinalizer(tracer_provider.shutdown)
    instrument_app = FastAPIInstrumentor.instrument_app

    def instrument_with_local_providers(app, **kwargs):
        instrument_app(app, meter_provider=meter_provider, tracer_provider=tracer_provider, **kwargs)

    monkeypatch.setattr(FastAPIInstrumentor, "instrument_app", staticmethod(instrument_with_local_providers))
    monkeypatch.setattr(settings, "otel_metrics_exporter", "none")
    tracing.setup_fastapi_instrumentations(app)
    request.addfinalizer(lambda: FastAPIInstrumentor.uninstrument_app(app))

    agents = "/apis/agents/v2/workspaces/demo/deployments"
    models = "/apis/models/v2/workspaces/demo/deployments"
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get(f"{agents}/one?q=sample")).status_code == 200
        assert (await client.get(f"{agents}/denied")).status_code == 403
        assert (await client.get(f"{models}/bad-upstream")).status_code == 502
        assert (await client.get(f"{models}/crash")).status_code == 500
        assert (await client.get(f"{agents}/one", headers={"x-test-denied": "true"})).status_code == 403
        assert (await client.get("/apis/unknown/private-id")).status_code == 404
        stream = await client.get(f"{models}/stream")
        assert stream.status_code == 200
        assert stream.content == b"partial"

    recorded = reader.get_metrics_data()
    assert recorded is not None
    metrics = [
        metric
        for resource in recorded.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == "http.server.duration"
    ]
    [duration] = metrics
    assert duration.unit == "ms"
    assert isinstance(duration.data, Histogram)
    counts = Counter()
    for point in duration.data.data_points:
        assert point.attributes is not None
        counts[point.attributes.get("http.target"), point.attributes["http.status_code"]] += point.count

    agents_template = "/apis/agents/v2/workspaces/{workspace}/deployments/{name}"
    models_template = "/apis/models/v2/workspaces/{workspace}/deployments/{name}"
    assert counts == {
        (agents_template, 200): 1,
        (agents_template, 403): 1,
        (models_template, 200): 1,
        (models_template, 500): 1,
        (models_template, 502): 1,
        (None, 403): 1,
        (None, 404): 1,
    }
