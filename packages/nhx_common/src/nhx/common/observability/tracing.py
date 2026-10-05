# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry tracing, metrics, and instrumentation setup.

NOTE: Imports of gRPC exporters, Prometheus readers, and instrumentor
libraries are intentionally deferred inside functions for startup
performance. Do not hoist them to module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from nhx.common.observability.context import AppContextSpanProcessor
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from .otel import settings

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_otel_resource(attributes: Dict[str, str] | None = None) -> Resource:
    """Create the OTEL Resource for this process.

    The resource is attached to all spans (traces) and metrics. Pass optional
    attributes (e.g. platform version) so they appear on every span.

    Args:
        attributes: Optional key-value pairs to merge into the resource (e.g.
            {"nhx.platform.platform_version": "26.2.0"}). Merged attributes take precedence.
    """
    base = Resource.create()
    if attributes:
        return base.merge(Resource.create(attributes))
    return base


def initialize_metrics(resource: Resource | None = None):
    """
    Initialize OpenTelemetry metrics

    We support exporters based the environment variable OTEL_METRICS_EXPORTER. Options are:
    - prometheus: Exposes Prometheus metrics endpoint
    - otlp: Exports metrics to an OTLP endpoint specified in OTEL_EXPORTER_OTLP_METRICS_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT
    - none: No metrics, uses InMemoryMetricReader
    """
    if resource is None:
        resource = create_otel_resource()
    if settings.otel_metrics_exporter == "prometheus":
        # NOTE: deferred import -- only reached when prometheus exporter is configured
        from opentelemetry.exporter.prometheus import PrometheusMetricReader

        reader = PrometheusMetricReader()
    elif settings.otel_metrics_exporter == "otlp":
        # NOTE: deferred imports -- only reached when OTLP exporter is configured
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=settings.otel_exporter_otlp_metrics_endpoint or settings.otel_exporter_otlp_endpoint
            )
        )
    elif settings.otel_metrics_exporter in ("", "none"):
        reader = InMemoryMetricReader()
    else:
        raise ValueError(f"Unsupported OTEL_METRICS_EXPORTER: {settings.otel_metrics_exporter}")
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))


def initialize_tracing(resource: Resource | None = None):
    """
    Initialize OpenTelemetry tracing

    We support exporters based the environment variable OTEL_TRACES_EXPORTER. Options are:
    - none: No span export, but tracing context/baggage still works for propagation
    - otlp: Exports spans to an OTLP endpoint specified in OTEL_ENDPOINT

    Request context (auth, workspace) is attached to spans by
    ``AppContextSpanProcessor``.
    """
    if resource is None:
        resource = create_otel_resource()

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(AppContextSpanProcessor())

    if settings.otel_traces_exporter == "otlp":
        # NOTE: deferred imports -- only reached when OTLP exporter is configured
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_traces_endpoint or settings.otel_exporter_otlp_endpoint
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    elif settings.otel_traces_exporter not in ("", "none"):
        raise ValueError(f"Unsupported OTEL_TRACES_EXPORTER: {settings.otel_traces_exporter}")
    trace.set_tracer_provider(provider)


def setup_fastapi_instrumentations(app: FastAPI):
    """
    Setup FastAPI instrumentation with OpenTelemetry.

    This function instruments the provided FastAPI application for metrics and tracing,
    excluding health and metrics endpoints from instrumentation.
    """
    # NOTE: deferred imports -- middleware and instrumentors are only needed at
    # server startup time, not during module loading.
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    from .middleware import RequestLoggingMiddleware

    app.add_middleware(RequestLoggingMiddleware)

    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="/health/live,/health/ready,/metrics",
        exclude_spans=["receive", "send"],
    )

    if settings.otel_metrics_exporter == "prometheus":
        from prometheus_fastapi_instrumentator import Instrumentator as PrometheusInstrumentator

        PrometheusInstrumentator().instrument(app).expose(app)


def setup_global_instrumentations(
    httpx: bool = True,
    sqlalchemy: bool = True,
    system_metrics: bool = False,
):
    """
    Setup common global OpenTelemetry instrumentations.

    We provide options to instrument automatically, you might also choose to instrument
    manually in your application code:
    - System metrics
    - SQLAlchemy
    - HTTPX clients
    """
    # NOTE: instrumentor imports are intentionally deferred -- each library
    # pulls in its own dependency tree and is only needed when the flag is set.

    if sqlalchemy:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument()

    if httpx:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()

    if system_metrics:
        from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor

        SystemMetricsInstrumentor().instrument()
