# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from datetime import datetime, timezone

import pytest
from nhx.core.jobs.metrics import JobRunMetrics
from nhx.core.jobs.telemetry import build_job_run_observation, build_job_run_telemetry_from_observation
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families


def _observation(status="completed", details=None, timestamps=True):
    return build_job_run_observation(
        source="private-customer-job",
        status=status,
        status_details=details if details is not None else {"input_tokens": 3, "output_tokens": 5},
        custom_fields=None,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc) if timestamps else None,
        updated_at=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc) if timestamps else None,
    )


def _metrics(reader):
    data = reader.get_metrics_data()
    return {
        metric.name: metric
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }


@pytest.mark.parametrize("status", ["completed", "error", "cancelled"])
def test_metrics_record_all_jobs_without_outbound_session_or_opt_in(monkeypatch, status):
    monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
    observation = _observation(status=status)
    assert build_job_run_telemetry_from_observation(observation, None) is None
    reader = InMemoryMetricReader()
    with ExitStack() as stack:
        provider = MeterProvider(metric_readers=[reader])
        stack.callback(provider.shutdown)
        JobRunMetrics(provider.get_meter("test")).record(observation)
        recorded = _metrics(reader)
        events = recorded["nhx.jobs.run.events.total"].data.data_points
        assert len(events) == 1
        assert events[0].value == 1
        assert dict(events[0].attributes) == {"job_type": "custom", "tool_category": "unknown", "status": status}
        duration = recorded["nhx.jobs.run.duration"]
        assert duration.unit == "s"
        assert duration.data.data_points[0].count == 1
        assert duration.data.data_points[0].sum == 5
        tokens = recorded["nhx.jobs.run.tokens.total"].data.data_points
        assert {point.attributes["direction"]: point.value for point in tokens} == {"input": 3, "output": 5}
        for metric in recorded.values():
            for point in metric.data.data_points:
                assert set(point.attributes) <= {"job_type", "tool_category", "status", "direction"}
                assert "private-customer-job" not in point.attributes.values()


def test_pause_then_completion_does_not_double_count_duration_or_tokens():
    reader = InMemoryMetricReader()
    with ExitStack() as stack:
        provider = MeterProvider(metric_readers=[reader])
        stack.callback(provider.shutdown)
        instruments = JobRunMetrics(provider.get_meter("test"))
        instruments.record(_observation(status="paused"))
        paused = _metrics(reader)
        assert set(paused) == {"nhx.jobs.run.events.total"}
        instruments.record(_observation())
        recorded = _metrics(reader)
        assert sum(point.value for point in recorded["nhx.jobs.run.events.total"].data.data_points) == 2
        assert recorded["nhx.jobs.run.duration"].data.data_points[0].count == 1
        assert sum(point.value for point in recorded["nhx.jobs.run.tokens.total"].data.data_points) == 8


def test_missing_measurements_are_not_reported_as_zero_and_known_zero_is_preserved():
    reader = InMemoryMetricReader()
    with ExitStack() as stack:
        provider = MeterProvider(metric_readers=[reader])
        stack.callback(provider.shutdown)
        instruments = JobRunMetrics(provider.get_meter("test"))
        instruments.record(_observation(details={}, timestamps=False))
        assert set(_metrics(reader)) == {"nhx.jobs.run.events.total"}
        instruments.record(_observation(details={"input_tokens": 0}, timestamps=False))
        recorded = _metrics(reader)
        assert "nhx.jobs.run.duration" not in recorded
        tokens = recorded["nhx.jobs.run.tokens.total"].data.data_points
        assert len(tokens) == 1
        assert tokens[0].attributes["direction"] == "input"
        assert tokens[0].value == 0


def test_existing_prometheus_exporter_exposes_job_metrics():
    registry = CollectorRegistry()
    reader = PrometheusMetricReader(registry=registry)
    with ExitStack() as stack:
        provider = MeterProvider(metric_readers=[reader])
        stack.callback(provider.shutdown)
        JobRunMetrics(provider.get_meter("test")).record(_observation())
        text = generate_latest(registry).decode()
        samples = [sample for family in text_string_to_metric_families(text) for sample in family.samples]
        events = next(sample for sample in samples if sample.name == "nhx_jobs_run_events_total")
        assert events.value == 1
        assert events.labels["status"] == "completed"
        duration = next(sample for sample in samples if sample.name == "nhx_jobs_run_duration_seconds_sum")
        assert duration.value == 5
        tokens = next(
            sample
            for sample in samples
            if sample.name == "nhx_jobs_run_tokens_total" and sample.labels["direction"] == "input"
        )
        assert tokens.value == 3
        assert "private-customer-job" not in text
