# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator-controlled job metrics, independent of anonymous usage telemetry.

Use the platform MeterProvider/exporters. Never add usage-telemetry identifiers,
workspace/job names, arbitrary plugin references, or model names as attributes.
"""

from math import isfinite

from nhx.common.observability import create_counter
from nhx.core.jobs.telemetry import JobRunObservation
from opentelemetry import metrics


class JobRunMetrics:
    """Record accepted lifecycle transitions once, not outbound delivery retries."""

    def __init__(self, meter: metrics.Meter) -> None:
        self.events = create_counter(
            meter=meter,
            subsystem="jobs",
            name="run.events.total",
            description="Accepted job-attempt completion, error, cancellation, and pause transitions.",
        )
        self.duration = meter.create_histogram(
            "nhx.jobs.run.duration",
            unit="s",
            description="Final attempt wall-clock duration including queue and pause time; excludes pause observations.",
            explicit_bucket_boundaries_advisory=(1, 10, 60, 300, 900, 3600, 14400, 86400),
        )
        self.tokens = create_counter(
            meter=meter,
            subsystem="jobs",
            name="run.tokens.total",
            description="Known terminal job-summary token counts by direction; not complete inference usage.",
        )

    def record(self, observation: JobRunObservation) -> None:
        attributes = {
            "job_type": observation.job_type,
            "tool_category": observation.tool_category,
            "status": observation.status,
        }
        self.events.add(1, attributes=attributes)
        # A pause is not a finished run. Resuming and finishing the same attempt
        # must not count a partial duration/token summary in addition to the final one.
        if observation.status not in {"completed", "error", "cancelled"}:
            return
        if isfinite(observation.duration_sec) and observation.duration_sec >= 0:
            self.duration.record(observation.duration_sec, attributes=attributes)
        for direction, count in (("input", observation.input_tokens), ("output", observation.output_tokens)):
            if count >= 0:
                self.tokens.add(count, attributes={**attributes, "direction": direction})


_job_run_metrics = JobRunMetrics(metrics.get_meter(__name__))


def record_job_run_metrics(observation: JobRunObservation) -> None:
    _job_run_metrics.record(observation)
