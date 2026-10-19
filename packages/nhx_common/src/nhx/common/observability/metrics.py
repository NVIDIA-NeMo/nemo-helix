# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from opentelemetry import metrics


def _format_metric_name(namespace: str, subsystem: str, name: str) -> str:
    metric_name = f"{namespace}.{name}"
    if subsystem:
        metric_name = f"{namespace}.{subsystem}.{name}"

    # Replace any underscores with dots to follow OTEL naming conventions.
    # Prometheus will replace dots with underscores when scraping, so this will not affect Prometheus metrics.
    metric_name = metric_name.replace("_", ".")
    return metric_name


def create_counter(
    meter: metrics.Meter, subsystem: str, name: str, description: str, namespace: str = "nhx"
) -> metrics.Counter:
    """
    Helper function to create a counter metric with standardized naming.

    Args:
        meter (metrics.Meter): The OpenTelemetry meter instance.
        subsystem (str): The subsystem name for the metric.
        name (str): The specific name of the metric.
        description (str): A description of the metric.
        namespace (str, optional): The namespace for the metric. Defaults to "nhx".
    """

    return meter.create_counter(
        name=_format_metric_name(namespace, subsystem, name),
        description=description,
    )


def create_observable_gauge(
    meter: metrics.Meter,
    subsystem: str,
    name: str,
    description: str,
    callbacks: list,
    namespace: str = "nhx",
) -> metrics.ObservableGauge:
    """
    Create an observable gauge with standardized naming.

    Use this for metrics that report a value (and optional attributes) when observed,
    e.g. an "info" gauge that reports 1 with labels like revision or version for
    filtering in dashboards.

    Args:
        meter: The OpenTelemetry meter instance.
        subsystem: The subsystem name for the metric.
        name: The specific name of the metric.
        description: A description of the metric.
        callbacks: List of callbacks that accept CallbackOptions and yield Observation.
        namespace: The namespace for the metric. Defaults to "nhx".

    Returns:
        The created ObservableGauge instrument.
    """
    return meter.create_observable_gauge(
        name=_format_metric_name(namespace, subsystem, name),
        callbacks=callbacks,
        description=description,
    )
