# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for the MetricInput/MetricOutput protocol."""

from typing import Any

from nhx_evals_sdk.execution.samples import build_metric_input
from nhx_evals_sdk.metrics.protocol import CorpusMetric, Metric
from nhx_evals_sdk.values import MetricInput, MetricResult


def metric_input(item: dict[str, Any], sample: dict[str, Any] | None = None) -> MetricInput:
    """Build a MetricInput for concise metric unit tests."""
    return build_metric_input(item, sample or {})


async def compute_scores(
    metric: Metric,
    item: dict[str, Any] | None = None,
    sample: dict[str, Any] | None = None,
) -> MetricResult:
    """Invoke the new metric protocol using test row/sample dictionaries."""
    return await metric.compute_scores(metric_input(item or {}, sample or {}))


async def compute_corpus_scores(
    metric: CorpusMetric,
    items: list[dict[str, Any]] | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> MetricResult | None:
    """Invoke the new corpus metric protocol using test row/sample dictionaries."""
    rows = items or []
    sample_rows = samples or [{} for _ in rows]
    return await metric.compute_corpus_scores(
        [build_metric_input(item, sample) for item, sample in zip(rows, sample_rows)]
    )


def output_names(metric: Metric) -> list[str]:
    """Return declared metric output names."""
    return [output.name for output in metric.output_spec()]
