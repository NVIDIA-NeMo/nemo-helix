# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local metric preparation helpers for evaluator SDK runtime."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TypeGuard, cast

from nemo_evaluator_sdk.execution._protocols import JobParamsConfigurableMetric
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricWithModels, MetricWithPreflight, MetricWithSecrets
from nemo_evaluator_sdk.metrics.utils import metric_type_name
from nemo_evaluator_sdk.resolver_protocols import ModelResolver, SecretResolver
from nemo_evaluator_sdk.values.params import RunConfig, RunConfigOnline, RunConfigOnlineModel
from pydantic import BaseModel


def unique_metric_keys(metrics: Sequence[Metric]) -> list[str]:
    """Assign stable unique keys to a sequence of metrics.

    Args:
        metrics: Metrics submitted in one evaluator call.

    Returns:
        Unique metric keys in the same order as the input metrics.
    """

    # A type name ending in ``_<n>`` can equal another type's suffixed key, so bump past collisions.
    seen: dict[str, int] = {}
    used: set[str] = set()
    keys: list[str] = []
    for metric in metrics:
        base = metric_type_name(metric)
        suffix = seen.get(base, 0) + 1
        key = base if suffix == 1 else f"{base}_{suffix}"
        while key in used:
            suffix += 1
            key = f"{base}_{suffix}"
        seen[base] = suffix
        used.add(key)
        keys.append(key)
    return keys


def is_metric(metrics: object) -> TypeGuard[Metric]:
    """Return whether a value is the single-metric form."""
    if isinstance(metrics, Metric):
        return True
    return False


def is_metric_sequence(metrics: object) -> TypeGuard[Sequence[Metric]]:
    """Return whether a value is the benchmark/multi-metric form."""
    if not isinstance(metrics, Metric) and isinstance(metrics, Sequence) and not isinstance(metrics, (str, bytes)):
        return all(isinstance(metric, Metric) for metric in metrics)
    return False


def copy_metric(metric: Metric) -> Metric:
    """Create a best-effort isolated copy of a metric instance.

    Preparation mutates metrics in place (runtime params, resolver hydration,
    preflight state), so backends copy first to avoid side effects on the
    caller's original metric object.
    """
    if isinstance(metric, BaseModel):
        return cast(Metric, metric.model_copy(deep=True))

    try:
        return copy.deepcopy(metric)
    except Exception as exc:
        raise TypeError(
            f"Cannot copy metric {type(metric).__name__}; use a Pydantic model or ensure the metric "
            "supports copy.deepcopy() (for example, by implementing __deepcopy__)."
        ) from exc


async def prepare_metric_for_execution(
    metric: Metric,
    *,
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel,
    model_resolver: ModelResolver,
    secret_resolver: SecretResolver,
) -> Metric:
    """Copy and prepare one metric for execution.

    Args:
        metric: User-provided metric instance.
        params: Materialized execution params for this run.
        model_resolver: Resolver used for any ``ModelRef`` fields.
        secret_resolver: Resolver used for any ``SecretRef`` fields.

    Returns:
        A copied metric ready for execution.
    """
    prepared_metric = copy_metric(metric)
    if isinstance(prepared_metric, JobParamsConfigurableMetric):
        prepared_metric.apply_evaluation_job_params(params)
    if isinstance(prepared_metric, MetricWithModels):
        await prepared_metric.resolve_models(model_resolver)
    if isinstance(prepared_metric, MetricWithSecrets):
        await prepared_metric.resolve_secrets(secret_resolver)
    if isinstance(prepared_metric, MetricWithPreflight):
        await prepared_metric.preflight()
    return prepared_metric
