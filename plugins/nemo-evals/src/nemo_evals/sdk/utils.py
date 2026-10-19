# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evals plugin SDK helpers."""

from __future__ import annotations

from nhx_evals_sdk.values.results import (
    AggregatedMetricResult,
    AggregateFieldName,
)


def filter_aggregate_scores(
    aggregate_scores: AggregatedMetricResult,
    aggregate_fields: tuple[AggregateFieldName, ...] | None,
) -> AggregatedMetricResult:
    """Return aggregate scores shaped by the requested fields."""
    if not aggregate_fields:
        return aggregate_scores
    fields = frozenset(aggregate_fields)
    return AggregatedMetricResult(scores=[score.with_fields(fields) for score in aggregate_scores.scores])
