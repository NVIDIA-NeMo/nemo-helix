# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evals SDK metrics for normalized captured agent runs."""

from collections.abc import Sequence

from evaluator_agent_eval.metrics.surface import (
    LEGACY_SURFACE,
    LEGACY_TEXT_PATTERNS,
    LegacySurfaceAvoidanceMetric,
    SurfaceAdherenceMetric,
)
from evaluator_agent_eval.metrics.trajectory import TrajectoryEvidenceMetric
from nhx_evals_sdk.metrics.base import Metric

SURFACE_FIELD_KEYS = {
    "observed_surfaces_key": "observed_surfaces",
    "allowed_surfaces_key": "allowed_surfaces",
    "forbidden_surfaces_key": "forbidden_surfaces",
    "forbidden_surface_hits_key": "forbidden_surface_hits",
}
TRAJECTORY_FIELD_KEYS = {
    "trajectory_summary_key": "trajectory_summary",
    "tool_call_count_key": "tool_call_count",
    "failed_command_count_key": "failed_command_count",
    "recovery_event_count_key": "recovery_event_count",
}

__all__ = [
    "LegacySurfaceAvoidanceMetric",
    "SurfaceAdherenceMetric",
    "TrajectoryEvidenceMetric",
    "default_agent_eval_metrics",
]


def default_agent_eval_metrics() -> Sequence[Metric]:
    """Return the MVP metric set for captured agent-run scoring."""
    return [
        SurfaceAdherenceMetric(**SURFACE_FIELD_KEYS),
        LegacySurfaceAvoidanceMetric(
            observed_surfaces_key="observed_surfaces",
            forbidden_surface_hits_key="forbidden_surface_hits",
            output_text_key="output_text",
            legacy_surface=LEGACY_SURFACE,
            legacy_text_patterns=LEGACY_TEXT_PATTERNS,
        ),
        TrajectoryEvidenceMetric(**TRAJECTORY_FIELD_KEYS),
    ]
