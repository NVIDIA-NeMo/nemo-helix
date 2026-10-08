# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Surface-adherence metrics for evaluator agent benchmark rows."""

from math import nan

from evaluator_agent_eval.surfaces import contains_nonnegated_substring
from nhx_evals_sdk.values.results import MetricResult, MetricScore

LEGACY_SURFACE = "legacy_service"
LEGACY_TEXT_PATTERNS = ("services/", "services\\")


class SurfaceAdherenceMetric:
    """Score whether observed surfaces stayed within the task constraint."""

    def __init__(
        self,
        *,
        observed_surfaces_key: str,
        allowed_surfaces_key: str,
        forbidden_surfaces_key: str,
        forbidden_surface_hits_key: str,
    ) -> None:
        self.observed_surfaces_key = observed_surfaces_key
        self.allowed_surfaces_key = allowed_surfaces_key
        self.forbidden_surfaces_key = forbidden_surfaces_key
        self.forbidden_surface_hits_key = forbidden_surface_hits_key

    @property
    def type(self) -> str:
        return "agent_eval/surface_adherence"

    def score_names(self) -> list[str]:
        return ["surface_adherence", "surface_violation_count"]

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        try:
            observed_surfaces = item[self.observed_surfaces_key]
            allowed_surfaces = item[self.allowed_surfaces_key]
            forbidden_surfaces = item[self.forbidden_surfaces_key]
            forbidden_surface_hits = item[self.forbidden_surface_hits_key]
            if (
                not isinstance(observed_surfaces, list)
                or not isinstance(allowed_surfaces, list)
                or not isinstance(forbidden_surfaces, list)
                or not isinstance(forbidden_surface_hits, list)
                or not all(isinstance(entry, str) for entry in observed_surfaces)
                or not all(isinstance(entry, str) for entry in allowed_surfaces)
                or not all(isinstance(entry, str) for entry in forbidden_surfaces)
                or not all(isinstance(entry, str) for entry in forbidden_surface_hits)
            ):
                raise TypeError("Surface fields must be list[str]")
            observed = set(observed_surfaces)
            allowed = set(allowed_surfaces)
            forbidden = set(forbidden_surfaces)
            disallowed_observed = observed.difference(allowed)
            forbidden_observed = observed.intersection(forbidden)
            violation_count = len(disallowed_observed.union(forbidden_observed)) + len(forbidden_surface_hits)
            adherence = 1.0 if violation_count == 0 else 0.0
        except (KeyError, TypeError):
            adherence = nan
            violation_count = nan
        return MetricResult(
            scores=[
                MetricScore(name="surface_adherence", value=adherence),
                MetricScore(name="surface_violation_count", value=float(violation_count)),
            ]
        )


class LegacySurfaceAvoidanceMetric:
    """Penalize legacy Evals service evidence separately from raw success."""

    def __init__(
        self,
        *,
        observed_surfaces_key: str,
        forbidden_surface_hits_key: str,
        output_text_key: str,
        legacy_surface: str,
        legacy_text_patterns: tuple[str, ...],
    ) -> None:
        self.observed_surfaces_key = observed_surfaces_key
        self.forbidden_surface_hits_key = forbidden_surface_hits_key
        self.output_text_key = output_text_key
        self.legacy_surface = legacy_surface
        self.legacy_text_patterns = legacy_text_patterns

    @property
    def type(self) -> str:
        return "agent_eval/legacy_surface_avoidance"

    def score_names(self) -> list[str]:
        return ["legacy_surface_avoidance", "legacy_surface_hit_count"]

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        try:
            observed_surfaces = item[self.observed_surfaces_key]
            forbidden_surface_hits = item[self.forbidden_surface_hits_key]
            output_text = item[self.output_text_key]
            if (
                not isinstance(observed_surfaces, list)
                or not isinstance(forbidden_surface_hits, list)
                or not isinstance(output_text, str)
                or not all(isinstance(entry, str) for entry in observed_surfaces)
                or not all(isinstance(entry, str) for entry in forbidden_surface_hits)
            ):
                raise TypeError("Legacy surface fields must be observed list[str], hit list[str], and output text str")
            hits = list(forbidden_surface_hits)
            lowered = output_text.lower()
            hits.extend(
                pattern for pattern in self.legacy_text_patterns if contains_nonnegated_substring(lowered, pattern)
            )
            if self.legacy_surface in observed_surfaces:
                hits.append(self.legacy_surface)
            hit_count = len(hits)
            avoidance = float(hit_count == 0)
        except (KeyError, TypeError):
            hit_count = nan
            avoidance = nan
        return MetricResult(
            scores=[
                MetricScore(name="legacy_surface_avoidance", value=avoidance),
                MetricScore(name="legacy_surface_hit_count", value=float(hit_count)),
            ]
        )
