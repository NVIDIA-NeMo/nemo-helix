# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-match metric runtime implementation."""

from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.template_rendering import render_reference_and_candidate, template_metric_repr
from nhx_evals_sdk.metrics.utils import normalize_text
from nhx_evals_sdk.values.metrics import ExactMatch

__all__ = ["ExactMatchMetric"]


class ExactMatchMetric(ExactMatch):
    """Exact-match metric runtime for evaluator-driven execution.

    Evaluator-driven runs expose dataset values through ``item`` and generated
    model outputs through ``sample.output_text`` for online execution.
    """

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return outputs emitted by this metric."""
        return [MetricOutputSpec.continuous_score(self.type.value)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute structured metric output for one item/sample pair.

        The algorithm renders reference and candidate text from templates, then
        normalizes both strings (case, punctuation, articles, and whitespace)
        before equality comparison.

        Args:
            input: Original dataset row paired with candidate output.

        Returns:
            ``MetricResult`` with one exact-match score entry.

        Raises:
            TypeError: If rendered reference or candidate is not a string.
            ValueError: If template rendering fails or ``candidate`` is omitted
                and ``sample.output_text`` is missing.
        """
        ground_truth, prediction = render_reference_and_candidate(
            metric_repr=template_metric_repr(self),
            metric_name=self.__class__.__name__,
            reference_template=self.reference,
            candidate_template=self.candidate,
            item=input.row.data,
            sample=input.candidate,
        )
        score = int(normalize_text(prediction) == normalize_text(ground_truth))
        return MetricResult(outputs=[MetricOutput(name=self.type.value, value=score)])
