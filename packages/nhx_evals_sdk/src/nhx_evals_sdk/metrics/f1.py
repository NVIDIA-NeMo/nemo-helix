# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F1 metric runtime implementation."""

import collections

from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.template_rendering import render_reference_and_candidate, template_metric_repr
from nhx_evals_sdk.metrics.utils import normalize_text
from nhx_evals_sdk.values.metrics import F1

__all__ = ["F1Metric"]


class F1Metric(F1):
    """F1 metric for token-overlap similarity scoring."""

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return outputs emitted by this metric."""
        return [MetricOutputSpec.continuous_score(self.type.value)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute the scores for the metric."""
        ground_truth, prediction = render_reference_and_candidate(
            metric_repr=template_metric_repr(self),
            metric_name=self.__class__.__name__,
            reference_template=self.reference,
            candidate_template=self.candidate,
            item=input.row.data,
            sample=input.candidate,
        )

        prediction_tokens = normalize_text(prediction).split()
        ground_truth_tokens = normalize_text(ground_truth).split()

        # If either token list is empty, the F1 is 1.0 if they agree, 0.0 otherwise.
        if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
            score = float(ground_truth_tokens == prediction_tokens)
            return MetricResult(outputs=[MetricOutput(name=self.type.value, value=score)])

        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return MetricResult(outputs=[MetricOutput(name=self.type.value, value=0.0)])

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        score = (2 * precision * recall) / (precision + recall)
        return MetricResult(outputs=[MetricOutput(name=self.type.value, value=score)])
