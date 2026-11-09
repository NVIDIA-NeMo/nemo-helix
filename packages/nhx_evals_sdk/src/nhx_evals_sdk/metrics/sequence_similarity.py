# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequence similarity metric runtime implementation."""

import difflib

from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.template_rendering import render_reference_and_candidate, template_metric_repr
from nhx_evals_sdk.values.metrics import SequenceSimilarity

__all__ = ["SequenceSimilarityMetric"]


class SequenceSimilarityMetric(SequenceSimilarity):
    """Character-level similarity from ``difflib.SequenceMatcher``, between 0 and 1.

    The score is ``SequenceMatcher(None, candidate, reference).ratio()``, with no normalization, so it
    matches scores computed that way elsewhere. Keep the argument order: once the reference is 200
    characters or longer, ``SequenceMatcher``'s autojunk rule ignores characters that make up more than
    1% of it, so swapping the arguments can change the score.
    """

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return outputs emitted by this metric."""
        return [MetricOutputSpec.continuous_score(self.type.value)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute the scores for the metric."""
        reference, candidate = render_reference_and_candidate(
            metric_repr=template_metric_repr(self),
            metric_name=self.__class__.__name__,
            reference_template=self.reference,
            candidate_template=self.candidate,
            item=input.row.data,
            sample=input.candidate,
        )
        score = difflib.SequenceMatcher(None, candidate, reference).ratio()
        return MetricResult(outputs=[MetricOutput(name=self.type.value, value=score)])
