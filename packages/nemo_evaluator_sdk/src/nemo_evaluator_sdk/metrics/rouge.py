# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ROUGE metric runtime implementation."""

import asyncio
from functools import cached_property
from typing import ClassVar, Literal

from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.metrics.template_rendering import (
    TemplateSample,
    render_reference_and_candidate,
    template_metric_repr,
)
from nemo_evaluator_sdk.values.metrics import ROUGE

__all__ = ["ROUGEMetric", "RougeScoreName"]

RougeScoreName = Literal["rouge_1_score", "rouge_2_score", "rouge_3_score", "rouge_L_score"]


class ROUGEMetric(ROUGE):
    """ROUGE metric for overlap-based summarization quality scoring.

    Evaluator-driven runs expose dataset fields through ``item`` and generated
    model outputs through ``sample.output_text`` for online execution.
    """

    scores_mapping: ClassVar[dict[RougeScoreName, str]] = {
        # Maps the public MetricResult score name to the underlying rouge_scorer key.
        "rouge_1_score": "rouge1",
        "rouge_2_score": "rouge2",
        "rouge_3_score": "rouge3",
        "rouge_L_score": "rougeL",
    }

    @cached_property
    def _scorer(self):
        """Lazily initialize the ROUGE scorer to avoid expensive import-time setup."""
        # The RougeScorer loads NLTK's stemmer/tokenizer machinery, so keeping
        # this lazy avoids unnecessary startup cost for callers that never use
        # ROUGE in a given process.
        from rouge_score import rouge_scorer

        return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3", "rougeL"], use_stemmer=True)

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return outputs emitted by this metric."""
        return [MetricOutputSpec.continuous_score(score_name) for score_name in self.scores_mapping]

    def _metric(self, item: dict, sample: TemplateSample) -> dict:
        """Compute raw ROUGE scores for one item/sample pair."""
        ground_truth, prediction = render_reference_and_candidate(
            metric_repr=template_metric_repr(self),
            metric_name=self.__class__.__name__,
            reference_template=self.reference,
            candidate_template=self.candidate,
            item=item,
            sample=sample,
        )
        return self._scorer.score(ground_truth, prediction)

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute structured score output for one item/sample pair.

        Scoring runs in a worker thread because building the scorer pulls in NLTK's stemmer and
        tokenizer, which takes about a second and would otherwise hold the event loop for any
        caller scoring inside a request handler.
        """
        scores = await asyncio.to_thread(self._metric, input.row.data, input.candidate)
        return MetricResult(
            outputs=[
                MetricOutput(name=score_name, value=scores[score_key].fmeasure)
                for score_name, score_key in self.scores_mapping.items()
            ]
        )
