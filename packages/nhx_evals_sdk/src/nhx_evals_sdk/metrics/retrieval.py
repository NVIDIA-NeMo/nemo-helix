# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEIR-compatible retrieval metrics scored with pytrec-eval."""

from __future__ import annotations

from collections.abc import Callable

import pytrec_eval
from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.values.metrics import (
    RetrievalMAP,
    RetrievalNDCG,
    RetrievalPrecision,
    RetrievalRecall,
)

__all__ = [
    "RetrievalMAPMetric",
    "RetrievalNDCGMetric",
    "RetrievalPrecisionMetric",
    "RetrievalRecallMetric",
]

_QRELS_FIELD = "qrels"
_SCORES_FIELD = "retrieval_scores"
_QUERY_ID = "q"


class RetrievalNDCGMetric(RetrievalNDCG):
    """Per-query nDCG at configured cutoffs (pytrec ``ndcg_cut_k``)."""

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return per-query nDCG outputs."""
        return _output_spec("ndcg_cut", self.k)

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute nDCG for one query."""
        return _pytrec_scores(input, "ndcg_cut", self.k, lambda cutoff: f"ndcg_cut_{cutoff}")


class RetrievalRecallMetric(RetrievalRecall):
    """Per-query recall at configured cutoffs (pytrec ``recall_k``)."""

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return per-query recall outputs."""
        return _output_spec("recall", self.k)

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute recall for one query."""
        return _pytrec_scores(input, "recall", self.k, lambda cutoff: f"recall_{cutoff}")


class RetrievalPrecisionMetric(RetrievalPrecision):
    """Per-query precision at configured cutoffs (pytrec ``P_k``)."""

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return per-query precision outputs."""
        return [MetricOutputSpec.continuous_score(f"P_{cutoff}") for cutoff in self.k]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute precision for one query."""
        return _pytrec_scores(input, "P", self.k, lambda cutoff: f"P_{cutoff}")


class RetrievalMAPMetric(RetrievalMAP):
    """Per-query average precision at configured cutoffs (pytrec ``map_cut_k``)."""

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return per-query MAP outputs."""
        return _output_spec("map_cut", self.k)

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute average precision for one query."""
        return _pytrec_scores(input, "map_cut", self.k, lambda cutoff: f"map_cut_{cutoff}")


def _output_spec(name: str, cutoffs: list[int]) -> list[MetricOutputSpec]:
    return [MetricOutputSpec.continuous_score(f"{name}_{cutoff}") for cutoff in cutoffs]


def _extract_run(input: MetricInput) -> tuple[dict[str, int], dict[str, float]]:
    qrels_raw = input.row.data.get(_QRELS_FIELD)
    scores_raw = input.candidate.metadata.get(_SCORES_FIELD)
    if not isinstance(qrels_raw, dict) or not isinstance(scores_raw, dict):
        raise ValueError(
            f"retrieval metrics require row.data[{_QRELS_FIELD!r}] and candidate.metadata[{_SCORES_FIELD!r}] mappings"
        )
    try:
        qrels = {str(document_id): int(relevance) for document_id, relevance in qrels_raw.items()}
        scores = {str(document_id): float(score) for document_id, score in scores_raw.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("retrieval qrels and scores must contain numeric values") from error
    return qrels, scores


def _pytrec_scores(
    input: MetricInput,
    name: str,
    cutoffs: list[int],
    measure: Callable[[int], str],
) -> MetricResult:
    qrels, scores = _extract_run(input)
    measures = {measure(cutoff) for cutoff in cutoffs}
    evaluator = pytrec_eval.RelevanceEvaluator({_QUERY_ID: qrels}, measures)
    results = evaluator.evaluate({_QUERY_ID: scores}).get(_QUERY_ID, {})
    prefix = f"{name}_" if name != "P" else "P_"
    return MetricResult(
        outputs=[
            MetricOutput(name=f"{prefix}{cutoff}", value=float(results.get(measure(cutoff), 0.0))) for cutoff in cutoffs
        ]
    )
