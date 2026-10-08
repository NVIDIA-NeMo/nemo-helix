# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import pytrec_eval
from nhx_evals_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput
from nhx_evals_sdk.metrics.retrieval import (
    RetrievalMAPMetric,
    RetrievalNDCGMetric,
    RetrievalPrecisionMetric,
    RetrievalRecallMetric,
)
from nhx_evals_sdk.metrics.types import MetricsUnion
from pydantic import TypeAdapter

_QRELS = {"d1": 2, "d2": 1}
_RUN = {"d2": 2.0, "d1": 1.0}


def _input(qrels: dict[str, int], scores: dict[str, float]) -> MetricInput:
    return MetricInput(
        row=DatasetRow(data={"qrels": qrels}),
        candidate=CandidateOutput(metadata={"retrieval_scores": scores}),
    )


def _pytrec(qrels: dict[str, int], run: dict[str, float], measures: set[str]) -> dict[str, float]:
    return pytrec_eval.RelevanceEvaluator({"q": qrels}, measures).evaluate({"q": run})["q"]


@pytest.mark.asyncio
async def test_ndcg_matches_pytrec_eval() -> None:
    expected = _pytrec(_QRELS, _RUN, {"ndcg_cut_1", "ndcg_cut_2"})
    result = await RetrievalNDCGMetric(k=[1, 2]).compute_scores(_input(_QRELS, _RUN))
    scores = {output.name: output.value for output in result.outputs}

    assert scores["ndcg_cut_1"] == pytest.approx(expected["ndcg_cut_1"])
    assert scores["ndcg_cut_2"] == pytest.approx(expected["ndcg_cut_2"])


@pytest.mark.asyncio
async def test_recall_precision_and_map_match_pytrec_eval() -> None:
    qrels = {"d1": 1, "d2": 1}
    run = {"d1": 3.0, "d3": 2.0, "d2": 1.0}
    expected = _pytrec(qrels, run, {"recall_1", "recall_2", "P_2", "map_cut_3"})

    recall = await RetrievalRecallMetric(k=[1, 2]).compute_scores(_input(qrels, run))
    precision = await RetrievalPrecisionMetric(k=[2]).compute_scores(_input(qrels, run))
    average_precision = await RetrievalMAPMetric(k=[3]).compute_scores(_input(qrels, run))

    recall_scores = {output.name: output.value for output in recall.outputs}
    assert recall_scores["recall_1"] == pytest.approx(expected["recall_1"])
    assert recall_scores["recall_2"] == pytest.approx(expected["recall_2"])
    assert precision.outputs[0].value == pytest.approx(expected["P_2"])
    assert average_precision.outputs[0].value == pytest.approx(expected["map_cut_3"])


def test_retrieval_metrics_are_registered_variants() -> None:
    adapter = TypeAdapter(MetricsUnion)

    metric = adapter.validate_python({"type": "retrieval-ndcg", "k": [10, 1]})

    assert isinstance(metric, RetrievalNDCGMetric)
    assert metric.k == [1, 10]


@pytest.mark.parametrize("k", [[], [0], [1, 1]])
def test_retrieval_metric_rejects_invalid_cutoffs(k: list[int]) -> None:
    with pytest.raises(ValueError):
        RetrievalRecallMetric(k=k)
