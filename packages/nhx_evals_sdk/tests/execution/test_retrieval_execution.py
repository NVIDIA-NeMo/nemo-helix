# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nhx_evals_sdk.execution import evaluator as evaluator_module
from nhx_evals_sdk.execution.evaluator import Evaluator
from nhx_evals_sdk.metrics.retrieval import RetrievalNDCGMetric, RetrievalRecallMetric
from nhx_evals_sdk.retrieval.beir import BeirCorpusDocument, BeirDataset, BeirQuery
from nhx_evals_sdk.values.models import Model
from nhx_evals_sdk.values.retrieval import Retrieval


def _dataset(tmp_path: Path) -> BeirDataset:
    return BeirDataset(
        root=tmp_path,
        corpus={
            "d1": BeirCorpusDocument(id="d1", text="one"),
            "d2": BeirCorpusDocument(id="d2", text="two"),
        },
        queries={
            "q1": BeirQuery(id="q1", text="one?"),
            "q2": BeirQuery(id="q2", text="two?"),
        },
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    )


async def _fake_retrieve(*args, **kwargs) -> dict[str, dict[str, float]]:
    return {
        "q1": {"d1": 1.0, "d2": 0.0},
        "q2": {"d1": 1.0, "d2": 0.0},
    }


def _target() -> Retrieval:
    return Retrieval(embeddings=Model(url="https://embed.example.test/v1", name="embed"))


@pytest.mark.asyncio
async def test_run_retrieval_target_returns_query_rows_and_range_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch the module object rather than a dotted string: the string form re-resolves
    # `evaluator` through the parent package, which is not necessarily the module this file
    # imported `Evaluator` from.
    monkeypatch.setattr(evaluator_module, "retrieve", _fake_retrieve)

    result = await Evaluator().run(
        dataset=_dataset(tmp_path),
        target=_target(),
        metrics=[RetrievalNDCGMetric(k=[1]), RetrievalRecallMetric(k=[1])],
    )

    assert len(result.row_scores) == 2
    scores = {score.name: score.mean for score in result.aggregate_scores.scores}
    assert scores["retrieval-ndcg.ndcg_cut_1"] == pytest.approx(0.5)
    assert scores["retrieval-recall.recall_1"] == pytest.approx(0.5)


def test_run_sync_supports_retrieval_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator_module, "retrieve", _fake_retrieve)

    result = Evaluator().run_sync(
        dataset=_dataset(tmp_path),
        target=_target(),
        metrics=[RetrievalRecallMetric(k=[1])],
    )

    assert result.aggregate_scores.scores[-1].name == "retrieval-recall.recall_1"


@pytest.mark.asyncio
async def test_retrieval_target_rejects_inline_rows(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="BeirDataset or BEIR fileset path"):
        await Evaluator().run(
            dataset=[],
            target=_target(),
            metrics=[RetrievalRecallMetric(k=[1])],
        )
