# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sparse declared metric outputs and their aggregate opportunities."""

from __future__ import annotations

import math

import pytest
from nemo_evaluator_sdk.metrics.aggregation import add_corpus_scores, aggregate_metrics
from nemo_evaluator_sdk.metrics.protocol import MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.values.results import AggregateRangeScore, AggregateRubricScore, RubricScoreStat


def _result(*outputs: tuple[str, object]) -> MetricResult:
    return MetricResult(outputs=[MetricOutput(name=name, value=value) for name, value in outputs])


def _range_specs() -> list[MetricOutputSpec]:
    return [
        MetricOutputSpec.continuous_score("reward"),
        MetricOutputSpec.continuous_score("format_ok", required=False),
    ]


def _assert_opportunities(aggregate, opportunities: int) -> None:
    assert sum((score.count or 0) + score.nan_count for score in aggregate.scores) == opportunities


def test_sparse_range_outputs_preserve_opportunities_and_sample_stats() -> None:
    aggregate = aggregate_metrics(
        [_result(("reward", 1), ("format_ok", 1)), _result(("reward", 0))],
        _range_specs(),
    )

    reward, format_ok = aggregate.scores
    assert isinstance(reward, AggregateRangeScore)
    assert (reward.mean, reward.count, reward.nan_count) == (0.5, 2, 0)
    assert reward.sample_variance == 0.5
    assert reward.sample_std_dev == math.sqrt(0.5)
    assert isinstance(format_ok, AggregateRangeScore)
    assert (format_ok.mean, format_ok.count, format_ok.nan_count) == (1.0, 1, 1)
    assert format_ok.sample_variance is None
    assert format_ok.sample_std_dev is None
    _assert_opportunities(aggregate, opportunities=4)


def test_all_optional_range_outputs_are_retained_as_unestimable() -> None:
    specs = [MetricOutputSpec.continuous_score("format_ok", required=False)]
    aggregate = aggregate_metrics([_result(), _result()], specs)

    assert len(aggregate.scores) == 1
    score = aggregate.scores[0]
    assert isinstance(score, AggregateRangeScore)
    assert score.count == 0
    assert score.nan_count == 2
    assert score.sum is None
    assert score.mean is None
    assert score.min is None
    assert score.max is None
    assert score.median is None
    assert score.variance is None
    assert score.std_dev is None
    assert score.sample_variance is None
    assert score.sample_std_dev is None
    assert score.percentiles is None
    assert score.histogram is not None and score.histogram.bins == []
    _assert_opportunities(aggregate, opportunities=2)


def test_row_aggregation_rejects_missing_required_aggregateable_output() -> None:
    with pytest.raises(ValueError, match="Missing required metric outputs: \\['reward'\\]"):
        aggregate_metrics([_result()], [MetricOutputSpec.continuous_score("reward")])


def test_failed_row_may_omit_required_non_aggregateable_output() -> None:
    specs = [
        MetricOutputSpec.continuous_score("quality"),
        MetricOutputSpec.label("quality.label"),
    ]
    definitions = {"quality": [RubricScoreStat(label="good", value=1, count=0)]}

    aggregate = aggregate_metrics(
        [_result(("quality", float("nan")))],
        specs,
        rubric_definitions=definitions,
    )

    score = aggregate.scores[0]
    assert isinstance(score, AggregateRubricScore)
    assert (score.count, score.nan_count, score.mean, score.mode_category) == (0, 1, None, None)
    assert [(bucket.label, bucket.count) for bucket in score.rubric_distribution] == [("good", 0)]


def test_sparse_rubric_outputs_keep_declared_categories_and_mode() -> None:
    specs = [
        MetricOutputSpec.continuous_score("quality", required=False),
        MetricOutputSpec.label("quality.label", required=False),
    ]
    definitions = {
        "quality": [
            RubricScoreStat(label="good", value=1, count=0),
            RubricScoreStat(label="bad", value=0, count=0),
        ]
    }

    aggregate = aggregate_metrics(
        [_result(("quality", 1), ("quality.label", "good")), _result()],
        specs,
        rubric_definitions=definitions,
    )
    score = aggregate.scores[0]
    assert isinstance(score, AggregateRubricScore)
    assert (score.count, score.nan_count, score.mean) == (1, 1, 1.0)
    assert [(bucket.label, bucket.count) for bucket in score.rubric_distribution] == [("good", 1), ("bad", 0)]
    assert score.mode_category == "good"
    _assert_opportunities(aggregate, opportunities=2)

    all_omitted = aggregate_metrics([_result(), _result()], specs, rubric_definitions=definitions)
    score = all_omitted.scores[0]
    assert isinstance(score, AggregateRubricScore)
    assert (score.count, score.nan_count, score.mean, score.mode_category) == (0, 2, None, None)
    assert [(bucket.label, bucket.count) for bucket in score.rubric_distribution] == [("good", 0), ("bad", 0)]
    _assert_opportunities(all_omitted, opportunities=2)


def test_corpus_scores_count_omitted_and_nan_opportunities() -> None:
    specs = [
        MetricOutputSpec.continuous_score("present"),
        MetricOutputSpec.continuous_score("omitted", required=False),
        MetricOutputSpec.continuous_score("nan", required=False),
    ]
    aggregate = aggregate_metrics([], [])
    add_corpus_scores(aggregate, _result(("present", 0.5), ("nan", float("nan"))), specs)

    assert [score.name for score in aggregate.scores] == ["present", "omitted", "nan"]
    present, omitted, nan = aggregate.scores
    assert isinstance(present, AggregateRangeScore)
    assert (present.count, present.nan_count, present.mean) == (1, 0, 0.5)
    assert present.sample_variance is None
    for score in (omitted, nan):
        assert isinstance(score, AggregateRangeScore)
        assert (score.count, score.nan_count, score.mean) == (0, 1, None)
        assert score.sum is None
        assert score.min is None
        assert score.max is None
        assert score.median is None
        assert score.variance is None
        assert score.std_dev is None
        assert score.sample_variance is None
        assert score.sample_std_dev is None
        assert score.percentiles is None
        assert score.histogram is not None and score.histogram.bins == []

    _assert_opportunities(aggregate, opportunities=3)


def test_corpus_scores_reject_missing_required_output() -> None:
    aggregate = aggregate_metrics([], [])

    with pytest.raises(ValueError, match="Missing required metric outputs: \\['required'\\]"):
        add_corpus_scores(
            aggregate,
            _result(),
            [MetricOutputSpec.continuous_score("required")],
        )


def test_empty_aggregation_has_no_scores() -> None:
    assert aggregate_metrics([], _range_specs()).scores == []
