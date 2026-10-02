# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shape-agnostic accessor every aggregate exposes."""

from __future__ import annotations

from nemo_evaluator_sdk.values.results import (
    AggregateRangeScore,
    AggregateRubricScore,
    AggregateScalarScore,
    AggregateScore,
    RubricScoreStat,
    summary_aggregate_record,
)


def test_a_range_score_reports_its_mean() -> None:
    score = AggregateRangeScore(name="reward", count=4, nan_count=0, mean=0.75)

    assert score.headline_value == 0.75


def test_a_scalar_score_reports_its_value_rather_than_its_absent_mean() -> None:
    # The whole point of the accessor: a backend-reported figure carries `value` and no `mean`, so
    # a caller reading `mean` drops a real measurement instead of reporting it.
    score = AggregateScalarScore(name="pass@1/accuracy", count=None, nan_count=0, value=0.8)

    assert score.mean is None
    assert score.headline_value == 0.8


def test_a_scalar_score_prefers_its_value_even_when_a_mean_is_also_supplied() -> None:
    # A producer may fill both. `value` is the number the backend reported; inheriting the base
    # property here would silently report the other one.
    score = AggregateScalarScore(name="elo", count=None, nan_count=0, value=1523.0, mean=0.5)

    assert score.headline_value == 1523.0


def test_a_rubric_score_reports_its_mean_when_it_has_one() -> None:
    score = AggregateRubricScore(
        name="helpfulness",
        count=2,
        nan_count=0,
        mean=2.5,
        rubric_distribution=[RubricScoreStat(label="good", value=3.0, count=2)],
    )

    assert score.headline_value == 2.5


def test_every_aggregate_shape_answers_the_accessor() -> None:
    # Guards the union: a fourth score type added without an accessor would read as None here
    # rather than failing at the call site of whichever consumer reaches for it first.
    scores: list[AggregateScore] = [
        AggregateRangeScore(name="a", count=1, nan_count=0, mean=0.1),
        AggregateScalarScore(name="b", count=None, nan_count=0, value=0.2),
        AggregateRubricScore(name="c", count=1, nan_count=0, mean=0.3, rubric_distribution=[]),
    ]

    assert [score.headline_value for score in scores] == [0.1, 0.2, 0.3]


def test_the_summary_row_carries_a_scalars_value_beside_the_mean_it_has_none_of() -> None:
    # The printed "Aggregate scores" table is built from this row. Reporting only `mean` left every
    # backend-reported figure blank in the SDK's own summary output.
    record = summary_aggregate_record(AggregateScalarScore(name="pass@1/accuracy", count=None, nan_count=0, value=0.8))

    assert record["value"] == 0.8
    assert record["mean"] is None


def test_a_range_score_row_has_no_value_column_to_confuse_with_its_mean() -> None:
    record = summary_aggregate_record(AggregateRangeScore(name="reward", count=4, nan_count=0, mean=0.25))

    assert "value" not in record
    assert record["mean"] == 0.25
