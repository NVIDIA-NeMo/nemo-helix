# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure paired comparison shared by the API, CLI, and Studio."""

from __future__ import annotations

import pytest
from nhx.intake.experiments.compare import (
    EvaluationMetricSession,
    InvalidEvaluationMetricError,
    compare_evaluation_sessions,
    compare_test_case_means,
    mean_by_test_case,
    parse_compare_metric,
)


def session(
    test_case: str | None,
    value: float | None,
    *,
    evaluation: str = "eval",
    failed: bool = False,
) -> EvaluationMetricSession:
    return EvaluationMetricSession(
        evaluation_name=evaluation,
        test_case_name=test_case,
        value=value,
        failed=failed,
    )


@pytest.mark.parametrize(
    ("metric", "higher_is_better"),
    [("evaluators.accuracy", True), ("cost_usd", False), ("latency_ms", False), ("tokens", False)],
)
def test_parse_metric_directions(metric: str, higher_is_better: bool) -> None:
    assert parse_compare_metric(metric).higher_is_better is higher_is_better


@pytest.mark.parametrize("metric", ["", "evaluators.", "accuracy", "duration_ms", "evaluators", "cost"])
def test_parse_metric_rejects_unknown_metrics(metric: str) -> None:
    with pytest.raises(InvalidEvaluationMetricError):
        parse_compare_metric(metric)


@pytest.mark.parametrize(
    ("sessions", "expected"),
    [
        pytest.param(
            [session("tc1", 1.0), session("tc1", 0.0), session("tc2", 0.5)],
            {"tc1": 0.5, "tc2": 0.5},
            id="repeated attempts are averaged",
        ),
        pytest.param(
            [session("tc1", 1.0), session("tc1", None, failed=True)],
            {"tc1": 0.5},
            id="a failed attempt scores zero",
        ),
        pytest.param(
            [session("tc1", 0.6), session("tc1", None), session("tc2", None)],
            {"tc1": 0.6},
            id="an unscored attempt is omitted, and so is a test case with none scored",
        ),
        pytest.param([session(None, 1.0), session("", 1.0)], {}, id="no test case name cannot pair"),
    ],
)
def test_mean_by_test_case(sessions: list[EvaluationMetricSession], expected: dict[str, float]) -> None:
    assert mean_by_test_case(sessions) == expected


@pytest.mark.parametrize(
    ("baseline", "candidate", "metric", "counts"),
    [
        pytest.param(
            {"a": 0.5, "b": 0.5, "c": 0.5, "only-baseline": 1.0},
            {"a": 0.9, "b": 0.5, "c": 0.1, "only-candidate": 1.0},
            "evaluators.acc",
            (1, 1, 1),
            id="only shared test cases pair",
        ),
        pytest.param(
            {"a": 0.5, "b": 0.5},
            {"a": 0.5 + 5e-10, "b": 0.5 - 5e-10},
            "evaluators.acc",
            (0, 2, 0),
            id="a difference within tolerance is the same",
        ),
        pytest.param({"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 0.5}, "cost_usd", (2, 0, 0), id="lower cost is better"),
        pytest.param(
            {"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 0.5}, "evaluators.acc", (0, 0, 2), id="lower score is worse"
        ),
        pytest.param(
            {"a": 0.8, "b": 0.9, "c": 0.7},
            {"a": 0.1, "b": 0.2, "c": 0.3},
            "evaluators.acc",
            (0, 0, 3),
            id="worse on every pair",
        ),
    ],
)
def test_compare_counts_pairs_by_direction(
    baseline: dict[str, float], candidate: dict[str, float], metric: str, counts: tuple[int, int, int]
) -> None:
    result = compare_test_case_means(baseline, candidate, parse_compare_metric(metric))

    assert result.n_pairs == sum(counts)
    assert (result.better, result.same, result.worse) == counts


@pytest.mark.parametrize(
    ("baseline", "candidate", "expected"),
    [
        pytest.param({"a": 0.2, "b": 0.4}, {"a": 0.5, "b": 0.6}, 0.25, id="candidate higher is positive"),
        pytest.param({"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 0.5}, -0.75, id="candidate lower is negative"),
    ],
)
def test_compare_mean_paired_difference_is_signed_candidate_minus_baseline(
    baseline: dict[str, float], candidate: dict[str, float], expected: float
) -> None:
    result = compare_test_case_means(baseline, candidate, parse_compare_metric("evaluators.acc"))

    assert result.mean_paired_difference == pytest.approx(expected)


def test_compare_without_shared_test_cases_returns_empty_interval() -> None:
    result = compare_test_case_means({"a": 1.0}, {"b": 1.0}, parse_compare_metric("evaluators.acc"))

    assert result.n_pairs == 0
    assert result.better == result.same == result.worse == 0
    assert result.mean_paired_difference is None
    assert result.interval_low is None
    assert result.interval_high is None
    assert result.p_candidate_better is None
    assert result.interval_caveat is not None


def test_compare_interval_is_deterministic_for_the_fixed_seed() -> None:
    baseline = {"a": 0.4, "b": 0.5, "c": 0.6, "d": 0.5, "e": 0.4}
    candidate = {"a": 0.6, "b": 0.4, "c": 0.7, "d": 0.6, "e": 0.5}
    metric = parse_compare_metric("evaluators.acc")

    first = compare_test_case_means(baseline, candidate, metric)
    second = compare_test_case_means(baseline, candidate, metric)

    assert first == second
    assert (first.better, first.same, first.worse) == (4, 0, 1)
    assert first.mean_paired_difference == pytest.approx(0.08)
    assert first.interval_low == pytest.approx(-0.02)
    assert first.interval_high == pytest.approx(0.16)
    assert first.p_candidate_better == pytest.approx(0.9495)


def test_compare_interval_brackets_a_clear_improvement() -> None:
    baseline = {f"tc{i}": 0.2 for i in range(12)}
    candidate = {f"tc{i}": 0.8 for i in range(12)}
    result = compare_test_case_means(baseline, candidate, parse_compare_metric("evaluators.acc"))

    assert result.n_pairs == 12
    assert (result.better, result.same, result.worse) == (12, 0, 0)
    assert result.interval_low == pytest.approx(0.6)
    assert result.interval_high == pytest.approx(0.6)
    assert result.p_candidate_better == pytest.approx(1.0)
    assert result.interval_caveat is None


def test_compare_flags_a_small_pair_count_as_a_weak_interval() -> None:
    baseline = {"a": 0.5, "b": 0.4}
    candidate = {"a": 0.6, "b": 0.5}
    result = compare_test_case_means(baseline, candidate, parse_compare_metric("evaluators.acc"))

    assert result.n_pairs == 2
    assert (result.better, result.same, result.worse) == (2, 0, 0)
    assert result.interval_low == pytest.approx(0.1)
    assert result.interval_high == pytest.approx(0.1)
    assert result.interval_caveat is not None


def test_compare_evaluation_sessions_applies_the_paired_mean_rules() -> None:
    baseline_sessions = [
        session("tc1", 1.0, evaluation="base"),
        session("tc1", None, failed=True, evaluation="base"),  # scores 0
        session("tc2", 0.5, evaluation="base"),
        session("tc3", 0.5, evaluation="base"),  # not shared with candidate
        session(None, 1.0, evaluation="base"),  # no test case: dropped
    ]
    candidate_sessions = [
        session("tc1", 0.5, evaluation="cand"),
        session("tc2", None, evaluation="cand"),  # unscored: omitted
        session("tc2", 1.0, evaluation="cand"),
        session("tc4", 1.0, evaluation="cand"),  # not shared with baseline
    ]

    result = compare_evaluation_sessions(
        baseline_sessions,
        candidate_sessions,
        parse_compare_metric("evaluators.acc"),
    )

    assert result.n_pairs == 2
    assert (result.better, result.same, result.worse) == (1, 1, 0)
    assert result.mean_paired_difference == pytest.approx(0.25)


def test_compare_evaluation_sessions_counts_cost_direction_from_raw_sessions() -> None:
    baseline_sessions = [session("tc1", 2.0, evaluation="base"), session("tc2", 1.0, evaluation="base")]
    candidate_sessions = [session("tc1", 1.0, evaluation="cand"), session("tc2", 0.5, evaluation="cand")]

    result = compare_evaluation_sessions(
        baseline_sessions,
        candidate_sessions,
        parse_compare_metric("cost_usd"),
    )

    assert (result.better, result.same, result.worse) == (2, 0, 0)
