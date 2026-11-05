# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure paired comparison shared by the API, CLI, and Studio."""

from __future__ import annotations

import pytest
from nhx.intake.experiments.compare import (
    BOOTSTRAP_INTERVAL_LEVEL,
    BOOTSTRAP_RESAMPLES,
    CompareMetric,
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


# ---------------------------------------------------------------------------
# parse_compare_metric
# ---------------------------------------------------------------------------


def test_parse_metric_directions() -> None:
    assert parse_compare_metric("evaluators.accuracy").higher_is_better is True
    assert parse_compare_metric("evaluators.harbor.verifier").id == "evaluators.harbor.verifier"
    assert parse_compare_metric("cost_usd").higher_is_better is False
    assert parse_compare_metric("latency_ms").higher_is_better is False
    assert parse_compare_metric("tokens").higher_is_better is False


@pytest.mark.parametrize("metric", ["", "evaluators.", "accuracy", "duration_ms", "evaluators", "cost"])
def test_parse_metric_rejects_unknown_metrics(metric: str) -> None:
    with pytest.raises(InvalidEvaluationMetricError):
        parse_compare_metric(metric)


# ---------------------------------------------------------------------------
# mean_by_test_case
# ---------------------------------------------------------------------------


def test_mean_by_test_case_averages_repeated_attempts() -> None:
    means = mean_by_test_case([session("tc1", 1.0), session("tc1", 0.0), session("tc2", 0.5)])
    assert means == {"tc1": 0.5, "tc2": 0.5}


def test_mean_by_test_case_scores_failed_attempts_as_zero() -> None:
    means = mean_by_test_case([session("tc1", 1.0), session("tc1", None, failed=True)])
    assert means == {"tc1": 0.5}


def test_mean_by_test_case_omits_unscored_non_failed_attempts() -> None:
    means = mean_by_test_case([session("tc1", 0.6), session("tc1", None), session("tc2", None)])
    # tc1 averages only its scored attempt; tc2 has no scored attempt at all,
    # so it is absent from the map and cannot pair.
    assert means == {"tc1": 0.6}


def test_mean_by_test_case_drops_sessions_without_a_test_case_name() -> None:
    assert mean_by_test_case([session(None, 1.0), session("", 1.0)]) == {}


# ---------------------------------------------------------------------------
# compare_test_case_means
# ---------------------------------------------------------------------------


def test_compare_pairs_only_shared_test_cases() -> None:
    baseline = {"a": 0.5, "b": 0.5, "c": 0.5, "only-baseline": 1.0}
    candidate = {"a": 0.9, "b": 0.5, "c": 0.1, "only-candidate": 1.0}
    result = compare_test_case_means(baseline, candidate, parse_compare_metric("evaluators.acc"))

    assert result.n_pairs == 3
    assert (result.better, result.same, result.worse) == (1, 1, 1)
    assert result.metric == "evaluators.acc"
    assert result.higher_is_better is True


def test_compare_counts_a_tie_within_tolerance_as_same() -> None:
    baseline = {"a": 0.5, "b": 0.5}
    candidate = {"a": 0.5 + 5e-10, "b": 0.5 - 5e-10}
    result = compare_test_case_means(baseline, candidate, parse_compare_metric("evaluators.acc"))

    assert result.n_pairs == 2
    assert (result.better, result.same, result.worse) == (0, 2, 0)


def test_compare_direction_for_lower_is_better_metrics() -> None:
    # Candidate is cheaper on every shared test case: better for cost_usd.
    baseline = {"a": 2.0, "b": 1.0}
    candidate = {"a": 1.0, "b": 0.5}
    result = compare_test_case_means(baseline, candidate, parse_compare_metric("cost_usd"))

    assert result.higher_is_better is False
    assert (result.better, result.same, result.worse) == (2, 0, 0)
    assert result.mean_paired_difference == pytest.approx(-0.75)
    # The same differences under a higher-is-better metric are all worse.
    flipped = compare_test_case_means(baseline, candidate, parse_compare_metric("evaluators.acc"))
    assert (flipped.better, flipped.same, flipped.worse) == (0, 0, 2)


def test_compare_reports_candidate_worse_on_every_pair() -> None:
    baseline = {"a": 0.8, "b": 0.9, "c": 0.7}
    candidate = {"a": 0.1, "b": 0.2, "c": 0.3}
    result = compare_test_case_means(baseline, candidate, parse_compare_metric("evaluators.acc"))

    assert (result.better, result.same, result.worse) == (0, 0, 3)
    assert result.mean_paired_difference == pytest.approx(-0.6)
    assert result.p_candidate_better == pytest.approx(0.0)


def test_compare_mean_paired_difference_is_signed_candidate_minus_baseline() -> None:
    baseline = {"a": 1.0, "b": 0.0}
    candidate = {"a": 0.5, "b": 0.5}
    result = compare_test_case_means(baseline, candidate, parse_compare_metric("evaluators.acc"))

    assert result.mean_paired_difference == pytest.approx(0.0)


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
    mean = first.mean_paired_difference
    low = first.interval_low
    high = first.interval_high
    probability = first.p_candidate_better
    assert mean is not None
    assert low is not None
    assert high is not None
    assert probability is not None
    assert low <= mean <= high
    assert 0.0 <= probability <= 1.0


def test_compare_interval_brackets_a_clear_improvement() -> None:
    baseline = {f"tc{i}": 0.2 for i in range(12)}
    candidate = {f"tc{i}": 0.8 for i in range(12)}
    result = compare_test_case_means(baseline, candidate, parse_compare_metric("evaluators.acc"))

    assert result.n_pairs == 12
    assert result.interval_low == pytest.approx(0.6)
    assert result.interval_high == pytest.approx(0.6)
    assert result.p_candidate_better == pytest.approx(1.0)
    assert result.interval_caveat is None


def test_compare_flags_a_small_pair_count_as_a_weak_interval() -> None:
    baseline = {"a": 0.5, "b": 0.4}
    candidate = {"a": 0.6, "b": 0.5}
    result = compare_test_case_means(baseline, candidate, parse_compare_metric("evaluators.acc"))

    assert result.n_pairs == 2
    assert result.interval_caveat is not None
    assert "2" in result.interval_caveat


def test_compare_resamples_are_configured_for_a_95_percent_interval() -> None:
    assert BOOTSTRAP_RESAMPLES == 2000
    assert BOOTSTRAP_INTERVAL_LEVEL == 0.95


# ---------------------------------------------------------------------------
# compare_evaluation_sessions
# ---------------------------------------------------------------------------


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

    # tc1: baseline (1.0 + 0)/2 = 0.5 vs candidate 0.5 -> same.
    # tc2: baseline 0.5 vs candidate 1.0 -> better.
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


def test_compare_metric_dataclass_is_comparable_for_equality_checks() -> None:
    assert CompareMetric(id="tokens", higher_is_better=False) == parse_compare_metric("tokens")
