# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure paired baseline-vs-candidate comparison of Evaluation metrics.

This module is the single pairing implementation shared by the compare API,
``nemo intake evaluations compare``, and Studio's EvaluationCompareChart. It
mirrors the rules Studio's comparison view applies when it pairs two runs by
test case:

- Sessions without a ``test_case_name`` cannot be paired and are dropped.
- A failed evaluator result scores ``0``.
- An unscored attempt that is not a failure is omitted from its test case's
  mean (unlike the evaluation rollup, which counts unscored attempts as zero).
- Only test cases present on both sides are paired.
- An absolute difference within ``1e-9`` counts as ``same``.
- For an evaluator score, higher is better; for cost, latency, and tokens,
  lower is better.

The uncertainty estimate is a bootstrap percentile interval over the paired
test-case mean differences, resampled with stdlib ``random`` on a fixed seed,
so a given input always yields the same interval.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# Two paired values closer than this count as the same, matching Studio's comparison tolerance.
SAME_TOLERANCE = 1e-9

# Bootstrap configuration. The seed is fixed so an (input, metric) pair always
# produces the same interval; callers cannot steer it.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_INTERVAL_LEVEL = 0.95
BOOTSTRAP_SEED = 0

# A bootstrap percentile interval from very few paired test cases is a weak
# estimate; the comparison result says so instead of implying false confidence.
WEAK_INTERVAL_MAX_PAIRS = 9

_EVALUATOR_METRIC_PREFIX = "evaluators."
# Studio's direction rule (``higherIsBetter`` in the EvaluationCompareChart):
# evaluator scores go up when things get better; every rollup metric goes down.
_LOWER_IS_BETTER_METRICS = frozenset({"cost_usd", "latency_ms", "tokens"})


class InvalidEvaluationMetricError(ValueError):
    """The requested metric is not one this comparison can pair by test case."""


@dataclass(frozen=True)
class CompareMetric:
    """A comparable evaluation metric and its direction.

    ``id`` uses Studio's metric vocabulary: ``evaluators.<name>``, ``cost_usd``,
    ``latency_ms``, or ``tokens``.
    """

    id: str
    higher_is_better: bool

    @property
    def evaluator_name(self) -> str | None:
        """The evaluator this metric reads, or None for a rollup metric."""
        if not self.id.startswith(_EVALUATOR_METRIC_PREFIX):
            return None
        return self.id[len(_EVALUATOR_METRIC_PREFIX) :]


def parse_compare_metric(metric: str) -> CompareMetric:
    """Parse a metric id into a :class:`CompareMetric`.

    Evaluator metrics (``evaluators.<name>``) are higher-is-better; the rollup
    metrics (``cost_usd``, ``latency_ms``, ``tokens``) are lower-is-better.
    Anything else is not comparable per test case.
    """
    if metric.startswith(_EVALUATOR_METRIC_PREFIX) and len(metric) > len(_EVALUATOR_METRIC_PREFIX):
        return CompareMetric(id=metric, higher_is_better=True)
    if metric in _LOWER_IS_BETTER_METRICS:
        return CompareMetric(id=metric, higher_is_better=False)
    raise InvalidEvaluationMetricError(
        f"Unsupported metric {metric!r}: use 'evaluators.<name>', 'cost_usd', 'latency_ms', or 'tokens'."
    )


@dataclass(frozen=True)
class EvaluationMetricSession:
    """One session's value for the compared metric.

    ``value`` is the session's mean score for the named evaluator (or its
    cost, latency, or token total) and is ``None`` when the session produced
    no value. ``failed`` marks a session whose evaluator recorded a FAILED
    result, which the shared pairing rules score as ``0``.
    """

    evaluation_name: str
    test_case_name: str | None
    value: float | None
    failed: bool = False


@dataclass(frozen=True)
class PairedComparison:
    """Result of pairing two evaluations' per-test-case means on one metric."""

    metric: str
    higher_is_better: bool
    n_pairs: int
    better: int
    same: int
    worse: int
    mean_paired_difference: float | None
    interval_low: float | None
    interval_high: float | None
    p_candidate_better: float | None
    interval_caveat: str | None


def mean_by_test_case(sessions: Iterable[EvaluationMetricSession]) -> dict[str, float]:
    """Mean metric value per test case, applying the shared pairing rules.

    ``sessions`` are one evaluation's attempts; each contributes its value to
    its test case's mean per the rules in the module docstring. A test case
    with no contributing attempts is absent from the result, so it cannot pair.
    """
    totals: dict[str, list[float]] = {}
    for session in sessions:
        if not session.test_case_name:
            continue
        if session.failed:
            value = 0.0
        elif session.value is None:
            continue
        else:
            value = session.value
        if not math.isfinite(value):
            continue
        totals.setdefault(session.test_case_name, []).append(value)
    return {test_case: sum(values) / len(values) for test_case, values in totals.items()}


def _favoring(difference: float, metric: CompareMetric) -> bool:
    if abs(difference) <= SAME_TOLERANCE:
        return False
    return (difference > 0) if metric.higher_is_better else (difference < 0)


def compare_test_case_means(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    metric: CompareMetric,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> PairedComparison:
    """Pair per-test-case means and compute the comparison with its interval.

    Differences are signed ``candidate - baseline``; ``mean_paired_difference``
    and the interval bounds keep that sign. ``p_candidate_better`` is the
    fraction of bootstrap resamples whose mean difference favors the candidate
    under the metric's direction. ``n_pairs == 0`` yields an empty interval.
    """
    # Sorted so the resampling sequence — and therefore the interval — does not
    # depend on input ordering.
    paired_names = sorted(name for name in candidate if name in baseline)
    differences = [candidate[name] - baseline[name] for name in paired_names]
    n_pairs = len(differences)

    better = sum(1 for difference in differences if _favoring(difference, metric))
    same = sum(1 for difference in differences if abs(difference) <= SAME_TOLERANCE)
    worse = n_pairs - better - same

    if n_pairs == 0:
        return PairedComparison(
            metric=metric.id,
            higher_is_better=metric.higher_is_better,
            n_pairs=0,
            better=0,
            same=0,
            worse=0,
            mean_paired_difference=None,
            interval_low=None,
            interval_high=None,
            p_candidate_better=None,
            interval_caveat="No test cases are paired between these evaluations, so no interval was computed.",
        )

    mean_difference = sum(differences) / n_pairs

    rng = random.Random(seed)
    resampled_means: list[float] = []
    favoring = 0
    for _ in range(resamples):
        total = 0.0
        for _ in range(n_pairs):
            total += differences[rng.randrange(n_pairs)]
        resample_mean = total / n_pairs
        resampled_means.append(resample_mean)
        if _favoring(resample_mean, metric):
            favoring += 1
    resampled_means.sort()

    # Percentile interval: the (1-level)/2 and (1+level)/2 quantiles of the
    # resampled mean differences.
    low_index = math.floor((1 - BOOTSTRAP_INTERVAL_LEVEL) / 2 * resamples)
    high_index = math.ceil((1 + BOOTSTRAP_INTERVAL_LEVEL) / 2 * resamples) - 1
    interval_low = resampled_means[low_index]
    interval_high = resampled_means[high_index]

    caveat = None
    if n_pairs <= WEAK_INTERVAL_MAX_PAIRS:
        pairs_word = "test case" if n_pairs == 1 else "test cases"
        caveat = (
            f"Only {n_pairs} paired {pairs_word}: this 95% bootstrap interval is a weak estimate "
            "and can move sharply with one more run."
        )

    return PairedComparison(
        metric=metric.id,
        higher_is_better=metric.higher_is_better,
        n_pairs=n_pairs,
        better=better,
        same=same,
        worse=worse,
        mean_paired_difference=mean_difference,
        interval_low=interval_low,
        interval_high=interval_high,
        p_candidate_better=favoring / resamples,
        interval_caveat=caveat,
    )


def compare_evaluation_sessions(
    baseline_sessions: Iterable[EvaluationMetricSession],
    candidate_sessions: Iterable[EvaluationMetricSession],
    metric: CompareMetric,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> PairedComparison:
    """Compare two evaluations from their raw per-session metric values.

    Applies the shared pairing rules to reduce each side's sessions to
    per-test-case means, then pairs and bootstraps.
    """
    return compare_test_case_means(
        mean_by_test_case(baseline_sessions),
        mean_by_test_case(candidate_sessions),
        metric,
        resamples=resamples,
        seed=seed,
    )
