# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregation data structures and computations for metric results."""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from nemo_evaluator_sdk.metrics.protocol import (
    BooleanValue,
    ContinuousScore,
    DiscreteScore,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
    validate_metric_result,
)
from nemo_evaluator_sdk.values.results import (
    AggregatedMetricResult,
    AggregateRangeScore,
    AggregateRubricScore,
    AggregateScore,
    Histogram,
    HistogramBin,
    MetricScore,
    Percentiles,
    RubricScoreStat,
    ScoreStats,
)

if TYPE_CHECKING:
    # Importing the score configuration module loads jsonschema. Aggregation's lightweight helpers
    # (notably compute_percentiles) do not need it, so keep this typing-only edge deferred.
    from nemo_evaluator_sdk.values.scores import Score


def is_aggregateable_output_spec(output_spec: MetricOutputSpec) -> bool:
    """Return whether an output should contribute aggregate statistics."""
    return issubclass(output_spec.value_schema, (ContinuousScore, DiscreteScore, BooleanValue))


@runtime_checkable
class MetricWithScores(Protocol):
    """Metric config protocol for metrics that carry rubric score definitions."""

    @property
    def scores(self) -> Sequence[Score]: ...


def _coerce_aggregate_output(output: MetricOutput, output_spec: MetricOutputSpec) -> MetricScore:
    """Convert one declared metric output into MetricScore form.

    ``output_spec`` must be aggregateable; callers select those with
    :func:`is_aggregateable_output_spec` before they get here.
    """
    if (
        issubclass(output_spec.value_schema, BooleanValue)
        and isinstance(output.value, float)
        and math.isnan(output.value)
    ):
        return MetricScore(name=output.name, value=output.value)
    coerced = cast(ContinuousScore | DiscreteScore | BooleanValue, output_spec.coerce_output(output))
    value = coerced.root
    if isinstance(value, bool):
        value = 1.0 if value else 0.0
    return MetricScore(name=output.name, value=value)


def _score_or_nan(output: MetricOutput | None, output_spec: MetricOutputSpec) -> MetricScore:
    """The spec's score for this row, or NaN when an optional output was not emitted."""
    if output is None:
        return MetricScore(name=output_spec.name, value=float("nan"))
    return _coerce_aggregate_output(output, output_spec)


def _attach_rubric_stats(
    score: MetricScore,
    output_by_name: Mapping[str, MetricOutput],
    rubric_definitions: Mapping[str, Sequence[RubricScoreStat]],
) -> MetricScore:
    """Attach per-row rubric bucket stats from companion label outputs."""
    rubric_definition = rubric_definitions.get(score.name)
    if not rubric_definition:
        return score

    label_output = output_by_name.get(f"{score.name}.label")
    selected_label = label_output.value if label_output is not None else None
    if isinstance(score.value, float) and math.isnan(score.value):
        selected_label = None

    rubric_distribution = [
        RubricScoreStat(
            label=rubric.label,
            description=rubric.description,
            value=rubric.value,
            count=int(isinstance(selected_label, str) and selected_label == rubric.label),
        )
        for rubric in rubric_definition
    ]
    return MetricScore(
        name=score.name,
        value=score.value,
        stats=ScoreStats(rubric_distribution=rubric_distribution),
    )


def add_corpus_scores(
    aggregated_result: AggregatedMetricResult,
    corpus_result: MetricResult,
    output_specs: list[MetricOutputSpec],
) -> None:
    """Append corpus-level scores using aggregate-score schema fields.

    Args:
        aggregated_result: Aggregate result object to mutate.
        corpus_result: Corpus-level metric output with one or more scores.

    Returns:
        ``None``. The ``aggregated_result`` object is updated in place.
    """
    validated_result = validate_metric_result(corpus_result, output_specs)
    outputs_by_name = {output.name: output for output in validated_result.outputs}
    aggregateable_specs = [output_spec for output_spec in output_specs if is_aggregateable_output_spec(output_spec)]
    for output_spec in aggregateable_specs:
        score = _score_or_nan(outputs_by_name.get(output_spec.name), output_spec)
        value = score.value
        if isinstance(value, float) and math.isnan(value):
            # Every statistic is undefined, so all of them stay at their model default of None;
            # `count=0` asserts that nothing was evaluated, which is not the same as unknown.
            aggregated_result.scores.append(
                AggregateRangeScore(name=score.name, count=0, nan_count=1, histogram=Histogram(bins=[]))
            )
            continue
        # Corpus-level metrics contribute one already-aggregated value, so
        # expose them through the same aggregate schema with count=1.
        corpus_score = AggregateRangeScore(
            name=score.name,
            count=1,
            nan_count=0,
            sum=value,
            mean=value,
            min=value,
            max=value,
            std_dev=0.0,
            variance=0.0,
            percentiles=Percentiles(
                p10=value,
                p20=value,
                p30=value,
                p40=value,
                p50=value,
                p60=value,
                p70=value,
                p80=value,
                p90=value,
                p100=value,
            ),
            histogram=Histogram(bins=[HistogramBin(lower_bound=value, upper_bound=value, count=1)]),
        )
        aggregated_result.scores.append(corpus_score)


def _compute_percentile(sorted_values: list[float], percentile: float) -> float:
    """Compute a percentile using linearly interpolated rank position.

    The rank position is computed as ``(p/100) * (n + 1) - 1`` and then
    interpolated between neighboring points when the position is fractional.

    Args:
        sorted_values: Score values sorted in ascending order.
        percentile: Percentile in the inclusive range ``[0, 100]``.

    Returns:
        The interpolated percentile value, or ``0.0`` when no values exist.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    pos = (percentile / 100.0) * (n + 1) - 1
    if pos <= 0:
        return sorted_values[0]
    if pos >= n - 1:
        return sorted_values[-1]
    lower_idx = int(pos)
    frac = pos - lower_idx
    return sorted_values[lower_idx] + frac * (sorted_values[lower_idx + 1] - sorted_values[lower_idx])


def compute_percentiles(sorted_values: list[float]) -> Percentiles:
    """Compute the fixed percentile set used by SDK aggregate output.

    Args:
        sorted_values: Score values sorted in ascending order.

    Returns:
        ``Percentiles`` containing p10 through p100.
    """
    return Percentiles(
        p10=_compute_percentile(sorted_values, 10),
        p20=_compute_percentile(sorted_values, 20),
        p30=_compute_percentile(sorted_values, 30),
        p40=_compute_percentile(sorted_values, 40),
        p50=_compute_percentile(sorted_values, 50),
        p60=_compute_percentile(sorted_values, 60),
        p70=_compute_percentile(sorted_values, 70),
        p80=_compute_percentile(sorted_values, 80),
        p90=_compute_percentile(sorted_values, 90),
        p100=_compute_percentile(sorted_values, 100),
    )


def _compute_histogram(values: list[float], num_bins: int = 10) -> Histogram:
    """Build a fixed-width histogram for numeric score distribution.

    The algorithm uses equal-width bins between the global minimum and maximum.
    All bins except the last are half-open ``[lower, upper)``, while the final
    bin is closed ``[lower, upper]`` so the maximum value is always counted.

    Args:
        values: Numeric score values to bucket.
        num_bins: Number of equally sized bins.

    Returns:
        Histogram object containing ordered bin counts.
    """
    if not values:
        return Histogram(bins=[])

    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return Histogram(bins=[HistogramBin(lower_bound=min_val, upper_bound=max_val, count=len(values))])

    bin_width = (max_val - min_val) / num_bins
    bins: list[HistogramBin] = []
    for i in range(num_bins):
        lower = min_val + i * bin_width
        upper = min_val + (i + 1) * bin_width
        if i == num_bins - 1:
            count = sum(1 for v in values if lower <= v <= upper)
        else:
            count = sum(1 for v in values if lower <= v < upper)
        bins.append(HistogramBin(lower_bound=lower, upper_bound=upper, count=count))
    return Histogram(bins=bins)


def aggregate_metrics(
    items: list[MetricResult],
    output_specs: list[MetricOutputSpec],
    rubric_definitions: Mapping[str, Sequence[RubricScoreStat]] | None = None,
) -> AggregatedMetricResult:
    """Aggregate row-level metric results into range or rubric summaries.

    This function performs two logical passes:
    1. Incremental accumulation of count/sum/min/max and rubric occurrences.
    2. Finalization of derived statistics (variance, percentiles, histogram,
       and rubric mode category) once complete value sets are known.

    Args:
        items: Row-level metric results to aggregate.
        output_specs: Declared outputs for the metric. Only continuous,
            discrete, and boolean output values contribute to aggregate scores.
        rubric_definitions: Optional rubric bucket definitions keyed by numeric
            output name. This is aggregation metadata, not metric protocol
            metadata, and is usually derived from LLM judge score config.

    Returns:
        Aggregate metric result with one aggregate score per score name.
    """
    score_values: dict[str, list[float]] = defaultdict(list)

    aggregated_results: dict[str, MetricScore] = {}
    rubric_distribution: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(OrderedDict))
    has_rubric: dict[str, bool] = {}

    aggregateable_specs = [output_spec for output_spec in output_specs if is_aggregateable_output_spec(output_spec)]
    if not items:
        return AggregatedMetricResult(scores=[])

    for output_spec in aggregateable_specs:
        score_name = output_spec.name
        aggregated_results[score_name] = MetricScore(
            name=score_name,
            value=0.0,
            stats=ScoreStats(
                count=0,
                sum=0,
                sum_squared=0,
                min=None,
                max=None,
                mean=0,
                variance=None,
                stddev=None,
                nan_count=0,
            ),
        )
        rubric_definition = (rubric_definitions or {}).get(score_name)
        has_rubric[score_name] = bool(rubric_definition)
        if rubric_definition:
            for rubric_stat in rubric_definition:
                rubric_distribution[score_name][rubric_stat.label] = {
                    "label": rubric_stat.label,
                    "value": rubric_stat.value,
                    "count": 0,
                }

    for item in items:
        output_by_name = {output.name: output for output in item.outputs}
        missing = [
            output_spec.name
            for output_spec in aggregateable_specs
            if output_spec.required and output_spec.name not in output_by_name
        ]
        if missing:
            raise ValueError(f"Missing required metric outputs: {missing}")
        for output_spec in aggregateable_specs:
            score = _score_or_nan(output_by_name.get(output_spec.name), output_spec)
            if has_rubric.get(score.name):
                score = _attach_rubric_stats(score, output_by_name, rubric_definitions or {})

            if score.stats and score.stats.rubric_distribution:
                for rubric_stat in score.stats.rubric_distribution:
                    if rubric_stat.count:
                        rubric_distribution[score.name][rubric_stat.label]["count"] += 1

            results = aggregated_results[score.name]
            assert results.stats is not None

            # int values can never be NaN in Python, so skipping int and
            # float checks is sufficient.
            # float('nan') is the only way to get a NaN in Python.
            if isinstance(score.value, float) and math.isnan(score.value):
                results.stats.nan_count = (results.stats.nan_count or 0) + 1
                continue

            score_values[score.name].append(score.value)
            results.stats.count = (results.stats.count or 0) + 1
            results.stats.sum = (results.stats.sum or 0) + score.value
            results.stats.sum_squared = (results.stats.sum_squared or 0) + score.value**2
            results.stats.mean = results.stats.sum / results.stats.count

            if results.stats.min is None or score.value < results.stats.min:
                results.stats.min = score.value
            if results.stats.max is None or score.value > results.stats.max:
                results.stats.max = score.value

            results.value = results.stats.mean

    for score_name, values in score_values.items():
        if not values:
            continue
        results = aggregated_results[score_name]
        assert results.stats is not None

        n = len(values)
        mean = results.stats.mean or 0
        # Report both conventions under explicit names rather than leaving the divisor implicit.
        # `variance`/`stddev` stay population (divide by n): these rows are the full evaluation set,
        # not a sample intended to estimate a larger population. The sample (n-1) figures are also
        # provided for callers estimating the spread of the process the values were drawn from, and
        # are undefined for a single value.
        sum_sq_dev = sum((v - mean) ** 2 for v in values)
        variance = sum_sq_dev / n if n > 0 else 0
        results.stats.variance = variance
        results.stats.stddev = math.sqrt(variance)
        sample_variance = sum_sq_dev / (n - 1) if n > 1 else None
        results.stats.sample_variance = sample_variance
        results.stats.sample_stddev = math.sqrt(sample_variance) if sample_variance is not None else None

    aggregated_scores: list[AggregateScore] = []
    for score_name, metric_score in aggregated_results.items():
        stats = metric_score.stats
        assert stats is not None

        values = score_values[score_name]
        base_name = metric_score.name
        base_count = stats.count or 0
        base_nan_count = stats.nan_count or 0
        base_mean = stats.mean
        base_sum = stats.sum if stats.sum is not None else (None if base_mean is None else (base_mean * base_count))
        base_min = stats.min if stats.min is not None else base_mean
        base_max = stats.max if stats.max is not None else base_mean
        base_variance = stats.variance if stats.variance is not None else (None if base_mean is None else 0.0)
        base_std_dev = stats.stddev if stats.stddev is not None else (None if base_mean is None else 0.0)
        # Sample stats stay None when undefined (fewer than two values) rather than defaulting to 0.0.
        base_sample_variance = stats.sample_variance
        base_sample_std_dev = stats.sample_stddev
        # Derived from the same helper that produces p50, so `median` and `percentiles.p50` agree
        # exactly wherever both are present (rubric scores carry no percentiles but still get a median).
        base_median = _compute_percentile(sorted(values), 50) if values else None

        if base_count == 0:
            base_sum = None
            base_mean = None
            base_min = None
            base_max = None
            base_variance = None
            base_std_dev = None
            base_sample_variance = None
            base_sample_std_dev = None
            base_median = None

        if has_rubric.get(score_name):
            rubric_dist = [
                RubricScoreStat(label=r["label"], value=r["value"], count=r["count"])
                for r in rubric_distribution[score_name].values()
            ]
            mode_category = None
            if any(r.count for r in rubric_dist):
                # Break ties deterministically so aggregate output is stable.
                max_count = max(r.count for r in rubric_dist)
                tied = sorted(r.label for r in rubric_dist if r.count == max_count)
                mode_category = tied[0]

            aggregated_scores.append(
                AggregateRubricScore(
                    name=base_name,
                    count=base_count,
                    nan_count=base_nan_count,
                    sum=base_sum,
                    mean=base_mean,
                    min=base_min,
                    max=base_max,
                    median=base_median,
                    variance=base_variance,
                    std_dev=base_std_dev,
                    sample_variance=base_sample_variance,
                    sample_std_dev=base_sample_std_dev,
                    rubric_distribution=rubric_dist,
                    mode_category=mode_category,
                )
            )
        else:
            if values:
                # Range scores get richer distribution metadata than rubric scores.
                sorted_values = sorted(values)
                percentiles = compute_percentiles(sorted_values)
                histogram = _compute_histogram(values)
            else:
                percentiles = None
                histogram = Histogram(bins=[])

            aggregated_scores.append(
                AggregateRangeScore(
                    name=base_name,
                    count=base_count,
                    nan_count=base_nan_count,
                    sum=base_sum,
                    mean=base_mean,
                    min=base_min,
                    max=base_max,
                    median=base_median,
                    variance=base_variance,
                    std_dev=base_std_dev,
                    sample_variance=base_sample_variance,
                    sample_std_dev=base_sample_std_dev,
                    percentiles=percentiles,
                    histogram=histogram,
                )
            )

    return AggregatedMetricResult(scores=aggregated_scores)


def rubric_definitions_from_scores(scores: Sequence[Score]) -> dict[str, list[RubricScoreStat]]:
    """Return declared rubric buckets keyed by score name."""
    from nemo_evaluator_sdk.values.scores import RubricScore

    definitions: dict[str, list[RubricScoreStat]] = {}
    for score in scores:
        if not isinstance(score, RubricScore):
            continue
        definitions[score.name] = [
            RubricScoreStat(
                label=rubric.label,
                description=rubric.description,
                value=rubric.value,
                count=0,
            )
            for rubric in score.rubric
        ]
    return definitions


def rubric_definitions_from_metric(metric: object) -> dict[str, list[RubricScoreStat]]:
    """Return rubric bucket definitions for metrics that carry score config."""
    if not isinstance(metric, MetricWithScores):
        return {}
    scores = metric.scores
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        return {}
    return rubric_definitions_from_scores(scores)
