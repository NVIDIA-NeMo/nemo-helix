# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared row-scoring and result-finalization primitives used during execution."""

from collections.abc import Iterable, Sequence
from logging import Logger, getLogger
from typing import Any

from nhx_evals_sdk.execution.samples import build_metric_input
from nhx_evals_sdk.execution.values import EvaluationError, EvaluationPhase
from nhx_evals_sdk.inference import requests_log_var
from nhx_evals_sdk.metrics.aggregation import (
    add_corpus_scores,
    aggregate_metrics,
    is_aggregateable_output_spec,
    rubric_definitions_from_metric,
)
from nhx_evals_sdk.metrics.protocol import (
    CorpusMetric,
    Metric,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
    validate_metric_result,
)
from nhx_evals_sdk.metrics.utils import metric_type_name
from nhx_evals_sdk.values import (
    EvaluationResult,
    RowScore,
)

logger = getLogger(__name__)


def nan_metric_result(outputs: Iterable[MetricOutputSpec]) -> MetricResult:
    """Build the NaN output payload used for ignored scoring failures.

    Only aggregateable outputs receive NaN placeholders; non-score outputs are
    not synthesized for failed rows.
    """
    return MetricResult(
        outputs=[
            MetricOutput(name=output.name, value=float("nan"))
            for output in outputs
            if is_aggregateable_output_spec(output)
        ]
    )


def corpus_output_spec(metric: Metric, fallback: list[MetricOutputSpec] | None = None) -> list[MetricOutputSpec]:
    """Return corpus-level output specs when a metric declares them."""
    corpus_spec = getattr(metric, "corpus_output_spec", None)
    if callable(corpus_spec):
        return list(corpus_spec())
    return list(fallback if fallback is not None else metric.output_spec())


CompletedRowEvaluation = tuple[int, MetricResult | None, RowScore]


def empty_evaluation_result() -> EvaluationResult:
    """Return the canonical empty evaluation result payload."""
    return EvaluationResult(row_scores=[], aggregate_scores=aggregate_metrics([], []))


async def finalize_evaluation_result(
    metric: Metric,
    eval_results: Sequence[CompletedRowEvaluation],
    *,
    skip_errored: bool = False,
) -> EvaluationResult:
    """Build the final evaluation result from eval_results row-level outputs.

    Callers are expected to pass ``eval_results`` in the original row order; the
    upstream pipeline (``run_indexed_tasks`` / ``run_generated_sample_scoring_pipeline``)
    already writes results by index, so no re-sorting is performed here.

    When ``skip_errored`` is true, rows whose ``RowScore.metric_errors`` is
    populated are excluded from aggregation so the NaN placeholder produced
    for ignored failures does not contribute to ``nan_count``, and the same
    rows are excluded from the ``items``/``samples`` passed to
    :meth:`CorpusMetric.compute_corpus_scores` so corpus-level aggregation
    (e.g. BLEU/ROUGE-corpus) isn't skewed by failed rows with empty samples.
    Errored rows still appear in ``row_scores``.
    """
    valid_eval_results = [
        (result, row_score)
        for _, result, row_score in eval_results
        if result is not None and not (skip_errored and row_score.metric_errors)
    ]
    metric_results = [result for result, _ in valid_eval_results]
    # Keep all rows in the reported ``row_scores`` (including errored ones); only
    # aggregation and corpus inputs honor ``skip_errored``.
    row_scores = [row_score for _, _, row_score in eval_results]

    output_spec = metric.output_spec()
    rubric_definitions = rubric_definitions_from_metric(metric)
    if rubric_definitions:
        aggregated_result = aggregate_metrics(metric_results, output_spec, rubric_definitions=rubric_definitions)
    else:
        aggregated_result = aggregate_metrics(metric_results, output_spec)

    if valid_eval_results and isinstance(metric, CorpusMetric):
        corpus_metric_result = await metric.compute_corpus_scores(
            inputs=[
                build_metric_input(row_score.item, row_score.sample, row_score.row_index)
                for _, row_score in valid_eval_results
            ],
        )
        if corpus_metric_result:
            add_corpus_scores(aggregated_result, corpus_metric_result, corpus_output_spec(metric, output_spec))

    return EvaluationResult(
        row_scores=row_scores,
        aggregate_scores=aggregated_result,
    )


async def score_row(
    metric: Metric,
    row: dict[str, Any],
    sample: dict[str, Any],
    index: int,
    metric_key: str,
    fail_fast: bool,
    generation_requests: list[dict[str, Any]],
    logger: Logger | None = None,
) -> tuple[int, MetricResult | None, RowScore]:
    """Score an already-prepared sample for one row.

    Args:
        metric: Metric object used for scoring.
        row: Input row from the dataset.
        sample: Prepared sample payload passed to the metric.
        index: Row position in the original dataset.
        metric_key: Key used to place score output in ``RowScore.metrics``.
        fail_fast: Whether metric errors should raise immediately. When
            ``True``, the exception is wrapped in ``EvaluationError`` and
            raised. When ``False``, a metric exception yields a NaN score row.
        generation_requests: Requests collected before metric scoring,
            such as online generation requests.
        logger: Optional logger override for row-scoring logs.
    Returns:
        Tuple of ``(index, metric_result_or_none, row_score_payload)``.

    Raises:
        EvaluationError: If row evaluation fails and ``fail_fast`` is ``True``.
    """

    metric_requests: list[dict[str, Any]] = []
    requests_log_var.set(metric_requests)
    active_logger = logger or globals()["logger"]

    try:
        output_spec = metric.output_spec()
        result = validate_metric_result(
            await metric.compute_scores(build_metric_input(row, sample, index)), output_spec
        )
        active_logger.debug(
            "Computed metric",
            extra={
                "item_index": index,
                "metric_type": metric_type_name(metric),
                "outputs": [output.model_dump() for output in result.outputs],
            },
        )
        return (
            index,
            result,
            RowScore(
                row_index=index,
                item=row,
                sample=sample,
                metrics={metric_key: result.outputs},
                requests=[*generation_requests, *metric_requests],
                metric_diagnostics={metric_key: result.diagnostics} if result.diagnostics else None,
            ),
        )
    except Exception as e:
        if fail_fast:
            raise EvaluationError(
                index,
                str(e),
                phase=EvaluationPhase.METRIC_SCORING,
                metric_key=metric_key,
            ) from e
        active_logger.warning("Evaluation failed, marking as NaN", extra={"item_index": index, "error": str(e)})
        result = nan_metric_result(metric.output_spec())
        return (
            index,
            result,
            RowScore(
                row_index=index,
                item=row,
                sample=sample,
                metrics={metric_key: result.outputs},
                requests=[*generation_requests, *metric_requests],
                metric_errors={metric_key: str(e)},
            ),
        )
