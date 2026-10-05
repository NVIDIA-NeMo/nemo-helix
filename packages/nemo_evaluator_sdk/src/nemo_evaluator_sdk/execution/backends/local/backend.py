# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local backend implementation for completed-result evaluator execution."""

from __future__ import annotations

from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from typing import Any

from nemo_evaluator_sdk.dataset_schemas.common import _MISSING, get_value_at_path
from nemo_evaluator_sdk.dataset_schemas.compatibility import apply_column_mapping_to_row
from nemo_evaluator_sdk.datasets.loader import prepare_dataset_rows
from nemo_evaluator_sdk.execution import benchmark_execution
from nemo_evaluator_sdk.execution.backends.base import BackendParams
from nemo_evaluator_sdk.execution.benchmark_execution import evaluate_benchmark as sdk_evaluate_benchmark
from nemo_evaluator_sdk.execution.metric_execution import (
    _merge_online_hooks,
    evaluate_metric,
    resolve_target_structured_output_mode,
)
from nemo_evaluator_sdk.execution.utils import prepare_metric_for_execution, unique_metric_keys
from nemo_evaluator_sdk.inference import PostprocessResponse, PreprocessRequest
from nemo_evaluator_sdk.metrics.protocol import Metric
from nemo_evaluator_sdk.resolvers import LocalModelResolver, LocalSecretResolver
from nemo_evaluator_sdk.session import begin_evaluation_session
from nemo_evaluator_sdk.values import DatasetInput, FieldMapping, Model
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult, namespace_result
from nemo_evaluator_sdk.values.results import AggregateFieldName, EvaluationResult
from nemo_evaluator_sdk.values.retrieval import Retrieval
from nemo_evaluator_sdk.values.targets import EvalTarget

log = getLogger(__name__)


def _prepare_rows(
    dataset: DatasetInput | str | Path,
    params: BackendParams,
    field_mapping: FieldMapping | None,
) -> list[dict[str, Any]]:
    """Load dataset rows, apply sampling limits, and project mapped fields."""
    rows = prepare_dataset_rows(
        dataset,
        None,
        params.limit_samples,
    )
    if field_mapping is None:
        return rows
    mapped = [apply_column_mapping_to_row(row, field_mapping) for row in rows]
    # Tested against the raw rows rather than by asking whether the canonical name is present:
    # mapping starts from a copy of the row, so a path that resolves nowhere leaves any same-named
    # raw column in place and that column gets scored. A binding that never resolves on any row is
    # a mistake worth naming; one that misses on some rows is not.
    unresolved = sorted(
        name
        for name, path in field_mapping.mapping().items()
        if all(get_value_at_path(row, path) is _MISSING for row in rows)
    )
    if rows and unresolved:
        log.warning(
            "field_mapping %s resolved no value in any of %d rows, so those canonical fields were "
            "not set and any same-named raw column was left in place. Check the path, and for a "
            "predicate such as messages[role=assistant] that some element matches.",
            unresolved,
            len(rows),
        )
    return mapped


class LocalBackend:
    """Local backend that executes metrics in-process."""

    def __init__(self) -> None:
        """Create a local backend with local resolver defaults."""
        self.secret_resolver = LocalSecretResolver()
        self.model_resolver = LocalModelResolver()

    async def _evaluate_one(
        self,
        *,
        metric: Metric,
        metric_key: str,
        params: BackendParams,
        target: EvalTarget,
        prompt_template: str | dict[str, Any] | None,
        aggregate_fields: tuple[AggregateFieldName, ...] | None,
        preprocess_hooks: tuple[PreprocessRequest, ...] | None,
        postprocess_hooks: tuple[PostprocessResponse, ...] | None,
        rows: list[dict[str, Any]],
    ) -> EvaluationResult:
        """Prepare one metric and execute it through the local runtime.

        Args:
            metric: Metric to execute.
            metric_key: Public metric key used to namespace the result.
            params: Validated run configuration for the selected target mode.
            target: Optional model or agent used to generate candidate responses before scoring.
            prompt_template: Optional prompt template for online target generation.
            aggregate_fields: Optional aggregate score fields to keep in the returned result.
            preprocess_hooks: Optional request preprocess hooks for online execution.
            postprocess_hooks: Optional response postprocess hooks for online execution.
            rows: Precomputed dataset rows shared across metrics in the request.

        Returns:
            A namespaced single-metric evaluation result.
        """
        # Preparation runs each metric's preflight, so it belongs inside the session.
        async with begin_evaluation_session():
            return await self._evaluate_one_in_session(
                metric=metric,
                metric_key=metric_key,
                params=params,
                target=target,
                prompt_template=prompt_template,
                aggregate_fields=aggregate_fields,
                preprocess_hooks=preprocess_hooks,
                postprocess_hooks=postprocess_hooks,
                rows=rows,
            )

    async def _evaluate_one_in_session(
        self,
        *,
        metric: Metric,
        metric_key: str,
        params: BackendParams,
        target: EvalTarget,
        prompt_template: str | dict[str, Any] | None,
        aggregate_fields: tuple[AggregateFieldName, ...] | None,
        preprocess_hooks: tuple[PreprocessRequest, ...] | None,
        postprocess_hooks: tuple[PostprocessResponse, ...] | None,
        rows: list[dict[str, Any]],
    ) -> EvaluationResult:
        prepared_metric = await prepare_metric_for_execution(
            metric,
            params=params,
            model_resolver=self.model_resolver,
            secret_resolver=self.secret_resolver,
        )

        result = await evaluate_metric(
            metric=prepared_metric,
            target=target,
            rows=rows,
            prompt_template=prompt_template,
            params=params,
            preprocess_hooks=preprocess_hooks,
            postprocess_hooks=postprocess_hooks,
        )

        return namespace_result(metric_key, result, aggregate_fields)

    async def evaluate_dataset(
        self,
        *,
        metrics: Sequence[Metric],
        dataset: DatasetInput | str | Path,
        params: BackendParams,
        target: EvalTarget = None,
        field_mapping: FieldMapping | None = None,
        prompt_template: str | dict[str, Any] | None = None,
        aggregate_fields: tuple[AggregateFieldName, ...] | None = None,
        preprocess_hooks: tuple[PreprocessRequest, ...] | None = None,
        postprocess_hooks: tuple[PostprocessResponse, ...] | None = None,
    ) -> BenchmarkEvaluationResult:
        """Execute multiple metrics locally using the shared streaming pipeline.

        Delegates to :func:`sdk_evaluate_benchmark` so that each dataset row runs
        target inference exactly once, regardless of metric count.

        Args:
            metrics: Metrics to prepare and execute together.
            dataset: Inline dataset rows, a dataset file, or a dataset directory/glob path.
            params: Validated run configuration for the selected target mode.
            target: Optional model or agent used to generate candidate responses before scoring.
            field_mapping: Optional mapping from canonical evaluator fields to dataset columns.
            prompt_template: Optional prompt template for online target generation.
            aggregate_fields: Optional aggregate score fields to keep in the returned result.
            preprocess_hooks: Optional request preprocess hooks for online execution.
            postprocess_hooks: Optional response postprocess hooks for online execution.

        Returns:
            A completed multi-metric result.
        """
        rows = _prepare_rows(dataset, params, field_mapping)
        metric_keys = unique_metric_keys(metrics)
        # Preparation runs each metric's preflight, so it belongs inside the session.
        async with begin_evaluation_session():
            return await self._evaluate_dataset_in_session(
                metrics=metrics,
                metric_keys=metric_keys,
                rows=rows,
                params=params,
                target=target,
                prompt_template=prompt_template,
                aggregate_fields=aggregate_fields,
                preprocess_hooks=preprocess_hooks,
                postprocess_hooks=postprocess_hooks,
            )

    async def _evaluate_dataset_in_session(
        self,
        *,
        metrics: Sequence[Metric],
        metric_keys: Sequence[str],
        rows: list[dict[str, Any]],
        params: BackendParams,
        target: EvalTarget,
        prompt_template: str | dict[str, Any] | None,
        aggregate_fields: tuple[AggregateFieldName, ...] | None,
        preprocess_hooks: Sequence[PreprocessRequest] | None,
        postprocess_hooks: Sequence[PostprocessResponse] | None,
    ) -> BenchmarkEvaluationResult:
        prepared_metrics = [
            await prepare_metric_for_execution(
                metric,
                params=params,
                model_resolver=self.model_resolver,
                secret_resolver=self.secret_resolver,
            )
            for metric in metrics
        ]
        metrics_built: list[tuple[str, Metric]] = list(zip(metric_keys, prepared_metrics, strict=True))
        if target is not None and not isinstance(target, Retrieval):
            merged_preprocess_hooks, merged_postprocess_hooks = _merge_online_hooks(
                params=params,
                target=target,
                preprocess_hooks=preprocess_hooks,
                postprocess_hooks=postprocess_hooks,
            )
        else:
            merged_preprocess_hooks = tuple(preprocess_hooks or ())
            merged_postprocess_hooks = tuple(postprocess_hooks or ())
        # This path builds target hooks itself and bypasses evaluate_metric, so it resolves here.
        if isinstance(target, Model):
            await resolve_target_structured_output_mode(
                preprocess_hooks=merged_preprocess_hooks,
                model=target,
                # Attribute lookup, not a bound import: the probe must use the same transport
                # evaluate_benchmark defaults to, including when that binding is swapped.
                inference_fn=benchmark_execution.make_inference_request,
                params=params,
            )
        return await sdk_evaluate_benchmark(
            metrics=metrics_built,
            rows=rows,
            target=target,
            params=params,
            prompt_template=prompt_template,
            preprocess_hooks=merged_preprocess_hooks,
            postprocess_hooks=merged_postprocess_hooks,
            aggregate_fields=aggregate_fields,
            logger=log,
        )
