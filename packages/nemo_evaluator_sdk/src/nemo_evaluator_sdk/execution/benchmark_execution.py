# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming benchmark execution pipeline for evaluator SDK runtime.

Generates each sample once and fans it out to every metric worker so that
multi-metric online benchmarks do not duplicate target inference per metric.

Pipeline shape:
1. Producer workers generate per-row samples and broadcast each sample
   directly to every per-metric queue (``_run_producer_workers``).
2. Metric workers consume their queue and emit row-level metric results.
3. The orchestrator awaits one failure-propagating ``asyncio.TaskGroup`` and
   assembles a :class:`BenchmarkEvaluationResult`.

Shutdown signaling for metric workers flows through ``_put_pipeline_sentinels``
with a cancellation-safe ``put_nowait`` fallback so that task-group cancellation
does not deadlock on a full per-metric queue.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger
from types import MappingProxyType
from typing import Any, Protocol, cast

import httpx
from nemo_evaluator_sdk.agent_inference import AgentInferenceFn, new_agent_inference_client
from nemo_evaluator_sdk.execution.config import fail_fast_from_params
from nemo_evaluator_sdk.execution.metric_execution import (
    generate_online_sample,
    generate_online_sample_agent,
)
from nemo_evaluator_sdk.execution.samples import build_metric_input, build_offline_sample, build_retrieval_sample
from nemo_evaluator_sdk.execution.scoring import (
    corpus_output_spec,
    nan_metric_result,
)
from nemo_evaluator_sdk.execution.values import EvaluationError, EvaluationPhase
from nemo_evaluator_sdk.inference import (
    InferenceFn,
    PostprocessResponse,
    PreprocessRequest,
    make_inference_request,
    new_inference_client,
    requests_log_var,
)
from nemo_evaluator_sdk.metrics.aggregation import (
    add_corpus_scores,
    aggregate_metrics,
    rubric_definitions_from_metric,
)
from nemo_evaluator_sdk.metrics.protocol import (
    CorpusMetric,
    Metric,
    MetricOutputSpec,
    MetricResult,
    validate_metric_result,
)
from nemo_evaluator_sdk.resilience.api import use_resilience_session
from nemo_evaluator_sdk.resilience.errors import first_failure_cause, iter_leaf_causes
from nemo_evaluator_sdk.values import (
    Agent,
    AgentBase,
    AggregatedMetricResult,
    AggregateFieldName,
    EvaluationResult,
    MetricDiagnostic,
    Model,
    RowScore,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)
from nemo_evaluator_sdk.values.multi_metric_results import (
    BenchmarkEvaluationResult,
    namespace_result,
)
from nemo_evaluator_sdk.values.retrieval import Retrieval
from nemo_evaluator_sdk.values.targets import EvalTarget
from openai import AsyncOpenAI

_log = getLogger(__name__)

_QUEUE_END: object = object()

BenchmarkParams = RunConfig | RunConfigOnline | RunConfigOnlineModel


class ProgressReporter(Protocol):
    """Progress-callback surface used by the benchmark pipeline.

    The pipeline always invokes ``increment_work()`` with no positional
    arguments. The protocol accepts an optional ``increment`` and an arbitrary
    return type so richer service-side implementations (e.g., ones that batch
    updates or return an HTTP response) satisfy the structural contract.
    """

    def increment_work(self, increment: int = 1, /) -> Any:
        """Signal that ``increment`` units of work completed."""
        ...


@dataclass(frozen=True)
class _SampleEvent:
    """One generated row sample ready to fan out to metric workers."""

    row_index: int
    item: MappingProxyType
    sample: MappingProxyType
    requests: list[dict]


@dataclass
class _MetricPipeline:
    """Queue + result storage for one benchmark metric."""

    metric_ref: str
    metric: Metric
    output_spec: list[MetricOutputSpec]
    queue: asyncio.Queue
    results: list[MetricResult | None]


def _normalize_metric_result(metric_result: MetricResult, expected_outputs: list[MetricOutputSpec]) -> MetricResult:
    """Validate output names and normalize output ordering to the declared spec."""
    validated = validate_metric_result(metric_result, expected_outputs)
    actual_outputs = {output.name: output for output in validated.outputs}
    return MetricResult(
        outputs=[actual_outputs[output.name] for output in expected_outputs if output.name in actual_outputs],
        diagnostics=validated.diagnostics,
    )


def _benchmark_error_from_exception(exc: BaseException) -> EvaluationError | None:
    """Return the deterministic benchmark error leaf from an exception tree."""
    errors = [leaf for leaf in iter_leaf_causes(exc) if isinstance(leaf, EvaluationError)]
    if not errors:
        return None
    return min(errors, key=lambda error: (error.index, error.metric_key or ""))


def _initialize_row_scores(items: list[dict]) -> list[RowScore]:
    """Create deterministic per-row output slots aligned with ``items``."""
    return [RowScore(row_index=idx, item=item, sample={}, metrics={}, requests=[]) for idx, item in enumerate(items)]


def _build_metric_pipelines(
    metrics: Sequence[tuple[str, Metric]],
    *,
    item_count: int,
    queue_capacity: int,
) -> list[_MetricPipeline]:
    """Attach one bounded queue per pre-built metric.

    Callers pass an ordered ``(metric_ref, metric)`` sequence so that service
    and SDK callers can build concrete metrics with their own factories before
    entering the pipeline.
    """
    pipelines: list[_MetricPipeline] = []
    for metric_ref, metric in metrics:
        output_spec = list(metric.output_spec())
        if not output_spec:
            raise RuntimeError(f"Metric '{metric_ref}' does not declare any outputs")
        pipelines.append(
            _MetricPipeline(
                metric_ref=metric_ref,
                metric=metric,
                output_spec=output_spec,
                queue=asyncio.Queue(maxsize=queue_capacity),
                results=[None] * item_count,
            )
        )
    return pipelines


def _finalize_row_request_logs(
    *,
    row_scores: list[RowScore],
    row_metric_requests: list[dict[str, list[dict]]],
    metric_refs_in_order: list[str],
) -> None:
    """Append metric requests to each row in deterministic metric order."""
    for row_idx, row_score in enumerate(row_scores):
        by_metric = row_metric_requests[row_idx]
        for metric_ref in metric_refs_in_order:
            row_score.requests.extend(by_metric.get(metric_ref, []))


def _metric_errors_for_ref(metric_errors: dict[str, str] | None, metric_ref: str) -> dict[str, str] | None:
    """Return the metric-specific error payload for a per-metric row.

    Combined benchmark rows can contain errors for multiple metric refs. When
    building an individual metric result, keep only that metric's error so
    per-metric exports and summaries report the same failed row status without
    leaking sibling metric failures.
    """
    if not metric_errors:
        return None
    error = metric_errors.get(metric_ref)
    if error is None:
        return None
    return {metric_ref: error}


def _metric_diagnostics_for_ref(
    metric_diagnostics: dict[str, list[MetricDiagnostic]] | None, metric_ref: str
) -> dict[str, list[MetricDiagnostic]] | None:
    """Return the metric-specific diagnostics payload for a per-metric row."""
    if not metric_diagnostics:
        return None
    diagnostics = metric_diagnostics.get(metric_ref)
    if diagnostics is None:
        return None
    return {metric_ref: diagnostics}


async def _finalize_benchmark_metric_result(
    *,
    metric: Metric,
    results: Sequence[MetricResult | None],
    row_scores: list[RowScore],
) -> EvaluationResult:
    """Build one metric's benchmark result while preserving NaN fallback rows.

    Benchmark lenient failures materialize as NaN metric results and
    must remain in aggregate statistics. Corpus-level scoring is different: it
    should only see successful rows so failed rows with empty samples do not
    skew corpus metrics.
    """
    output_spec = metric.output_spec()
    metric_results = [result for result in results if result is not None]
    rubric_definitions = rubric_definitions_from_metric(metric)
    if rubric_definitions:
        aggregated = aggregate_metrics(metric_results, output_spec, rubric_definitions=rubric_definitions)
    else:
        aggregated = aggregate_metrics(metric_results, output_spec)
    if isinstance(metric, CorpusMetric):
        # Ignored sample-generation failures intentionally keep metric_errors
        # empty to match the previous service benchmark row artifacts, so
        # exclude their placeholder samples from corpus scoring explicitly.
        corpus_rows = [
            row_score
            for row_score in row_scores
            if not row_score.metric_errors and "inference_error" not in row_score.sample
        ]
        if corpus_rows:
            corpus_result = await metric.compute_corpus_scores(
                inputs=[
                    build_metric_input(row_score.item, row_score.sample, row_score.row_index)
                    for row_score in corpus_rows
                ],
            )
            if corpus_result is not None:
                add_corpus_scores(aggregated, corpus_result, corpus_output_spec(metric, output_spec))
    return EvaluationResult(row_scores=row_scores, aggregate_scores=aggregated)


async def _put_pipeline_sentinels(
    *,
    pipelines: list[_MetricPipeline],
    worker_count: int,
) -> None:
    """Signal metric workers to stop by pushing one sentinel per worker.

    During TaskGroup cancellation, awaiting on a full queue can deadlock
    shutdown. Best-effort sentinel insertion is sufficient because sibling
    tasks are being cancelled anyway.
    """
    current = asyncio.current_task()
    cancelling = current is not None and current.cancelling() > 0
    for pipeline in pipelines:
        for _ in range(worker_count):
            if cancelling:
                try:
                    pipeline.queue.put_nowait(_QUEUE_END)
                except asyncio.QueueFull:
                    pass
            else:
                await pipeline.queue.put(_QUEUE_END)


async def _run_producer_workers(
    *,
    items: list[dict],
    target: EvalTarget,
    inference_fn: InferenceFn | AgentInferenceFn | None,
    client: AsyncOpenAI | httpx.AsyncClient | None = None,
    params: BenchmarkParams,
    prompt_template: str | dict[str, Any] | None,
    row_scores: list[RowScore],
    pipelines: list[_MetricPipeline],
    worker_count: int,
    default_headers: dict[str, str] | None,
    preprocess_hooks: Sequence[PreprocessRequest],
    postprocess_hooks: Sequence[PostprocessResponse],
    logger: logging.Logger,
) -> None:
    """Generate sample events concurrently and broadcast to every metric queue.

    Each worker pops a row index, runs inference for that row, writes the
    row's sample and request log into ``row_scores``, then pushes the resulting
    event onto every metric pipeline queue. The only backpressure point is the
    per-metric queues, whose shutdown path is cancellation-safe via
    :func:`_put_pipeline_sentinels`.
    """
    index_queue: asyncio.Queue[int] = asyncio.Queue()
    for idx in range(len(items)):
        index_queue.put_nowait(idx)

    is_online = target is not None and not isinstance(target, Retrieval)
    tolerate_failure = not fail_fast_from_params(params)
    online_params = params if isinstance(params, RunConfigOnline) else None
    online_model_params = params if isinstance(params, RunConfigOnlineModel) else None

    async def _produce_worker() -> None:
        while True:
            try:
                idx = index_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            item = items[idx]
            requests_log: list[dict] = []
            requests_log_var.set(requests_log)
            try:
                if isinstance(target, Retrieval):
                    sample = build_retrieval_sample(item, target.rankings)
                elif is_online:
                    assert target is not None
                    if prompt_template is None:
                        raise ValueError("prompt_template is required for online benchmark evaluation")
                    if isinstance(target, AgentBase):
                        agent_target = cast(Agent, target)
                        sample = await generate_online_sample_agent(
                            agent=agent_target,
                            row=item,
                            index=idx,
                            prompt_template=prompt_template,
                            params=online_params,
                            preprocess_hooks=preprocess_hooks,
                            postprocess_hooks=postprocess_hooks,
                            agent_inference_fn=cast(AgentInferenceFn | None, inference_fn),
                            client=cast(httpx.AsyncClient | None, client),
                            default_headers=default_headers,
                        )
                    else:
                        model_inference_fn = (
                            cast(InferenceFn, inference_fn) if inference_fn is not None else make_inference_request
                        )
                        sample = await generate_online_sample(
                            target=target,
                            row=item,
                            index=idx,
                            prompt_template=prompt_template,
                            params=online_model_params,
                            preprocess_hooks=preprocess_hooks,
                            postprocess_hooks=postprocess_hooks,
                            inference_fn=model_inference_fn,
                            client=cast(AsyncOpenAI | None, client),
                            default_headers=default_headers,
                        )
                else:
                    sample = build_offline_sample(item)
                    # TODO: Consider matching ComputeMetricPipeline by applying
                    # offline postprocess hooks after reconciling service
                    # progress tracking so sample hooks do not double count.
            except Exception as e:
                if not tolerate_failure:
                    raise EvaluationError(
                        index=idx,
                        message=str(e),
                        phase=EvaluationPhase.SAMPLE_GENERATION,
                    ) from e
                logger.warning(
                    "Online sample generation failed, marking row as NaN-eligible",
                    extra={"item_index": idx, "error": str(e)},
                )
                # TODO: Consider short-circuiting metric workers for ignored
                # sample-generation failures instead of forwarding the
                # inference_error placeholder sample to each metric.
                sample = {"output_text": None, "response": {}, "inference_error": str(e)}
            finally:
                index_queue.task_done()

            row_scores[idx].sample = dict(sample)
            row_scores[idx].requests.extend(requests_log)

            event = _SampleEvent(
                row_index=idx,
                item=MappingProxyType(item),
                sample=MappingProxyType(sample),
                requests=requests_log,
            )
            for pipeline in pipelines:
                await pipeline.queue.put(event)

    producer_workers = min(worker_count, len(items))
    async with asyncio.TaskGroup() as producer_tg:
        for _ in range(producer_workers):
            producer_tg.create_task(_produce_worker())


async def _metric_worker(
    *,
    params: BenchmarkParams,
    pipeline: _MetricPipeline,
    row_scores: list[RowScore],
    row_metric_requests: list[dict[str, list[dict]]],
    logger: logging.Logger,
    progress: ProgressReporter | None = None,
) -> None:
    """Consume one metric queue and compute row-level metric results."""
    tolerate_failure = not fail_fast_from_params(params)
    while True:
        queued_event = await pipeline.queue.get()
        if queued_event is _QUEUE_END:
            pipeline.queue.task_done()
            return
        if not isinstance(queued_event, _SampleEvent):
            raise ValueError(f"Expected _SampleEvent, got: {type(queued_event).__name__}")

        event = queued_event
        requests_log: list[dict] = []
        requests_log_var.set(requests_log)
        try:
            metric_result = _normalize_metric_result(
                await pipeline.metric.compute_scores(
                    build_metric_input(dict(event.item), dict(event.sample), event.row_index)
                ),
                pipeline.output_spec,
            )
        except Exception as e:
            if not tolerate_failure:
                raise EvaluationError(
                    index=event.row_index,
                    message=str(e),
                    phase=EvaluationPhase.METRIC_SCORING,
                    metric_key=pipeline.metric_ref,
                ) from e
            error_message = str(e)
            logger.warning(
                "Evaluation failed, marking as NaN",
                extra={"metric_ref": pipeline.metric_ref, "item_index": event.row_index, "error": error_message},
            )
            metric_result = nan_metric_result(pipeline.output_spec)
            # Record the swallowed metric exception on the row while keeping
            # the NaN score result. Example: if the "judge" metric raises
            # "bad output", the row gets metric_errors={"judge": "bad output"}.
            metric_errors = row_scores[event.row_index].metric_errors or {}
            metric_errors[pipeline.metric_ref] = error_message
            row_scores[event.row_index].metric_errors = metric_errors
        finally:
            row_metric_requests[event.row_index][pipeline.metric_ref] = list(requests_log)
            pipeline.queue.task_done()

        pipeline.results[event.row_index] = metric_result
        row_scores[event.row_index].metrics[pipeline.metric_ref] = metric_result.outputs
        if metric_result.diagnostics:
            row_diagnostics = row_scores[event.row_index].metric_diagnostics or {}
            row_diagnostics[pipeline.metric_ref] = metric_result.diagnostics
            row_scores[event.row_index].metric_diagnostics = row_diagnostics
        if progress is not None:
            progress.increment_work()


async def _run_streaming_pipeline(
    *,
    items: list[dict],
    target: EvalTarget,
    inference_fn: InferenceFn | AgentInferenceFn | None,
    client: AsyncOpenAI | httpx.AsyncClient | None = None,
    params: BenchmarkParams,
    prompt_template: str | dict[str, Any] | None,
    row_scores: list[RowScore],
    pipelines: list[_MetricPipeline],
    row_metric_requests: list[dict[str, list[dict]]],
    worker_count: int,
    default_headers: dict[str, str] | None,
    preprocess_hooks: Sequence[PreprocessRequest],
    postprocess_hooks: Sequence[PostprocessResponse],
    progress: ProgressReporter | None,
    logger: logging.Logger,
) -> None:
    """Run producer + metric consumers under one failure-propagating task group.

    Producers broadcast directly to every per-metric queue, so the only
    shutdown-signaling path goes through :func:`_put_pipeline_sentinels`, which
    is cancellation-safe. A producer only blocks on ``pipeline.queue.put`` when
    that specific metric queue is full, which surfaces backpressure one stage
    earlier than a buffered intermediate queue would.
    """

    async def _produce_and_signal() -> None:
        try:
            await _run_producer_workers(
                items=items,
                target=target,
                inference_fn=inference_fn,
                client=client,
                params=params,
                prompt_template=prompt_template,
                row_scores=row_scores,
                pipelines=pipelines,
                worker_count=worker_count,
                default_headers=default_headers,
                preprocess_hooks=preprocess_hooks,
                postprocess_hooks=postprocess_hooks,
                logger=logger,
            )
        finally:
            await _put_pipeline_sentinels(pipelines=pipelines, worker_count=worker_count)

    try:
        async with asyncio.TaskGroup() as tg:
            for pipeline in pipelines:
                for _ in range(worker_count):
                    tg.create_task(
                        _metric_worker(
                            params=params,
                            pipeline=pipeline,
                            row_scores=row_scores,
                            row_metric_requests=row_metric_requests,
                            logger=logger,
                            progress=progress,
                        )
                    )
            tg.create_task(_produce_and_signal())
    except BaseException as exc:
        root = first_failure_cause(exc)
        logger.error(
            "Benchmark streaming pipeline failed",
            extra={
                "root_error_type": type(root).__name__,
                "root_error": str(root),
                "raw_error_type": type(exc).__name__,
            },
        )
        raise


async def evaluate_benchmark(
    *,
    metrics: Sequence[tuple[str, Metric]],
    rows: list[dict],
    target: EvalTarget = None,
    inference_fn: InferenceFn | AgentInferenceFn | None = None,
    params: BenchmarkParams,
    prompt_template: str | dict[str, Any] | None = None,
    preprocess_hooks: Sequence[PreprocessRequest] = (),
    postprocess_hooks: Sequence[PostprocessResponse] = (),
    default_headers: dict[str, str] | None = None,
    progress: ProgressReporter | None = None,
    aggregate_fields: tuple[AggregateFieldName, ...] | None = None,
    logger: logging.Logger | None = None,
) -> BenchmarkEvaluationResult:
    """Run one benchmark evaluation using the shared streaming pipeline.

    Each dataset row runs inference exactly once regardless of how many metrics
    are configured; each sample is fanned out to every metric worker through a
    bounded per-metric queue. Row-level failures (inference *or* metric scoring)
    are mapped to NaN only when ``params`` is an online params object with
    ``ignore_request_failure=True``; otherwise row failures abort the run.

    Args:
        metrics: Ordered ``(metric_ref, metric)`` tuples. ``metric_ref`` is the
            public identifier used to namespace aggregate score names.
        rows: Dataset rows to evaluate.
        target: Model or agent used for online inference, or a retrieval pipeline
            whose rankings are attached to each query row. Pass ``None`` for
            offline benchmarks; metric workers then receive the offline sample
            built from each row.
        inference_fn: Optional inference callable. Defaults to SDK's
            ``make_inference_request`` / ``make_agent_inference_request`` based
            on the ``target`` type.
        params: Task-level execution parameters. Online-only fields
            (``ignore_request_failure``, ``request_timeout``, ``max_retries``)
            are read only when ``params`` is an :class:`RunConfigOnline`.
        prompt_template: Jinja template used to render per-row requests; required
            when ``target`` is a model or agent.
        preprocess_hooks: Request preprocessors applied before each online
            inference call.
        postprocess_hooks: Response postprocessors applied after each online
            inference call.
        default_headers: Optional headers appended to every online target request
            (mirrors :class:`ComputeMetricPipeline.default_headers`).
        progress: Optional progress reporter; ``increment_work`` fires once per
            metric/row result.
        aggregate_fields: Optional aggregate score fields to keep in the returned result.
        logger: Optional logger for pipeline-level warnings and errors.

    Returns:
        A :class:`BenchmarkEvaluationResult` combining per-row scores, per-metric
        aggregates (with ``metric_ref.`` namespaced score names), and a flattened
        top-level aggregate view.

    Raises:
        ValueError: If ``metrics`` is empty.
        EvaluationError: If a row fails during strict benchmark
            execution (``fail_fast=True``).
        RuntimeError: If any metric result slot is missing after pipeline
            completion (an internal invariant check).
    """
    log = logger or _log
    if not metrics:
        raise ValueError("metrics must contain at least one (metric_ref, metric) tuple")

    queue_capacity = max(1, params.parallelism * 2)
    worker_count = min(len(rows), max(1, params.parallelism)) if rows else max(1, params.parallelism)

    row_scores = _initialize_row_scores(rows)
    row_metric_requests: list[dict[str, list[dict]]] = [dict() for _ in range(len(rows))]
    pipelines = _build_metric_pipelines(metrics, item_count=len(rows), queue_capacity=queue_capacity)

    client = None
    client_close_fn = None
    if isinstance(target, Model):
        client = new_inference_client(target)
        client_close_fn = client.close
    elif isinstance(target, AgentBase):
        client = new_agent_inference_client()
        client_close_fn = client.aclose

    async with use_resilience_session():
        try:
            await _run_streaming_pipeline(
                items=rows,
                target=target,
                inference_fn=inference_fn,
                client=client,
                params=params,
                prompt_template=prompt_template,
                row_scores=row_scores,
                pipelines=pipelines,
                row_metric_requests=row_metric_requests,
                worker_count=worker_count,
                default_headers=default_headers,
                preprocess_hooks=preprocess_hooks,
                postprocess_hooks=postprocess_hooks,
                progress=progress,
                logger=log,
            )
        except Exception as exc:
            if benchmark_error := _benchmark_error_from_exception(exc):
                # TaskGroup wraps worker failures in ExceptionGroup. Re-raise
                # the typed SDK error as the public exception while preserving
                # the original row-level failure as its cause.
                raise benchmark_error from benchmark_error.__cause__
            raise

    if client_close_fn:
        await client_close_fn()

    row_generation_requests = [list(row.requests) for row in row_scores]

    _finalize_row_request_logs(
        row_scores=row_scores,
        row_metric_requests=row_metric_requests,
        metric_refs_in_order=[metric_ref for metric_ref, _ in metrics],
    )

    per_metric: dict[str, EvaluationResult] = {}
    for pipeline in pipelines:
        if any(r is None for r in pipeline.results):
            raise RuntimeError(f"Internal error: missing metric results for '{pipeline.metric_ref}'")
        completed_rows = [
            RowScore(
                row_index=row.row_index,
                item=row.item,
                sample=row.sample,
                metrics={pipeline.metric_ref: row.metrics.get(pipeline.metric_ref, [])},
                requests=[
                    *row_generation_requests[row_idx],
                    *row_metric_requests[row_idx].get(pipeline.metric_ref, []),
                ],
                metric_errors=_metric_errors_for_ref(row.metric_errors, pipeline.metric_ref),
                metric_diagnostics=_metric_diagnostics_for_ref(row.metric_diagnostics, pipeline.metric_ref),
            )
            for row_idx, row in enumerate(row_scores)
        ]
        raw_result = await _finalize_benchmark_metric_result(
            metric=pipeline.metric,
            results=pipeline.results,
            row_scores=completed_rows,
        )
        per_metric[pipeline.metric_ref] = namespace_result(pipeline.metric_ref, raw_result, aggregate_fields)

    top_aggregate_scores = AggregatedMetricResult(
        scores=[score for result in per_metric.values() for score in result.aggregate_scores.scores]
    )
    return BenchmarkEvaluationResult(
        row_scores=row_scores,
        aggregate_scores=top_aggregate_scores,
        per_metric=per_metric,
    )
