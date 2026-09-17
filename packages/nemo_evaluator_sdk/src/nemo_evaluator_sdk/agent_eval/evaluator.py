# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone agent evaluation orchestration."""

# ruff: noqa: I001 - the vendored SDK mirror uses different import-order settings.

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from logging import getLogger
from pathlib import Path
from typing import Any, cast, overload
from urllib.parse import urlparse

import httpx
import nemo_evaluator_sdk.inference as inference
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary, RunMetadata
from nemo_evaluator_sdk.agent_eval.scores import (
    AgentEvalDiagnostic,
    AgentEvalDiagnosticSeverity,
    AgentEvalScoreStatus,
    AgentEvalTaskScore,
    TRIAL_STATUS_DETAIL,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTarget,
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    AgentTaskRunner,
    RunAggregationsProvider,
    RunnerInfo,
    TrialAwareMetricsProvider,
    TrialMeasurements,
)
from nemo_evaluator_sdk.agent_eval.usage_keys import (
    CACHE_CREATION_INPUT_TOKENS_KEY,
    CACHE_READ_INPUT_TOKENS_KEY,
    CACHED_TOKENS_KEY,
    COMPLETION_TOKEN_KEYS,
    PROMPT_TOKEN_KEYS,
    SEPARATE_CACHE_KEYS,
    USAGE_COUNT_KEYS,
    USAGE_DETAILS_KEYS,
    _first_nonnegative_int,
    _first_usage_details,
)
from nemo_evaluator_sdk.agent_inference import (
    AgentInferenceContext,
    AgentInferenceFn,
    AgentInferenceFnFactory,
    make_agent_inference_fn,
    new_agent_inference_client,
)
from nemo_evaluator_sdk.execution.metric_execution import (
    generate_online_sample,
    resolve_target_structured_output_mode,
    run_sync,
)
from nemo_evaluator_sdk.session import begin_evaluation_session
from nemo_evaluator_sdk.execution.samples import build_metric_input
from nemo_evaluator_sdk.inference import InferenceFn
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricWithPreflight, validate_metric_result
from nemo_evaluator_sdk.metrics.utils import metric_type_name
from nemo_evaluator_sdk.values import (
    Agent,
    AgentBase,
    GenericAgent,
    Model,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)
from nemo_evaluator_sdk.values.results import AggregateScore
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_FORMAT_JSON,
    EVIDENCE_TRACE,
    CandidateEvidence,
    EvidenceDescriptor,
)
from openai import AsyncOpenAI

log = getLogger(__name__)

_SAMPLE_KEYS_EXCLUDED_FROM_OUTPUT_METADATA = frozenset(
    {
        "evidence",
        "invocation_metadata",
        "invocation_status",
        "output_text",
        "response",
        "trajectory",
    }
)


class AgentEvaluator:
    """Run stored-trial or live-target agent evaluations.

    The online inference seam (an optional ``inference_fn``, transport ``client``, and
    ``default_headers``) is injected on the evaluator instance rather than the run config,
    because these are runtime transport concerns rather than declarative run settings. A
    single ``inference_fn``/``client`` pair serves both model and agent targets; leave them
    unset to let the evaluator build a default client for the resolved target type.
    """

    @overload
    def __init__(
        self,
        *,
        inference_fn: InferenceFn | AgentInferenceFn | None = None,
        agent_inference_fn_factory: None = None,
        client: AsyncOpenAI | httpx.AsyncClient | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        inference_fn: None = None,
        agent_inference_fn_factory: AgentInferenceFnFactory,
        client: AsyncOpenAI | httpx.AsyncClient | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        inference_fn: InferenceFn | AgentInferenceFn | None = None,
        agent_inference_fn_factory: AgentInferenceFnFactory | None = None,
        client: AsyncOpenAI | httpx.AsyncClient | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """Configure runtime dependencies for live target generation.

        Args:
            inference_fn: Optional model or agent inference override. When omitted, the
                evaluator selects the default implementation for the target type.
            agent_inference_fn_factory: Optional per-task factory for agent inference.
                The evaluator supplies persistence and invocation identity through an
                :class:`AgentInferenceContext`.
            client: Optional transport client matching the target type: ``AsyncOpenAI`` for
                models or ``httpx.AsyncClient`` for agents.
            default_headers: Additional HTTP headers forwarded to live inference requests.
        """
        if inference_fn is not None and agent_inference_fn_factory is not None:
            raise ValueError("provide either inference_fn or agent_inference_fn_factory, not both")
        self.inference_fn = inference_fn
        self.agent_inference_fn_factory = agent_inference_fn_factory
        self.client = client
        self.default_headers = default_headers

    async def run(
        self,
        *,
        tasks: Sequence[AgentEvalTask],
        trials: Sequence[AgentEvalTrial] | None = None,
        target: AgentEvalTarget | None = None,
        config: AgentEvalRunConfig | None = None,
    ) -> AgentEvalResult:
        """Evaluate imported trials or generate live trials before scoring.

        Exactly one of ``trials`` or ``target`` must be provided.
        """
        resolved_config = config or AgentEvalRunConfig()
        task_list = list(tasks)
        if not task_list:
            raise ValueError("at least one task is required")

        run_id = resolved_config.run_id or _new_run_id()
        runtime_config = resolved_config.model_copy(update={"run_id": run_id})
        started_at = datetime.now(UTC)

        seam_error = "provide exactly one of trials or target"
        if trials is not None and target is not None:
            raise ValueError(seam_error)

        async with begin_evaluation_session():
            # Branch on which seam was supplied so the type checker narrows each of ``trials`` and
            # ``target`` to a concrete type without a cast. The final arm is the "neither" case.
            if trials is not None:
                source_trial_list = list(trials)
            elif target is not None:
                source_trial_list = await self._generate_trials(tasks=task_list, target=target, config=runtime_config)
            else:
                raise ValueError(seam_error)
            # ``model_copy(update=...)`` skips nested validation. Revalidate so corrupted
            # measurements cannot reach scoring, and derived fields (e.g. total_tokens) are filled.
            trial_list = [AgentEvalTrial.model_validate(trial) for trial in source_trial_list]
            if isinstance(target, TrialAwareMetricsProvider):
                # Some runners only know their full metric list after trials exist
                # (Harbor extras in reward_details). Ask for this task's metrics and
                # copy the task with that list before scoring.
                trials_by_task = _trials_by_task(task_list, trial_list)
                original_task_ids = tuple(trial.task_id for trial in trial_list)
                # Revalidation creates normalized copies. Audit the runner-returned instances too:
                # a provider can retain and mutate those references instead of the sequence it receives.
                source_task_ids = tuple(trial.task_id for trial in source_trial_list)
                task_list = [
                    _task_with_updated_metrics(task, target.scoring_metrics(task, trials_by_task[task.id]))
                    for task in task_list
                ]
                changed_assignments = _changed_trial_assignments(trial_list, original_task_ids)
                changed_assignments.extend(_changed_trial_assignments(source_trial_list, source_task_ids))
                if changed_assignments:
                    raise ValueError(_scoring_metrics_assignment_error(changed_assignments))
            scores = await self._score_trials(
                tasks=task_list,
                trials=trial_list,
                config=runtime_config,
                run_id=run_id,
            )
        runner_scores = _collect_runner_aggregate_scores(target) if target is not None else []
        finished_at = datetime.now(UTC)
        metadata = RunMetadata(
            labels=dict(runtime_config.labels),
            target=_describe_target(target, runtime_config.params),
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=(finished_at - started_at).total_seconds(),
            sdk_version=_sdk_version(),
        )
        result = AgentEvalResult(
            run_id=run_id,
            tasks=task_list,
            trials=trial_list,
            scores=scores,
            summary=AgentEvalSummary.from_scores(
                scores, tasks=task_list, trials=trial_list, extra_scores=runner_scores
            ),
            metadata=metadata,
            work_dir=runtime_config.work_dir,
        )

        return result

    def run_sync(
        self,
        *,
        tasks: Sequence[AgentEvalTask],
        trials: Sequence[AgentEvalTrial] | None = None,
        target: AgentEvalTarget | None = None,
        config: AgentEvalRunConfig | None = None,
    ) -> AgentEvalResult:
        """Synchronous bridge for :meth:`run`."""
        return run_sync(lambda: self.run(tasks=tasks, trials=trials, target=target, config=config))

    async def _score_trials(
        self,
        *,
        tasks: list[AgentEvalTask],
        trials: list[AgentEvalTrial],
        config: AgentEvalRunConfig,
        run_id: str,
    ) -> list[AgentEvalTaskScore]:
        task_index_by_id = {task.id: index for index, task in enumerate(tasks)}
        # Regrouped here even when `run` already grouped for the metric-finalization hook. This also
        # validates direct callers of `_score_trials`; `run` separately rejects hook-side assignment
        # changes, including balanced swaps that regrouping alone cannot detect.
        trials_by_task = _trials_by_task(tasks, trials)

        for task in tasks:
            if not task.metrics:
                raise ValueError(f"task {task.id!r} does not declare any metrics")

        await _preflight_task_metrics(tasks, trials_by_task)

        semaphore = asyncio.Semaphore(config.parallelism)

        async def guarded_score(task: AgentEvalTask, trial: AgentEvalTrial, metric: Metric) -> AgentEvalTaskScore:
            async with semaphore:
                row_index = task_index_by_id[task.id]
                if trial.status == AgentEvalTrialStatus.FAILED:
                    return _failed_metric_score(
                        run_id=run_id,
                        task=task,
                        trial=trial,
                        metric=metric,
                        row_index=row_index,
                        diagnostic=AgentEvalDiagnostic(
                            severity=AgentEvalDiagnosticSeverity.ERROR,
                            message=f"trial {trial.id!r} is failed",
                            source=metric_type_name(metric),
                            # The key pass@k reads to tell "the agent produced nothing" (a failed
                            # attempt) from "the metric raised" (an unusable measurement).
                            details={TRIAL_STATUS_DETAIL: trial.status.value},
                        ),
                    )
                try:
                    return await _score_metric(
                        run_id=run_id,
                        task=task,
                        trial=trial,
                        metric=metric,
                        row_index=row_index,
                    )
                except Exception as exc:
                    if config.fail_fast:
                        raise
                    log.warning(
                        "metric %s failed for trial %r (task %r): %s",
                        metric_type_name(metric),
                        trial.id,
                        task.id,
                        exc,
                    )
                    return _failed_metric_score(
                        run_id=run_id,
                        task=task,
                        trial=trial,
                        metric=metric,
                        row_index=row_index,
                        diagnostic=_exception_diagnostic(exc, metric_type_name(metric)),
                    )

        return await asyncio.gather(
            *[
                guarded_score(task, trial, metric)
                for task in tasks
                for trial in trials_by_task.get(task.id, [])
                for metric in task.metrics
            ]
        )

    async def _generate_trials(
        self,
        *,
        tasks: list[AgentEvalTask],
        target: AgentEvalTarget,
        config: AgentEvalRunConfig,
    ) -> list[AgentEvalTrial]:
        if isinstance(target, AgentTaskRunner):
            return list(await target.run_tasks(tasks, config=config))
        if not isinstance(target, (Model, AgentBase)):
            raise NotImplementedError(f"unsupported agent-eval target type: {type(target).__name__}")

        params = _resolve_live_params(config, target)
        prompt_template = config.prompt_template or _default_prompt_template(target)
        semaphore = asyncio.Semaphore(params.parallelism)

        # Use the injected transport client when provided; otherwise build a default for the
        # resolved target type and close it when generation finishes.
        client = self.client
        close_client: Callable[[], Awaitable[Any]] | None = None
        if client is None and self.inference_fn is None:
            if isinstance(target, Model):
                client = inference.new_inference_client(target)
                close_client = client.close
            else:
                client = new_agent_inference_client()
                close_client = client.aclose

        try:
            # When config.params.ignore_request_failure is set, convert a failed generation request
            # into a FAILED trial (which the scorer turns into failed metric scores) instead of
            # aborting the whole run. This matches the existing online-evaluator contract.
            async def generate_one(index: int, task: AgentEvalTask) -> AgentEvalTrial:
                async with semaphore:
                    # Keep evaluator-owned runtime identity separate from task inputs.
                    # ``_generate_sample`` exposes these values to request templates under
                    # ``agent_eval``. For agent targets, the same values are supplied to the
                    # inference factory so stream translators and evidence can carry stable
                    # evaluation identifiers without coupling them to this evaluator.
                    agent_eval_context = {
                        "run_id": config.run_id,
                        "task_id": task.id,
                        "invocation_id": f"{config.run_id}:{task.id}:{target.name}",
                    }
                    evidence_dir = (
                        _task_evidence_dir(Path(config.work_dir), index=index, task_id=task.id)
                        if config.work_dir is not None and isinstance(target, AgentBase)
                        else None
                    )
                    resolved_inference_fn = self.inference_fn
                    if isinstance(target, AgentBase) and resolved_inference_fn is None:
                        factory = self.agent_inference_fn_factory or make_agent_inference_fn
                        resolved_inference_fn = factory(
                            AgentInferenceContext(
                                evidence_dir=evidence_dir,
                                metadata=agent_eval_context,
                            )
                        )
                    try:
                        sample = await _generate_sample(
                            target=target,
                            row=_task_row(task),
                            index=index,
                            prompt_template=prompt_template,
                            params=params,
                            inference_fn=resolved_inference_fn,
                            client=client,
                            default_headers=self.default_headers,
                            agent_eval_context=agent_eval_context,
                        )
                    except Exception as exc:
                        if params.ignore_request_failure:
                            return _failed_generation_trial(task, target, exc)
                        raise
                    return _trial_from_sample(task, target, sample)

            return await asyncio.gather(*(generate_one(index, task) for index, task in enumerate(tasks)))
        finally:
            if close_client is not None:
                await close_client()


def _trials_by_task(
    tasks: Sequence[AgentEvalTask],
    trials: Sequence[AgentEvalTrial],
) -> dict[str, list[AgentEvalTrial]]:
    """Group trials by ``task_id``. Two hard checks:

    Check 1 — unknown task. A trial names a ``task_id`` that is not in ``tasks``.
    Typical cause: imported ``trials.jsonl`` mixed with a different taskset, or a
    runner stamp typo. Fail the run; there is no task in this eval to attach it to.

    Check 2 — task with no trial. ``tasks`` asked for work the runner/import did
    not produce. Fail the run rather than ranking on a quieter subset: dropping
    the missing task would shrink the mean (easy tasks succeed, ``hard`` never
    ran). Missing expected tasks fail explicitly
    instead of silently shrinking coverage, and an absent Harbor ``result.json``
    must not be fabricated into a ``FAILED`` trial. That is distinct from an
    errored trial that *did* land: those stay ``PARTIAL`` and remain
    in the denominator.
    """
    task_ids = {task.id for task in tasks}
    trials_by_task: dict[str, list[AgentEvalTrial]] = defaultdict(list)
    for trial in trials:
        if trial.task_id not in task_ids:
            raise ValueError(f"trial {trial.id!r} references unknown task {trial.task_id!r}")
        trials_by_task[trial.task_id].append(trial)
    tasks_without_trials = [task.id for task in tasks if not trials_by_task.get(task.id)]
    if tasks_without_trials:
        raise ValueError(f"no trials produced for tasks: {sorted(tasks_without_trials)}")
    return dict(trials_by_task)


def _changed_trial_assignments(
    trials: Sequence[AgentEvalTrial],
    original_task_ids: Sequence[str],
) -> list[tuple[int, str, str, str]]:
    """Rows whose ``task_id`` changed from the snapshot in ``original_task_ids``.

    ``scoring_metrics`` must not reassign trials. Compare after the hook so a
    balanced swap (which regrouping would miss) still fails.
    """
    return [
        (index, trial.id, original_task_id, trial.task_id)
        for index, (trial, original_task_id) in enumerate(zip(trials, original_task_ids, strict=True))
        if trial.task_id != original_task_id
    ]


def _scoring_metrics_assignment_error(changed: Sequence[tuple[int, str, str, str]]) -> str:
    """Explain a ``task_id`` mutation and what ``scoring_metrics`` should do instead."""
    seen: set[tuple[str, str, str]] = set()
    parts: list[str] = []
    for _, trial_id, original_task_id, task_id in changed:
        key = (trial_id, original_task_id, task_id)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"trial {trial_id!r} task_id {original_task_id!r} -> {task_id!r}")
    return (
        "scoring_metrics must not mutate trials "
        f"({'; '.join(parts)}). Return a new metric list; leave each trial.task_id "
        "as the runner assigned it."
    )


def _task_with_updated_metrics(task: AgentEvalTask, metrics: Sequence[Metric]) -> AgentEvalTask:
    """Copy ``task`` with ``metrics`` replaced, through the constructor rather than ``model_copy``.

    ``model_copy(update=...)`` skips validation; the constructor re-runs the check that rejects
    duplicate metric types and views referencing outputs the new metrics do not declare. Every
    other field is carried over by value, so a field added to :class:`AgentEvalTask` later cannot
    quietly revert to its default here -- ``extra="forbid"`` catches an unknown key, not an
    omitted one.
    """
    fields: dict[str, Any] = dict(task)
    fields["metrics"] = list(metrics)
    return AgentEvalTask(**fields)


async def _preflight_task_metrics(
    tasks: Sequence[AgentEvalTask],
    trials_by_task: Mapping[str, Sequence[AgentEvalTrial]],
) -> None:
    """Run each distinct metric's preflight once, skipping metrics with nothing to score.

    Agent-eval scores metrics directly rather than through ``prepare_metric_for_execution``, so
    nothing else runs their preflight. Failed trials short-circuit to a failed score without
    invoking the metric, so a task whose trials all failed must not trigger one: preflight resolves
    the judge endpoint, making it both a wasted request and a way for the run to abort.
    """
    preflighted: set[int] = set()
    for task in tasks:
        scoreable = any(trial.status != AgentEvalTrialStatus.FAILED for trial in trials_by_task.get(task.id, ()))
        if not scoreable:
            continue
        for metric in task.metrics:
            if isinstance(metric, MetricWithPreflight) and id(metric) not in preflighted:
                preflighted.add(id(metric))
                await metric.preflight()


async def _generate_sample(
    *,
    target: Model | Agent,
    row: dict[str, Any],
    index: int,
    prompt_template: str | dict[str, Any],
    params: RunConfigOnline | RunConfigOnlineModel,
    inference_fn: InferenceFn | AgentInferenceFn | None,
    client: AsyncOpenAI | httpx.AsyncClient | None,
    default_headers: dict[str, str] | None,
    agent_eval_context: dict[str, Any],
) -> dict[str, Any]:
    # InferenceFn and AgentInferenceFn are callable protocols, so isinstance cannot discriminate
    # the injected fn; narrow it per target type with a cast (matching execution/benchmark_execution).
    # The transport client is a real class union, so isinstance narrowing is enough there.
    if isinstance(target, Model):
        model_params = cast(RunConfigOnlineModel, params)
        preprocess_hooks, postprocess_hooks = inference.new_hooks(model_params)
        model_inference_fn = (
            cast(InferenceFn, inference_fn) if inference_fn is not None else inference.make_inference_request
        )
        await resolve_target_structured_output_mode(
            preprocess_hooks=preprocess_hooks,
            model=target,
            inference_fn=model_inference_fn,
            params=model_params,
        )
        return await generate_online_sample(
            target=target,
            row=row,
            index=index,
            prompt_template=prompt_template,
            params=model_params,
            inference_fn=model_inference_fn,
            client=client if isinstance(client, AsyncOpenAI) else None,
            preprocess_hooks=preprocess_hooks,
            postprocess_hooks=postprocess_hooks,
            default_headers=default_headers,
            template_context={"agent_eval": agent_eval_context},
        )

    if inference_fn is None:
        raise TypeError("expected AgentInferenceFn for Agent target")
    agent_inference_fn = cast(AgentInferenceFn, inference_fn)
    return await generate_online_sample(
        target=target,
        row=row,
        index=index,
        prompt_template=prompt_template,
        params=params,
        inference_fn=agent_inference_fn,
        client=client if isinstance(client, httpx.AsyncClient) else None,
        default_headers=default_headers,
        template_context={"agent_eval": agent_eval_context},
    )


def _trial_from_sample(task: AgentEvalTask, target: Model | Agent, sample: dict[str, Any]) -> AgentEvalTrial:
    output_text = sample.get("output_text")
    if not (isinstance(output_text, str) and output_text.strip()):
        # Reasoning models that exhaust the token budget can return only
        # `reasoning_content` with empty `content`. Fall back to that text so the
        # trial stays scorable instead of being dropped as empty output.
        output_text = _reasoning_content_fallback(sample.get("response"))
    evidence = sample.get("evidence")
    if evidence is not None and not isinstance(evidence, CandidateEvidence):
        evidence = CandidateEvidence.model_validate(evidence)

    # Evidence precedence:
    # - trajectory exists: merge it without replacing a typed trace.
    # - no trajectory, but typed evidence exists: preserve that evidence unchanged.
    # - neither exists: synthesize the fallback trace.
    if "trajectory" in sample:
        trace = EvidenceDescriptor(kind=EVIDENCE_TRACE, format=EVIDENCE_FORMAT_JSON, data=sample["trajectory"])
        descriptors = dict(evidence.descriptors) if evidence is not None else {}
        descriptors.setdefault(EVIDENCE_TRACE, trace)
        evidence = CandidateEvidence(
            descriptors=descriptors,
            metadata=dict(evidence.metadata) if evidence is not None else {},
        )
    elif evidence is None:
        evidence = CandidateEvidence(
            descriptors={
                EVIDENCE_TRACE: EvidenceDescriptor(
                    kind=EVIDENCE_TRACE,
                    format=EVIDENCE_FORMAT_JSON,
                    data={"task_id": task.id, "target": target.name},
                )
            }
        )

    status_value = sample.get("invocation_status", AgentEvalTrialStatus.COMPLETED.value)
    status = AgentEvalTrialStatus(status_value)
    invocation_metadata = sample.get("invocation_metadata")
    if not isinstance(invocation_metadata, dict):
        invocation_metadata = {}
    trial_id = f"{task.id}:{target.name}"

    return AgentEvalTrial(
        id=trial_id,
        task_id=task.id,
        status=status,
        output=AgentOutput(
            output_text=output_text if isinstance(output_text, str) else None,
            response=sample.get("response"),
            metadata={
                **invocation_metadata,
                **{
                    key: value for key, value in sample.items() if key not in _SAMPLE_KEYS_EXCLUDED_FROM_OUTPUT_METADATA
                },
            },
        ),
        evidence=evidence,
        measurements=_live_response_usage_measurements(sample.get("response"), trial_id=trial_id),
        metadata={
            **invocation_metadata,
            "model_id": target.name,
            "target_name": target.name,
            "generated": True,
        },
    )


def _live_response_usage_measurements(response: Any, *, trial_id: str) -> TrialMeasurements:
    """Normalize a live model or agent response's usage into typed measurements."""
    if not isinstance(response, Mapping):
        return TrialMeasurements()
    usage = response.get("usage")
    if usage is None:
        return TrialMeasurements()
    if not isinstance(usage, Mapping):
        log.warning("Trial %r has invalid live response usage=%r; expected an object and omitted it", trial_id, usage)
        return TrialMeasurements()

    _warn_invalid_live_usage_counts(trial_id, usage)
    separate_cache = any(usage.get(key) is not None for key in SEPARATE_CACHE_KEYS)
    details = _first_usage_details(usage)
    inclusive_cache = details is not None and details.get(CACHED_TOKENS_KEY) is not None
    completion = _first_nonnegative_int(usage, *COMPLETION_TOKEN_KEYS)
    if separate_cache and inclusive_cache:
        log.warning(
            "Trial %r live response usage contains both inclusive cache details and separate cache fields; "
            "omitting ambiguous prompt/cache measurements",
            trial_id,
        )
        return TrialMeasurements(completion_tokens=completion)

    prompt = _first_nonnegative_int(usage, *PROMPT_TOKEN_KEYS)
    cache_read = _first_nonnegative_int(usage, CACHE_READ_INPUT_TOKENS_KEY) if separate_cache else None
    cache_creation = _first_nonnegative_int(usage, CACHE_CREATION_INPUT_TOKENS_KEY) if separate_cache else None
    invalid_separate_cache = any(
        key in usage and usage[key] is not None and _first_nonnegative_int(usage, key) is None
        for key in SEPARATE_CACHE_KEYS
    )
    if separate_cache and prompt is not None:
        prompt = None if invalid_separate_cache else prompt + (cache_creation or 0) + (cache_read or 0)
    elif details is not None and CACHED_TOKENS_KEY in details:
        cache_read = _first_nonnegative_int(details, CACHED_TOKENS_KEY)

    return TrialMeasurements(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cache_creation_tokens=cache_creation,
        cache_read_tokens=cache_read,
    )


def _warn_invalid_live_usage_counts(trial_id: str, usage: Mapping[str, Any]) -> None:
    """Log malformed live usage fields that the adapter omits."""
    for key in USAGE_COUNT_KEYS:
        if key in usage and usage[key] is not None and _first_nonnegative_int(usage, key) is None:
            log.warning(
                "Trial %r has invalid live response usage %s=%r; expected a non-negative integer and omitted it",
                trial_id,
                key,
                usage[key],
            )
    for details_key in USAGE_DETAILS_KEYS:
        details = usage.get(details_key)
        if details is not None and not isinstance(details, Mapping):
            log.warning(
                "Trial %r has invalid live response usage %s=%r; expected an object and omitted it",
                trial_id,
                details_key,
                details,
            )
        elif (
            isinstance(details, Mapping)
            and CACHED_TOKENS_KEY in details
            and details[CACHED_TOKENS_KEY] is not None
            and _first_nonnegative_int(details, CACHED_TOKENS_KEY) is None
        ):
            log.warning(
                "Trial %r has invalid live response usage %s.cached_tokens=%r; "
                "expected a non-negative integer and omitted it",
                trial_id,
                details_key,
                details[CACHED_TOKENS_KEY],
            )


def _reasoning_content_fallback(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
    return None


def _failed_generation_trial(task: AgentEvalTask, target: Model | Agent, exc: Exception) -> AgentEvalTrial:
    return AgentEvalTrial(
        id=f"{task.id}:{target.name}",
        task_id=task.id,
        status=AgentEvalTrialStatus.FAILED,
        output=None,
        evidence=CandidateEvidence(
            descriptors={
                "error": EvidenceDescriptor(
                    kind="error",
                    data={"error_type": exc.__class__.__name__, "error": str(exc)},
                )
            }
        ),
        metadata={
            "model_id": target.name,
            "target_name": target.name,
            "generated": True,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        },
    )


async def _score_metric(
    *,
    run_id: str,
    task: AgentEvalTask,
    trial: AgentEvalTrial,
    metric: Metric,
    row_index: int,
) -> AgentEvalTaskScore:
    output_spec = metric.output_spec()
    metric_result = validate_metric_result(
        await metric.compute_scores(build_metric_input(_metric_row(task, trial), _trial_sample(trial), row_index)),
        output_spec,
    )
    metric_type = metric_type_name(metric)
    return AgentEvalTaskScore(
        id=_score_id(run_id, task.id, trial.id, metric_type),
        run_id=run_id,
        task_id=task.id,
        trial_id=trial.id,
        metric_type=metric_type,
        status=AgentEvalScoreStatus.COMPLETED,
        outputs=metric_result.outputs,
        # Persist the metric's own diagnostics (e.g. per-criterion judge verdicts) — the failure path
        # already records diagnostics; the success path dropped them.
        diagnostics=[
            AgentEvalDiagnostic(
                severity=AgentEvalDiagnosticSeverity.INFO,
                message=diagnostic.message,
                source=metric_type,
                details=diagnostic.details or {},
            )
            for diagnostic in metric_result.diagnostics
        ],
        metadata={
            "row_index": row_index,
            "trial_metadata": trial.metadata,
        },
    )


def _failed_metric_score(
    *,
    run_id: str,
    task: AgentEvalTask,
    trial: AgentEvalTrial,
    metric: Metric,
    row_index: int,
    diagnostic: AgentEvalDiagnostic,
) -> AgentEvalTaskScore:
    metric_type = metric_type_name(metric)
    return AgentEvalTaskScore(
        id=_score_id(run_id, task.id, trial.id, metric_type),
        run_id=run_id,
        task_id=task.id,
        trial_id=trial.id,
        metric_type=metric_type,
        status=AgentEvalScoreStatus.FAILED,
        outputs=[],
        diagnostics=[diagnostic],
        metadata={
            "row_index": row_index,
            "trial_metadata": trial.metadata,
        },
    )


def _exception_diagnostic(exc: Exception, metric_type: str) -> AgentEvalDiagnostic:
    return AgentEvalDiagnostic(
        severity=AgentEvalDiagnosticSeverity.ERROR,
        message=str(exc) or exc.__class__.__name__,
        source=metric_type,
        details={"exception_type": exc.__class__.__name__},
    )


def _score_id(run_id: str, task_id: str, trial_id: str, metric_type: str) -> str:
    return f"{run_id}:{task_id}:{trial_id}:{metric_type}"


def _trial_sample(trial: AgentEvalTrial) -> dict[str, Any]:
    """Flatten one trial into the sample dict that becomes ``MetricInput.candidate``.

    A metric never calls this. Scoring path, once per ``(task, trial, metric)``::

        _score_metric
          -> build_metric_input(_metric_row(task, trial), _trial_sample(trial), ...)
          -> metric.compute_scores(MetricInput)
    """
    if trial.output is None:
        return {}
    sample: dict[str, Any] = {
        **trial.metadata,
        **trial.output.metadata,
    }
    if trial.output.output_text is not None:
        sample["output_text"] = trial.output.output_text
    if trial.output.response is not None:
        sample["response"] = trial.output.response
    if trial.evidence is not None:
        sample["evidence"] = trial.evidence
    return sample


def _resolve_live_params(
    config: AgentEvalRunConfig,
    target: Model | Agent,
) -> RunConfigOnline | RunConfigOnlineModel:
    params = config.params
    if isinstance(target, Model):
        if params is None:
            return RunConfigOnlineModel(parallelism=config.parallelism)
        if isinstance(params, RunConfigOnlineModel):
            return params
        if isinstance(params, RunConfigOnline):
            return RunConfigOnlineModel(**params.model_dump(mode="python"))
        if isinstance(params, RunConfig):
            return RunConfigOnlineModel(**params.model_dump(mode="python"))

    if params is None:
        return RunConfigOnline(parallelism=config.parallelism)
    if isinstance(params, RunConfigOnlineModel):
        return RunConfigOnline(
            **params.model_dump(
                mode="python",
                exclude={"inference", "system_prompt", "reasoning", "structured_output"},
            )
        )
    if isinstance(params, RunConfigOnline):
        return params
    return RunConfigOnline(**params.model_dump(mode="python"))


def _default_prompt_template(target: Model | Agent) -> dict[str, Any] | str:
    # Every default renders against the single canonical task input, ``instruction`` (see
    # ``AgentEvalTask.agent_prompt``); no other input key is special.
    if isinstance(target, GenericAgent):
        # A generic HTTP agent defines its own request entirely through its `body` template, which
        # renders against the task inputs (e.g. `{{ instruction }}`). Pass the task row through
        # unchanged so `body` — not a chat/completions assumption — shapes the payload. See
        # `_resolve_http_agent_invocation`, which renders `body` against this request.
        return "{{ item }}"
    if isinstance(target, Model) and _is_completions_endpoint(target.url):
        return {"prompt": "{{item.instruction}}"}
    return {"messages": [{"role": "user", "content": "{{item.instruction}}"}]}


def _task_row(task: AgentEvalTask) -> dict[str, Any]:
    # The task inputs verbatim, plus the task id. `instruction` is the single canonical input the
    # target is prompted with (see `AgentEvalTask.agent_prompt` and `_default_prompt_template`); no
    # input key is synthesized or aliased here.
    return {**task.inputs, "task_id": task.id}


def _metric_row(task: AgentEvalTask, trial: AgentEvalTrial) -> dict[str, Any]:
    return {
        "task": {
            "id": task.id,
            "intent": task.intent,
            "metadata": task.metadata,
        },
        "inputs": task.inputs,
        # Grader-only ground truth: available to metrics here but never seeded into the agent's
        # workspace (see AgentEvalTask.reference), so a metric can grade against held-out artifacts.
        "reference": task.reference,
        "trial": {
            "id": trial.id,
            "task_id": trial.task_id,
            "status": trial.status.value,
            # How the trial failed, for a metric that grades on it. None when the producer reported no failure.
            "error": trial.error.model_dump(mode="json") if trial.error is not None else None,
            "measurements": trial.measurements.model_dump(mode="json"),
            "metadata": dict(trial.metadata),
        },
    }


def _is_completions_endpoint(url: str) -> bool:
    path = urlparse(url).path.rstrip("/")
    return path.endswith("/completions") and not path.endswith("/chat/completions")


def _sdk_version() -> str | None:
    try:
        return package_version("nemo-evaluator-sdk")
    except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
        return None


def _describe_target(
    target: AgentEvalTarget | None,
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel | None = None,
) -> RunnerInfo:
    """Identify what produced the trials, for the run's provenance.

    Runners identify themselves via the required :meth:`AgentTaskRunner.runner_info`; trials supplied
    directly have no runner.

    Models and agents are described by name *and* the settings they were invoked with — the endpoint
    ``url``, plus the whole ``params`` object (temperature, max_tokens, reasoning effort, system prompt,
    retries, ...). A name alone is not an identity: the same model name served from two different URLs,
    or at two different temperatures, would otherwise record identical provenance. ``params`` is dumped
    whole rather than cherry-picked, because a filtered subset is what bites you later when the omitted
    field turns out to be the one that mattered. It carries no credentials — ``Model.api_key_secret`` is
    a reference on the model, and ``default_headers`` is excluded from serialization.
    """
    if target is None:
        return RunnerInfo(name="imported", kind="imported")
    if isinstance(target, (Model, AgentBase)):
        config: dict[str, Any] = {"url": getattr(target, "url", None)}
        if params is not None:
            config["params"] = params.model_dump(mode="json", exclude_none=True)
        return RunnerInfo(name=target.name, kind="model" if isinstance(target, Model) else "agent", config=config)
    return target.runner_info()


def _collect_runner_aggregate_scores(target: object) -> list[AggregateScore]:
    """The typed subset of a runner's own aggregations, for merging into ``summary.scores``.

    A runner that maps its numbers onto aggregate scores namespaces them under ``runner.<name>.``, so
    they sit alongside the SDK's own without being mistaken for them. That namespace is *enforced*, not
    merely documented: ``RunAggregationsProvider`` is a public extension point, ``summary.scores`` is a
    flat list, and a third-party runner returning ``gym_reward.reward`` would not overwrite the SDK's
    own aggregate but sit next to it under the same name, leaving any lookup to pick one arbitrarily.

    Offending entries are dropped with a warning rather than raised on. This runs *after* ``run_tasks``,
    so raising would sink a completed run — potentially hours of collection — over a naming bug, while
    the numbers themselves remain in the runner's own files inside the bundle.
    """
    if not isinstance(target, RunAggregationsProvider):
        return []
    runner_info = getattr(target, "runner_info", None)  # structurally optional: the protocol is a companion
    runner_name = runner_info().name if callable(runner_info) else None
    prefix = f"runner.{runner_name}." if runner_name else "runner."
    collected: list[AggregateScore] = []
    for score in target.run_aggregate_scores():
        if not score.name.startswith(prefix):
            log.warning(
                "Dropping runner-contributed aggregate %r: RunAggregationsProvider names must be "
                "namespaced %r so an imported figure is never mistaken for one the SDK computed.",
                score.name,
                prefix,
            )
            continue
        collected.append(score)
    return collected


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"agent-eval-{timestamp}-{uuid.uuid4().hex[:8]}"


def _task_evidence_dir(output_dir: Path, *, index: int, task_id: str) -> Path:
    safe_task_id = _safe_path_component(task_id)
    task_dir = f"{index:06d}-{safe_task_id}" if safe_task_id else f"task-{index:06d}"
    return output_dir / "evidence" / task_dir


def _safe_path_component(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in "-_." else "-" for char in value)
    return sanitized.strip("-_.")[:120]
