# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a completed agent evaluation to Intake.

``publish_to_intake`` is the explicit, post-run consumer of ``AgentEvalResult``
(see AALGO-290). It is **not** a side effect of ``AgentEvaluator.run()`` and
there is no feature flag — optionality is structural: you make the call or you
don't, and the platform client is a required argument.

It references an **existing** Experiment (created by the caller via the platform
Experiments SDK) and never creates one. Per Trial it: POSTs the ATIF trajectory,
resolves the trajectory's root span, then POSTs one evaluator-result per metric
output. All request shapes come from :mod:`nemo_evaluator.intake.mapping`; the
HTTP calls go through the typed Intake client.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta
from typing import cast

from nemo_evaluator.intake import mapping
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, TrialMeasurements
from nemo_platform_plugin.client.errors import NotFoundError, UnprocessableEntityError
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import (
    AtifCreateParams,
    AtifCreateRequest,
    AtifFinalMetricsParam,
    AtifStepParam,
    EvaluatorResultCreateRequest,
    TraceFilterParam,
)
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

#: Default ceiling on concurrent per-trial publishes.
DEFAULT_MAX_CONCURRENCY = 8


class PublishError(RuntimeError):
    """Raised when one or more trials fail to publish (or a span never resolves).

    Carries the partial :class:`PublishReport` of trials that *did* publish, so the
    caller can see what landed before re-running.
    """

    def __init__(self, message: str, *, report: PublishReport | None = None) -> None:
        super().__init__(message)
        self.report = report


class PublishedTrial(BaseModel):
    """Record of one Trial written to Intake."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str = Field(description="Identifier of the published trial.")
    session_id: str = Field(description="Intake session id minted for the trajectory.")
    span_id: str = Field(description="Resolved root AGENT span id the scores were attached to.")
    evaluator_result_count: int = Field(description="Number of evaluator-result rows written for this trial.")


class SkippedScore(BaseModel):
    """A score output omitted from publish because Intake can't represent it yet (cross-team ask X6)."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str = Field(description="Trial whose score output was omitted.")
    name: str = Field(description='"{metric_type}.{output}" of the omitted output.')
    reason: str = Field(description="Why it was omitted (e.g. 'scoring failed', 'non-finite value').")


class PublishReport(BaseModel):
    """Summary of a ``publish_to_intake`` run."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(description="Experiment the results were published under.")
    workspace: str = Field(description="Workspace the writes targeted.")
    run_id: str = Field(description="Source AgentEvalResult run id.")
    published_trials: list[PublishedTrial] = Field(
        default_factory=list, description="Per-trial records of what was written."
    )
    skipped: list[SkippedScore] = Field(
        default_factory=list,
        description="Score outputs omitted because Intake can't represent failed/non-finite scores (cross-team ask X6).",
    )

    @property
    def trial_count(self) -> int:
        """Number of trials published."""
        return len(self.published_trials)

    @property
    def evaluator_result_count(self) -> int:
        """Total evaluator-result rows written across all trials."""
        return sum(trial.evaluator_result_count for trial in self.published_trials)


def _token_final_metrics(measurements: TrialMeasurements) -> AtifFinalMetricsParam | None:
    """Project the trial's recorded token usage onto ATIF ``final_metrics``.

    Intake promotes these root totals onto the trajectory's root span, where the evaluation rollup
    reads them — so publishing them is what makes token counts show up under the Evaluation. Only the
    fields the trial actually recorded are set; a trial that recorded neither token counts nor a
    cost yields ``None`` (no ``final_metrics`` block). A recorded cost alone is enough to produce
    one.
    """
    final_metrics: AtifFinalMetricsParam = {}
    if measurements.prompt_tokens is not None:
        final_metrics["total_prompt_tokens"] = measurements.prompt_tokens
    if measurements.completion_tokens is not None:
        final_metrics["total_completion_tokens"] = measurements.completion_tokens
    if measurements.cache_read_tokens is not None:
        final_metrics["total_cached_tokens"] = measurements.cache_read_tokens
    if measurements.cost_usd is not None:
        final_metrics["total_cost_usd"] = measurements.cost_usd
    return final_metrics or None


def _atif_request_body(body: AtifCreateParams) -> AtifCreateParams:
    """Return the ATIF JSON body, excluding generated-SDK-only path parameters."""
    return cast(AtifCreateParams, {key: value for key, value in body.items() if key != "workspace"})


def _evaluator_result_request_body(row: mapping.EvaluatorResultCreateParams) -> EvaluatorResultCreateRequest:
    """Return the evaluator-result JSON body, excluding path parameters."""
    return EvaluatorResultCreateRequest.model_validate({key: value for key, value in row.items() if key != "workspace"})


async def publish_to_intake(
    result: AgentEvalResult,
    *,
    client: AsyncIntakeClient,
    experiment_id: str,
    workspace: str | None = None,
    agent_name: str = "agent",
    agent_version: str = mapping.DEFAULT_AGENT_VERSION,
    model_name: str | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> PublishReport:
    """Publish a completed ``AgentEvalResult`` to Intake under an existing Experiment.

    For each trial: POST its OTLP trace when it has one, else fall back to the ATIF
    trajectory, then POST one evaluator-result per metric output against the trial's
    root span. Trials are published concurrently up to ``max_concurrency``.

    Publishing is **not atomic** and Intake has no rollback, so a per-trial failure
    must not abort the others: every trial that can land does, and the failures are
    collected and raised together as a :class:`PublishError` (carrying the partial
    report). The evaluation's local bundle is the system of record and is never
    touched, so the caller can re-run ``publish_to_intake`` once the issue is fixed
    to publish the remaining trials.

    Re-publish is **idempotent**: the session id is derived from the run and trial
    (``mapping.session_id_for``), the step timestamp from the run's ``started_at``,
    and each evaluator-result id from its target — so every key Intake replaces on is
    a function of the result, not of when it was published. Re-sending a trial that
    already landed replaces its rows instead of duplicating them, which is what makes
    a worker retry after a partial publish safe.

    ``experiment_id`` must reference an evaluation that already exists and is not
    deleted; it is checked once up front. ATIF ingest enforces this server-side, but
    OTLP ingest does not, so nothing else would stop the primary path from writing
    spans against an evaluation that cannot hold them. Creating the evaluation is a
    separate, caller-side step via the platform Experiments SDK.

    Agent identity (``agent_name``/``agent_version``/``model_name``) is taken as
    arguments because it lives on the run *target*, which ``AgentEvalResult`` does
    not carry (design §3.9 #6).
    """
    validated_trials = [AgentEvalTrial.model_validate(trial) for trial in result.trials]
    intake = client
    resolved_workspace = intake.require_workspace(workspace)

    # Required, not defaulted to "now": a fallback would silently reintroduce the
    # publish-time clock that makes re-ingest duplicate rows (see the ingest note on
    # ``mapping.trial_to_atif_ingest``). A real run always sets it; a hand-built result
    # must say when it ran.
    started_at = result.metadata.started_at
    if started_at is None:
        raise PublishError(
            f"Cannot publish run {result.run_id!r}: metadata.started_at is unset, and publishing "
            "without it would write trajectories that duplicate on re-publish."
        )

    try:
        await intake.get_evaluation(name=experiment_id, workspace=resolved_workspace)
    except NotFoundError as error:
        raise PublishError(
            f"Cannot publish run {result.run_id!r}: evaluation {experiment_id!r} does not exist in "
            f"workspace {resolved_workspace!r}, or has been deleted."
        ) from error

    scores_by_trial: dict[str, list[AgentEvalTaskScore]] = defaultdict(list)
    for score in result.scores:
        scores_by_trial[score.trial_id].append(score)

    semaphore = asyncio.Semaphore(max_concurrency)
    skipped: list[SkippedScore] = []

    async def _publish_otlp(trial: AgentEvalTrial, *, measurements: TrialMeasurements, session_id: str) -> str | None:
        """Publish the trial's OTLP trace and return its root span id, or ``None`` when it has none."""
        otlp = await mapping.otlp_ingest_for_trial(
            trial,
            session_id=session_id,
            evaluation_name=experiment_id,
            measurements=measurements,
            started_at=started_at,
        )
        if otlp is None:
            return None
        payload, span_id = otlp
        response = await intake.create_otlp_traces(content=payload, workspace=resolved_workspace)
        ingest_response = response.data()
        # OTLP ingest answers 200 with a per-span error list, so a partly-stored trace is
        # invisible unless the body is read — including the case where the span the scores
        # are about to reference is the one that was dropped.
        if ingest_response.errors:
            raise PublishError(
                f"Intake rejected {len(ingest_response.errors)} span(s) of trial {trial.id!r}: {ingest_response.errors}"
            )
        return span_id

    async def _publish_atif(trial: AgentEvalTrial, *, measurements: TrialMeasurements, session_id: str) -> str:
        """Publish the trial's ATIF trajectory and return the root span id Intake minted for it."""
        # A recorded runtime gives the trajectory a real end so Intake's latency rollup sees the
        # trial's wall-clock duration instead of 0; absent → no window, latency stays unmeasured.
        ended_at = (
            started_at + timedelta(seconds=measurements.runtime_sec) if measurements.runtime_sec is not None else None
        )

        def build_body(steps: Sequence[AtifStepParam] | None) -> AtifCreateParams:
            body = mapping.trial_to_atif_ingest(
                trial,
                run_id=result.run_id,
                evaluation_name=experiment_id,
                agent_name=agent_name,
                started_at=started_at,
                agent_version=agent_version,
                model_name=model_name,
                final_metrics=_token_final_metrics(measurements),
                ended_at=ended_at,
                steps=steps,
            )
            body["workspace"] = resolved_workspace
            return body

        recorded_steps = await mapping.atif_steps_from_trial(trial, started_at=started_at)
        try:
            await intake.create_atif(
                workspace=resolved_workspace,
                body=AtifCreateRequest(root=_atif_request_body(build_body(recorded_steps))),
            )
        except UnprocessableEntityError as error:
            # The SDK's ATIF read model accepts shapes ingest does not, so a trajectory that
            # parsed locally can still be refused. Losing the trial's scores over a trace
            # detail is the wrong trade — fall back to the single-step trajectory a runner
            # with no trace would have sent. Guard on emptiness rather than ``is None``: with no
            # steps to drop, the retry would resend the body ingest just refused.
            if not recorded_steps:
                raise
            logger.warning(
                "Intake rejected the recorded trajectory for trial %s (%s); publishing its final output instead.",
                trial.id,
                error,
            )
            await intake.create_atif(
                workspace=resolved_workspace,
                body=AtifCreateRequest(root=_atif_request_body(build_body(None))),
            )

        # ATIF span ids are minted by Intake from its own identity scheme, so unlike the
        # OTLP path this one has to read the id back rather than know it locally.
        return await _resolve_root_span_id(intake, workspace=resolved_workspace, session_id=session_id)

    async def _publish_trial(trial: AgentEvalTrial) -> PublishedTrial:
        async with semaphore:
            measurements = trial.measurements
            session_id = mapping.session_id_for(result.run_id, trial.id)
            span_id = await _publish_otlp(trial, measurements=measurements, session_id=session_id)
            if span_id is None:
                span_id = await _publish_atif(trial, measurements=measurements, session_id=session_id)
            return await _publish_scores(trial, session_id=session_id, span_id=span_id)

    async def _publish_scores(trial: AgentEvalTrial, *, session_id: str, span_id: str) -> PublishedTrial:
        written = 0
        for score in scores_by_trial.get(trial.id, []):
            rows, omitted = mapping.score_to_evaluator_results(score, session_id=session_id, span_id=span_id)
            for row in rows:
                row["workspace"] = resolved_workspace
                await intake.create_evaluator_result(
                    workspace=resolved_workspace,
                    body=_evaluator_result_request_body(row),
                )
                written += 1
            skipped.extend(SkippedScore(trial_id=trial.id, name=item.name, reason=item.reason) for item in omitted)

        return PublishedTrial(
            trial_id=trial.id,
            session_id=session_id,
            span_id=span_id,
            evaluator_result_count=written,
        )

    outcomes = await asyncio.gather(*(_publish_trial(trial) for trial in validated_trials), return_exceptions=True)

    published: list[PublishedTrial] = []
    failures: list[tuple[str, BaseException]] = []
    for trial, outcome in zip(validated_trials, outcomes, strict=True):
        if isinstance(outcome, PublishedTrial):
            published.append(outcome)
        else:
            failures.append((trial.id, outcome))

    report = PublishReport(
        experiment_id=experiment_id,
        workspace=resolved_workspace,
        run_id=result.run_id,
        published_trials=published,
        skipped=skipped,
    )
    if failures:
        raise PublishError(_publish_failure_message(result, report, failures), report=report)
    return report


def _publish_failure_message(
    result: AgentEvalResult,
    report: PublishReport,
    failures: list[tuple[str, BaseException]],
) -> str:
    """Build an actionable error: what failed, what survives locally, how to recover."""
    # work_dir is where the runtimes wrote trial evidence, NOT a bundle location: a bundle exists only
    # if the caller chose to persist(), and persist() can be pointed elsewhere. Say only what is true.
    location = (
        f"still in memory, and the run's trial evidence is under {result.work_dir}"
        if result.work_dir is not None
        else "still in memory and have not been written to disk (call result.persist() to keep them)"
    )
    detail = "\n  ".join(f"{trial_id}: {type(error).__name__}: {error}" for trial_id, error in failures)
    return (
        f"publish_to_intake: {len(failures)} of {len(result.trials)} trial(s) failed to publish "
        f"({report.trial_count} succeeded). The evaluation results are {location}; re-run "
        f"publish_to_intake(result, ...) once the issue is resolved to publish the rest.\n"
        f"Failed trials:\n  {detail}"
    )


async def _resolve_root_span_id(intake: AsyncIntakeClient, *, workspace: str, session_id: str) -> str:
    """Return the root AGENT span id for a freshly-ingested trajectory (design §3.5, option 1)."""
    trace_filter: TraceFilterParam = {"session_id": session_id}
    traces = await intake.list_traces(workspace=workspace, query_params={"filter": trace_filter})
    async for trace in traces.items():
        if trace.root_span_id:
            return trace.root_span_id
    raise PublishError(f"No root span resolved for session {session_id!r} after ATIF ingest")
