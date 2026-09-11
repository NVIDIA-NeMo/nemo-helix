# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a finished evaluation run to Intake, when the spec asked for it.

Covers both shapes: ``publish_agent_eval_result`` for an agent-eval run, and
``publish_row_eval_result`` for a dataset-driven one, which is adapted to the same
``AgentEvalResult`` shape first (see ``intake.row_adapter``).

``publish_to_intake`` is deliberately not a side effect of ``AgentEvaluator.run()`` (AALGO-290):
optionality is structural, you call it or you don't. ``spec.publication.intake`` keeps that shape —
absent means no publish, and nothing here runs — while giving the job API a way to request it, which
is what Studio needs to get evaluation runs into Experiments.

``run`` is synchronous and the publisher is async, so the call goes through the same
``run_with_isolated_async_client`` bridge ``result_persistence`` uses for the entity write.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from nemo_evaluator.intake.publish import PublishError, PublishReport, publish_to_intake
from nemo_evaluator.intake.row_adapter import RowIdentityError, row_result_to_agent_eval_result
from nemo_evaluator.jobs.agent_spec import Target, target_agent_identity
from nemo_evaluator.jobs.publication_spec import IntakePublicationSpec, RowIntakePublicationSpec
from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.values import Model
from nemo_evaluator_sdk.values.agents import AgentBase
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.results import EvaluationResult
from nemo_platform_plugin.client.errors import NemoClientError, NotFoundError
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import EvaluationPatchRequest, EvaluationResponse
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

#: Evaluation metadata keys carrying how long the run took and how long publishing it took, both as
#: string-encoded seconds (the API takes ``dict[str, str]``). Two keys rather than one ``duration``
#: because a reader cannot tell which of the two a single key means.
EVAL_DURATION_KEY = "eval_duration_sec"
PUBLISH_DURATION_KEY = "publish_duration_sec"


class PublicationOutcome(BaseModel):
    """What publication did, as reported in the job output.

    Deliberately not ``PublishReport``: that is the publisher's internal shape and names the
    Evaluation ``experiment_id``, which contradicts the ``evaluation_id`` the job API accepts. This
    is the public contract, so it uses the API's vocabulary and omits the rest.
    """

    model_config = ConfigDict(extra="forbid")

    status: PlatformJobStatus = Field(
        description="Platform job status vocabulary: COMPLETED when everything published, ERROR "
        "otherwise. Only those two are ever emitted here.",
    )
    evaluation_id: str = Field(description="Evaluation the results were published under.")
    trial_count: int = Field(default=0, description="Trials actually published (partial on failure).")
    evaluator_result_count: int = Field(default=0, description="Evaluator-result rows written.")
    skipped: list[str] = Field(
        default_factory=list,
        description="Score outputs Intake cannot represent, as 'trial_id: name (reason)'.",
    )
    error: str | None = Field(default=None, description="Why publication failed; absent on success.")


class PublicationFailedError(RuntimeError):
    """Publication failed and the spec marked it required, so the job fails with it.

    Carries the outcome so the caller can still report what landed before the failure.
    """

    def __init__(self, outcome: PublicationOutcome) -> None:
        super().__init__(outcome.error or "publication failed")
        self.outcome = outcome


def _skipped_lines(report: PublishReport) -> list[str]:
    return [f"{item.trial_id}: {item.name} ({item.reason})" for item in report.skipped]


def _completed(evaluation_id: str, report: PublishReport) -> PublicationOutcome:
    return PublicationOutcome(
        status=PlatformJobStatus.COMPLETED,
        evaluation_id=evaluation_id,
        trial_count=report.trial_count,
        evaluator_result_count=report.evaluator_result_count,
        skipped=_skipped_lines(report),
    )


def _failed(evaluation_id: str, error: str, report: PublishReport | None) -> PublicationOutcome:
    return PublicationOutcome(
        status=PlatformJobStatus.ERROR,
        evaluation_id=evaluation_id,
        trial_count=report.trial_count if report is not None else 0,
        evaluator_result_count=report.evaluator_result_count if report is not None else 0,
        skipped=_skipped_lines(report) if report is not None else [],
        error=error,
    )


def _eval_duration_sec(result: AgentEvalResult) -> float | None:
    """Wall-clock seconds the run itself took, or ``None`` when it cannot be known."""
    if result.metadata.duration_sec is not None:
        return result.metadata.duration_sec
    if result.metadata.started_at is not None:
        # A row-eval result carries only a start time (see ``intake.row_adapter``), so its length is
        # measured here instead. Call this before publishing, not after, or the run duration swallows
        # the publish it is meant to be reported alongside. It still slightly overcounts: it covers
        # persisting the bundle between the run finishing and this call.
        return (datetime.now(UTC) - result.metadata.started_at).total_seconds()
    return None


async def _record_durations(
    intake: AsyncIntakeClient,
    *,
    evaluation: EvaluationResponse,
    spec: IntakePublicationSpec,
    workspace: str,
    eval_duration_sec: float | None,
    publish_duration_sec: float,
) -> None:
    """Stamp the two durations onto the Evaluation's metadata.

    PATCH replaces the metadata dict wholesale, so the merge with what is already there is
    mandatory — a blind write would drop producer keys such as ``eval_config_fileset``.
    """
    metadata = dict(evaluation.metadata or {})
    if eval_duration_sec is not None:
        metadata[EVAL_DURATION_KEY] = f"{eval_duration_sec:.1f}"
    metadata[PUBLISH_DURATION_KEY] = f"{publish_duration_sec:.1f}"
    await intake.patch_evaluation(
        name=spec.evaluation_id,
        workspace=workspace,
        body=EvaluationPatchRequest(metadata=metadata),
    )


async def _publish(
    result: AgentEvalResult,
    *,
    intake: AsyncIntakeClient,
    spec: IntakePublicationSpec,
    workspace: str,
    agent_name: str,
    model_name: str | None,
) -> PublishReport:
    """Check the Evaluation exists, publish under it, then record how long both took."""
    # Measured before the publish so the two durations stay disjoint — for a result that carries only
    # a start time, reading this afterwards would fold the whole publish into the run's own length.
    eval_duration_sec = _eval_duration_sec(result)
    # The Evaluation must pre-exist — ATIF ingest rejects an unknown one per trial, so without this
    # a typo would surface as N failed writes after a partial publish instead of one clear stop.
    # This reads the entity store, so it says nothing about whether Intake's span storage is up;
    # that surfaces on the first ingest below, and re-publish is idempotent, so it needs no probe.
    evaluation = (await intake.get_evaluation(name=spec.evaluation_id, workspace=workspace)).data()
    publish_started = time.monotonic()
    report = await publish_to_intake(
        result,
        client=intake,
        experiment_id=spec.evaluation_id,
        workspace=workspace,
        agent_name=agent_name,
        agent_version=spec.agent_version,
        model_name=model_name,
    )
    publish_duration_sec = time.monotonic() - publish_started
    try:
        await _record_durations(
            intake,
            evaluation=evaluation,
            spec=spec,
            workspace=workspace,
            eval_duration_sec=eval_duration_sec,
            publish_duration_sec=publish_duration_sec,
        )
    except Exception:
        # The durations are informational, and the publish already succeeded. Every caller-side
        # handler turns an exception here into a failed job (and a raise when `required`), so letting
        # one escape would report a successful publish as a failure.
        logger.warning(
            "Published to Intake but could not record durations on evaluation %r",
            spec.evaluation_id,
            exc_info=True,
        )
    return report


def publish_agent_eval_result(
    result: AgentEvalResult,
    *,
    spec: IntakePublicationSpec,
    target: Target | Model | AgentBase | None,
    workspace: str,
    intake: AsyncIntakeClient | None,
) -> PublicationOutcome:
    """Publish a finished run to Intake and describe what happened.

    Raises :class:`PublicationFailedError` when publication fails and ``spec.required`` is set;
    otherwise returns a failed outcome for the job output. Either way the result bundle has already
    been saved by the caller, so nothing is lost — a failure costs a re-publish, not a re-run.
    """
    derived_agent_name, model_name = target_agent_identity(target)
    # Spec validation guarantees one of these is set (see `_require_resolvable_publication_identity`).
    agent_name = spec.agent_name or derived_agent_name or ""

    def fail(error: str, report: PublishReport | None = None) -> PublicationOutcome:
        outcome = _failed(spec.evaluation_id, error, report)
        if spec.required:
            raise PublicationFailedError(outcome)
        logger.warning(
            "Publication to Intake failed for evaluation %r but was not required; continuing: %s",
            spec.evaluation_id,
            error,
        )
        return outcome

    if intake is None:
        return fail("No platform client available for platformless run; cannot publish with Intake.")

    logger.info(
        "Publishing %d trial(s) to Intake under evaluation %r in workspace %r",
        len(result.trials),
        spec.evaluation_id,
        workspace,
    )
    try:
        report = run_with_isolated_async_client(
            intake,
            lambda cloned_intake: _publish(
                result,
                intake=cloned_intake,
                spec=spec,
                workspace=workspace,
                agent_name=agent_name,
                model_name=model_name,
            ),
        )
    except PublishError as error:
        return fail(str(error), error.report)
    except NotFoundError:
        return fail(
            f"Evaluation {spec.evaluation_id!r} does not exist in workspace {workspace!r}. "
            "Create it before submitting the job; the evaluation does not create it."
        )
    except NemoClientError as error:
        return fail(f"{type(error).__name__}: {error}")
    except Exception as error:
        # `required=False` promises the evaluation survives a failed publish. Letting an unforeseen
        # error escape would break that promise for exactly the failures nobody anticipated, so the
        # catch-all is the point rather than an oversight. Logged with a traceback because, unlike
        # the handlers above, there is no known cause to report.
        logger.exception("Unexpected error publishing to Intake for evaluation %r", spec.evaluation_id)
        return fail(f"Unexpected {type(error).__name__}: {error}")

    logger.info(
        "Published %d trial(s) and %d evaluator result(s) to Intake under evaluation %r",
        report.trial_count,
        report.evaluator_result_count,
        spec.evaluation_id,
    )
    return _completed(spec.evaluation_id, report)


def publish_row_eval_result(
    result: EvaluationResult | BenchmarkEvaluationResult,
    *,
    spec: RowIntakePublicationSpec,
    target: Model | AgentBase | None,
    run_id: str | None,
    started_at: datetime,
    workspace: str,
    intake: AsyncIntakeClient | None,
) -> PublicationOutcome:
    """Publish a finished dataset-driven run to Intake and describe what happened.

    Adapts the row result to the shape the publisher consumes, then hands off to
    :func:`publish_agent_eval_result` — the failure semantics, the outcome shape, and the
    ``required`` behaviour are all the same.

    ``run_id`` is the job id and may be ``None`` for in-process calls. Unlike agent eval, whose
    result always carries a generated run id, a row result has none, so there is nothing stable to key
    published sessions on and the run cannot be published.
    """
    if run_id is None:
        error = "No job id to publish under; a dataset-driven evaluation takes its run identity from the job."
        outcome = _failed(spec.evaluation_id, error, None)
        if spec.required:
            raise PublicationFailedError(outcome)
        logger.warning("Publication to Intake failed for evaluation %r: %s", spec.evaluation_id, error)
        return outcome

    try:
        adapted = row_result_to_agent_eval_result(
            result,
            run_id=run_id,
            started_at=started_at,
            test_case_id_field=spec.test_case_id_field,
        )
    except RowIdentityError as error:
        outcome = _failed(spec.evaluation_id, str(error), None)
        if spec.required:
            raise PublicationFailedError(outcome) from error
        logger.warning("Publication to Intake failed for evaluation %r: %s", spec.evaluation_id, error)
        return outcome

    return publish_agent_eval_result(adapted, spec=spec, target=target, workspace=workspace, intake=intake)
