# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A taskset run's Intake Evaluation: one per run, named after it, in the Experiment the request named.

Submit creates the Evaluation (``ensure_run_evaluation``) before it accepts the run;
each task Job publishes its own trials under it when it finishes; the controller stamps the run's
outcome and duration onto it once (``stamp_run_evaluation``). Nothing here publishes trials.
"""

import logging
from datetime import UTC, datetime
from typing import Protocol

from nemo_evals.jobs.publication import EVAL_DURATION_KEY
from nemo_evals.taskset_runs.entities import ResultState, TasksetRunEntity, TasksetRunPhase
from nemo_evals.taskset_runs.specs import RunIntakePublication, TasksetRunPlan
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.intake.types import EvaluationCreateRequest, EvaluationPatchRequest, EvaluationResponse

logger = logging.getLogger(__name__)

RUN_ID_KEY = "taskset_run_id"
"""Evaluation metadata key naming the run; with the Experiment it marks an Evaluation as the run's own."""
SOURCE_KEY = "source"
EVALUATION_SOURCE = "nemo-evals.taskset-run"
RESULT_STATE_KEY = "result_state"
QUEUE_DURATION_KEY = "queue_duration_sec"
"""Seconds the run waited for its first task slot; its duration (``eval_duration_sec``) counts from acceptance."""


class RunEvaluations(Protocol):
    """The Intake operations a run's Evaluation needs, acting as the run's caller or creator."""

    async def experiment_id(self, workspace: str, name: str) -> str | None:
        """The Experiment's id, or None when no Experiment has that name."""
        ...

    async def create(self, workspace: str, body: EvaluationCreateRequest) -> None:
        """Create the Evaluation. Raises ``ConflictError`` when the name is taken."""
        ...

    async def get(self, workspace: str, name: str) -> EvaluationResponse | None:
        """The Evaluation, or None when it doesn't exist."""
        ...

    async def patch(self, workspace: str, name: str, body: EvaluationPatchRequest) -> None: ...


class IntakeRunEvaluations:
    """:class:`RunEvaluations` over the Intake service."""

    def __init__(self, intake: AsyncIntakeClient) -> None:
        self._intake = intake

    async def experiment_id(self, workspace: str, name: str) -> str | None:
        try:
            return (await self._intake.get_experiment(workspace=workspace, name=name)).data().id
        except NotFoundError:
            return None

    async def create(self, workspace: str, body: EvaluationCreateRequest) -> None:
        await self._intake.create_evaluation(workspace=workspace, body=body)

    async def get(self, workspace: str, name: str) -> EvaluationResponse | None:
        try:
            return (await self._intake.get_evaluation(workspace=workspace, name=name)).data()
        except NotFoundError:
            return None

    async def patch(self, workspace: str, name: str, body: EvaluationPatchRequest) -> None:
        await self._intake.patch_evaluation(workspace=workspace, name=name, body=body)


class ForeignEvaluationError(Exception):
    """An Evaluation under the run's name isn't this run's: task Jobs must not publish into it."""

    def __init__(self, workspace: str, name: str) -> None:
        super().__init__(
            f"Intake Evaluation {workspace}/{name} exists and belongs to something else; "
            "delete it, or submit with another idempotency_key"
        )


class EvaluationDatasetMismatchError(Exception):
    """The run's Evaluation exists with another dataset: an earlier attempt resolved another taskset revision,
    or asked for another ``dataset_name``/``dataset_version``. No retry of this request can match it."""

    def __init__(self, workspace: str, name: str) -> None:
        super().__init__(
            f"Intake Evaluation {workspace}/{name} was created with different dataset metadata; "
            "delete it, or resubmit with a new idempotency_key"
        )


def _names_run(evaluation: EvaluationResponse, run_id: str, intake: RunIntakePublication) -> bool:
    return evaluation.metadata.get(RUN_ID_KEY) == run_id and intake.experiment_id in evaluation.experiment_ids


def _same_dataset(evaluation: EvaluationResponse, intake: RunIntakePublication) -> bool:
    return (evaluation.dataset_name, evaluation.dataset_version) == (intake.dataset_name, intake.dataset_version)


def is_run_evaluation(evaluation: EvaluationResponse, run_id: str, intake: RunIntakePublication) -> bool:
    """True when ``evaluation`` is the one submit created for run ``run_id`` in the plan's Experiment, with its dataset.

    Evaluation names are predictable (the run id), so the name alone proves nothing.
    """
    return _names_run(evaluation, run_id, intake) and _same_dataset(evaluation, intake)


async def ensure_run_evaluation(evaluations: RunEvaluations, workspace: str, run_id: str, plan: TasksetRunPlan) -> None:
    """Create run ``run_id``'s Evaluation, or adopt the one an earlier attempt created; safe to repeat.

    Submit calls it before accepting the run, so no run record exists yet.

    Raises:
        ForeignEvaluationError: An Evaluation under the run's name isn't this run's.
        EvaluationDatasetMismatchError: The run's Evaluation exists with another dataset.
        NemoClientError: Intake failed; a retry of the same submit creates it.
    """
    intake = plan.intake
    assert intake is not None
    body = EvaluationCreateRequest(
        name=run_id,
        experiment_ids=[intake.experiment_id],
        dataset_name=intake.dataset_name,
        dataset_version=intake.dataset_version,
        description=f"Taskset run {run_id} of {plan.taskset_ref} ({len(plan.spec.tasks)} tasks)",
        metadata={RUN_ID_KEY: run_id, SOURCE_KEY: EVALUATION_SOURCE},
        status=TasksetRunPhase.RUNNING.value,
    )
    try:
        await evaluations.create(workspace, body)
        logger.info("Created Intake Evaluation %s/%s in Experiment %s", workspace, run_id, intake.experiment)
        return
    except ConflictError:
        pass  # an earlier attempt created it, or someone else did; checked below
    existing = await evaluations.get(workspace, run_id)
    if existing is None or not _names_run(existing, run_id, intake):
        raise ForeignEvaluationError(workspace, run_id)
    if not _same_dataset(existing, intake):
        raise EvaluationDatasetMismatchError(workspace, run_id)


async def stamp_run_evaluation(
    evaluations: RunEvaluations,
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    *,
    phase: TasksetRunPhase,
    result_state: ResultState,
    now: datetime,
) -> None:
    """Record the run's outcome and duration on its Evaluation; best effort, never raises.

    One read-merge-PATCH: PATCH replaces ``metadata`` whole, so keys others wrote must be carried over.
    Task Jobs never write the Evaluation's metadata, so this is its only writer after submit.
    """
    intake = plan.intake
    assert intake is not None
    try:
        existing = await evaluations.get(run.workspace, run.name)
        if existing is None or not is_run_evaluation(existing, run.name, intake):
            logger.warning("Run %s: no Intake Evaluation of its own to stamp", run.name)
            return
        metadata = dict(existing.metadata)
        accepted_at = _aware(run.accepted_at)
        metadata[EVAL_DURATION_KEY] = f"{(now - accepted_at).total_seconds():.1f}"
        started_at = _aware(run.started_at) if run.started_at is not None else now  # never started: queued throughout
        metadata[QUEUE_DURATION_KEY] = f"{(started_at - accepted_at).total_seconds():.1f}"
        metadata[RESULT_STATE_KEY] = result_state.value
        await evaluations.patch(run.workspace, run.name, EvaluationPatchRequest(status=phase.value, metadata=metadata))
    except Exception:
        # The run's own state is authoritative; a missing stamp only leaves the Evaluation "running".
        logger.warning("Run %s: stamping its Intake Evaluation failed", run.name, exc_info=True)


def _aware(value: datetime) -> datetime:
    """Entity Store may return naive timestamps; they are UTC."""
    return value if value.utcoffset() is not None else value.replace(tzinfo=UTC)
