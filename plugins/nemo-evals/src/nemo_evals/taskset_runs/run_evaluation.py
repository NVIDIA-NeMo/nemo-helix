# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A taskset run's Intake Evaluation: one per run, named after it, in the Experiment the request named.

M0 story 1 Part 1. Submit creates the Evaluation (``ensure_run_evaluation``) before any task Job exists;
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


def is_run_evaluation(evaluation: EvaluationResponse, run: TasksetRunEntity, intake: RunIntakePublication) -> bool:
    """True when ``evaluation`` is the one submit created for ``run`` in the plan's Experiment.

    Evaluation names are predictable (the run id), so the name alone proves nothing.
    """
    return evaluation.metadata.get(RUN_ID_KEY) == run.name and intake.experiment_id in evaluation.experiment_ids


async def ensure_run_evaluation(evaluations: RunEvaluations, run: TasksetRunEntity, plan: TasksetRunPlan) -> None:
    """Create the run's Evaluation, or adopt the one an earlier attempt created; safe to repeat.

    Raises:
        ForeignEvaluationError: An Evaluation under the run's name isn't this run's.
        NemoClientError: Intake failed; a retry of the same submit creates it.
    """
    intake = plan.intake
    assert intake is not None
    body = EvaluationCreateRequest(
        name=run.name,
        experiment_ids=[intake.experiment_id],
        dataset_name=intake.dataset_name,
        dataset_version=intake.dataset_version,
        description=f"Taskset run {run.name} of {plan.taskset_ref} ({len(plan.spec.tasks)} tasks)",
        metadata={RUN_ID_KEY: run.name, SOURCE_KEY: EVALUATION_SOURCE},
        status=TasksetRunPhase.RUNNING.value,
    )
    try:
        await evaluations.create(run.workspace, body)
        logger.info("Created Intake Evaluation %s/%s in Experiment %s", run.workspace, run.name, intake.experiment)
        return
    except ConflictError:
        pass  # an earlier attempt created it, or someone else did; checked below
    existing = await evaluations.get(run.workspace, run.name)
    if existing is None or not is_run_evaluation(existing, run, intake):
        raise ForeignEvaluationError(run.workspace, run.name)


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
        if existing is None or not is_run_evaluation(existing, run, intake):
            logger.warning("Run %s: no Intake Evaluation of its own to stamp", run.name)
            return
        metadata = dict(existing.metadata)
        accepted_at = (
            run.accepted_at if run.accepted_at.utcoffset() is not None else run.accepted_at.replace(tzinfo=UTC)
        )
        metadata[EVAL_DURATION_KEY] = f"{(now - accepted_at).total_seconds():.1f}"
        metadata[RESULT_STATE_KEY] = result_state.value
        await evaluations.patch(run.workspace, run.name, EvaluationPatchRequest(status=phase.value, metadata=metadata))
    except Exception:
        # The run's own state is authoritative; a missing stamp only leaves the Evaluation "running".
        logger.warning("Run %s: stamping its Intake Evaluation failed", run.name, exc_info=True)
