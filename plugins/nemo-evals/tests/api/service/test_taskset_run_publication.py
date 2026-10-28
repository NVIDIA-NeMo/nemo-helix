# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit with publication.intake: one Intake Evaluation per run, created before the run is accepted."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from nemo_evals.api.service.taskset_run_service import (
    RunEvaluationDeniedError,
    RunEvaluationError,
    RunEvaluationMismatchError,
    TasksetRunResolutionError,
)
from nemo_evals.api.taskset_run_schemas import TasksetRunInput
from nemo_evals.jobs.harbor_task_spec import HarborTaskJobSpec
from nemo_evals.taskset_runs.entities import TasksetRunEntity
from nemo_evals.taskset_runs.run_evaluation import EVALUATION_SOURCE, RUN_ID_KEY
from nemo_evals.taskset_runs.task_jobs import task_job_request
from nemo_helix_plugin.client.errors import InternalServerError
from nemo_helix_plugin.intake.types import EvaluationResponse

pytestmark = pytest.mark.usefixtures("published")

RUN = "run-11f77762178b0e05"  # derived from (default, user:alice, ci-run)


@pytest.fixture
def publishing(request_body):
    request_body["publication"] = {"intake": {"experiment": "swe-agent-tuning"}}
    return request_body


async def submit(service, body):
    return await service.submit(TasksetRunInput.model_validate(body), workspace="default", caller="user:alice")


async def runs(entity_store):
    return (await entity_store.list(TasksetRunEntity, workspace="default")).data


async def test_no_publication_makes_no_intake_call(service, request_body, fake_evaluations, run_store):
    fake_evaluations.fail = {"create", "get", "patch"}
    fake_evaluations.experiments = {}
    outcome = await submit(service, request_body)
    plan = await run_store.load_plan(await run_store.find("default", outcome.status.run_id))
    assert plan.intake is None and not fake_evaluations.created
    assert not any(line.startswith("Intake") for line in outcome.status.warnings)


async def test_a_missing_experiment_is_rejected_before_acceptance(service, publishing, fake_evaluations, entity_store):
    fake_evaluations.experiments = {}
    with pytest.raises(TasksetRunResolutionError, match="Experiment default/swe-agent-tuning doesn't exist"):
        await submit(service, publishing)
    assert not await runs(entity_store) and not fake_evaluations.created


async def test_submit_creates_the_runs_evaluation_before_accepting_it(
    service, publishing, fake_evaluations, fake_task_jobs, run_store, entity_store
):
    runs_at_create: list[int] = []
    create = fake_evaluations.create

    async def recording_create(workspace, body):
        runs_at_create.append(len(await runs(entity_store)))
        await create(workspace, body)

    fake_evaluations.create = recording_create
    outcome = await submit(service, publishing)

    assert runs_at_create == [0]  # task Jobs publish to it, so every stored run already has it
    (body,) = fake_evaluations.created
    run = await run_store.find("default", RUN)
    taskset, _, digest = run.taskset_ref.partition("#")
    assert (body.name, body.experiment_ids, body.status) == (
        RUN,
        ["exp-1"],  # Intake takes the Experiment's id; submit resolved it from the name
        "running",
    )
    # Without a dataset in the request, the dataset is the pinned taskset and its digest the version.
    assert (body.dataset_name, body.dataset_version) == (taskset, digest)
    assert body.metadata == {RUN_ID_KEY: RUN, "source": EVALUATION_SOURCE}
    plan = await run_store.load_plan(run)
    assert plan.intake is not None
    assert (plan.intake.experiment_id, plan.intake.agent_name) == ("exp-1", "oracle")
    assert outcome.status.warnings[-1] == f"Intake: Evaluation {RUN} in Experiment swe-agent-tuning"
    job = await task_job_request(plan, run, 0, plan.task_environments[0])  # what the controller will create
    publication = HarborTaskJobSpec.model_validate(job.spec).spec.publication
    assert publication is not None and publication.intake is not None
    assert (publication.intake.evaluation_id, publication.intake.required) == (RUN, False)


async def test_intake_setup_doesnt_come_out_of_the_runs_execution_budget(
    service, publishing, fake_evaluations, run_store
):
    """The queue timeout starts once the Evaluation exists: a slow Intake call can't accept a run already expired."""
    now = [datetime(2026, 10, 14, 12, tzinfo=UTC)]
    service._clock = lambda: now[0]
    create = fake_evaluations.create

    async def slow_create(workspace, body):
        now[0] += timedelta(seconds=45)
        await create(workspace, body)

    fake_evaluations.create = slow_create
    publishing["timeout_seconds"] = 30
    await submit(service, publishing)

    run = await run_store.find("default", RUN)
    assert run.accepted_at == datetime(2026, 10, 14, 12, 0, 45, tzinfo=UTC)
    assert run.timeout_seconds == 30  # counted from the run's first slot, not from acceptance
    assert run.execution_deadline_at == run.accepted_at + timedelta(seconds=service._config.runs.queue_timeout_seconds)


async def test_a_requested_dataset_names_the_evaluations_dataset(service, publishing, fake_evaluations):
    publishing["publication"]["intake"] |= {
        "dataset_name": "datacurve/deep-swe",
        "dataset_version": "sha256:" + "a" * 64,
    }
    await submit(service, publishing)
    (body,) = fake_evaluations.created
    assert (body.dataset_name, body.dataset_version) == ("datacurve/deep-swe", "sha256:" + "a" * 64)


async def test_a_keyed_retry_after_a_failed_accept_adopts_the_evaluation(
    service, publishing, fake_evaluations, entity_store, monkeypatch
):
    """An accept that fails after the Evaluation was created leaves it behind; the same key's retry adopts it."""
    accept = service._store.accept

    async def failing_accept(acceptance):
        raise RuntimeError("entity store unavailable")

    monkeypatch.setattr(service._store, "accept", failing_accept)
    with pytest.raises(RuntimeError, match="entity store unavailable"):
        await submit(service, publishing)
    assert not await runs(entity_store) and len(fake_evaluations.created) == 1

    monkeypatch.setattr(service._store, "accept", accept)
    outcome = await submit(service, publishing)
    assert outcome.created is True and len(await runs(entity_store)) == 1 and len(fake_evaluations.created) == 1
    assert outcome.status.warnings[-1] == f"Intake: Evaluation {RUN} in Experiment swe-agent-tuning"


async def test_a_keyed_retry_asking_for_another_dataset_is_refused(
    service, publishing, fake_evaluations, entity_store, monkeypatch
):
    """The Evaluation an earlier attempt left names another dataset: nothing is adopted, no run is stored."""
    monkeypatch.setattr(service._store, "accept", AsyncMock(side_effect=RuntimeError("entity store unavailable")))
    with pytest.raises(RuntimeError, match="entity store unavailable"):
        await submit(service, publishing)
    monkeypatch.undo()
    publishing["publication"]["intake"] |= {"dataset_name": "datacurve/deep-swe", "dataset_version": "v2"}
    with pytest.raises(RunEvaluationMismatchError, match="created with different dataset metadata"):
        await submit(service, publishing)
    assert not await runs(entity_store) and len(fake_evaluations.created) == 1


@pytest.mark.parametrize(
    "metadata, experiments",
    [({RUN_ID_KEY: "run-someone-else"}, ["exp-1"]), ({RUN_ID_KEY: RUN}, ["exp-other"])],
    ids=["other-run", "other-experiment"],
)
async def test_a_foreign_evaluation_under_the_runs_name_is_never_adopted(
    service, publishing, fake_evaluations, entity_store, metadata, experiments
):
    fake_evaluations.evaluations[RUN] = EvaluationResponse(
        id="eval-x", name=RUN, workspace="default", experiment_ids=experiments, dataset_name="d", metadata=metadata
    )
    with pytest.raises(RunEvaluationError, match="belongs to something else"):
        await submit(service, publishing)
    assert not await runs(entity_store)  # no task Job may publish into someone else's Evaluation


async def test_an_intake_outage_stores_no_run_and_a_keyed_retry_adopts_the_evaluation(
    service, publishing, fake_evaluations, fake_task_jobs, entity_store, files_service
):
    """Intake saved the Evaluation but its reply was lost: submit fails, and the retry adopts that Evaluation."""
    create = fake_evaluations.create

    async def lost_reply(workspace, body):
        await create(workspace, body)
        raise InternalServerError(httpx.Response(503, json={"detail": "intake unavailable"}))

    fake_evaluations.create = lost_reply
    with pytest.raises(RunEvaluationError, match="creating its Intake Evaluation failed"):
        await submit(service, publishing)
    assert not await runs(entity_store) and not files_service.filesets

    fake_evaluations.create = create
    outcome = await submit(service, publishing)
    assert outcome.created is True and len(await runs(entity_store)) == 1
    assert len(fake_evaluations.created) == 1


async def test_a_caller_who_may_not_create_evaluations_is_refused_before_acceptance(
    service, publishing, authorizer, entity_store, fake_task_jobs
):
    async def check(method: str, path: str) -> None:
        if path == "/apis/intake/v2/workspaces/default/evaluations":
            raise HTTPException(status_code=403, detail="denied")

    authorizer.side_effect = check
    with pytest.raises(HTTPException) as exc:
        await submit(service, publishing)
    assert exc.value.status_code == 403
    assert exc.value.detail == "you may not create Intake Evaluations in workspace default"
    assert not await runs(entity_store)


async def test_a_caller_who_may_not_read_the_experiment_is_refused_before_the_lookup(
    service, publishing, authorizer, fake_evaluations, entity_store
):
    """The lookup runs as the evals service, so the caller's own read is checked first."""

    async def check(method: str, path: str) -> None:
        if (method, path) == ("GET", "/apis/intake/v2/workspaces/default/experiments/swe-agent-tuning"):
            raise HTTPException(status_code=403, detail="denied")

    authorizer.side_effect = check
    fake_evaluations.fail = {"experiment"}  # a lookup would 503 instead
    with pytest.raises(HTTPException) as exc:
        await submit(service, publishing)
    assert exc.value.status_code == 403
    assert exc.value.detail == "you can't read Intake Experiment default/swe-agent-tuning"
    assert not await runs(entity_store)


async def test_an_evaluation_create_denied_stores_no_run(service, publishing, fake_evaluations, entity_store):
    """Submit checked the permission first, so a 403 here means it was revoked since: no unpublished run."""
    fake_evaluations.fail = {"create-denied"}
    with pytest.raises(RunEvaluationDeniedError, match=f"run {RUN}: Intake refused to create its Evaluation"):
        await submit(service, publishing)
    assert not await runs(entity_store)
