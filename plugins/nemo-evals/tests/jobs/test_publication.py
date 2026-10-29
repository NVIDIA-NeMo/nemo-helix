# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agent-eval job's Intake publication step."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest
import respx
from nemo_evals.api.schemas import MetricInline
from nemo_evals.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evals.intake.publish import publish_to_intake
from nemo_evals.jobs.agent_evaluate import AsyncAgentEvalJob
from nemo_evals.jobs.agent_spec import (
    AgentEvalInputSpec,
    AgentEvalSpec,
    AgentEvalTaskInput,
    AgentTarget,
    FabricConfigSource,
    FabricRunnerTarget,
    GymAgentSource,
    GymRunnerTarget,
    HarborBuiltinAgentSource,
    HarborImportedAgentSource,
    HarborModelName,
    HarborRunnerTarget,
    ModelTarget,
    ResolvedTask,
    Target,
    target_agent_identity,
)
from nemo_evals.jobs.evaluate import AsyncEvaluateJob, EvaluateInputSpec, EvaluateSpec
from nemo_evals.jobs.publication import (
    EVAL_DURATION_KEY,
    PUBLISH_DURATION_KEY,
    PublicationFailedError,
    PublicationOutcome,
    publish_agent_eval_result,
)
from nemo_evals.jobs.publication_spec import (
    IntakePublicationSpec,
    PublicationSpec,
    RowIntakePublicationSpec,
    RowPublicationSpec,
)
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.hybrid import HybridMetricBundlePackager
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nhx.testing.intake import INTAKE_TEST_BASE_URL, InMemoryIntakeService
from nhx_evals_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary, RunMetadata
from nhx_evals_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nhx_evals_sdk.execution.metric_execution import run_sync
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.metrics.protocol import MetricOutput
from nhx_evals_sdk.values import Model, RunConfigOnline, RunConfigOnlineModel
from nhx_evals_sdk.values.agents import NemoAgentToolkitAgent
from nhx_evals_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from nhx_evals_sdk.values.multi_metric_results import BenchmarkEvaluationResult  # noqa: F401
from nhx_evals_sdk.values.results import AggregatedMetricResult, EvaluationResult, RowScore
from pydantic import ValidationError
from pytest_mock import MockerFixture

STARTED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

_INLINE_METRIC = MetricInline.model_validate(
    bundle_metric(
        ExactMatchMetric(reference="{{item.question}}", candidate="{{item.question}}"),
        HybridMetricBundlePackager(),
    ).model_dump(mode="json")
)

#: Aliased because `_FakeTraces` defines a `list` method, which shadows the builtin in class scope.


def _response(status: int) -> httpx.Response:
    """A real httpx response; typed client errors read ``response.request`` in their constructor."""
    return httpx.Response(status, request=httpx.Request("GET", "http://platform/evaluations"))


# --- fakes ------------------------------------------------------------------


@pytest.fixture
def intake() -> Iterator[InMemoryIntakeService]:
    """In-memory Intake, also reachable from the fresh HTTP clients publication opens per event loop."""
    service = InMemoryIntakeService()
    service.evaluation_metadata = {"eval_config_fileset": "fs-1"}
    with respx.mock(assert_all_called=False) as router:
        router.route(host=urlsplit(INTAKE_TEST_BASE_URL).hostname).mock(side_effect=service)
        yield service


def _result() -> AgentEvalResult:
    return AgentEvalResult(
        run_id="run-1",
        tasks=[],
        trials=[
            AgentEvalTrial(
                id="trial-1",
                task_id="task-1",
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="answer"),
            )
        ],
        scores=[
            AgentEvalTaskScore(
                id="score-1",
                run_id="run-1",
                task_id="task-1",
                trial_id="trial-1",
                metric_type="accuracy",
                status=AgentEvalScoreStatus.COMPLETED,
                outputs=[MetricOutput(name="score", value=1.0)],
            )
        ],
        summary=AgentEvalSummary(),
        metadata=RunMetadata(started_at=STARTED_AT),
    )


def _publish(
    client: AsyncIntakeClient | None, *, required: bool = True, agent_name: str | None = "a"
) -> PublicationOutcome:
    return publish_agent_eval_result(
        _result(),
        spec=IntakePublicationSpec(evaluation_id="eval-1", agent_name=agent_name, required=required),
        target=None,
        workspace="default",
        intake=client,
    )


# --- identity resolution ----------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (AgentTarget(agent=NemoAgentToolkitAgent(name="my-agent", url="http://agent")), ("my-agent", None)),
        (ModelTarget(model=Model(name="gpt-4o", url="http://model")), (None, "gpt-4o")),
        (
            HarborRunnerTarget(source=HarborBuiltinAgentSource(name="oracle", model=HarborModelName(name="m"))),
            ("oracle", "m"),
        ),
        (HarborRunnerTarget(source=HarborImportedAgentSource(import_path="pkg:Agent")), ("pkg:Agent", None)),
        (
            GymRunnerTarget(
                source=GymAgentSource(component="simple_agent", config="conf/agent.yaml"), resources_server="mcqa"
            ),
            ("simple_agent", None),
        ),
        (FabricRunnerTarget(source=FabricConfigSource(config={}, model="p/m")), (None, "p/m")),
        (None, (None, None)),
    ],
)
def test_target_agent_identity(target: Target | None, expected: tuple[str | None, str | None]) -> None:
    assert target_agent_identity(target) == expected


# --- submit-time validation -------------------------------------------------


def _input_spec(target: Target | None, publication: PublicationSpec | None, **kwargs: Any) -> AgentEvalInputSpec:
    trials = None if target is not None else [_result().trials[0]]
    return AgentEvalInputSpec(
        tasks=[AgentEvalTaskInput(id="task-1", intent="do it")],
        target=target,
        trials=trials,
        publication=publication,
        **kwargs,
    )


def test_publication_is_optional() -> None:
    spec = _input_spec(ModelTarget(model=Model(name="m", url="http://m")), None)
    assert spec.publication is None


def test_publication_defaults_to_required(intake: InMemoryIntakeService) -> None:
    spec = _input_spec(
        AgentTarget(agent=NemoAgentToolkitAgent(name="a", url="http://a")),
        PublicationSpec(intake=IntakePublicationSpec(evaluation_id="eval-1")),
    )
    assert spec.publication is not None
    assert spec.publication.intake is not None
    assert spec.publication.intake.required is True


def test_agent_name_derived_from_agent_target_needs_no_override(intake: InMemoryIntakeService) -> None:
    spec = _input_spec(
        AgentTarget(agent=NemoAgentToolkitAgent(name="derived", url="http://a")),
        PublicationSpec(intake=IntakePublicationSpec(evaluation_id="eval-1")),
    )
    assert spec.publication is not None
    assert spec.publication.intake is not None
    assert spec.publication.intake.agent_name is None
    assert target_agent_identity(spec.target)[0] == "derived"


def test_agent_name_derived_from_gym_target_needs_no_override(intake: InMemoryIntakeService) -> None:
    spec = _input_spec(
        GymRunnerTarget(
            source=GymAgentSource(component="simple_agent", config="conf/agent.yaml"), resources_server="mcqa"
        ),
        PublicationSpec(intake=IntakePublicationSpec(evaluation_id="eval-1")),
    )
    assert spec.publication is not None
    assert spec.publication.intake is not None
    assert spec.publication.intake.agent_name is None
    assert target_agent_identity(spec.target)[0] == "simple_agent"


@pytest.mark.parametrize(
    "target",
    [
        ModelTarget(model=Model(name="gpt-4o", url="http://model")),
        FabricRunnerTarget(source=FabricConfigSource(config={})),
        None,
    ],
)
def test_undeducible_agent_name_is_rejected_at_submit(target: Target | None) -> None:
    with pytest.raises(ValidationError, match="agent_name` is required"):
        _input_spec(target, PublicationSpec(intake=IntakePublicationSpec(evaluation_id="eval-1")))


def test_blank_identity_fields_are_rejected() -> None:
    # An empty `agent_name` satisfies `is not None` in the identity validator, so it would skip the
    # derivation that validator exists to require and resolve back to "" at publish time.
    with pytest.raises(ValidationError):
        IntakePublicationSpec(evaluation_id="eval-1", agent_name="")
    with pytest.raises(ValidationError):
        IntakePublicationSpec(evaluation_id="eval-1", agent_version="")


@pytest.mark.parametrize(
    "target",
    [
        ModelTarget(model=Model(name="gpt-4o", url="http://model")),
        FabricRunnerTarget(source=FabricConfigSource(config={})),
        None,
    ],
)
def test_explicit_agent_name_satisfies_undeducible_targets(target: Target | None) -> None:
    spec = _input_spec(
        target, PublicationSpec(intake=IntakePublicationSpec(evaluation_id="eval-1", agent_name="explicit"))
    )
    assert spec.publication is not None


# --- publishing -------------------------------------------------------------


def test_publishes_and_reports_what_landed(intake: InMemoryIntakeService, mocker: MockerFixture) -> None:
    client = intake.client()
    with_http_client = mocker.spy(AsyncIntakeClient, "with_http_client")
    outcome = _publish(client)

    # Publication runs on a throwaway HTTP client so the injected one never binds to its event loop.
    assert with_http_client.call_count == 1
    assert outcome.status == HelixJobStatus.COMPLETED
    assert outcome.evaluation_id == "eval-1"
    assert outcome.trial_count == 1
    assert outcome.evaluator_result_count == 1
    assert outcome.error is None
    # Twice: this path needs the entity to stamp durations onto, and ``publish_to_intake``
    # re-checks for itself because it is public and cannot assume its caller did.
    assert [lookup["name"] for lookup in intake.evaluation_lookups] == ["eval-1", "eval-1"]


def test_durations_are_stamped_without_dropping_existing_metadata(intake: InMemoryIntakeService) -> None:
    client = intake.client()
    _publish(client)

    (patched,) = intake.evaluation_patches
    metadata = patched["metadata"]
    # PATCH replaces the metadata dict wholesale, so stamping the durations has to merge with what
    # the producer already wrote. A blind write would silently drop `eval_config_fileset`.
    assert metadata["eval_config_fileset"] == "fs-1"
    assert float(metadata[EVAL_DURATION_KEY]) > 0
    assert float(metadata[PUBLISH_DURATION_KEY]) >= 0


def test_a_taskset_task_job_publishes_without_touching_the_shared_evaluation(intake: InMemoryIntakeService) -> None:
    # A taskset run's task Jobs share one Evaluation; each stamping its own duration would race.
    outcome = publish_agent_eval_result(
        _result(),
        spec=IntakePublicationSpec(evaluation_id="eval-1", agent_name="a", required=False),
        target=None,
        workspace="default",
        intake=intake.client(),
        record_durations=False,
    )
    assert outcome.status == HelixJobStatus.COMPLETED and outcome.trial_count == 1
    assert intake.evaluation_patches == []


def test_a_started_at_only_result_does_not_count_publish_time_as_run_time(intake: InMemoryIntakeService) -> None:
    # A row-eval result carries no `duration_sec`, so the run's length is derived from `started_at`.
    # Deriving it after the publish instead of before would fold the publish into the run.
    publish_sec = 0.3
    intake.atif_delay_sec = publish_sec
    client = intake.client()
    result = _result()
    result.metadata.started_at = datetime.now(UTC)

    publish_agent_eval_result(
        result,
        spec=IntakePublicationSpec(evaluation_id="eval-1", agent_name="a", required=True),
        target=None,
        workspace="default",
        intake=client,
    )

    (patched,) = intake.evaluation_patches
    metadata = patched["metadata"]
    assert float(metadata[PUBLISH_DURATION_KEY]) >= publish_sec
    assert float(metadata[EVAL_DURATION_KEY]) < publish_sec / 2


def test_a_failed_duration_stamp_does_not_fail_the_publish(intake: InMemoryIntakeService) -> None:
    intake.patch_failure = httpx.ConnectError("patch failed")
    client = intake.client()
    outcome = _publish(client, required=True)

    # The durations are informational and the trials already landed, so losing them must not turn a
    # successful publish into a failed job — which is what every caller-side handler would do.
    assert outcome.status == HelixJobStatus.COMPLETED
    assert outcome.trial_count == 1
    assert outcome.error is None


def test_outcome_does_not_leak_experiment_id(intake: InMemoryIntakeService) -> None:
    outcome = _publish(intake.client())
    assert "experiment_id" not in outcome.model_dump()


def test_stamps_the_run_start_time_so_republish_is_idempotent(intake: InMemoryIntakeService) -> None:
    client = intake.client()
    _publish(client)
    assert datetime.fromisoformat(intake.atif_requests[0]["steps"][0]["timestamp"]) == STARTED_AT


def test_missing_evaluation_fails_before_any_ingest(intake: InMemoryIntakeService) -> None:
    intake.evaluation_exists = False
    client = intake.client()
    with pytest.raises(PublicationFailedError) as excinfo:
        _publish(client)

    assert intake.atif_requests == []
    outcome = excinfo.value.outcome
    assert outcome.status == HelixJobStatus.ERROR
    assert "does not exist" in (outcome.error or "")


def test_required_failure_raises_with_partial_outcome(intake: InMemoryIntakeService) -> None:
    intake.atif_failure = httpx.ConnectError("ingest failed")
    client = intake.client()
    with pytest.raises(PublicationFailedError) as excinfo:
        _publish(client)

    outcome = excinfo.value.outcome
    assert outcome.status == HelixJobStatus.ERROR
    assert outcome.trial_count == 0


def test_optional_failure_returns_outcome_instead_of_raising(intake: InMemoryIntakeService) -> None:
    intake.atif_failure = httpx.ConnectError("ingest failed")
    client = intake.client()
    outcome = _publish(client, required=False)

    assert outcome.status == HelixJobStatus.ERROR
    assert outcome.error


def test_unexpected_failure_still_honours_required_false(intake: InMemoryIntakeService) -> None:
    # `required=False` promises the evaluation survives a failed publish. An error outside the known
    # taxonomy — a bug, a transport quirk, anything unforeseen — must not be the one case that
    # escapes and fails the job anyway.
    intake.evaluation_failure = ValueError("something nobody planned for")
    client = intake.client()
    outcome = _publish(client, required=False)

    assert outcome.status == HelixJobStatus.ERROR
    assert "ValueError" in (outcome.error or "")
    assert "something nobody planned for" in (outcome.error or "")


def test_unexpected_failure_fails_the_job_when_required(intake: InMemoryIntakeService) -> None:
    intake.evaluation_failure = ValueError("something nobody planned for")
    client = intake.client()
    with pytest.raises(PublicationFailedError) as excinfo:
        _publish(client)

    assert excinfo.value.outcome.status == HelixJobStatus.ERROR


def test_platformless_run_is_a_failure_not_a_crash() -> None:
    outcome = _publish(None, required=False)
    assert outcome.status == HelixJobStatus.ERROR
    assert "platformless" in (outcome.error or "")


def test_platformless_run_fails_the_job_when_required() -> None:
    with pytest.raises(PublicationFailedError):
        _publish(None)


# --- job wiring -------------------------------------------------------------


class _FakeEvaluator:
    """Stand-in for AgentEvaluator returning one completed trial per task."""

    def __init__(self, *, started_at: datetime | None = STARTED_AT) -> None:
        self._started_at = started_at
        self.loop: asyncio.AbstractEventLoop | None = None

    def run_sync(self, *, tasks: Sequence[AgentEvalTask], **kwargs: Any) -> AgentEvalResult:
        # Drives a real loop to completion the way `AgentEvaluator.run_sync` does, so publication
        # afterwards runs against an already-closed loop.
        return run_sync(lambda: self._run(tasks))

    async def _run(self, tasks: Sequence[AgentEvalTask]) -> AgentEvalResult:
        self.loop = asyncio.get_running_loop()
        return AgentEvalResult(
            run_id="run-1",
            tasks=list(tasks),
            trials=[
                AgentEvalTrial(
                    id=f"{task.id}:trial",
                    task_id=task.id,
                    status=AgentEvalTrialStatus.COMPLETED,
                    output=AgentOutput(output_text="4"),
                )
                for task in tasks
            ],
            scores=[
                AgentEvalTaskScore(
                    id=f"{task.id}:score",
                    run_id="run-1",
                    task_id=task.id,
                    trial_id=f"{task.id}:trial",
                    metric_type="accuracy",
                    status=AgentEvalScoreStatus.COMPLETED,
                    outputs=[MetricOutput(name="score", value=1.0)],
                )
                for task in tasks
            ],
            summary=AgentEvalSummary(),
            metadata=RunMetadata(started_at=self._started_at),
        )


def _job_context(tmp_path: Path, *, job_id: str | None = None) -> JobContext:
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    return JobContext(
        workspace="default",
        storage=storage,
        results=LocalJobResults(root=storage.persistent / "results"),
        job_id=job_id,
    )


def _job_spec(*, required: bool = True) -> AgentEvalSpec:
    return AgentEvalSpec(
        tasks=[ResolvedTask(id="task-1", spec=ResolvedEvaluatorTaskDefinition(kind="evaluator", intent="Answer."))],
        target=FabricRunnerTarget(source=FabricConfigSource(config={}, model="p/m")),
        publication=PublicationSpec(
            intake=IntakePublicationSpec(evaluation_id="eval-1", agent_name="a", required=required)
        ),
    )


def test_job_does_not_publish_without_a_publication_spec(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    mocker.patch.object(AsyncAgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator())
    client = intake.client()

    spec = AgentEvalSpec(
        tasks=[ResolvedTask(id="task-1", spec=ResolvedEvaluatorTaskDefinition(kind="evaluator", intent="Answer."))],
        target=FabricRunnerTarget(source=FabricConfigSource(config={})),
    )
    result = AsyncAgentEvalJob().run(spec.model_dump(), ctx=_job_context(tmp_path), async_client=client)

    assert "publication" not in result
    assert intake.atif_requests == []


def test_job_publishes_through_the_real_sync_bridge(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    evaluator = _FakeEvaluator()
    mocker.patch.object(AsyncAgentEvalJob, "_build_evaluator", return_value=evaluator)
    client = intake.client()
    ctx = _job_context(tmp_path)

    result = AsyncAgentEvalJob().run(_job_spec().model_dump(), ctx=ctx, async_client=client)

    # The evaluator drove a loop to completion first; publication then ran on a different one,
    # reusing the same injected SDK. That crossing is what raises "Event loop is closed" when the
    # client is bound to a dead loop (cf. nhx-1hr.2). It does not distinguish `run_sync` from a bare
    # `asyncio.run` — no loop is running at this point, so both behave the same here.
    ingest_loop = intake.atif_loops[0]
    assert evaluator.loop is not None
    assert evaluator.loop.is_closed()
    assert ingest_loop is not None
    assert ingest_loop is not evaluator.loop

    assert result["status"] == HelixJobStatus.COMPLETED
    assert result["publication"] == {
        "status": HelixJobStatus.COMPLETED,
        "evaluation_id": "eval-1",
        "trial_count": 1,
        "evaluator_result_count": 1,
        "skipped": [],
    }
    assert len(intake.atif_requests) == 1


def test_job_keeps_the_bundle_when_required_publication_fails(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    # Publication is the only step that can fail the job, so it runs last: the bundle and summary
    # artifacts must survive for a later re-publish.
    mocker.patch.object(AsyncAgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator())
    intake.evaluation_exists = False
    client = intake.client()
    ctx = _job_context(tmp_path)

    with pytest.raises(PublicationFailedError):
        AsyncAgentEvalJob().run(_job_spec().model_dump(), ctx=ctx, async_client=client)

    assert (ctx.storage.persistent / "agent-eval" / "trials.jsonl").exists()
    assert (ctx.storage.persistent / "results" / "agent-eval-results").exists()


def test_job_completes_when_optional_publication_fails(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    mocker.patch.object(AsyncAgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator())
    intake.evaluation_exists = False
    client = intake.client()

    result = AsyncAgentEvalJob().run(
        _job_spec(required=False).model_dump(),
        ctx=_job_context(tmp_path),
        async_client=client,
    )

    assert result["status"] == HelixJobStatus.COMPLETED
    assert result["publication"]["status"] == HelixJobStatus.ERROR
    assert "does not exist" in result["publication"]["error"]


def test_job_publication_requires_a_run_start_time(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    # Without `started_at` the trajectory would fall back to Intake's per-request ingest clock, and
    # re-publishing would duplicate spans instead of replacing them. Refuse rather than write rows
    # that can never be collapsed.
    mocker.patch.object(AsyncAgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator(started_at=None))

    result = AsyncAgentEvalJob().run(
        _job_spec(required=False).model_dump(),
        ctx=_job_context(tmp_path),
        async_client=intake.client(),
    )

    assert result["publication"]["status"] == HelixJobStatus.ERROR
    assert "started_at" in result["publication"]["error"]


# --- dataset-driven (row) eval wiring ---------------------------------------


class _FakeRowEvaluator:
    """Stand-in for the row Evaluator, returning one scored row."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None

    def run_sync(self, **kwargs: Any) -> EvaluationResult:
        # Drives a real loop to completion the way the row `Evaluator` does, so publication
        # afterwards runs against an already-closed loop.
        return run_sync(self._run)

    async def _run(self) -> EvaluationResult:
        self.loop = asyncio.get_running_loop()
        return EvaluationResult(
            row_scores=[
                RowScore(
                    row_index=0,
                    item={"question": "2+2?", "qid": "q-1"},
                    sample={"output_text": "4", "response": {}},
                    metrics={"exact_match": [MetricOutput(name="score", value=1.0)]},
                    requests=[],
                )
            ],
            aggregate_scores=AggregatedMetricResult(scores=[]),
        )


def _evaluate_spec(*, required: bool = True, **intake: Any) -> EvaluateSpec:
    return EvaluateSpec(
        metrics=[_INLINE_METRIC],
        dataset=[{"question": "2+2?", "qid": "q-1"}],
        target=Model(name="gpt-4o", url="http://model"),
        params=RunConfigOnlineModel(),
        publication=RowPublicationSpec(
            intake=RowIntakePublicationSpec(evaluation_id="eval-1", agent_name="a", required=required, **intake)
        ),
    )


def test_evaluate_job_does_not_publish_without_a_publication_spec(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    mocker.patch("nemo_evals.jobs.utils.Evaluator", return_value=_FakeRowEvaluator())
    client = intake.client()

    spec = EvaluateSpec(metrics=[_INLINE_METRIC], dataset=[{"question": "2+2?"}])
    result = AsyncEvaluateJob().run(spec.model_dump(), ctx=_job_context(tmp_path, job_id="job-1"), async_client=client)

    assert "publication" not in result
    assert intake.atif_requests == []


def test_evaluate_job_persists_the_run_identity_it_published_under(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    # `EvaluationResult` carries no timings, so without this artifact a re-publish would have to mint
    # a new `started_at` — a different span `start_time` for the same session, which writes a second
    # trajectory rather than replacing the first.
    mocker.patch("nemo_evals.jobs.utils.Evaluator", return_value=_FakeRowEvaluator())
    client = intake.client()
    ctx = _job_context(tmp_path, job_id="job-1")

    AsyncEvaluateJob().run(_evaluate_spec().model_dump(), ctx=ctx, async_client=client)

    # Registered as its own artifact, like the sibling score files — a re-publish should not have to
    # unpack the artifacts directory to recover the identity it must reuse.
    registered = ctx.storage.persistent / "results" / "run-metadata"
    assert registered.exists()

    persisted = json.loads((ctx.storage.persistent / "artifacts" / "run-metadata.json").read_text())
    assert persisted["run_id"] == "job-1"
    published_session = intake.atif_requests[0]["session_id"]
    assert published_session.startswith(f"{persisted['run_id']}:")
    assert datetime.fromisoformat(intake.atif_requests[0]["steps"][0]["timestamp"]) == datetime.fromisoformat(
        persisted["started_at"]
    )


def test_evaluate_job_publishes_rows_through_the_real_sync_bridge(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    evaluator = _FakeRowEvaluator()
    mocker.patch("nemo_evals.jobs.utils.Evaluator", return_value=evaluator)
    client = intake.client()

    result = AsyncEvaluateJob().run(
        _evaluate_spec().model_dump(),
        ctx=_job_context(tmp_path, job_id="job-1"),
        async_client=client,
    )

    # The evaluator drove a loop to completion first; publication then ran on a different one,
    # reusing the same injected SDK. That crossing is what raises "Event loop is closed" when the
    # client is bound to a dead loop. It does not distinguish `run_sync` from a bare `asyncio.run` —
    # no loop is running at this point, so both behave the same here.
    ingest_loop = intake.atif_loops[0]
    assert evaluator.loop is not None
    assert evaluator.loop.is_closed()
    assert ingest_loop is not None
    assert ingest_loop is not evaluator.loop

    assert result["publication"]["status"] == HelixJobStatus.COMPLETED
    assert result["publication"]["trial_count"] == 1
    assert len(intake.atif_requests) == 1
    # The run identity is the job id, so re-publishing the same job replaces rather than duplicates.
    # The trial half is the row content hash, stable across runs and dataset reorderings.
    session_id = intake.atif_requests[0]["session_id"]
    assert session_id.startswith("job-1:")
    assert len(session_id.removeprefix("job-1:")) == 64


def test_evaluate_job_uses_the_configured_test_case_id_column(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    mocker.patch("nemo_evals.jobs.utils.Evaluator", return_value=_FakeRowEvaluator())
    client = intake.client()

    AsyncEvaluateJob().run(
        _evaluate_spec(test_case_id_field="qid").model_dump(),
        ctx=_job_context(tmp_path, job_id="job-1"),
        async_client=client,
    )

    assert intake.atif_requests[0]["session_id"] == "job-1:q-1"
    assert intake.atif_requests[0]["evaluation_context"]["test_case_name"] == "q-1"


def test_evaluate_job_without_a_job_id_cannot_publish(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    # A row result carries no run id of its own, so without a job id there is nothing stable to key
    # sessions on.
    mocker.patch("nemo_evals.jobs.utils.Evaluator", return_value=_FakeRowEvaluator())
    client = intake.client()

    result = AsyncEvaluateJob().run(
        _evaluate_spec(required=False).model_dump(),
        ctx=_job_context(tmp_path, job_id=None),
        async_client=client,
    )

    assert result["publication"]["status"] == HelixJobStatus.ERROR
    assert "job id" in result["publication"]["error"]
    assert intake.atif_requests == []


def test_evaluate_job_reports_a_bad_test_case_id_column(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    mocker.patch("nemo_evals.jobs.utils.Evaluator", return_value=_FakeRowEvaluator())
    client = intake.client()

    result = AsyncEvaluateJob().run(
        _evaluate_spec(required=False, test_case_id_field="missing").model_dump(),
        ctx=_job_context(tmp_path, job_id="job-1"),
        async_client=client,
    )

    assert result["publication"]["status"] == HelixJobStatus.ERROR
    assert "missing" in result["publication"]["error"]
    assert intake.atif_requests == []


def test_row_target_without_a_derivable_agent_name_is_rejected_at_submit() -> None:
    # A Model target names a model, not an agent. Without this the run would publish every
    # trajectory under an empty agent name.
    with pytest.raises(ValidationError, match="agent_name` is required"):
        EvaluateInputSpec(
            metrics=[_INLINE_METRIC],
            dataset=[{"question": "2+2?"}],
            target=Model(name="gpt-4o", url="http://model"),
            params=RunConfigOnlineModel(),
            publication=RowPublicationSpec(intake=RowIntakePublicationSpec(evaluation_id="eval-1")),
        )


def test_row_agent_target_derives_its_name() -> None:
    spec = EvaluateInputSpec(
        metrics=[_INLINE_METRIC],
        dataset=[{"question": "2+2?"}],
        target=NemoAgentToolkitAgent(name="my-agent", url="http://agent"),
        params=RunConfigOnline(),
        prompt_template="{{item.question}}",
        publication=RowPublicationSpec(intake=RowIntakePublicationSpec(evaluation_id="eval-1")),
    )
    assert target_agent_identity(spec.target) == ("my-agent", None)


# --- OTLP request budget ------------------------------------------------------


@pytest.mark.parametrize("spec_type", [IntakePublicationSpec, RowIntakePublicationSpec])
def test_otlp_request_budget_defaults_to_4_mib(spec_type: type[IntakePublicationSpec]) -> None:
    assert spec_type(evaluation_id="eval-1").max_otlp_request_bytes == 4 * 1024 * 1024


@pytest.mark.parametrize("spec_type", [IntakePublicationSpec, RowIntakePublicationSpec])
@pytest.mark.parametrize("budget", [0, -1])
def test_a_non_positive_otlp_request_budget_is_rejected(spec_type: type[IntakePublicationSpec], budget: int) -> None:
    with pytest.raises(ValidationError, match="max_otlp_request_bytes"):
        spec_type(evaluation_id="eval-1", max_otlp_request_bytes=budget)


def test_agent_publication_splits_otlp_traces_at_the_spec_budget(intake: InMemoryIntakeService) -> None:
    trace_id, root_id = "0123456789abcdef0123456789abcdef", "0102030405060708"
    spans = [{"traceId": trace_id, "spanId": root_id, "name": "agent run"}] + [
        {
            "traceId": trace_id,
            "spanId": f"{index:016x}",
            "parentSpanId": root_id,
            "name": f"child-{index}",
            "attributes": [{"key": "payload", "value": {"stringValue": "x" * 1000}}],
        }
        for index in range(1, 4)
    ]
    trace = EvidenceDescriptor(
        kind="trace", format="otlp", data={"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}
    )
    result = _result()
    result = result.model_copy(
        update={
            "trials": [
                result.trials[0].model_copy(update={"evidence": CandidateEvidence(descriptors={"trace": trace})})
            ]
        }
    )

    outcome = publish_agent_eval_result(
        result,
        spec=IntakePublicationSpec(evaluation_id="eval-1", agent_name="a", max_otlp_request_bytes=2500),
        target=None,
        workspace="default",
        intake=intake.client(),
    )

    assert outcome.status == HelixJobStatus.COMPLETED
    bodies = [request["body"] for request in intake.otlp_requests]
    assert len(bodies) > 1
    assert all(isinstance(body, bytes) and len(body) <= 2500 for body in bodies)


def test_evaluate_job_forwards_the_otlp_request_budget(
    tmp_path: Path, mocker: MockerFixture, intake: InMemoryIntakeService
) -> None:
    mocker.patch("nemo_evals.jobs.utils.Evaluator", return_value=_FakeRowEvaluator())
    forwarded = mocker.patch("nemo_evals.jobs.publication.publish_to_intake", wraps=publish_to_intake)

    AsyncEvaluateJob().run(
        _evaluate_spec(max_otlp_request_bytes=2_097_152).model_dump(),
        ctx=_job_context(tmp_path, job_id="job-1"),
        async_client=intake.client(),
    )

    assert forwarded.call_args.kwargs["max_otlp_request_bytes"] == 2_097_152
