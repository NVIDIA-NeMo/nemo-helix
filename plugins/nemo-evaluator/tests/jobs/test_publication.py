# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agent-eval job's Intake publication step."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from nemo_evaluator.api.schemas import MetricInline
from nemo_evaluator.jobs.agent_evaluate import AsyncAgentEvalJob
from nemo_evaluator.jobs.agent_spec import (
    AgentEvalInputSpec,
    AgentEvalSpec,
    AgentEvalTaskInput,
    AgentEvalTaskSpec,
    AgentTarget,
    FabricRunnerTarget,
    GymRunnerTarget,
    HarborRunnerTarget,
    ModelTarget,
    Target,
    target_agent_identity,
)
from nemo_evaluator.jobs.evaluate import AsyncEvaluateJob, EvaluateInputSpec, EvaluateSpec
from nemo_evaluator.jobs.publication import (
    EVAL_DURATION_KEY,
    PUBLISH_DURATION_KEY,
    PublicationFailedError,
    PublicationOutcome,
    publish_agent_eval_result,
)
from nemo_evaluator.jobs.publication_spec import (
    IntakePublicationSpec,
    PublicationSpec,
    RowIntakePublicationSpec,
    RowPublicationSpec,
)
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary, RunMetadata
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nemo_evaluator_sdk.execution.metric_execution import run_sync
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.protocol import MetricOutput
from nemo_evaluator_sdk.values import Model, RunConfigOnline, RunConfigOnlineModel
from nemo_evaluator_sdk.values.agents import NemoAgentToolkitAgent
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult  # noqa: F401
from nemo_evaluator_sdk.values.results import AggregatedMetricResult, EvaluationResult, RowScore
from nemo_platform_plugin.client.errors import NemoTransportError, NotFoundError
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import (
    AtifCreateRequest,
    EvaluationPatchRequest,
    EvaluatorResultCreateRequest,
    ListTracesQueryParams,
)
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from pydantic import ValidationError
from pytest_mock import MockerFixture

STARTED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

_INLINE_METRIC = MetricInline.model_validate(
    bundle_metric(
        ExactMatchMetric(reference="{{item.question}}", candidate="{{item.question}}"),
        CloudpickleMetricBundlePackager(),
    ).model_dump(mode="json")
)

#: Aliased because `_FakeTraces` defines a `list` method, which shadows the builtin in class scope.
_SessionIds = list[str]


def _response(status: int) -> httpx.Response:
    """A real httpx response; typed client errors read ``response.request`` in their constructor."""
    return httpx.Response(status, request=httpx.Request("GET", "http://platform/evaluations"))


# --- fakes ------------------------------------------------------------------


class _FakeTraces:
    def __init__(self, calls: _SessionIds) -> None:
        self._calls = calls

    def list(self, *, workspace: str, filter: dict[str, Any]) -> AsyncIterator[object]:  # noqa: A002
        session_id = filter["session_id"]
        self._calls.append(session_id)

        async def _gen() -> AsyncIterator[object]:
            yield SimpleNamespace(session_id=session_id, root_span_id=f"span:{session_id}")

        return _gen()


class _FakeEvaluations:
    def __init__(
        self,
        *,
        missing: bool = False,
        error: Exception | None = None,
        patch_error: Exception | None = None,
    ) -> None:
        self.missing = missing
        self.error = error
        self.patch_error = patch_error
        self.retrieved: _SessionIds = []
        self.patched: list[dict[str, Any]] = []
        self.metadata: dict[str, str] = {"eval_config_fileset": "fs-1"}

    async def retrieve(self, name: str, *, workspace: str | None = None) -> object:
        self.retrieved.append(name)
        if self.error is not None:
            raise self.error
        if self.missing:
            raise NotFoundError(_response(404))
        return SimpleNamespace(name=name, metadata=dict(self.metadata))

    async def patch(self, name: str, *, workspace: str | None = None, **kwargs: Any) -> object:
        if self.patch_error is not None:
            raise self.patch_error
        self.patched.append({"name": name, **kwargs})
        return SimpleNamespace(name=name)


class _FakeIngest:
    def __init__(self, calls: list[dict[str, Any]], *, error: Exception | None = None, delay_sec: float = 0.0) -> None:
        self._calls = calls
        self._error = error
        self._delay_sec = delay_sec
        self.loop: asyncio.AbstractEventLoop | None = None

    async def create(self, **kwargs: Any) -> None:
        self.loop = asyncio.get_running_loop()
        if self._delay_sec:
            await asyncio.sleep(self._delay_sec)
        if self._error is not None:
            raise self._error
        self._calls.append(kwargs)


class _FakePaginatedResponse:
    def __init__(self, items: AsyncIterator[object]) -> None:
        self._items = items

    def items(self) -> AsyncIterator[object]:
        return self._items


class _FakeClient(AsyncIntakeClient):
    """Minimal stand-in for the typed Intake client publication touches."""

    def __init__(
        self,
        *,
        missing_evaluation: bool = False,
        ingest_error: Exception | None = None,
        preflight_error: Exception | None = None,
        patch_error: Exception | None = None,
        ingest_delay_sec: float = 0.0,
    ) -> None:
        self._workspace = "default"
        self._http = httpx.AsyncClient()
        self.atif_calls: list[dict[str, Any]] = []
        self.eval_result_calls: list[dict[str, Any]] = []
        self.trace_calls: _SessionIds = []
        self.with_http_client_calls = 0
        self.evaluations = _FakeEvaluations(missing=missing_evaluation, error=preflight_error, patch_error=patch_error)
        self._intake = SimpleNamespace(
            ingest=SimpleNamespace(atif=_FakeIngest(self.atif_calls, error=ingest_error, delay_sec=ingest_delay_sec)),
            evaluator_results=_FakeIngest(self.eval_result_calls),
            traces=_FakeTraces(self.trace_calls),
        )

    @property
    def intake(self) -> SimpleNamespace:
        return self._intake

    def with_http_client(self, http_client: httpx.AsyncClient) -> _FakeClient:
        del http_client
        self.with_http_client_calls += 1
        return self

    async def get_evaluation(self, *, workspace: str | None = None, name: str) -> object:
        result = await self.evaluations.retrieve(name, workspace=workspace)
        return SimpleNamespace(data=lambda: result)

    async def patch_evaluation(
        self, *, workspace: str | None = None, name: str, body: EvaluationPatchRequest
    ) -> object:
        result = await self.evaluations.patch(
            name,
            workspace=workspace,
            **body.model_dump(mode="json", exclude_none=True),
        )
        return SimpleNamespace(data=lambda: result)

    async def create_atif(self, *, workspace: str | None = None, body: AtifCreateRequest) -> object:
        payload = dict(body.root)
        payload["workspace"] = workspace or self.workspace
        await self.intake.ingest.atif.create(**payload)
        return SimpleNamespace(data=lambda: None)

    async def create_evaluator_result(
        self, *, workspace: str | None = None, body: EvaluatorResultCreateRequest
    ) -> object:
        payload = body.model_dump(mode="json", exclude_none=True)
        payload["workspace"] = workspace or self.workspace
        result = await self.intake.evaluator_results.create(**payload)
        return SimpleNamespace(data=lambda: result)

    async def list_traces(
        self, *, workspace: str | None = None, query_params: ListTracesQueryParams | None = None
    ) -> _FakePaginatedResponse:
        params = query_params or {}
        return _FakePaginatedResponse(
            self.intake.traces.list(
                workspace=workspace or self.workspace or "", filter=dict(params.get("filter") or {})
            )
        )


def _client(**kwargs: Any) -> _FakeClient:
    return _FakeClient(**kwargs)


def _fake_intake_from_client(client: _FakeClient) -> _FakeClient:
    return client


@pytest.fixture(autouse=True)
def _preserve_fake_intake_clients(mocker: MockerFixture) -> None:
    mocker.patch(
        "nemo_evaluator.jobs.agent_evaluate.AsyncIntakeClient.from_client", side_effect=_fake_intake_from_client
    )
    mocker.patch("nemo_evaluator.jobs.evaluate.AsyncIntakeClient.from_client", side_effect=_fake_intake_from_client)


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


def _publish(client: _FakeClient | None, *, required: bool = True, agent_name: str | None = "a") -> PublicationOutcome:
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
        (HarborRunnerTarget(agent_name="oracle", agent_model_name="m"), ("oracle", "m")),
        (HarborRunnerTarget(agent_name="oracle", agent_import_path="pkg:Agent"), ("pkg:Agent", None)),
        (
            GymRunnerTarget(agent="simple_agent", agent_config="conf/agent.yaml", resources_server="mcqa"),
            ("simple_agent", None),
        ),
        (FabricRunnerTarget(config={}, model="p/m"), (None, "p/m")),
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


def test_publication_defaults_to_required() -> None:
    spec = _input_spec(
        AgentTarget(agent=NemoAgentToolkitAgent(name="a", url="http://a")),
        PublicationSpec(intake=IntakePublicationSpec(evaluation_id="eval-1")),
    )
    assert spec.publication is not None
    assert spec.publication.intake is not None
    assert spec.publication.intake.required is True


def test_agent_name_derived_from_agent_target_needs_no_override() -> None:
    spec = _input_spec(
        AgentTarget(agent=NemoAgentToolkitAgent(name="derived", url="http://a")),
        PublicationSpec(intake=IntakePublicationSpec(evaluation_id="eval-1")),
    )
    assert spec.publication is not None
    assert spec.publication.intake is not None
    assert spec.publication.intake.agent_name is None
    assert target_agent_identity(spec.target)[0] == "derived"


def test_agent_name_derived_from_gym_target_needs_no_override() -> None:
    spec = _input_spec(
        GymRunnerTarget(agent="simple_agent", agent_config="conf/agent.yaml", resources_server="mcqa"),
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
        FabricRunnerTarget(config={}),
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
    [ModelTarget(model=Model(name="gpt-4o", url="http://model")), FabricRunnerTarget(config={}), None],
)
def test_explicit_agent_name_satisfies_undeducible_targets(target: Target | None) -> None:
    spec = _input_spec(
        target, PublicationSpec(intake=IntakePublicationSpec(evaluation_id="eval-1", agent_name="explicit"))
    )
    assert spec.publication is not None


# --- publishing -------------------------------------------------------------


def test_publishes_and_reports_what_landed() -> None:
    client = _FakeClient()
    outcome = _publish(client)

    assert client.with_http_client_calls == 1
    assert outcome.status == PlatformJobStatus.COMPLETED
    assert outcome.evaluation_id == "eval-1"
    assert outcome.trial_count == 1
    assert outcome.evaluator_result_count == 1
    assert outcome.error is None
    # Twice: this path needs the entity to stamp durations onto, and ``publish_to_intake``
    # re-checks for itself because it is public and cannot assume its caller did.
    assert client.evaluations.retrieved == ["eval-1", "eval-1"]


def test_durations_are_stamped_without_dropping_existing_metadata() -> None:
    client = _FakeClient()
    _publish(client)

    (patched,) = client.evaluations.patched
    metadata = patched["metadata"]
    # PATCH replaces the metadata dict wholesale, so stamping the durations has to merge with what
    # the producer already wrote. A blind write would silently drop `eval_config_fileset`.
    assert metadata["eval_config_fileset"] == "fs-1"
    assert float(metadata[EVAL_DURATION_KEY]) > 0
    assert float(metadata[PUBLISH_DURATION_KEY]) >= 0


def test_a_started_at_only_result_does_not_count_publish_time_as_run_time() -> None:
    # A row-eval result carries no `duration_sec`, so the run's length is derived from `started_at`.
    # Deriving it after the publish instead of before would fold the publish into the run.
    publish_sec = 0.3
    client = _FakeClient(ingest_delay_sec=publish_sec)
    result = _result()
    result.metadata.started_at = datetime.now(UTC)

    publish_agent_eval_result(
        result,
        spec=IntakePublicationSpec(evaluation_id="eval-1", agent_name="a", required=True),
        target=None,
        workspace="default",
        intake=client,
    )

    (patched,) = client.evaluations.patched
    metadata = patched["metadata"]
    assert float(metadata[PUBLISH_DURATION_KEY]) >= publish_sec
    assert float(metadata[EVAL_DURATION_KEY]) < publish_sec / 2


def test_a_failed_duration_stamp_does_not_fail_the_publish() -> None:
    client = _FakeClient(
        patch_error=NemoTransportError(httpx.ConnectError("patch failed", request=httpx.Request("PATCH", "http://x")))
    )
    outcome = _publish(client, required=True)

    # The durations are informational and the trials already landed, so losing them must not turn a
    # successful publish into a failed job — which is what every caller-side handler would do.
    assert outcome.status == PlatformJobStatus.COMPLETED
    assert outcome.trial_count == 1
    assert outcome.error is None


def test_outcome_does_not_leak_experiment_id() -> None:
    outcome = _publish(_client())
    assert "experiment_id" not in outcome.model_dump()


def test_stamps_the_run_start_time_so_republish_is_idempotent() -> None:
    client = _FakeClient()
    _publish(client)
    assert client.atif_calls[0]["steps"][0]["timestamp"] == STARTED_AT


def test_missing_evaluation_fails_before_any_ingest() -> None:
    client = _FakeClient(missing_evaluation=True)
    with pytest.raises(PublicationFailedError) as excinfo:
        _publish(client)

    assert client.atif_calls == []
    outcome = excinfo.value.outcome
    assert outcome.status == PlatformJobStatus.ERROR
    assert "does not exist" in (outcome.error or "")


def test_required_failure_raises_with_partial_outcome() -> None:
    client = _FakeClient(
        ingest_error=NemoTransportError(
            httpx.ConnectError("ingest failed", request=httpx.Request("POST", "http://platform/intake"))
        )
    )
    with pytest.raises(PublicationFailedError) as excinfo:
        _publish(client)

    outcome = excinfo.value.outcome
    assert outcome.status == PlatformJobStatus.ERROR
    assert outcome.trial_count == 0


def test_optional_failure_returns_outcome_instead_of_raising() -> None:
    client = _FakeClient(
        ingest_error=NemoTransportError(
            httpx.ConnectError("ingest failed", request=httpx.Request("POST", "http://platform/intake"))
        )
    )
    outcome = _publish(client, required=False)

    assert outcome.status == PlatformJobStatus.ERROR
    assert outcome.error


def test_unexpected_failure_still_honours_required_false() -> None:
    # `required=False` promises the evaluation survives a failed publish. An error outside the known
    # taxonomy — a bug, a transport quirk, anything unforeseen — must not be the one case that
    # escapes and fails the job anyway.
    client = _FakeClient(preflight_error=ValueError("something nobody planned for"))
    outcome = _publish(client, required=False)

    assert outcome.status == PlatformJobStatus.ERROR
    assert "ValueError" in (outcome.error or "")
    assert "something nobody planned for" in (outcome.error or "")


def test_unexpected_failure_fails_the_job_when_required() -> None:
    client = _FakeClient(preflight_error=ValueError("something nobody planned for"))
    with pytest.raises(PublicationFailedError) as excinfo:
        _publish(client)

    assert excinfo.value.outcome.status == PlatformJobStatus.ERROR


def test_platformless_run_is_a_failure_not_a_crash() -> None:
    outcome = _publish(None, required=False)
    assert outcome.status == PlatformJobStatus.ERROR
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
            scores=[],
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
        tasks=[AgentEvalTaskSpec(id="task-1", intent="Answer.")],
        target=FabricRunnerTarget(config={}, model="p/m"),
        publication=PublicationSpec(
            intake=IntakePublicationSpec(evaluation_id="eval-1", agent_name="a", required=required)
        ),
    )


def test_job_does_not_publish_without_a_publication_spec(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.object(AsyncAgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator())
    client = _FakeClient()

    spec = AgentEvalSpec(tasks=[AgentEvalTaskSpec(id="task-1", intent="Answer.")], target=FabricRunnerTarget(config={}))
    result = AsyncAgentEvalJob().run(spec.model_dump(), ctx=_job_context(tmp_path), async_client=client)

    assert "publication" not in result
    assert client.atif_calls == []


def test_job_publishes_through_the_real_sync_bridge(tmp_path: Path, mocker: MockerFixture) -> None:
    evaluator = _FakeEvaluator()
    mocker.patch.object(AsyncAgentEvalJob, "_build_evaluator", return_value=evaluator)
    client = _FakeClient()
    ctx = _job_context(tmp_path)

    result = AsyncAgentEvalJob().run(_job_spec().model_dump(), ctx=ctx, async_client=client)

    # The evaluator drove a loop to completion first; publication then ran on a different one,
    # reusing the same injected SDK. That crossing is what raises "Event loop is closed" when the
    # client is bound to a dead loop (cf. nmp-1hr.2). It does not distinguish `run_sync` from a bare
    # `asyncio.run` — no loop is running at this point, so both behave the same here.
    ingest_loop = client.intake.ingest.atif.loop
    assert evaluator.loop is not None
    assert evaluator.loop.is_closed()
    assert ingest_loop is not None
    assert ingest_loop is not evaluator.loop

    assert result["status"] == PlatformJobStatus.COMPLETED
    assert result["publication"] == {
        "status": PlatformJobStatus.COMPLETED,
        "evaluation_id": "eval-1",
        "trial_count": 1,
        "evaluator_result_count": 0,
        "skipped": [],
    }
    assert len(client.atif_calls) == 1


def test_job_keeps_the_bundle_when_required_publication_fails(tmp_path: Path, mocker: MockerFixture) -> None:
    # Publication is the only step that can fail the job, so it runs last: the bundle and summary
    # artifacts must survive for a later re-publish.
    mocker.patch.object(AsyncAgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator())
    client = _FakeClient(missing_evaluation=True)
    ctx = _job_context(tmp_path)

    with pytest.raises(PublicationFailedError):
        AsyncAgentEvalJob().run(_job_spec().model_dump(), ctx=ctx, async_client=client)

    assert (ctx.storage.persistent / "agent-eval" / "trials.jsonl").exists()
    assert (ctx.storage.persistent / "results" / "agent-eval-results").exists()


def test_job_completes_when_optional_publication_fails(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.object(AsyncAgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator())
    client = _FakeClient(missing_evaluation=True)

    result = AsyncAgentEvalJob().run(
        _job_spec(required=False).model_dump(),
        ctx=_job_context(tmp_path),
        async_client=client,
    )

    assert result["status"] == PlatformJobStatus.COMPLETED
    assert result["publication"]["status"] == PlatformJobStatus.ERROR
    assert "does not exist" in result["publication"]["error"]


def test_job_publication_requires_a_run_start_time(tmp_path: Path, mocker: MockerFixture) -> None:
    # Without `started_at` the trajectory would fall back to Intake's per-request ingest clock, and
    # re-publishing would duplicate spans instead of replacing them. Refuse rather than write rows
    # that can never be collapsed.
    mocker.patch.object(AsyncAgentEvalJob, "_build_evaluator", return_value=_FakeEvaluator(started_at=None))

    result = AsyncAgentEvalJob().run(
        _job_spec(required=False).model_dump(),
        ctx=_job_context(tmp_path),
        async_client=_FakeClient(),
    )

    assert result["publication"]["status"] == PlatformJobStatus.ERROR
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


def test_evaluate_job_does_not_publish_without_a_publication_spec(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=_FakeRowEvaluator())
    client = _FakeClient()

    spec = EvaluateSpec(metrics=[_INLINE_METRIC], dataset=[{"question": "2+2?"}])
    result = AsyncEvaluateJob().run(spec.model_dump(), ctx=_job_context(tmp_path, job_id="job-1"), async_client=client)

    assert "publication" not in result
    assert client.atif_calls == []


def test_evaluate_job_persists_the_run_identity_it_published_under(tmp_path: Path, mocker: MockerFixture) -> None:
    # `EvaluationResult` carries no timings, so without this artifact a re-publish would have to mint
    # a new `started_at` — a different span `start_time` for the same session, which writes a second
    # trajectory rather than replacing the first.
    mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=_FakeRowEvaluator())
    client = _FakeClient()
    ctx = _job_context(tmp_path, job_id="job-1")

    AsyncEvaluateJob().run(_evaluate_spec().model_dump(), ctx=ctx, async_client=client)

    # Registered as its own artifact, like the sibling score files — a re-publish should not have to
    # unpack the artifacts directory to recover the identity it must reuse.
    registered = ctx.storage.persistent / "results" / "run-metadata"
    assert registered.exists()

    persisted = json.loads((ctx.storage.persistent / "artifacts" / "run-metadata.json").read_text())
    assert persisted["run_id"] == "job-1"
    published_session = client.atif_calls[0]["session_id"]
    assert published_session.startswith(f"{persisted['run_id']}:")
    assert client.atif_calls[0]["steps"][0]["timestamp"] == datetime.fromisoformat(persisted["started_at"])


def test_evaluate_job_publishes_rows_through_the_real_sync_bridge(tmp_path: Path, mocker: MockerFixture) -> None:
    evaluator = _FakeRowEvaluator()
    mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=evaluator)
    client = _FakeClient()

    result = AsyncEvaluateJob().run(
        _evaluate_spec().model_dump(),
        ctx=_job_context(tmp_path, job_id="job-1"),
        async_client=client,
    )

    # The evaluator drove a loop to completion first; publication then ran on a different one,
    # reusing the same injected SDK. That crossing is what raises "Event loop is closed" when the
    # client is bound to a dead loop. It does not distinguish `run_sync` from a bare `asyncio.run` —
    # no loop is running at this point, so both behave the same here.
    ingest_loop = client.intake.ingest.atif.loop
    assert evaluator.loop is not None
    assert evaluator.loop.is_closed()
    assert ingest_loop is not None
    assert ingest_loop is not evaluator.loop

    assert result["publication"]["status"] == PlatformJobStatus.COMPLETED
    assert result["publication"]["trial_count"] == 1
    assert len(client.atif_calls) == 1
    # The run identity is the job id, so re-publishing the same job replaces rather than duplicates.
    # The trial half is the row content hash, stable across runs and dataset reorderings.
    session_id = client.atif_calls[0]["session_id"]
    assert session_id.startswith("job-1:")
    assert len(session_id.removeprefix("job-1:")) == 64


def test_evaluate_job_uses_the_configured_test_case_id_column(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=_FakeRowEvaluator())
    client = _FakeClient()

    AsyncEvaluateJob().run(
        _evaluate_spec(test_case_id_field="qid").model_dump(),
        ctx=_job_context(tmp_path, job_id="job-1"),
        async_client=client,
    )

    assert client.atif_calls[0]["session_id"] == "job-1:q-1"
    assert client.atif_calls[0]["evaluation_context"]["test_case_name"] == "q-1"


def test_evaluate_job_without_a_job_id_cannot_publish(tmp_path: Path, mocker: MockerFixture) -> None:
    # A row result carries no run id of its own, so without a job id there is nothing stable to key
    # sessions on.
    mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=_FakeRowEvaluator())
    client = _FakeClient()

    result = AsyncEvaluateJob().run(
        _evaluate_spec(required=False).model_dump(),
        ctx=_job_context(tmp_path, job_id=None),
        async_client=client,
    )

    assert result["publication"]["status"] == PlatformJobStatus.ERROR
    assert "job id" in result["publication"]["error"]
    assert client.atif_calls == []


def test_evaluate_job_reports_a_bad_test_case_id_column(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch("nemo_evaluator.jobs.evaluate.Evaluator", return_value=_FakeRowEvaluator())
    client = _FakeClient()

    result = AsyncEvaluateJob().run(
        _evaluate_spec(required=False, test_case_id_field="missing").model_dump(),
        ctx=_job_context(tmp_path, job_id="job-1"),
        async_client=client,
    )

    assert result["publication"]["status"] == PlatformJobStatus.ERROR
    assert "missing" in result["publication"]["error"]
    assert client.atif_calls == []


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
