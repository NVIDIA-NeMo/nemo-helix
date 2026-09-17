# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for publish_to_intake — the explicit Evaluator -> Intake publish step."""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from nemo_evaluator.intake.publish import PublishError, _token_final_metrics, publish_to_intake
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary, RunMetadata
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    TrialError,
    TrialMeasurements,
)
from nemo_evaluator_sdk.metrics.protocol import MetricOutput
from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from nemo_platform_plugin.client.errors import NotFoundError, UnprocessableEntityError
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import (
    AtifCreateRequest,
    EvaluatorResultCreateRequest,
    IngestResponse,
    ListTracesQueryParams,
)
from pydantic import ValidationError

# --- fakes ------------------------------------------------------------------


class _FakeAtif:
    def __init__(self, calls: list[dict[str, Any]], *, fail: bool = False) -> None:
        self._calls = calls
        self._fail = fail

    async def create(self, **kwargs: Any) -> None:
        if self._fail:
            raise RuntimeError("atif ingest 400")
        self._calls.append(kwargs)


class _FakeOtlpTraces:
    def __init__(self, calls: list[dict[str, Any]], *, errors: list[str] | None = None) -> None:
        self._calls = calls
        self._errors = errors or []

    async def create(self, **kwargs: Any) -> object:
        self._calls.append(kwargs)
        return SimpleNamespace(data=lambda: IngestResponse(errors=list(self._errors)))


class _FakeEvaluatorResults:
    def __init__(self, calls: list[dict[str, Any]], *, fail_session: str | None = None) -> None:
        self._calls = calls
        self._fail_session = fail_session

    async def create(self, **kwargs: Any) -> object:
        if self._fail_session is not None and kwargs.get("session_id") == self._fail_session:
            raise RuntimeError(f"evaluator-results 500 for {kwargs['session_id']}")
        self._calls.append(kwargs)
        return SimpleNamespace(evaluator_result_id="eval-1")


class _FakeTraces:
    """Returns one root-span trace per requested session id (or none, to test resolution failure)."""

    def __init__(self, *, root_span_id: str | None) -> None:
        self._root_span_id = root_span_id

    def list(self, *, workspace: str, filter: dict[str, Any]) -> AsyncIterator[object]:  # noqa: A002
        root_span_id = self._root_span_id
        session_id = filter["session_id"]

        async def _gen() -> AsyncIterator[object]:
            if root_span_id is not None:
                yield SimpleNamespace(session_id=session_id, root_span_id=f"{root_span_id}:{session_id}")

        return _gen()


class _FakeEvaluations:
    """Resolves the run's evaluation, or 404s the way ingest does for a missing or deleted one."""

    def __init__(self, calls: list[dict[str, Any]], *, missing: bool = False) -> None:
        self._calls = calls
        self._missing = missing

    async def get(self, name: str, *, workspace: str) -> object:
        self._calls.append({"name": name, "workspace": workspace})
        if self._missing:
            raise NotFoundError(
                httpx.Response(404, request=httpx.Request("GET", "http://intake/evaluations")),
            )
        return SimpleNamespace(data=lambda: SimpleNamespace(name=name))


class _FakePaginatedResponse:
    def __init__(self, items: AsyncIterator[object]) -> None:
        self._items = items

    def items(self) -> AsyncIterator[object]:
        return self._items


class _FakeClient(AsyncIntakeClient):
    def __init__(
        self,
        *,
        workspace: str | None = "default",
        root_span_id: str | None = "span",
        atif_fail: bool = False,
        fail_eval_session: str | None = None,
        atif: Any | None = None,
        otlp_errors: list[str] | None = None,
        evaluation_missing: bool = False,
    ) -> None:
        self._workspace = workspace
        self.evaluation_calls: list[dict[str, Any]] = []
        self._evaluations = _FakeEvaluations(self.evaluation_calls, missing=evaluation_missing)
        self.atif_calls: list[dict[str, Any]] = []
        self.otlp_calls: list[dict[str, Any]] = []
        self.eval_calls: list[dict[str, Any]] = []
        self._atif = atif or _FakeAtif(self.atif_calls, fail=atif_fail)
        self._otlp_traces = _FakeOtlpTraces(self.otlp_calls, errors=otlp_errors)
        self._evaluator_results = _FakeEvaluatorResults(self.eval_calls, fail_session=fail_eval_session)
        self._traces = _FakeTraces(root_span_id=root_span_id)

    async def create_atif(self, *, workspace: str | None = None, body: AtifCreateRequest) -> object:
        payload = dict(body.root)
        payload["workspace"] = workspace or self.workspace
        await self._atif.create(**payload)
        return SimpleNamespace(data=lambda: None)

    async def create_otlp_traces(
        self,
        *,
        workspace: str | None = None,
        content: bytes | Iterable[bytes] | AsyncIterable[bytes],
    ) -> object:
        return await self._otlp_traces.create(workspace=workspace or self.workspace, body=content)

    async def create_evaluator_result(
        self, *, workspace: str | None = None, body: EvaluatorResultCreateRequest
    ) -> object:
        payload = body.model_dump(mode="json", exclude_none=True)
        payload["workspace"] = workspace or self.workspace
        result = await self._evaluator_results.create(**payload)
        return SimpleNamespace(data=lambda: result)

    async def list_traces(
        self, *, workspace: str | None = None, query_params: ListTracesQueryParams | None = None
    ) -> _FakePaginatedResponse:
        params = query_params or {}
        return _FakePaginatedResponse(
            self._traces.list(workspace=workspace or self.workspace or "", filter=dict(params.get("filter") or {}))
        )

    async def get_evaluation(self, *, workspace: str | None = None, name: str) -> object:
        return await self._evaluations.get(name, workspace=workspace or self.workspace or "")


def _client(**kwargs: Any) -> _FakeClient:
    return _FakeClient(**kwargs)


# --- fixtures ---------------------------------------------------------------


def _trial(trial_id: str, task_id: str = "task-1") -> AgentEvalTrial:
    return AgentEvalTrial(
        id=trial_id,
        task_id=task_id,
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text="answer"),
    )


def _score(
    trial_id: str,
    metric_type: str,
    outputs: list[MetricOutput],
    status: AgentEvalScoreStatus = AgentEvalScoreStatus.COMPLETED,
) -> AgentEvalTaskScore:
    return AgentEvalTaskScore(
        id=f"score-{trial_id}-{metric_type}",
        run_id="run-1",
        task_id="task-1",
        trial_id=trial_id,
        metric_type=metric_type,
        status=status,
        outputs=outputs,
    )


STARTED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _result(trials: list[AgentEvalTrial], scores: list[AgentEvalTaskScore]) -> AgentEvalResult:
    return AgentEvalResult(
        run_id="run-1",
        tasks=[],
        trials=trials,
        scores=scores,
        summary=AgentEvalSummary(),
        metadata=RunMetadata(started_at=STARTED_AT),
    )


def test_token_final_metrics_projects_recorded_token_usage() -> None:
    # Recorded token usage becomes ATIF final_metrics, which Intake promotes onto the root span.
    measurements = TrialMeasurements(prompt_tokens=120, completion_tokens=45, cache_read_tokens=30, cost_usd=0.134)
    assert _token_final_metrics(measurements) == {
        "total_prompt_tokens": 120,
        "total_completion_tokens": 45,
        "total_cached_tokens": 30,
        "total_cost_usd": 0.134,
    }


def test_token_final_metrics_carries_cost_without_token_counts() -> None:
    # A harness that reports spend but not usage still gets its cost onto the root span.
    assert _token_final_metrics(TrialMeasurements(cost_usd=0.5)) == {"total_cost_usd": 0.5}


def test_token_final_metrics_is_none_without_recorded_usage() -> None:
    # No recorded token usage → no final_metrics block (rather than zeros).
    assert _token_final_metrics(TrialMeasurements()) is None


# --- tests ------------------------------------------------------------------


async def test_publishes_trajectory_and_scores() -> None:
    result = _result(
        trials=[_trial("t-1")],
        scores=[
            _score("t-1", "accuracy", [MetricOutput(name="score", value=0.5), MetricOutput(name="passed", value=True)]),
            _score("t-1", "latency", [MetricOutput(name="p50", value=1.2)]),
        ],
    )
    client = _FakeClient()
    report = await publish_to_intake(result, client=client, experiment_id="exp-1")

    assert len(client.atif_calls) == 1
    assert client.atif_calls[0]["session_id"] == "run-1:t-1"
    assert client.atif_calls[0]["evaluation_context"] == {
        "evaluation_name": "exp-1",
        "test_case_name": "task-1",
    }
    # 3 metric outputs across the two score records -> 3 evaluator-result rows.
    assert len(client.eval_calls) == 3
    assert {call["name"] for call in client.eval_calls} == {"accuracy.score", "accuracy.passed", "latency.p50"}
    # span_id resolved from the trace and threaded into every row.
    assert {call["span_id"] for call in client.eval_calls} == {"span:run-1:t-1"}

    assert report.trial_count == 1
    assert report.evaluator_result_count == 3
    published = report.published_trials[0]
    assert (published.trial_id, published.session_id, published.span_id, published.evaluator_result_count) == (
        "t-1",
        "run-1:t-1",
        "span:run-1:t-1",
        3,
    )


async def test_publish_revalidates_model_copy_corrupted_measurements_before_network_calls() -> None:
    trial = _trial("t-1").model_copy(update={"measurements": TrialMeasurements(prompt_tokens=8, completion_tokens=2)})
    result = _result([trial], [])
    corrupted_measurements = trial.measurements.model_copy(update={"prompt_tokens": -1})
    corrupted_trial = trial.model_copy(update={"measurements": corrupted_measurements})
    corrupted_result = result.model_copy(update={"trials": [corrupted_trial]})
    client = _FakeClient()

    with pytest.raises(ValidationError):
        await publish_to_intake(
            corrupted_result,
            client=client,
            experiment_id="exp-1",
        )

    assert client.evaluation_calls == []
    assert (client.otlp_calls, client.atif_calls, client.eval_calls) == ([], [], [])


async def test_an_unknown_evaluation_fails_the_run_before_any_trial_is_written() -> None:
    # OTLP ingest does not check the evaluation exists the way ATIF ingest does, so without this
    # guard the primary path would write spans against an evaluation that cannot hold them.
    result = _result(
        trials=[_trial("t-1")],
        scores=[_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])],
    )
    client = _FakeClient(evaluation_missing=True)

    with pytest.raises(PublishError, match="does not exist"):
        await publish_to_intake(result, client=client, experiment_id="exp-1")

    assert client.evaluation_calls == [{"name": "exp-1", "workspace": "default"}]
    assert (client.otlp_calls, client.atif_calls, client.eval_calls) == ([], [], [])


async def test_multiple_trials_each_get_their_own_session_and_span() -> None:
    result = _result(
        trials=[_trial("t-1"), _trial("t-2")],
        scores=[
            _score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)]),
            _score("t-2", "accuracy", [MetricOutput(name="score", value=0.0)]),
        ],
    )
    client = _FakeClient()
    report = await publish_to_intake(result, client=client, experiment_id="exp-1")

    assert len(client.atif_calls) == 2
    assert report.trial_count == 2
    by_session = {call["session_id"]: call["span_id"] for call in client.eval_calls}
    assert by_session == {"run-1:t-1": "span:run-1:t-1", "run-1:t-2": "span:run-1:t-2"}


async def test_trial_without_scores_still_ingests_trajectory() -> None:
    result = _result(trials=[_trial("t-1")], scores=[])
    client = _FakeClient()
    report = await publish_to_intake(result, client=client, experiment_id="exp-1")

    assert len(client.atif_calls) == 1
    assert len(client.eval_calls) == 0
    assert report.published_trials[0].evaluator_result_count == 0


async def test_explicit_workspace_overrides_client_default() -> None:
    result = _result(trials=[_trial("t-1")], scores=[])
    client = _FakeClient(workspace="default")
    report = await publish_to_intake(result, client=client, experiment_id="exp-1", workspace="ws-2")
    assert report.workspace == "ws-2"
    assert client.atif_calls[0]["workspace"] == "ws-2"


async def test_missing_workspace_raises() -> None:
    result = _result(trials=[_trial("t-1")], scores=[])
    client = _FakeClient(workspace=None)
    with pytest.raises(ValueError, match="workspace"):
        await publish_to_intake(result, client=client, experiment_id="exp-1")


async def test_unresolvable_span_raises_publish_error() -> None:
    result = _result(
        trials=[_trial("t-1")],
        scores=[_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])],
    )
    client = _FakeClient(root_span_id=None)
    with pytest.raises(PublishError, match="No root span"):
        await publish_to_intake(result, client=client, experiment_id="exp-1")


async def test_ingest_failure_propagates() -> None:
    result = _result(trials=[_trial("t-1")], scores=[])
    client = _FakeClient(atif_fail=True)
    with pytest.raises(RuntimeError, match="atif ingest 400"):
        await publish_to_intake(result, client=client, experiment_id="exp-1")


async def test_failed_and_non_finite_scores_are_skipped_and_reported() -> None:
    # NaN can't be sent (not JSON-serializable) and a FAILED score is not a real measurement; both
    # are omitted but surfaced in the report so the omission is explicit, not silent (X6).
    result = _result(
        trials=[_trial("t-1")],
        scores=[
            _score(
                "t-1", "accuracy", [MetricOutput(name="score", value=1.0), MetricOutput(name="broken", value=math.nan)]
            ),
            _score("t-1", "judge", [MetricOutput(name="verdict", value=math.nan)], status=AgentEvalScoreStatus.FAILED),
        ],
    )
    client = _FakeClient()
    report = await publish_to_intake(result, client=client, experiment_id="exp-1")

    # Only the finite, completed output is sent to Intake.
    assert {call["name"] for call in client.eval_calls} == {"accuracy.score"}
    # The omissions are reported, with reasons.
    assert {(skip.name, skip.reason) for skip in report.skipped} == {
        ("accuracy.broken", "non-finite value"),
        ("judge.verdict", "scoring failed"),
    }


async def test_one_trial_failure_does_not_block_others_and_is_reported() -> None:
    # Partial uploads are acceptable (intake has no rollback), so a single trial's failure must NOT
    # abort the others — every trial that can land should land, leaving less for an idempotent retry.
    result = _result(
        trials=[_trial("t-1"), _trial("t-2")],
        scores=[
            _score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)]),
            _score("t-2", "accuracy", [MetricOutput(name="score", value=0.0)]),
        ],
    )
    client = _FakeClient(fail_eval_session="run-1:t-2")

    with pytest.raises(PublishError) as excinfo:
        await publish_to_intake(result, client=client, experiment_id="exp-1", max_concurrency=1)

    # The healthy trial still published despite the other failing.
    assert any(call["session_id"] == "run-1:t-1" for call in client.eval_calls)
    assert all(call["session_id"] != "run-1:t-2" for call in client.eval_calls)

    # The failure surfaces the affected trial and points the user at recovery.
    message = str(excinfo.value).lower()
    assert "t-2" in message
    assert "re-run" in message or "cached" in message or "publish" in message


class _RejectRichTrajectoryAtif:
    """Ingest that refuses any multi-step trajectory, the way a schema mismatch would."""

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self._calls = calls

    async def create(self, **kwargs: Any) -> None:
        self._calls.append(kwargs)
        if len(kwargs["steps"]) > 1:
            raise UnprocessableEntityError(
                httpx.Response(422, request=httpx.Request("POST", "http://test/atif")),
            )


def _trial_with_atif_trace(tmp_path: Path, trial_id: str = "t-1") -> AgentEvalTrial:
    trace = tmp_path / "trajectory.json"
    trace.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "s",
                "agent": {"name": "codex", "version": "1.0"},
                "steps": [
                    {"step_id": 1, "source": "user", "message": "q", "timestamp": "2026-01-01T00:00:00+00:00"},
                    {"step_id": 2, "source": "agent", "message": "a", "timestamp": "2026-01-01T00:00:01+00:00"},
                ],
            }
        )
    )
    return AgentEvalTrial(
        id=trial_id,
        task_id="task-1",
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text="a"),
        evidence=CandidateEvidence(
            descriptors={"trace": EvidenceDescriptor(kind="trace", format="atif", ref=str(trace))}
        ),
    )


@pytest.mark.asyncio
async def test_a_rejected_trajectory_falls_back_instead_of_failing_the_trial(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    client = _client(atif=_RejectRichTrajectoryAtif(calls))
    trial = _trial_with_atif_trace(tmp_path)

    report = await publish_to_intake(
        _result([trial], [_score("t-1", "m", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="eval-1",
        workspace="ws-1",
    )

    # The rich attempt is refused, the single-step retry lands, and the trial still publishes.
    assert [len(call["steps"]) for call in calls] == [2, 1]
    assert report.trial_count == 1


# --- OTLP publish path ------------------------------------------------------

_OTLP_TRACE_ID = "0123456789abcdef0123456789abcdef"
_OTLP_SPAN_ID = "0102030405060708"


def _otlp_trial(trial_id: str = "t-1", *, span_id: str = _OTLP_SPAN_ID, extra_spans: list[dict] | None = None):
    from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor

    spans = [{"traceId": _OTLP_TRACE_ID, "spanId": span_id, "name": "agent run"}, *(extra_spans or [])]
    trace = EvidenceDescriptor(
        kind="trace", format="otlp", data={"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}
    )
    trial = _trial(trial_id)
    return trial.model_copy(update={"evidence": CandidateEvidence(descriptors={"trace": trace})})


async def test_a_trial_with_an_otlp_trace_publishes_otlp_and_skips_atif() -> None:
    client = _client()

    report = await publish_to_intake(
        _result([_otlp_trial()], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    assert len(client.otlp_calls) == 1
    assert client.atif_calls == [], "the OTLP trace is the trial's trajectory; ATIF must not also be sent"
    assert client.otlp_calls[0]["workspace"] == "default"
    assert isinstance(client.otlp_calls[0]["body"], bytes)
    # The root span id comes off the payload, not from an Intake round trip.
    assert report.published_trials[0].span_id == _OTLP_SPAN_ID


async def test_the_published_otlp_payload_carries_the_stamped_identity() -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    client = _client()

    await publish_to_intake(
        _result([_otlp_trial()], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    sent = ExportTraceServiceRequest()
    sent.ParseFromString(client.otlp_calls[0]["body"])
    span = sent.resource_spans[0].scope_spans[0].spans[0]
    attributes = {attribute.key: attribute.value.string_value for attribute in span.attributes}
    assert attributes["gen_ai.conversation.id"] == "run-1:t-1"
    assert attributes["nemo.evaluation.name"] == "exp-1"
    assert attributes["nemo.test_case.name"] == "task-1"
    # The producer's ids survive the hex/base64 round trip, or scores attach to nothing.
    assert span.span_id.hex() == _OTLP_SPAN_ID
    assert span.trace_id.hex() == _OTLP_TRACE_ID


async def test_a_trial_without_an_otlp_trace_still_publishes_atif() -> None:
    client = _client()

    await publish_to_intake(
        _result([_trial("t-1")], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    assert client.otlp_calls == []
    assert len(client.atif_calls) == 1


async def test_an_otlp_trace_with_no_single_root_falls_back_to_atif() -> None:
    # Nothing stands for the whole run, so there is no span to attach a trial score to.
    second_root = {"traceId": _OTLP_TRACE_ID, "spanId": "aabbccddeeff0011", "name": "other root"}
    client = _client()

    await publish_to_intake(
        _result(
            [_otlp_trial(extra_spans=[second_root])],
            [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])],
        ),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    assert client.otlp_calls == []
    assert len(client.atif_calls) == 1


async def test_trial_totals_land_on_the_root_span_only() -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    child = {"traceId": _OTLP_TRACE_ID, "spanId": "aabbccddeeff0011", "parentSpanId": _OTLP_SPAN_ID, "name": "child"}
    trial = _otlp_trial(extra_spans=[child]).model_copy(
        update={
            "measurements": TrialMeasurements(
                prompt_tokens=120,
                completion_tokens=45,
                cache_read_tokens=30,
                cost_usd=0.134,
            ),
            "error": TrialError(type="Timeout", message="agent exceeded its budget"),
        }
    )
    client = _client()

    await publish_to_intake(
        _result([trial], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    sent = ExportTraceServiceRequest()
    sent.ParseFromString(client.otlp_calls[0]["body"])
    spans = {span.name: span for span in sent.resource_spans[0].scope_spans[0].spans}
    root = {attribute.key: attribute.value for attribute in spans["agent run"].attributes}
    assert root["gen_ai.usage.input_tokens"].int_value == 120
    assert root["gen_ai.usage.output_tokens"].int_value == 45
    assert root["gen_ai.usage.cached_tokens"].int_value == 30
    assert root["gen_ai.usage.cost"].double_value == pytest.approx(0.134)
    assert root["exception.type"].string_value == "Timeout"
    assert root["exception.message"].string_value == "agent exceeded its budget"

    # A rollup that sums across the trace would double-count these if every span carried them.
    child_keys = {attribute.key for attribute in spans["child"].attributes}
    assert not child_keys & {"gen_ai.usage.input_tokens", "gen_ai.usage.cost", "exception.type"}
    # Identity, by contrast, is on every span so any of them resolves back to the trial.
    assert "gen_ai.conversation.id" in child_keys


async def test_a_root_span_with_no_usable_id_falls_back_to_atif() -> None:
    # Intake drops a span whose id is absent or all zero, so scoring against that id would
    # point at a span it never stored.
    client = _client()

    await publish_to_intake(
        _result(
            [_otlp_trial(span_id="0000000000000000")],
            [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])],
        ),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    assert client.otlp_calls == []
    assert len(client.atif_calls) == 1


async def test_a_span_without_a_start_time_is_given_the_runs() -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    # Intake stores a span with no start time against its own ingest clock, and start time is
    # part of the key it replaces on, so a re-publish would insert instead of replacing.
    client = _client()

    await publish_to_intake(
        _result([_otlp_trial()], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    sent = ExportTraceServiceRequest()
    sent.ParseFromString(client.otlp_calls[0]["body"])
    assert sent.resource_spans[0].scope_spans[0].spans[0].start_time_unix_nano == int(
        STARTED_AT.timestamp() * 1_000_000_000
    )


async def test_a_failed_trial_marks_its_root_span_failed() -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.trace.v1.trace_pb2 import Status

    # Status is a span field, not an attribute; recorded only as text the trace reads as a success.
    trial = _otlp_trial().model_copy(update={"error": TrialError(type="Timeout", message="over budget")})
    client = _client()

    await publish_to_intake(
        _result([trial], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    sent = ExportTraceServiceRequest()
    sent.ParseFromString(client.otlp_calls[0]["body"])
    status = sent.resource_spans[0].scope_spans[0].spans[0].status
    assert status.code == Status.STATUS_CODE_ERROR
    assert status.message == "over budget"


async def test_a_partly_rejected_otlp_batch_fails_the_trial_instead_of_scoring_it() -> None:
    # Ingest answers 200 with a per-span error list, so a dropped span is invisible unless the
    # body is read -- including when the dropped span is the one the scores reference.
    client = _client(otlp_errors=["span 0102030405060708: parent_span_id must not be all zero"])

    with pytest.raises(PublishError, match="Intake rejected 1 span"):
        await publish_to_intake(
            _result([_otlp_trial()], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
            client=client,
            experiment_id="exp-1",
            agent_name="a",
        )

    assert client.eval_calls == [], "scores must not attach to a trace Intake only partly stored"
