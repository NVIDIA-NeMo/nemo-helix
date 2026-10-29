# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for publish_to_intake — the explicit Evaluator -> Intake publish step."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from nemo_evals.intake.publish import PublishError, _token_final_metrics, publish_to_intake
from nemo_evals.intake.row_adapter import row_result_to_agent_eval_result
from nhx.testing.intake import InMemoryIntakeService
from nhx_evals_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary, RunMetadata
from nhx_evals_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nhx_evals_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    TrialError,
    TrialMeasurements,
)
from nhx_evals_sdk.metrics.protocol import MetricOutput
from nhx_evals_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from nhx_evals_sdk.values.results import AggregatedMetricResult, EvaluationResult, RowScore
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from pydantic import ValidationError

# --- fakes ------------------------------------------------------------------


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
    intake = InMemoryIntakeService()
    client = intake.client()
    report = await publish_to_intake(result, client=client, experiment_id="exp-1")

    assert len(intake.atif_requests) == 1
    assert intake.atif_requests[0]["session_id"] == "run-1:t-1"
    assert intake.atif_requests[0]["evaluation_context"] == {
        "evaluation_name": "exp-1",
        "test_case_name": "task-1",
    }
    # 3 metric outputs across the two score records -> 3 evaluator-result rows.
    assert len(intake.evaluator_result_requests) == 3
    assert {call["name"] for call in intake.evaluator_result_requests} == {
        "accuracy.score",
        "accuracy.passed",
        "latency.p50",
    }
    # span_id resolved from the trace and threaded into every row.
    assert {call["span_id"] for call in intake.evaluator_result_requests} == {"span:run-1:t-1"}

    assert report.trial_count == 1
    assert report.evaluator_result_count == 3
    published = report.published_trials[0]
    assert (published.trial_id, published.session_id, published.span_id, published.evaluator_result_count) == (
        "t-1",
        "run-1:t-1",
        "span:run-1:t-1",
        3,
    )


async def test_an_agent_row_trial_publishes_its_runtime_as_the_trajectory_window() -> None:
    row = RowScore(
        row_index=0,
        item={"prompt": "2+2?"},
        sample={"output_text": "4", "response": {"choices": []}},
        metrics={},
        requests=[],
        runtime_sec=37.25,
    )
    result = row_result_to_agent_eval_result(
        EvaluationResult(row_scores=[row], aggregate_scores=AggregatedMetricResult(scores=[])),
        run_id="job-1",
        started_at=STARTED_AT,
    )
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(result, client=client, experiment_id="exp-1")

    (step,) = intake.atif_requests[0]["steps"]
    invocation = step["extra"]["invocation"]
    assert invocation["start_timestamp"] == STARTED_AT.timestamp()
    assert invocation["end_timestamp"] - invocation["start_timestamp"] == 37.25


async def test_a_row_whose_generation_failed_publishes_an_errored_trajectory() -> None:
    row = RowScore(
        row_index=0,
        item={"prompt": "2+2?"},
        sample={"output_text": None, "response": {}, "inference_error": "Row 0 failed inference: timed out"},
        metrics={},
        requests=[],
        runtime_sec=4.0,
    )
    result = row_result_to_agent_eval_result(
        EvaluationResult(row_scores=[row], aggregate_scores=AggregatedMetricResult(scores=[])),
        run_id="job-1",
        started_at=STARTED_AT,
    )
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(result, client=client, experiment_id="exp-1")

    assert intake.atif_requests[0]["extra"]["error"] == {
        "type": "InferenceError",
        "message": "Row 0 failed inference: timed out",
    }


async def test_publish_revalidates_model_copy_corrupted_measurements_before_network_calls() -> None:
    trial = _trial("t-1").model_copy(update={"measurements": TrialMeasurements(prompt_tokens=8, completion_tokens=2)})
    result = _result([trial], [])
    corrupted_measurements = trial.measurements.model_copy(update={"prompt_tokens": -1})
    corrupted_trial = trial.model_copy(update={"measurements": corrupted_measurements})
    corrupted_result = result.model_copy(update={"trials": [corrupted_trial]})
    intake = InMemoryIntakeService()
    client = intake.client()

    with pytest.raises(ValidationError):
        await publish_to_intake(
            corrupted_result,
            client=client,
            experiment_id="exp-1",
        )

    assert intake.evaluation_lookups == []
    assert (intake.otlp_requests, intake.atif_requests, intake.evaluator_result_requests) == ([], [], [])


async def test_an_unknown_evaluation_fails_the_run_before_any_trial_is_written() -> None:
    # OTLP ingest does not check the evaluation exists the way ATIF ingest does, so without this
    # guard the primary path would write spans against an evaluation that cannot hold them.
    result = _result(
        trials=[_trial("t-1")],
        scores=[_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])],
    )
    intake = InMemoryIntakeService()
    intake.evaluation_exists = False
    client = intake.client()

    with pytest.raises(PublishError, match="does not exist"):
        await publish_to_intake(result, client=client, experiment_id="exp-1")

    assert intake.evaluation_lookups == [{"name": "exp-1", "workspace": "default"}]
    assert (intake.otlp_requests, intake.atif_requests, intake.evaluator_result_requests) == ([], [], [])


async def test_multiple_trials_each_get_their_own_session_and_span() -> None:
    result = _result(
        trials=[_trial("t-1"), _trial("t-2")],
        scores=[
            _score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)]),
            _score("t-2", "accuracy", [MetricOutput(name="score", value=0.0)]),
        ],
    )
    intake = InMemoryIntakeService()
    client = intake.client()
    report = await publish_to_intake(result, client=client, experiment_id="exp-1")

    assert len(intake.atif_requests) == 2
    assert report.trial_count == 2
    by_session = {call["session_id"]: call["span_id"] for call in intake.evaluator_result_requests}
    assert by_session == {"run-1:t-1": "span:run-1:t-1", "run-1:t-2": "span:run-1:t-2"}


async def test_trial_without_scores_still_ingests_trajectory() -> None:
    result = _result(trials=[_trial("t-1")], scores=[])
    intake = InMemoryIntakeService()
    client = intake.client()
    report = await publish_to_intake(result, client=client, experiment_id="exp-1")

    assert len(intake.atif_requests) == 1
    assert len(intake.evaluator_result_requests) == 0
    assert report.published_trials[0].evaluator_result_count == 0


async def test_explicit_workspace_overrides_client_default() -> None:
    result = _result(trials=[_trial("t-1")], scores=[])
    intake = InMemoryIntakeService()
    client = intake.client(workspace="default")
    report = await publish_to_intake(result, client=client, experiment_id="exp-1", workspace="ws-2")
    assert report.workspace == "ws-2"
    assert intake.atif_requests[0]["workspace"] == "ws-2"


async def test_missing_workspace_raises() -> None:
    result = _result(trials=[_trial("t-1")], scores=[])
    intake = InMemoryIntakeService()
    client = intake.client(workspace=None)
    with pytest.raises(ValueError, match="workspace"):
        await publish_to_intake(result, client=client, experiment_id="exp-1")


async def test_unresolvable_span_raises_publish_error() -> None:
    result = _result(
        trials=[_trial("t-1")],
        scores=[_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])],
    )
    intake = InMemoryIntakeService()
    intake.root_span_prefix = None
    client = intake.client()
    with pytest.raises(PublishError, match="No root span"):
        await publish_to_intake(result, client=client, experiment_id="exp-1")


async def test_ingest_failure_propagates() -> None:
    result = _result(trials=[_trial("t-1")], scores=[])
    intake = InMemoryIntakeService()
    intake.atif_failure = 400
    client = intake.client()
    with pytest.raises(PublishError, match="BadRequestError: HTTP 400"):
        await publish_to_intake(result, client=client, experiment_id="exp-1")


async def test_failed_and_non_finite_scores_publish_as_failed_rows() -> None:
    """Intake counts a failed attempt from an explicit FAILED row, so nothing is dropped any more."""
    result = _result(
        trials=[_trial("t-1")],
        scores=[
            _score(
                "t-1", "accuracy", [MetricOutput(name="score", value=1.0), MetricOutput(name="broken", value=math.nan)]
            ),
            _score("t-1", "judge", [MetricOutput(name="verdict", value=math.nan)], status=AgentEvalScoreStatus.FAILED),
        ],
    )
    intake = InMemoryIntakeService()
    client = intake.client()
    report = await publish_to_intake(result, client=client, experiment_id="exp-1")

    assert {(call["name"], call.get("status", "SCORED")) for call in intake.evaluator_result_requests} == {
        ("accuracy.score", "SCORED"),
        ("accuracy.broken", "FAILED"),
        ("judge.verdict", "FAILED"),
    }
    assert report.skipped == []
    assert report.published_trials[0].evaluator_result_count == 3


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
    intake = InMemoryIntakeService()
    intake.failing_evaluator_result_sessions = {"run-1:t-2"}
    client = intake.client()

    with pytest.raises(PublishError) as excinfo:
        await publish_to_intake(result, client=client, experiment_id="exp-1", max_concurrency=1)

    # The healthy trial still published despite the other failing.
    assert any(call["session_id"] == "run-1:t-1" for call in intake.evaluator_result_requests)
    assert all(call["session_id"] != "run-1:t-2" for call in intake.evaluator_result_requests)

    # The failure surfaces the affected trial and points the user at recovery.
    message = str(excinfo.value).lower()
    assert "t-2" in message
    assert "re-run" in message or "cached" in message or "publish" in message


def _reject_rich_trajectories(attempts: list[int]) -> Callable[[dict[str, object]], bool]:
    """Refuse any multi-step trajectory, the way a schema mismatch would, recording each attempt's step count."""

    def reject(body: dict[str, object]) -> bool:
        steps = body["steps"]
        assert isinstance(steps, list)
        attempts.append(len(steps))
        return len(steps) > 1

    return reject


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
    attempts: list[int] = []
    intake = InMemoryIntakeService()
    intake.atif_rejects = _reject_rich_trajectories(attempts)
    client = intake.client()
    trial = _trial_with_atif_trace(tmp_path)

    report = await publish_to_intake(
        _result([trial], [_score("t-1", "m", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="eval-1",
        workspace="ws-1",
    )

    # The rich attempt is refused, the single-step retry lands, and the trial still publishes.
    assert attempts == [2, 1]
    assert report.trial_count == 1


async def test_a_trial_with_an_untimed_recorded_trajectory_publishes_its_runtime(tmp_path: Path) -> None:
    trace = tmp_path / "trajectory.json"
    trace.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "s",
                "agent": {"name": "codex", "version": "1.0"},
                "steps": [
                    {"step_id": 1, "source": "user", "message": "q"},
                    {"step_id": 2, "source": "agent", "message": "a"},
                ],
            }
        )
    )
    trial = AgentEvalTrial(
        id="t-1",
        task_id="task-1",
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text="a"),
        evidence=CandidateEvidence(
            descriptors={"trace": EvidenceDescriptor(kind="trace", format="atif", ref=str(trace))}
        ),
        measurements=TrialMeasurements(runtime_sec=12.5),
    )
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(_result([trial], []), client=client, experiment_id="exp-1")

    *_, last_step = intake.atif_requests[0]["steps"]
    invocation = last_step["extra"]["invocation"]
    assert invocation["end_timestamp"] - invocation["start_timestamp"] == 12.5


# --- OTLP publish path ------------------------------------------------------

_OTLP_TRACE_ID = "0123456789abcdef0123456789abcdef"
_OTLP_SPAN_ID = "0102030405060708"


def _otlp_trial(trial_id: str = "t-1", *, span_id: str = _OTLP_SPAN_ID, extra_spans: list[dict] | None = None):
    from nhx_evals_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor

    spans = [{"traceId": _OTLP_TRACE_ID, "spanId": span_id, "name": "agent run"}, *(extra_spans or [])]
    trace = EvidenceDescriptor(
        kind="trace", format="otlp", data={"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}
    )
    trial = _trial(trial_id)
    return trial.model_copy(update={"evidence": CandidateEvidence(descriptors={"trace": trace})})


async def test_a_trial_with_an_otlp_trace_publishes_otlp_and_skips_atif() -> None:
    intake = InMemoryIntakeService()
    client = intake.client()

    report = await publish_to_intake(
        _result([_otlp_trial()], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    assert len(intake.otlp_requests) == 1
    assert intake.atif_requests == [], "the OTLP trace is the trial's trajectory; ATIF must not also be sent"
    assert intake.otlp_requests[0]["workspace"] == "default"
    assert isinstance(intake.otlp_requests[0]["body"], bytes)
    # The root span id comes off the payload, not from an Intake round trip.
    assert report.published_trials[0].span_id == _OTLP_SPAN_ID


async def test_the_published_otlp_payload_carries_the_stamped_identity() -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    intake = InMemoryIntakeService()

    client = intake.client()

    await publish_to_intake(
        _result([_otlp_trial()], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    sent = ExportTraceServiceRequest()
    sent.ParseFromString(intake.otlp_requests[0]["body"])
    span = sent.resource_spans[0].scope_spans[0].spans[0]
    attributes = {attribute.key: attribute.value.string_value for attribute in span.attributes}
    assert attributes["gen_ai.conversation.id"] == "run-1:t-1"
    assert attributes["nemo.evaluation.name"] == "exp-1"
    assert attributes["nemo.test_case.name"] == "task-1"
    # The producer's ids survive the hex/base64 round trip, or scores attach to nothing.
    assert span.span_id.hex() == _OTLP_SPAN_ID
    assert span.trace_id.hex() == _OTLP_TRACE_ID


async def test_a_trial_without_an_otlp_trace_still_publishes_atif() -> None:
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(
        _result([_trial("t-1")], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    assert intake.otlp_requests == []
    assert len(intake.atif_requests) == 1


async def test_an_otlp_trace_with_no_single_root_falls_back_to_atif() -> None:
    # Nothing stands for the whole run, so there is no span to attach a trial score to.
    second_root = {"traceId": _OTLP_TRACE_ID, "spanId": "aabbccddeeff0011", "name": "other root"}
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(
        _result(
            [_otlp_trial(extra_spans=[second_root])],
            [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])],
        ),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    assert intake.otlp_requests == []
    assert len(intake.atif_requests) == 1


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
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(
        _result([trial], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    sent = ExportTraceServiceRequest()
    sent.ParseFromString(intake.otlp_requests[0]["body"])
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
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(
        _result(
            [_otlp_trial(span_id="0000000000000000")],
            [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])],
        ),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    assert intake.otlp_requests == []
    assert len(intake.atif_requests) == 1


async def test_a_span_without_a_start_time_is_given_the_runs() -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    # Intake stores a span with no start time against its own ingest clock, and start time is
    # part of the key it replaces on, so a re-publish would insert instead of replacing.
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(
        _result([_otlp_trial()], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    sent = ExportTraceServiceRequest()
    sent.ParseFromString(intake.otlp_requests[0]["body"])
    assert sent.resource_spans[0].scope_spans[0].spans[0].start_time_unix_nano == int(
        STARTED_AT.timestamp() * 1_000_000_000
    )


async def test_an_untimed_root_span_is_given_the_trials_runtime_as_its_end() -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    trial = _otlp_trial().model_copy(update={"measurements": TrialMeasurements(runtime_sec=45.0)})
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(_result([trial], []), client=client, experiment_id="exp-1", agent_name="a")

    sent = ExportTraceServiceRequest()
    sent.ParseFromString(intake.otlp_requests[0]["body"])
    root = sent.resource_spans[0].scope_spans[0].spans[0]
    assert root.end_time_unix_nano - root.start_time_unix_nano == 45_000_000_000


async def test_a_root_span_with_its_own_end_keeps_it_over_the_trials_runtime() -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    spans = [{"traceId": _OTLP_TRACE_ID, "spanId": _OTLP_SPAN_ID, "name": "agent run", "endTimeUnixNano": "2000"}]
    trace = EvidenceDescriptor(
        kind="trace", format="otlp", data={"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}
    )
    trial = _trial("t-1").model_copy(
        update={
            "evidence": CandidateEvidence(descriptors={"trace": trace}),
            "measurements": TrialMeasurements(runtime_sec=45.0),
        }
    )
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(_result([trial], []), client=client, experiment_id="exp-1", agent_name="a")

    sent = ExportTraceServiceRequest()
    sent.ParseFromString(intake.otlp_requests[0]["body"])
    assert sent.resource_spans[0].scope_spans[0].spans[0].end_time_unix_nano == 2000


async def test_a_failed_trial_marks_its_root_span_failed() -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.trace.v1.trace_pb2 import Status

    # Status is a span field, not an attribute; recorded only as text the trace reads as a success.
    trial = _otlp_trial().model_copy(update={"error": TrialError(type="Timeout", message="over budget")})
    intake = InMemoryIntakeService()
    client = intake.client()

    await publish_to_intake(
        _result([trial], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
        client=client,
        experiment_id="exp-1",
        agent_name="a",
    )

    sent = ExportTraceServiceRequest()
    sent.ParseFromString(intake.otlp_requests[0]["body"])
    status = sent.resource_spans[0].scope_spans[0].spans[0].status
    assert status.code == Status.STATUS_CODE_ERROR
    assert status.message == "over budget"


async def test_a_gym_rollout_whose_agent_never_ran_publishes_a_failed_root_span(tmp_path: Path) -> None:
    from nhx_evals_sdk.agent_eval.runtimes.gym import discover_gym_tasks
    from nhx_evals_sdk.agent_eval.runtimes.gym.dataset import materialize_dataset
    from nhx_evals_sdk.agent_eval.runtimes.gym.results import trials_from_rollouts
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.trace.v1.trace_pb2 import Status

    dataset = tmp_path / "ds.jsonl"
    dataset.write_text(
        "\n".join(json.dumps({"responses_create_params": {"input": question}}) for question in ("q0", "q1")) + "\n",
        encoding="utf-8",
    )
    tasks = discover_gym_tasks(dataset)
    index_map = materialize_dataset(tasks, tmp_path / "gym_input.jsonl")
    empty = {"output": [], "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
    answered = {
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "B"}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        json.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "response": empty, "reward": 0.0})
        + "\n"
        + json.dumps({"_ng_task_index": 1, "_ng_rollout_index": 0, "response": answered, "reward": 1.0})
        + "\n",
        encoding="utf-8",
    )
    trials = trials_from_rollouts(rollouts, tasks, index_map)
    intake = InMemoryIntakeService()

    await publish_to_intake(_result(trials, []), client=intake.client(), experiment_id="exp-1", agent_name="a")

    roots = []
    for request in intake.otlp_requests:
        sent = ExportTraceServiceRequest()
        sent.ParseFromString(request["body"])
        spans = [span for scope in sent.resource_spans[0].scope_spans for span in scope.spans]
        roots.extend(span for span in spans if not span.parent_span_id)
    assert sorted(root.status.code for root in roots) == [Status.STATUS_CODE_UNSET, Status.STATUS_CODE_ERROR]
    (failed,) = (root for root in roots if root.status.code == Status.STATUS_CODE_ERROR)
    assert "never called" in failed.status.message


async def test_a_partly_rejected_otlp_batch_fails_the_trial_instead_of_scoring_it() -> None:
    # Ingest answers 200 with a per-span error list, so a dropped span is invisible unless the
    # body is read -- including when the dropped span is the one the scores reference.
    intake = InMemoryIntakeService()
    intake.otlp_errors = ["span 0102030405060708: parent_span_id must not be all zero"]
    client = intake.client()

    with pytest.raises(PublishError, match="Intake rejected 1 span"):
        await publish_to_intake(
            _result([_otlp_trial()], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
            client=client,
            experiment_id="exp-1",
            agent_name="a",
        )

    assert intake.evaluator_result_requests == [], "scores must not attach to a trace Intake only partly stored"


# --- OTLP request budget ------------------------------------------------------

_CHILD_SPAN_IDS = ["1111111111111111", "2222222222222222", "3333333333333333"]


def _large_otlp_trial(child_payload_bytes: int, trial_id: str = "t-1") -> AgentEvalTrial:
    """A root span and three children, each child carrying ``child_payload_bytes`` of attribute text."""
    children = [
        {
            "traceId": _OTLP_TRACE_ID,
            "spanId": span_id,
            "parentSpanId": _OTLP_SPAN_ID,
            "name": f"child-{index}",
            "attributes": [{"key": "payload", "value": {"stringValue": "x" * child_payload_bytes}}],
        }
        for index, span_id in enumerate(_CHILD_SPAN_IDS)
    ]
    return _otlp_trial(trial_id, extra_spans=children)


def _sent_bodies(intake: InMemoryIntakeService) -> list[bytes]:
    bodies = []
    for request in intake.otlp_requests:
        body = request["body"]
        assert isinstance(body, bytes)
        bodies.append(body)
    return bodies


def _span_ids(body: bytes) -> list[str]:
    sent = ExportTraceServiceRequest()
    sent.ParseFromString(body)
    return [
        span.span_id.hex() for resource in sent.resource_spans for scope in resource.scope_spans for span in scope.spans
    ]


def _sent_span_ids(intake: InMemoryIntakeService) -> list[list[str]]:
    return [_span_ids(body) for body in _sent_bodies(intake)]


async def test_a_trace_over_the_default_budget_is_sent_as_several_requests() -> None:
    # Three ~2 MiB children: together over Intake's 5 MiB default body limit, each well under it.
    intake = InMemoryIntakeService()

    report = await publish_to_intake(
        _result(
            [_large_otlp_trial(2 * 1024 * 1024)], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]
        ),
        client=intake.client(),
        experiment_id="exp-1",
        agent_name="a",
    )

    assert len(intake.otlp_requests) > 1
    assert all(len(body) <= 4 * 1024 * 1024 for body in _sent_bodies(intake))
    assert [span_id for batch in _sent_span_ids(intake) for span_id in batch] == [_OTLP_SPAN_ID, *_CHILD_SPAN_IDS]
    assert report.published_trials[0].span_id == _OTLP_SPAN_ID
    assert len(intake.evaluator_result_requests) == 1


async def test_a_trace_within_the_budget_is_sent_as_one_request() -> None:
    intake = InMemoryIntakeService()

    await publish_to_intake(
        _result([_large_otlp_trial(1000)], []),
        client=intake.client(),
        experiment_id="exp-1",
        agent_name="a",
        max_otlp_request_bytes=1024 * 1024,
    )

    assert _sent_span_ids(intake) == [[_OTLP_SPAN_ID, *_CHILD_SPAN_IDS]]


async def test_an_overridden_budget_splits_a_trace_the_default_would_send_whole() -> None:
    intake = InMemoryIntakeService()

    await publish_to_intake(
        _result([_large_otlp_trial(1000)], []),
        client=intake.client(),
        experiment_id="exp-1",
        agent_name="a",
        max_otlp_request_bytes=2500,
    )

    assert len(intake.otlp_requests) > 1
    assert all(len(body) <= 2500 for body in _sent_bodies(intake))
    assert [span_id for batch in _sent_span_ids(intake) for span_id in batch] == [_OTLP_SPAN_ID, *_CHILD_SPAN_IDS]


async def test_a_span_too_large_for_the_budget_fails_the_trial_before_any_of_it_is_sent() -> None:
    intake = InMemoryIntakeService()

    with pytest.raises(PublishError, match="max_otlp_request_bytes") as raised:
        await publish_to_intake(
            _result(
                [_large_otlp_trial(1000), _trial("t-2")],
                [
                    _score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)]),
                    _score("t-2", "accuracy", [MetricOutput(name="score", value=1.0)]),
                ],
            ),
            client=intake.client(),
            experiment_id="exp-1",
            agent_name="a",
            max_otlp_request_bytes=800,
        )

    assert intake.otlp_requests == [], "no part of an unsendable trace may land"
    assert {row["session_id"] for row in intake.evaluator_result_requests} == {"run-1:t-2"}
    assert raised.value.report is not None
    assert [trial.trial_id for trial in raised.value.report.published_trials] == ["t-2"]


async def test_a_rejection_in_a_later_request_fails_the_trial_before_its_scores() -> None:
    last_child = _CHILD_SPAN_IDS[-1]
    intake = InMemoryIntakeService()
    intake.otlp_rejects = lambda body: [f"span {last_child}: rejected"] if last_child in _span_ids(body) else []

    with pytest.raises(PublishError, match=r"OTLP request (\d+) of \1:"):
        await publish_to_intake(
            _result([_large_otlp_trial(1000)], [_score("t-1", "accuracy", [MetricOutput(name="score", value=1.0)])]),
            client=intake.client(),
            experiment_id="exp-1",
            agent_name="a",
            max_otlp_request_bytes=2500,
        )

    sent = _sent_span_ids(intake)
    assert len(sent) > 1
    assert last_child not in sent[0], "the rejected span must arrive in a later request than the first"
    assert intake.evaluator_result_requests == []


@pytest.mark.parametrize("budget", [0, -1])
async def test_a_non_positive_otlp_budget_is_rejected_before_any_request(budget: int) -> None:
    intake = InMemoryIntakeService()

    with pytest.raises(ValueError, match="max_otlp_request_bytes must be positive"):
        await publish_to_intake(
            _result([_otlp_trial()], []),
            client=intake.client(),
            experiment_id="exp-1",
            agent_name="a",
            max_otlp_request_bytes=budget,
        )

    assert intake.evaluation_lookups == []
    assert intake.otlp_requests == []
