# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Evaluator -> Intake boundary mapping module."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from nemo_evals.intake.mapping import (
    ATIF_SCHEMA_VERSION,
    DEFAULT_AGENT_VERSION,
    RecordedSteps,
    atif_steps_from_trial,
    otlp_ingest_for_trial,
    run_task_to_evaluation_context,
    score_to_evaluator_results,
    session_id_for,
    trial_to_atif_ingest,
)
from nemo_helix_plugin.intake.types import EvaluatorResultCreateParams
from nhx_evals_sdk.agent_eval.scores import (
    AgentEvalDiagnostic,
    AgentEvalDiagnosticSeverity,
    AgentEvalScoreStatus,
    AgentEvalTaskScore,
)
from nhx_evals_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    TrialError,
    TrialMeasurements,
)
from nhx_evals_sdk.metrics.protocol import (
    BooleanValue,
    ContinuousScore,
    DiscreteScore,
    Label,
    MetricOutput,
)
from nhx_evals_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

STARTED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _trial(*, trial_id: str = "trial-1", task_id: str = "task-1", output_text: str | None = "hello") -> AgentEvalTrial:
    output = AgentOutput(output_text=output_text) if output_text is not None else None
    status = AgentEvalTrialStatus.COMPLETED if output is not None else AgentEvalTrialStatus.FAILED
    return AgentEvalTrial(id=trial_id, task_id=task_id, status=status, output=output)


def _score(
    *,
    outputs: list[MetricOutput],
    diagnostics: list[AgentEvalDiagnostic] | None = None,
    status: AgentEvalScoreStatus = AgentEvalScoreStatus.COMPLETED,
) -> AgentEvalTaskScore:
    return AgentEvalTaskScore(
        id="score-1",
        run_id="run-1",
        task_id="task-1",
        trial_id="trial-1",
        metric_type="accuracy",
        status=status,
        outputs=outputs,
        diagnostics=diagnostics or [],
    )


def _rows(
    score: AgentEvalTaskScore, *, session_id: str = "s", span_id: str = "sp"
) -> list[EvaluatorResultCreateParams]:
    """The publishable rows from a score, dropping the skipped list (for row-shape assertions)."""
    rows, _ = score_to_evaluator_results(score, session_id=session_id, span_id=span_id)
    return rows


# --- session_id_for ---------------------------------------------------------


def test_session_id_is_stable_per_trial() -> None:
    assert session_id_for("run-1", "trial-1") == "run-1:trial-1"


# --- run_task_to_evaluation_context -----------------------------------------


def test_evaluation_context_is_lean() -> None:
    context = run_task_to_evaluation_context(_trial(task_id="task-42"), evaluation_name="bench-x-variant")
    assert context == {"evaluation_name": "bench-x-variant", "test_case_name": "task-42"}


# --- trial_to_atif_ingest ---------------------------------------------------


def test_trial_to_atif_ingest_shape() -> None:
    body = trial_to_atif_ingest(
        _trial(trial_id="t-1", task_id="task-1", output_text="final answer"),
        run_id="run-1",
        evaluation_name="exp-1",
        agent_name="my-agent",
        started_at=STARTED_AT,
        model_name="gpt-4o",
    )
    assert body["schema_version"] == ATIF_SCHEMA_VERSION
    assert body["session_id"] == "run-1:t-1"
    assert body["agent"] == {"name": "my-agent", "version": DEFAULT_AGENT_VERSION, "model_name": "gpt-4o"}
    assert body["steps"] == [{"source": "agent", "step_id": 1, "message": "final answer", "timestamp": STARTED_AT}]
    assert body["evaluation_context"] == {"evaluation_name": "exp-1", "test_case_name": "task-1"}
    assert "final_metrics" not in body


def test_trial_to_atif_ingest_defaults_version_and_omits_model_name() -> None:
    body = trial_to_atif_ingest(
        _trial(), run_id="run-1", evaluation_name="exp-1", agent_name="a", started_at=STARTED_AT
    )
    assert body["agent"] == {"name": "a", "version": "unknown"}
    assert "model_name" not in body["agent"]


def test_trial_to_atif_ingest_handles_missing_output() -> None:
    body = trial_to_atif_ingest(
        _trial(output_text=None), run_id="run-1", evaluation_name="exp-1", agent_name="a", started_at=STARTED_AT
    )
    assert body["steps"] == [{"source": "agent", "step_id": 1, "message": "", "timestamp": STARTED_AT}]


def test_trial_to_atif_ingest_includes_trial_error_in_root_extra() -> None:
    trial = AgentEvalTrial(
        id="trial-timeout",
        task_id="task-1",
        status=AgentEvalTrialStatus.PARTIAL,
        output=AgentOutput(output_text="partial answer"),
        error=TrialError(
            type="AgentTimeoutError",
            message="Agent execution timed out after 900.0 seconds",
            occurred_at=STARTED_AT,
        ),
    )

    body = trial_to_atif_ingest(
        trial,
        run_id="run-1",
        evaluation_name="exp-1",
        agent_name="a",
        started_at=STARTED_AT,
    )

    assert body["extra"] == {
        "error": {
            "type": "AgentTimeoutError",
            "message": "Agent execution timed out after 900.0 seconds",
        }
    }


def test_trial_to_atif_ingest_omits_absent_error_message() -> None:
    trial = AgentEvalTrial(
        id="trial-failed",
        task_id="task-1",
        status=AgentEvalTrialStatus.FAILED,
        error=TrialError(type="RuntimeError"),
    )

    body = trial_to_atif_ingest(
        trial,
        run_id="run-1",
        evaluation_name="exp-1",
        agent_name="a",
        started_at=STARTED_AT,
    )

    assert body["extra"] == {"error": {"type": "RuntimeError"}}


def test_trial_to_atif_ingest_includes_final_metrics_when_given() -> None:
    body = trial_to_atif_ingest(
        _trial(),
        run_id="run-1",
        evaluation_name="exp-1",
        agent_name="a",
        started_at=STARTED_AT,
        final_metrics={"total_prompt_tokens": 10},
    )
    assert body["final_metrics"] == {"total_prompt_tokens": 10}


def test_trial_to_atif_ingest_adds_invocation_window_when_ended_at_given() -> None:
    # ended_at gives the single-step trajectory a real duration (via the invocation window) so Intake's
    # root-span latency is the trial's runtime instead of 0.
    ended = STARTED_AT + timedelta(seconds=12.5)
    body = trial_to_atif_ingest(
        _trial(), run_id="run-1", evaluation_name="exp-1", agent_name="a", started_at=STARTED_AT, ended_at=ended
    )
    (step,) = body["steps"]
    assert step["extra"] == {
        "invocation": {"start_timestamp": STARTED_AT.timestamp(), "end_timestamp": ended.timestamp()}
    }


def test_trial_to_atif_ingest_puts_the_runtime_window_on_the_last_untimed_recorded_step() -> None:
    ended = STARTED_AT + timedelta(seconds=12.5)
    steps = [
        {"step_id": 1, "source": "user", "message": "q", "timestamp": STARTED_AT.isoformat()},
        {"step_id": 2, "source": "agent", "message": "a", "timestamp": STARTED_AT.isoformat(), "extra": {"k": "v"}},
    ]

    body = trial_to_atif_ingest(
        _trial(),
        run_id="run-1",
        evaluation_name="exp-1",
        agent_name="a",
        started_at=STARTED_AT,
        ended_at=ended,
        steps=RecordedSteps(steps, timed=False),
    )

    first, last = body["steps"]
    assert first == steps[0]
    assert last["extra"] == {
        "k": "v",
        "invocation": {"start_timestamp": STARTED_AT.timestamp(), "end_timestamp": ended.timestamp()},
    }
    assert "invocation" not in steps[1]["extra"]


def test_trial_to_atif_ingest_leaves_recorded_step_timing_alone() -> None:
    steps = [
        {"step_id": 1, "source": "user", "message": "q", "timestamp": STARTED_AT.isoformat()},
        {"step_id": 2, "source": "agent", "message": "a", "timestamp": "2026-01-02T03:04:14+00:00"},
    ]

    body = trial_to_atif_ingest(
        _trial(),
        run_id="run-1",
        evaluation_name="exp-1",
        agent_name="a",
        started_at=STARTED_AT,
        ended_at=STARTED_AT + timedelta(seconds=12.5),
        steps=RecordedSteps(steps, timed=True),
    )

    assert body["steps"] == steps


def test_trial_to_atif_ingest_omits_invocation_window_without_ended_at() -> None:
    body = trial_to_atif_ingest(
        _trial(), run_id="run-1", evaluation_name="exp-1", agent_name="a", started_at=STARTED_AT
    )
    assert "extra" not in next(iter(body["steps"]))


# --- score_to_evaluator_results: data_type coercions ------------------------


def test_score_row_naming_and_targeting() -> None:
    rows = _rows(
        _score(outputs=[MetricOutput(name="score", value=0.5)]),
        session_id="run-1:trial-1",
        span_id="span-abc",
    )
    assert len(rows) == 1
    assert rows[0]["name"] == "accuracy.score"
    assert rows[0]["session_id"] == "run-1:trial-1"
    assert rows[0]["span_id"] == "span-abc"


def test_one_row_per_output() -> None:
    rows = _rows(
        _score(outputs=[MetricOutput(name="a", value=1.0), MetricOutput(name="b", value=2.0)]),
        span_id="span",
    )
    assert [row["name"] for row in rows] == ["accuracy.a", "accuracy.b"]


@pytest.mark.parametrize("value", [True, BooleanValue(True)])
def test_boolean_coercion_true(value: object) -> None:
    row = _rows(_score(outputs=[MetricOutput(name="passed", value=value)]))[0]
    assert row["data_type"] == "BOOLEAN"
    assert row["value"] == 1.0
    assert "string_value" not in row


@pytest.mark.parametrize("value", [False, BooleanValue(False)])
def test_boolean_coercion_false(value: object) -> None:
    row = _rows(_score(outputs=[MetricOutput(name="passed", value=value)]))[0]
    assert row["data_type"] == "BOOLEAN"
    assert row["value"] == 0.0


@pytest.mark.parametrize("value", [0.87, 3, ContinuousScore(0.87), DiscreteScore(3)])
def test_numeric_coercion(value: object) -> None:
    row = _rows(_score(outputs=[MetricOutput(name="m", value=value)]))[0]
    assert row["data_type"] == "NUMERIC"
    assert isinstance(row["value"], float)
    assert "string_value" not in row


@pytest.mark.parametrize("value", ["PASS", Label("PASS")])
def test_text_coercion(value: object) -> None:
    row = _rows(_score(outputs=[MetricOutput(name="verdict", value=value)]))[0]
    assert row["data_type"] == "TEXT"
    assert row["string_value"] == "PASS"
    assert "value" not in row


def test_comment_taken_from_first_diagnostic() -> None:
    score = _score(
        outputs=[MetricOutput(name="score", value=1.0)],
        diagnostics=[
            AgentEvalDiagnostic(severity=AgentEvalDiagnosticSeverity.WARNING, message="first"),
            AgentEvalDiagnostic(severity=AgentEvalDiagnosticSeverity.INFO, message="second"),
        ],
    )
    row = _rows(score)[0]
    assert row["comment"] == "first"


def test_comment_prefers_each_outputs_matching_diagnostic_over_general_fallback() -> None:
    score = _score(
        outputs=[
            MetricOutput(name="reward", value=1.0),
            MetricOutput(name="format_ok", value=1.0),
        ],
        diagnostics=[
            AgentEvalDiagnostic(
                severity=AgentEvalDiagnosticSeverity.WARNING,
                message="format detail",
                details={"output": "format_ok"},
            ),
            AgentEvalDiagnostic(severity=AgentEvalDiagnosticSeverity.INFO, message="general detail"),
            AgentEvalDiagnostic(
                severity=AgentEvalDiagnosticSeverity.WARNING,
                message="reward detail",
                details={"output": "reward"},
            ),
        ],
    )

    rows = _rows(score)

    assert [(row["name"], row["comment"]) for row in rows] == [
        ("accuracy.reward", "reward detail"),
        ("accuracy.format_ok", "format detail"),
    ]


def test_comment_absent_without_diagnostics() -> None:
    row = _rows(_score(outputs=[MetricOutput(name="score", value=1.0)]))[0]
    assert "comment" not in row


# --- score_to_evaluator_results: failed outputs -----------------------------


def test_non_finite_outputs_become_failed_rows() -> None:
    """NaN is the SDK's placeholder for a tolerated failure, so Intake gets a FAILED row, not silence."""
    rows, skipped = score_to_evaluator_results(
        _score(outputs=[MetricOutput(name="score", value=1.0), MetricOutput(name="broken", value=math.nan)]),
        session_id="s",
        span_id="sp",
    )
    assert skipped == []
    assert [(row["name"], row.get("status", "SCORED")) for row in rows] == [
        ("accuracy.score", "SCORED"),
        ("accuracy.broken", "FAILED"),
    ]
    failed = rows[1]
    assert "value" not in failed and "string_value" not in failed
    assert failed["comment"] == "non-finite value"


def test_failed_score_becomes_one_failed_row_per_declared_output() -> None:
    """A FAILED score has no outputs of its own, so the metric's declared outputs name the rows."""
    diagnostic = AgentEvalDiagnostic(
        severity=AgentEvalDiagnosticSeverity.ERROR, message="judge endpoint unreachable", source="accuracy"
    )
    rows, skipped = score_to_evaluator_results(
        _score(outputs=[], status=AgentEvalScoreStatus.FAILED, diagnostics=[diagnostic]),
        session_id="s",
        span_id="sp",
        output_names=["score", "passed"],
    )
    assert skipped == []
    assert [(row["name"], row["status"], row["comment"]) for row in rows] == [
        ("accuracy.score", "FAILED", "judge endpoint unreachable"),
        ("accuracy.passed", "FAILED", "judge endpoint unreachable"),
    ]
    assert all("value" not in row for row in rows)


def test_failed_score_with_partial_outputs_still_covers_every_declared_output() -> None:
    """A FAILED score that carries some outputs must not hide the declared ones it lacks."""
    rows, _ = score_to_evaluator_results(
        _score(outputs=[MetricOutput(name="score", value=math.nan)], status=AgentEvalScoreStatus.FAILED),
        session_id="s",
        span_id="sp",
        output_names=["score", "passed"],
    )
    assert [(row["name"], row["status"]) for row in rows] == [
        ("accuracy.score", "FAILED"),
        ("accuracy.passed", "FAILED"),
    ]


def test_failed_rows_keep_each_outputs_own_diagnostic() -> None:
    """A metric that failed per output (Harbor rewards) explains each FAILED row with its own cause."""
    rows, _ = score_to_evaluator_results(
        _score(
            outputs=[],
            status=AgentEvalScoreStatus.FAILED,
            diagnostics=[
                AgentEvalDiagnostic(
                    severity=AgentEvalDiagnosticSeverity.ERROR,
                    message="reward 'passed' was not measured",
                    details={"output": "passed"},
                ),
                AgentEvalDiagnostic(
                    severity=AgentEvalDiagnosticSeverity.ERROR,
                    message="reward 'score' was not measured",
                    details={"output": "score"},
                ),
            ],
        ),
        session_id="s",
        span_id="sp",
        output_names=["score", "passed", "unexplained"],
    )
    assert [(row["name"], row.get("comment")) for row in rows] == [
        ("accuracy.score", "reward 'score' was not measured"),
        ("accuracy.passed", "reward 'passed' was not measured"),
        ("accuracy.unexplained", None),
    ]


def test_diagnostic_tagged_with_no_output_counts_as_score_level() -> None:
    """``details["output"] = None`` is how the rewards metric marks a score-wide finding."""
    score = _score(
        outputs=[MetricOutput(name="score", value=1.0)],
        diagnostics=[
            AgentEvalDiagnostic(
                severity=AgentEvalDiagnosticSeverity.WARNING,
                message="reward entry was rejected",
                details={"output": None},
            )
        ],
    )
    assert _rows(score)[0]["comment"] == "reward entry was rejected"


def test_failed_score_without_declared_outputs_names_the_metric_itself() -> None:
    rows, _ = score_to_evaluator_results(
        _score(outputs=[], status=AgentEvalScoreStatus.FAILED), session_id="s", span_id="sp"
    )
    assert [(row["name"], row["status"]) for row in rows] == [("accuracy", "FAILED")]
    assert "comment" not in rows[0]


def _atif_document() -> dict[str, Any]:
    """An ATIF trajectory shaped like the one Harbor's ATIF-capable agents write."""
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "harbor-session",
        "agent": {"name": "codex", "version": "1.0"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "solve it", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"step_id": 2, "source": "agent", "message": "204", "timestamp": "2026-01-01T00:00:01+00:00"},
        ],
    }


def _trial_with_trace(tmp_path: Path, *, document: object, evidence_format: str = "atif") -> AgentEvalTrial:
    trace_path = tmp_path / "trajectory.json"
    trace_path.write_text(json.dumps(document))
    return AgentEvalTrial(
        id="trial-1",
        task_id="task-1",
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text="204"),
        evidence=CandidateEvidence(
            descriptors={"trace": EvidenceDescriptor(kind="trace", format=evidence_format, ref=str(trace_path))}
        ),
    )


@pytest.mark.asyncio
async def test_atif_trace_evidence_supplies_the_real_steps(tmp_path: Path) -> None:
    trial = _trial_with_trace(tmp_path, document=_atif_document())

    recorded = await atif_steps_from_trial(trial, started_at=STARTED_AT)

    assert recorded is not None
    assert [step["source"] for step in recorded.steps] == ["user", "agent"]
    assert [step["message"] for step in recorded.steps] == ["solve it", "204"]
    assert recorded.timed


@pytest.mark.asyncio
async def test_the_trajectorys_own_agent_version_is_published_unless_one_is_given(tmp_path: Path) -> None:
    recorded = await atif_steps_from_trial(
        _trial_with_trace(tmp_path, document=_atif_document()), started_at=STARTED_AT
    )
    assert recorded is not None and recorded.agent_version == "1.0"
    trial = _trial()

    def published_version(**kwargs: Any) -> str:
        body = trial_to_atif_ingest(
            trial, run_id="r", evaluation_name="e", agent_name="codex", started_at=STARTED_AT, **kwargs
        )
        return body["agent"]["version"]

    # Harbor records codex's release in the trajectory; the default would overwrite it with "unknown".
    assert published_version(steps=recorded) == "1.0"
    assert published_version(steps=recorded, agent_version="2.0") == "2.0"  # an explicit version still wins
    assert published_version() == DEFAULT_AGENT_VERSION  # no trajectory, nothing to take it from


@pytest.mark.asyncio
async def test_steps_without_a_timestamp_fall_back_to_the_run_start(tmp_path: Path) -> None:
    document = _atif_document()
    del document["steps"][1]["timestamp"]

    recorded = await atif_steps_from_trial(_trial_with_trace(tmp_path, document=document), started_at=STARTED_AT)

    # Intake keys spans on start_time; a step with no timestamp would take the ingest clock and
    # duplicate on re-publish instead of replacing.
    assert recorded is not None
    assert recorded.steps[1]["timestamp"] == STARTED_AT.isoformat()


@pytest.mark.asyncio
async def test_a_single_recorded_timestamp_conveys_no_duration(tmp_path: Path) -> None:
    document = _atif_document()
    del document["steps"][1]["timestamp"]

    recorded = await atif_steps_from_trial(_trial_with_trace(tmp_path, document=document), started_at=STARTED_AT)

    assert recorded is not None
    assert not recorded.timed


@pytest.mark.asyncio
async def test_a_recorded_timestamp_equal_to_the_run_start_still_counts_as_recorded(tmp_path: Path) -> None:
    document = _atif_document()
    document["steps"][0]["timestamp"] = STARTED_AT.isoformat()
    document["steps"][1]["timestamp"] = STARTED_AT.strftime("%Y-%m-%dT%H:%M:%SZ")

    recorded = await atif_steps_from_trial(_trial_with_trace(tmp_path, document=document), started_at=STARTED_AT)

    assert recorded is not None
    assert recorded.timed


@pytest.mark.asyncio
async def test_an_invocation_window_makes_untimestamped_steps_timed(tmp_path: Path) -> None:
    document = _atif_document()
    for step in document["steps"]:
        del step["timestamp"]
    document["steps"][1]["extra"] = {"invocation": {"start_timestamp": 1.0, "end_timestamp": 3.0}}

    recorded = await atif_steps_from_trial(_trial_with_trace(tmp_path, document=document), started_at=STARTED_AT)

    assert recorded is not None
    assert recorded.timed


@pytest.mark.asyncio
async def test_trace_evidence_that_is_not_atif_is_ignored(tmp_path: Path) -> None:
    trial = _trial_with_trace(tmp_path, document=_atif_document(), evidence_format="json")

    assert await atif_steps_from_trial(trial, started_at=STARTED_AT) is None


@pytest.mark.asyncio
async def test_unreadable_trace_evidence_does_not_fail_the_publish(tmp_path: Path) -> None:
    trial = _trial_with_trace(tmp_path, document=_atif_document())
    (tmp_path / "trajectory.json").unlink()

    assert await atif_steps_from_trial(trial, started_at=STARTED_AT) is None


@pytest.mark.asyncio
async def test_a_trial_with_no_evidence_has_no_steps() -> None:
    assert await atif_steps_from_trial(_trial(), started_at=STARTED_AT) is None


def test_supplied_steps_replace_the_synthetic_single_step() -> None:
    real_steps = [
        {"step_id": 1, "source": "user", "message": "solve it"},
        {"step_id": 2, "source": "agent", "message": "204"},
    ]

    body = trial_to_atif_ingest(
        _trial(),
        run_id="run-1",
        evaluation_name="eval-1",
        agent_name="codex",
        started_at=STARTED_AT,
        steps=RecordedSteps(real_steps, timed=False),
    )

    assert body["steps"] == real_steps


def test_no_supplied_steps_still_emits_the_final_output_text() -> None:
    body = trial_to_atif_ingest(
        _trial(output_text="only answer"),
        run_id="run-1",
        evaluation_name="eval-1",
        agent_name="oracle",
        started_at=STARTED_AT,
    )

    assert [step["message"] for step in body["steps"]] == ["only answer"]


@pytest.mark.asyncio
async def test_step_ids_are_renumbered_to_the_sequence_ingest_requires(tmp_path: Path) -> None:
    document = _atif_document()
    for step in document["steps"]:
        del step["step_id"]

    recorded = await atif_steps_from_trial(_trial_with_trace(tmp_path, document=document), started_at=STARTED_AT)

    # Ingest rejects a trajectory whose ids are not one-based and sequential.
    assert recorded is not None
    assert [step["step_id"] for step in recorded.steps] == [1, 2]


@pytest.mark.asyncio
async def test_a_trajectory_with_no_steps_is_treated_as_absent(tmp_path: Path) -> None:
    document = _atif_document()
    document["steps"] = []

    trial = _trial_with_trace(tmp_path, document=document)

    # Ingest needs at least one step, so an empty trajectory must fall back to the synthetic step.
    assert await atif_steps_from_trial(trial, started_at=STARTED_AT) is None


# --- otlp_ingest_for_trial: caller identity ---------------------------------

_OTLP_TRACE_ID = "0123456789abcdef0123456789abcdef"
_OTLP_ROOT_SPAN_ID = "0102030405060708"
_OTLP_CHILD_SPAN_ID = "aabbccddeeff0011"


def _otlp_document() -> dict[str, Any]:
    """A trace whose root already records the producer's agent and model identity."""
    return {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": _OTLP_TRACE_ID,
                                "spanId": _OTLP_ROOT_SPAN_ID,
                                "name": "agent run",
                                "attributes": [
                                    {"key": "gen_ai.agent.name", "value": {"stringValue": "relay"}},
                                    {"key": "gen_ai.agent.version", "value": {"stringValue": "0.162.1"}},
                                    {"key": "gen_ai.request.model", "value": {"stringValue": "relay-model"}},
                                ],
                            },
                            {
                                "traceId": _OTLP_TRACE_ID,
                                "spanId": _OTLP_CHILD_SPAN_ID,
                                "parentSpanId": _OTLP_ROOT_SPAN_ID,
                                "name": "child",
                                "attributes": [{"key": "gen_ai.agent.name", "value": {"stringValue": "subagent"}}],
                            },
                        ]
                    }
                ]
            }
        ]
    }


def _published_spans(trial: AgentEvalTrial, **identity: Any):
    """Publish the trial's OTLP trace and return the sent payload's spans, by name."""
    published = None

    async def run() -> dict[str, Any]:
        nonlocal published
        published = await otlp_ingest_for_trial(
            trial,
            session_id="run-1:t-1",
            evaluation_name="exp-1",
            measurements=TrialMeasurements(),
            started_at=STARTED_AT,
            **identity,
        )
        assert published is not None
        payload, _span_id = published
        return _parse_spans(payload)

    return run


def _parse_spans(payload: bytes) -> dict[str, Any]:
    request = ExportTraceServiceRequest()
    request.ParseFromString(payload)
    spans = [span for rs in request.resource_spans for ss in rs.scope_spans for span in ss.spans]
    return {span.name: span for span in spans}


@pytest.mark.asyncio
async def test_an_omitted_identity_keeps_the_producer_values(tmp_path: Path) -> None:
    # No agent name, no model, and the default version: the producer recorded who ran the trace,
    # and the caller passed nothing that should replace it. Intake reads the agent back from the
    # root span's gen_ai.agent.name, so overriding it with defaults would mislabel the trace.
    trial = _trial_with_trace(tmp_path, document=_otlp_document(), evidence_format="otlp")

    spans = await _published_spans(trial)()

    attributes = {attribute.key: attribute.value.string_value for attribute in spans["agent run"].attributes}
    assert attributes["gen_ai.agent.name"] == "relay"
    assert attributes["gen_ai.agent.version"] == "0.162.1"
    assert attributes["gen_ai.request.model"] == "relay-model"


@pytest.mark.asyncio
async def test_an_empty_identity_keeps_the_producer_values(tmp_path: Path) -> None:
    # An empty string is not a name. Intake reads a blank agent name as no agent, so stamping
    # "" would publish a trace that is worse than the one the producer recorded.
    trial = _trial_with_trace(tmp_path, document=_otlp_document(), evidence_format="otlp")

    spans = await _published_spans(trial, agent_name="", model_name="")()

    attributes = {attribute.key: attribute.value.string_value for attribute in spans["agent run"].attributes}
    assert attributes["gen_ai.agent.name"] == "relay"
    assert attributes["gen_ai.request.model"] == "relay-model"


@pytest.mark.asyncio
async def test_an_explicit_identity_overrides_the_root_only(tmp_path: Path) -> None:
    # An explicit name, version, and model replace the producer's on the root span. Child spans
    # keep their own identity: stamping them would clobber a subagent's recorded name.
    trial = _trial_with_trace(tmp_path, document=_otlp_document(), evidence_format="otlp")

    spans = await _published_spans(trial, agent_name="eval-agent", agent_version="3.1.4", model_name="gpt-x")()

    root = {attribute.key: attribute.value.string_value for attribute in spans["agent run"].attributes}
    assert root["gen_ai.agent.name"] == "eval-agent"
    assert root["gen_ai.agent.version"] == "3.1.4"
    assert root["gen_ai.request.model"] == "gpt-x"

    child_keys = {attribute.key: attribute.value.string_value for attribute in spans["child"].attributes}
    assert child_keys["gen_ai.agent.name"] == "subagent"
    assert "gen_ai.agent.version" not in child_keys
    assert "gen_ai.request.model" not in child_keys


@pytest.mark.asyncio
async def test_the_default_unknown_version_keeps_the_producer_version(tmp_path: Path) -> None:
    # DEFAULT_AGENT_VERSION is "I have no version to give", matching the ATIF path: a producer
    # version already on the root span wins over the placeholder.
    trial = _trial_with_trace(tmp_path, document=_otlp_document(), evidence_format="otlp")

    spans = await _published_spans(trial, agent_name="eval-agent")()

    root = {attribute.key: attribute.value.string_value for attribute in spans["agent run"].attributes}
    assert root["gen_ai.agent.version"] == "0.162.1"


@pytest.mark.asyncio
async def test_the_cached_trace_survives_a_publish_byte_for_byte(tmp_path: Path) -> None:
    # The evidence handle caches the parsed messages, so a publish that stamped them in place
    # would change what every later reader of the cached evidence sees.
    trial = _trial_with_trace(tmp_path, document=_otlp_document(), evidence_format="otlp")
    assert trial.evidence is not None
    handle = await trial.evidence.trace("trace", format="otlp")
    before = ExportTraceServiceRequest(resource_spans=await handle.resource_spans()).SerializeToString()

    run = _published_spans(trial, agent_name="eval-agent", model_name="gpt-x")
    await run()
    await run()  # a second publish reads the same cache again

    after = ExportTraceServiceRequest(resource_spans=await handle.resource_spans()).SerializeToString()
    assert after == before
