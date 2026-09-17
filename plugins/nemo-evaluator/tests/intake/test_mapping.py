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
from nemo_evaluator.intake.mapping import (
    ATIF_SCHEMA_VERSION,
    DEFAULT_AGENT_VERSION,
    atif_steps_from_trial,
    run_task_to_evaluation_context,
    score_to_evaluator_results,
    session_id_for,
    trial_to_atif_ingest,
)
from nemo_evaluator_sdk.agent_eval.scores import (
    AgentEvalDiagnostic,
    AgentEvalDiagnosticSeverity,
    AgentEvalScoreStatus,
    AgentEvalTaskScore,
)
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, TrialError
from nemo_evaluator_sdk.metrics.protocol import (
    BooleanValue,
    ContinuousScore,
    DiscreteScore,
    Label,
    MetricOutput,
)
from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from nemo_platform_plugin.intake.types import EvaluatorResultCreateParams

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


# --- score_to_evaluator_results: skipped outputs ----------------------------


def test_non_finite_outputs_are_skipped_not_dropped_silently() -> None:
    rows, skipped = score_to_evaluator_results(
        _score(outputs=[MetricOutput(name="score", value=1.0), MetricOutput(name="broken", value=math.nan)]),
        session_id="s",
        span_id="sp",
    )
    assert [row["name"] for row in rows] == ["accuracy.score"]
    assert [(item.name, item.reason) for item in skipped] == [("accuracy.broken", "non-finite value")]


def test_failed_score_yields_no_rows_and_skips_every_output() -> None:
    rows, skipped = score_to_evaluator_results(
        _score(
            outputs=[MetricOutput(name="score", value=1.0), MetricOutput(name="passed", value=True)],
            status=AgentEvalScoreStatus.FAILED,
        ),
        session_id="s",
        span_id="sp",
    )
    assert rows == []
    assert [(item.name, item.reason) for item in skipped] == [
        ("accuracy.score", "scoring failed"),
        ("accuracy.passed", "scoring failed"),
    ]


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

    steps = await atif_steps_from_trial(trial, started_at=STARTED_AT)

    assert steps is not None
    assert [step["source"] for step in steps] == ["user", "agent"]
    assert [step["message"] for step in steps] == ["solve it", "204"]


@pytest.mark.asyncio
async def test_steps_without_a_timestamp_fall_back_to_the_run_start(tmp_path: Path) -> None:
    document = _atif_document()
    del document["steps"][1]["timestamp"]

    steps = await atif_steps_from_trial(_trial_with_trace(tmp_path, document=document), started_at=STARTED_AT)

    # Intake keys spans on start_time; a step with no timestamp would take the ingest clock and
    # duplicate on re-publish instead of replacing.
    assert steps is not None
    assert steps[1]["timestamp"] == STARTED_AT.isoformat()


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
        steps=real_steps,  # ty: ignore[invalid-argument-type]
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

    steps = await atif_steps_from_trial(_trial_with_trace(tmp_path, document=document), started_at=STARTED_AT)

    # Ingest rejects a trajectory whose ids are not one-based and sequential.
    assert steps is not None
    assert [step["step_id"] for step in steps] == [1, 2]


@pytest.mark.asyncio
async def test_a_trajectory_with_no_steps_falls_back_to_the_synthetic_step(tmp_path: Path) -> None:
    document = _atif_document()
    document["steps"] = []

    trial = _trial_with_trace(tmp_path, document=document)
    body = trial_to_atif_ingest(
        trial,
        run_id="run-1",
        evaluation_name="eval-1",
        agent_name="codex",
        started_at=STARTED_AT,
        steps=await atif_steps_from_trial(trial, started_at=STARTED_AT),
    )

    # Ingest needs at least one step, so an empty trajectory must not empty the request.
    assert [step["message"] for step in body["steps"]] == ["204"]
