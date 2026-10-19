# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

from nhx.intake.api.v2.experiments.schemas import EvaluationSessionFilter, EvaluationSessionResponse
from nhx.intake.repository.evaluation_session import EvaluationSessionRow
from nhx.intake.spans.domain import SpanStatus


def test_evaluation_session_from_row_preserves_detailed_payloads() -> None:
    input_text = "x" * 1050
    output_text = "y" * 1050
    row = _session_row(input_text=input_text, output_text=output_text)

    response = EvaluationSessionResponse.from_row(row)

    assert response.input == input_text
    assert response.output == output_text
    assert response.test_case_name == "case"
    assert response.model_dump()["test_case_id"] == "case"


def test_evaluation_session_schema_deprecates_only_test_case_id() -> None:
    properties = EvaluationSessionResponse.model_json_schema()["properties"]

    assert properties["test_case_name"].get("deprecated") is not True
    assert properties["test_case_id"]["deprecated"] is True


def test_evaluation_session_filter_deprecates_only_test_case_id() -> None:
    properties = EvaluationSessionFilter.model_json_schema()["properties"]

    assert properties["test_case_name"].get("deprecated") is not True
    assert properties["test_case_id"]["deprecated"] is True


def test_evaluation_session_from_row_applies_payload_modes() -> None:
    row = _session_row(input_text="x" * 1050, output_text="y" * 1050)

    summary = EvaluationSessionResponse.from_row(row, mode="summary")
    preview = EvaluationSessionResponse.from_row(row, mode="preview")

    assert summary.input is None
    assert summary.output is None
    assert preview.input == "x" * 300
    assert preview.output == "y" * 300


def _session_row(input_text: str, output_text: str) -> EvaluationSessionRow:
    now = datetime.now(timezone.utc)
    return EvaluationSessionRow(
        workspace="default",
        evaluation_name="evaluation",
        session_id="session",
        test_case_name="case",
        trace_id="trace",
        root_span_id="root",
        started_at=now,
        ended_at=now,
        latency_ms=10,
        status=SpanStatus.SUCCESS,
        input=input_text,
        output=output_text,
        input_tokens=1,
        output_tokens=2,
        cached_tokens=3,
        cost_total_usd=0.01,
        evaluator_scores={"score": 1.0},
    )
