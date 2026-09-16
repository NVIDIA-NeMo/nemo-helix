# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility tests for evaluation-context identifier renames."""

import pytest
from nhx.intake.spans.ingest.evaluation_context import EvaluationContext
from pydantic import ValidationError

EXPECTED_CONTEXT = {
    "evaluation_name": "eval-a",
    "test_case_name": "case-a",
    "evaluation_id": "eval-a",
    "test_case_id": "case-a",
}


def test_evaluation_context_accepts_canonical_fields() -> None:
    context = EvaluationContext.model_validate({"evaluation_name": "eval-a", "test_case_name": "case-a"})

    assert context.model_dump() == EXPECTED_CONTEXT


@pytest.mark.parametrize(
    "payload",
    [
        {"evaluation_id": "eval-a", "test_case_id": "case-a"},
        {"evaluation_name": "eval-a", "test_case_id": "case-a"},
        {
            "evaluation_name": "eval-a",
            "evaluation_id": "eval-a",
            "test_case_name": "case-a",
            "test_case_id": "case-a",
        },
    ],
)
def test_evaluation_context_normalizes_deprecated_and_mixed_fields(payload: dict[str, str]) -> None:
    context = EvaluationContext.model_validate(payload)

    assert context.model_dump() == EXPECTED_CONTEXT


@pytest.mark.parametrize(
    ("canonical", "deprecated"),
    [
        ("evaluation_name", "evaluation_id"),
        ("test_case_name", "test_case_id"),
    ],
)
def test_evaluation_context_rejects_conflicting_names(canonical: str, deprecated: str) -> None:
    with pytest.raises(ValidationError, match=f"{canonical} and deprecated {deprecated} must match"):
        EvaluationContext.model_validate({canonical: "new-value", deprecated: "old-value"})


def test_evaluation_context_schema_marks_old_fields_deprecated() -> None:
    properties = EvaluationContext.model_json_schema()["properties"]

    assert properties["evaluation_name"].get("deprecated") is not True
    assert properties["test_case_name"].get("deprecated") is not True
    assert properties["evaluation_id"]["deprecated"] is True
    assert properties["test_case_id"]["deprecated"] is True
