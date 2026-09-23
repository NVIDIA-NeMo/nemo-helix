# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evaluation context models for span ingest endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationContext(BaseModel):
    """Identifies the Evaluation and optional test case associated with ingested telemetry."""

    evaluation_name: str | None = Field(default=None, description="Name of an existing Evaluation.")
    test_case_name: str | None = Field(default=None, description="Optional producer-supplied test case name.")
    evaluation_id: str | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated alias for evaluation_name. Use evaluation_name instead.",
    )
    test_case_id: str | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated alias for test_case_name. Use test_case_name instead.",
    )

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def normalize_deprecated_fields(cls, data: Any) -> Any:
        """Accept either spelling and keep both response fields consistent."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        cls._normalize_field_pair(normalized, canonical="evaluation_name", deprecated="evaluation_id")
        cls._normalize_field_pair(normalized, canonical="test_case_name", deprecated="test_case_id")
        return normalized

    @staticmethod
    def _normalize_field_pair(data: dict[str, Any], *, canonical: str, deprecated: str) -> None:
        canonical_value = data.get(canonical)
        deprecated_value = data.get(deprecated)
        if canonical_value is not None and deprecated_value is not None and canonical_value != deprecated_value:
            raise ValueError(f"{canonical} and deprecated {deprecated} must match when both are provided")
        value = canonical_value if canonical_value is not None else deprecated_value
        if value is not None:
            data[canonical] = value
            data[deprecated] = value

    def has_values(self) -> bool:
        return self.evaluation_name is not None or self.test_case_name is not None
