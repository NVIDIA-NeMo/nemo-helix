# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-adjacent value types for evaluator SDK metrics."""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, StringConstraints, field_serializer, field_validator

from nemo_evaluator_sdk.values.evidence import CandidateEvidence

MetricTypeName = Annotated[str, StringConstraints(min_length=1)]


class DatasetRow(BaseModel):
    """Original dataset row plus optional stable row identity."""

    model_config = ConfigDict(extra="forbid")

    row_index: int | None = None
    data: dict[str, Any]


class CandidateOutput(BaseModel):
    """Candidate or prediction output being scored for one dataset row."""

    model_config = ConfigDict(extra="forbid")

    output_text: str | None = None
    response: Any | None = None
    trajectory: Any | None = None
    evidence: CandidateEvidence | None = Field(
        default=None,
        description="Named evidence captured for the candidate (final state, traces, logs, ...) for agent eval.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_sample(self) -> dict[str, Any]:
        """Return a sample-shaped payload for template rendering helpers."""
        sample = dict(self.metadata)
        if self.output_text is not None:
            sample["output_text"] = self.output_text
        if self.response is not None:
            sample["response"] = self.response
        if self.trajectory is not None:
            sample["trajectory"] = self.trajectory
        if self.evidence is not None:
            sample["evidence"] = self.evidence
        return sample


class MetricInput(BaseModel):
    """Complete per-row scoring input passed to a metric."""

    model_config = ConfigDict(extra="forbid")

    row: DatasetRow
    candidate: CandidateOutput


class ContinuousScore(RootModel[float]):
    """Continuous numeric metric value."""


class DiscreteScore(RootModel[int]):
    """Discrete numeric metric value."""


class Label(RootModel[str]):
    """String label metric value."""


class BooleanValue(RootModel[bool]):
    """Boolean metric value."""


class MetricOutputSpec(BaseModel):
    """Schema for one named value emitted by a metric."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str | None = None
    value_schema: type[BaseModel]
    required: bool = True

    @field_validator("name")
    @classmethod
    def _name_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("metric output name must not be empty")
        return value

    @staticmethod
    def continuous_score(name: str, description: str | None = None, *, required: bool = True) -> "MetricOutputSpec":
        return MetricOutputSpec(name=name, description=description, value_schema=ContinuousScore, required=required)

    @staticmethod
    def discrete_score(name: str, description: str | None = None, *, required: bool = True) -> "MetricOutputSpec":
        return MetricOutputSpec(name=name, description=description, value_schema=DiscreteScore, required=required)

    @staticmethod
    def label(name: str, description: str | None = None, *, required: bool = True) -> "MetricOutputSpec":
        return MetricOutputSpec(name=name, description=description, value_schema=Label, required=required)

    @staticmethod
    def boolean(name: str, description: str | None = None, *, required: bool = True) -> "MetricOutputSpec":
        return MetricOutputSpec(name=name, description=description, value_schema=BooleanValue, required=required)

    @staticmethod
    def model(
        name: str,
        value_schema: type[BaseModel],
        description: str | None = None,
        *,
        required: bool = True,
    ) -> "MetricOutputSpec":
        return MetricOutputSpec(name=name, description=description, value_schema=value_schema, required=required)

    def coerce_value(self, value: Any) -> BaseModel:
        """Validate and coerce a raw output value to this spec's declared schema."""
        return self.value_schema.model_validate(value)

    def coerce_output(self, output: "MetricOutput") -> BaseModel:
        """Validate and coerce a named metric output against this spec."""
        if output.name != self.name:
            raise ValueError(f"Expected metric output {self.name!r}, got {output.name!r}")
        return self.coerce_value(output.value)

    def value_json_schema(self) -> dict[str, Any]:
        return self.value_schema.model_json_schema()


class MetricDescriptor(BaseModel):
    """Metadata describing a metric implementation and its declared outputs."""

    model_config = ConfigDict(extra="forbid")

    type: MetricTypeName
    outputs: list[MetricOutputSpec] = Field(min_length=1)

    @field_validator("outputs")
    @classmethod
    def _output_names_must_be_unique(cls, value: list[MetricOutputSpec]) -> list[MetricOutputSpec]:
        names = [output.name for output in value]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate metric output names: {duplicates}")
        return value


class MetricOutput(BaseModel):
    """One named value emitted by a metric."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: Any

    @field_serializer("value")
    def serialize_nan(self, value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return "NaN"
        return value


#: ``details`` keys with a meaning beyond the emitting metric, so a consumer can act on them
#: without knowing which metric wrote them. ``OUTPUT_DETAIL`` names the single metric output a
#: diagnostic is about (absent when it describes the result as a whole); ``REASON_DETAIL`` carries
#: the metric's own short machine-readable cause. Named for the same reason ``TRIAL_STATUS_DETAIL``
#: is: they cross a package boundary, and a rename must not degrade a reader silently.
OUTPUT_DETAIL = "output"
REASON_DETAIL = "reason"


class MetricDiagnostic(BaseModel):
    """One diagnostic finding explaining how a metric score was derived.

    ``message`` is the human-readable entry point. Metric-specific structured
    context (expected/actual values, diffs, check breakdowns, judge rationales,
    etc.) belongs in ``details`` -- see :data:`OUTPUT_DETAIL` and :data:`REASON_DETAIL` for the
    two keys that are read outside the metric that wrote them.
    """

    model_config = ConfigDict(extra="forbid")

    message: str = Field(description="Human-readable diagnostic message.")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional metric-specific structured context for debugging and inspection.",
    )


class MetricResult(BaseModel):
    """Structured row-level metric result."""

    model_config = ConfigDict(extra="forbid")

    outputs: list[MetricOutput]
    diagnostics: list[MetricDiagnostic] = Field(
        default_factory=list,
        description=(
            "Optional diagnostic findings that explain how the result was derived. For debugging and inspection only."
        ),
    )


def validate_metric_result(result: MetricResult, outputs: list[MetricOutputSpec]) -> MetricResult:
    """Validate a metric result against its declared outputs."""
    returned_names = [output.name for output in result.outputs]
    duplicates = sorted({name for name in returned_names if returned_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate metric output names: {duplicates}")

    outputs_by_name = {output.name: output for output in outputs}
    declared_names = [output.name for output in outputs]
    declared = set(declared_names)
    returned = set(returned_names)
    missing = [output.name for output in outputs if output.required and output.name not in returned]
    undeclared = [name for name in returned_names if name not in declared]

    if missing:
        raise ValueError(f"Missing required metric outputs: {missing}")
    if undeclared:
        raise ValueError(f"Undeclared metric outputs: {undeclared}")
    for output in result.outputs:
        outputs_by_name[output.name].coerce_output(output)
    return result
