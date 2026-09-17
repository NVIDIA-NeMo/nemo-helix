# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared MetricInput -> MetricResult contract."""

from __future__ import annotations

from typing import cast

import pytest
from nemo_evaluator_sdk.metrics.protocol import (
    CandidateOutput,
    ContinuousScore,
    DatasetRow,
    Label,
    Metric,
    MetricDescriptor,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
    validate_metric_result,
)
from pydantic import BaseModel, ValidationError


class JudgeDetails(BaseModel):
    label: str
    rationale: str
    confidence: float


def test_metric_input_groups_row_and_candidate() -> None:
    metric_input = MetricInput(
        row=DatasetRow(row_index=7, data={"answer": "Paris", "category": "geography"}),
        candidate=CandidateOutput(output_text="Paris", metadata={"model": "mock"}),
    )

    assert metric_input.row.row_index == 7
    assert metric_input.candidate.output_text == "Paris"
    assert metric_input.row.data["answer"] == "Paris"
    assert metric_input.candidate.as_sample() == {"model": "mock", "output_text": "Paris"}


def test_metric_output_spec_convenience_constructors_and_json_schema() -> None:
    score = MetricOutputSpec.continuous_score("reward", "Reward score")
    label = MetricOutputSpec.label("judge_label")
    details = MetricOutputSpec.model("judge_details", JudgeDetails)

    assert score.name == "reward"
    assert score.description == "Reward score"
    assert score.required is True
    assert score.value_schema is ContinuousScore
    assert score.value_json_schema()["type"] == "number"
    assert label.value_schema is Label
    assert details.value_schema is JudgeDetails
    schema_properties = cast(dict[str, object], details.value_json_schema()["properties"])
    confidence_schema = cast(dict[str, object], schema_properties["confidence"])
    assert confidence_schema["type"] == "number"


def test_metric_output_spec_coerces_values_to_declared_schema() -> None:
    reward = MetricOutputSpec.continuous_score("reward")
    details = MetricOutputSpec.model("judge_details", JudgeDetails)

    coerced_reward = reward.coerce_output(MetricOutput(name="reward", value=1))
    coerced_details = details.coerce_value({"label": "pass", "rationale": "all checks passed", "confidence": 0.9})

    assert isinstance(coerced_reward, ContinuousScore)
    assert coerced_reward.root == 1.0
    assert isinstance(coerced_details, JudgeDetails)
    assert coerced_details.label == "pass"

    with pytest.raises(ValueError, match="Expected metric output"):
        reward.coerce_output(MetricOutput(name="other", value=1))


def test_metric_descriptor_rejects_duplicate_outputs() -> None:
    with pytest.raises(ValueError, match="duplicate metric output names"):
        MetricDescriptor(
            type="tests.duplicate",
            outputs=[
                MetricOutputSpec.continuous_score("reward"),
                MetricOutputSpec.boolean("reward"),
            ],
        )


def test_metric_descriptor_rejects_empty_type() -> None:
    with pytest.raises(ValidationError):
        MetricDescriptor(type="", outputs=[MetricOutputSpec.continuous_score("reward")])


def test_validate_metric_result_accepts_declared_outputs() -> None:
    outputs = [
        MetricOutputSpec.continuous_score("reward"),
        MetricOutputSpec.boolean("correct"),
        MetricOutputSpec.label("label"),
    ]
    result = MetricResult(
        outputs=[
            MetricOutput(name="reward", value=True),
            MetricOutput(name="correct", value=True),
            MetricOutput(name="label", value="yes"),
        ]
    )

    validated = validate_metric_result(result, outputs)

    assert validated is result


def test_validate_metric_result_rejects_duplicate_output_names() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward")]
    result = MetricResult(
        outputs=[
            MetricOutput(name="reward", value=1.0),
            MetricOutput(name="reward", value=0.0),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate metric output"):
        validate_metric_result(result, outputs)


def test_validate_metric_result_rejects_missing_or_undeclared_outputs() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward"), MetricOutputSpec.continuous_score("format")]

    with pytest.raises(ValueError, match="Missing required metric outputs"):
        validate_metric_result(MetricResult(outputs=[MetricOutput(name="reward", value=1.0)]), outputs)

    with pytest.raises(ValueError, match="Undeclared metric outputs"):
        validate_metric_result(
            MetricResult(
                outputs=[
                    MetricOutput(name="reward", value=1.0),
                    MetricOutput(name="format", value=1.0),
                    MetricOutput(name="extra", value=1.0),
                ]
            ),
            outputs,
        )


def test_validate_metric_result_allows_missing_optional_outputs() -> None:
    outputs = [
        MetricOutputSpec.continuous_score("reward"),
        MetricOutputSpec.continuous_score("format", required=False),
    ]

    result = validate_metric_result(
        MetricResult(outputs=[MetricOutput(name="reward", value=1.0)]),
        outputs,
    )

    assert result.outputs == [MetricOutput(name="reward", value=1.0)]


def test_validate_metric_result_rejects_none_for_continuous_score() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward")]

    with pytest.raises(ValidationError):
        validate_metric_result(MetricResult(outputs=[MetricOutput(name="reward", value=None)]), outputs)


def test_validate_metric_result_rejects_value_that_does_not_match_schema() -> None:
    outputs = [MetricOutputSpec.model("judge_details", JudgeDetails)]
    result = MetricResult(outputs=[MetricOutput(name="judge_details", value={"label": "pass"})])

    with pytest.raises(ValidationError):
        validate_metric_result(result, outputs)


@pytest.mark.asyncio
async def test_minimal_metric_conforms_to_protocol() -> None:
    class MinimalMetric:
        type = "tests.minimal"

        def output_spec(self) -> list[MetricOutputSpec]:
            return [MetricOutputSpec.continuous_score("reward")]

        async def compute_scores(self, input: MetricInput) -> MetricResult:
            assert input.candidate.output_text == "candidate"
            return MetricResult(outputs=[MetricOutput(name="reward", value=1.0)])

    metric = MinimalMetric()

    assert isinstance(metric, Metric)
    result = await metric.compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(output_text="candidate"))
    )
    assert result.outputs == [MetricOutput(name="reward", value=1.0)]


def test_metric_result_rejects_legacy_scores_field() -> None:
    with pytest.raises(ValidationError):
        MetricResult.model_validate({"scores": [{"name": "reward", "value": 1.0}]})
