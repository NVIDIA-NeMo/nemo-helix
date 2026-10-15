# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.enums import MetricType
from nhx_evals_sdk.metrics.protocol import validate_metric_result
from nhx_evals_sdk.metrics.tunable_rag_defaults import normalize_score_weights
from nhx_evals_sdk.metrics.tunable_rag_evaluator import TunableRagEvaluatorMetric, _parse_json_object
from nhx_evals_sdk.values.models import Model


def _make_model() -> Model:
    return Model(
        url="https://judge.example.test/v1/chat/completions",
        name="judge-model",
        format="openai",
    )


def _judge_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": json.dumps(payload)}}],
    }


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ({"coverage": 0.5, "correctness": 0.3, "relevance": 0.2}, (0.5, 0.3, 0.2)),
        ({"coverage": 1.0, "correctness": 1.0, "relevance": 1.0}, pytest.approx((1 / 3, 1 / 3, 1 / 3))),
    ],
)
def test_normalize_score_weights(weights: dict[str, float], expected: tuple[float, float, float]) -> None:
    assert normalize_score_weights(weights) == expected


def test_parse_json_object_strips_markdown_fence() -> None:
    parsed = _parse_json_object('```json\n{"score": 0.8, "reasoning": "ok"}\n```')
    assert parsed == {"score": 0.8, "reasoning": "ok"}


def test_extract_eval_fields_reads_instruction() -> None:
    from nhx_evals_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput
    from nhx_evals_sdk.metrics.tunable_rag_evaluator import _extract_eval_fields

    metric_input = MetricInput(
        row=DatasetRow(
            data={
                "inputs": {"instruction": "What is 2+2?"},
                "reference": {"answer": "4"},
            }
        ),
        candidate=CandidateOutput(output_text="4"),
    )
    instruction, answer, generated = _extract_eval_fields(metric_input)
    assert instruction == "What is 2+2?"
    assert answer == "4"
    assert generated == "4"


@pytest.mark.asyncio
async def test_default_scoring_emits_weighted_average_and_subscores() -> None:
    metric = TunableRagEvaluatorMetric(
        model=_make_model(),
        default_scoring=True,
        default_score_weights={"coverage": 0.5, "correctness": 0.3, "relevance": 0.2},
    )

    async def fake_inference(model, request, max_retries, client=None):  # noqa: ANN001
        return _judge_response(
            {
                "coverage_score": 1.0,
                "correctness_score": 0.5,
                "relevance_score": 0.0,
                "reasoning": "partially correct",
            }
        )

    metric._inference_fn = fake_inference  # noqa: SLF001

    result = await compute_scores(
        metric,
        {
            "inputs": {"instruction": "Who invented the telephone?"},
            "reference": {"answer": "Alexander Graham Bell"},
        },
        {"output_text": "Bell invented the telephone."},
    )

    validate_metric_result(result, metric.output_spec())
    values = {output.name: output.value for output in result.outputs}
    assert values["coverage_score"] == 1.0
    assert values["correctness_score"] == 0.5
    assert values["relevance_score"] == 0.0
    assert values["average_score"] == pytest.approx(0.5 * 1.0 + 0.3 * 0.5 + 0.2 * 0.0)
    assert values["reasoning"] == "partially correct"


@pytest.mark.asyncio
async def test_default_scoring_clamps_scores_to_unit_scale() -> None:
    metric = TunableRagEvaluatorMetric(
        model=_make_model(),
        default_scoring=True,
        default_score_weights={"coverage": 0.5, "correctness": 0.3, "relevance": 0.2},
    )

    async def fake_inference(model, request, max_retries, client=None):  # noqa: ANN001
        return _judge_response(
            {
                "coverage_score": 100.0,
                "correctness_score": 10.0,
                "relevance_score": -2.0,
                "reasoning": "judge ignored the 0-1 scale",
            }
        )

    metric._inference_fn = fake_inference  # noqa: SLF001

    result = await compute_scores(
        metric,
        {"inputs": {"instruction": "q"}, "reference": {"answer": "a"}},
        {"output_text": "a"},
    )

    values = {output.name: output.value for output in result.outputs}
    assert values["coverage_score"] == 1.0
    assert values["correctness_score"] == 1.0
    assert values["relevance_score"] == 0.0
    assert values["average_score"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_default_scoring_rejects_non_finite_scores() -> None:
    metric = TunableRagEvaluatorMetric(model=_make_model(), default_scoring=True)

    async def fake_inference(model, request, max_retries, client=None):  # noqa: ANN001
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"coverage_score": NaN, "correctness_score": 1.0, "relevance_score": 1.0, "reasoning": "bad"}'
                    }
                }
            ]
        }

    metric._inference_fn = fake_inference  # noqa: SLF001

    result = await compute_scores(
        metric,
        {"inputs": {"instruction": "q"}, "reference": {"answer": "a"}},
        {"output_text": "a"},
    )

    values = {output.name: output.value for output in result.outputs}
    assert values["average_score"] == 0.0
    assert "unusable" in str(values["reasoning"]).lower()


@pytest.mark.asyncio
async def test_max_retries_comes_from_run_config_online() -> None:
    from nhx_evals_sdk.values.params import RunConfig, RunConfigOnline

    metric = TunableRagEvaluatorMetric(model=_make_model(), default_scoring=True)
    captured: dict[str, Any] = {}

    async def fake_inference(model, request, max_retries, client=None):  # noqa: ANN001
        captured["max_retries"] = max_retries
        return _judge_response(
            {
                "coverage_score": 1.0,
                "correctness_score": 1.0,
                "relevance_score": 1.0,
                "reasoning": "ok",
            }
        )

    metric._inference_fn = fake_inference  # noqa: SLF001
    metric.apply_evaluation_job_params(RunConfigOnline(max_retries=7))

    await compute_scores(
        metric,
        {"inputs": {"instruction": "q"}, "reference": {"answer": "a"}},
        {"output_text": "a"},
    )
    assert captured["max_retries"] == 7

    # Offline RunConfig has no max_retries; keep the last online value.
    metric.apply_evaluation_job_params(RunConfig())
    await compute_scores(
        metric,
        {"inputs": {"instruction": "q"}, "reference": {"answer": "a"}},
        {"output_text": "a"},
    )
    assert captured["max_retries"] == 7


@pytest.mark.asyncio
async def test_custom_scoring_emits_average_score_only() -> None:
    metric = TunableRagEvaluatorMetric(
        model=_make_model(),
        default_scoring=False,
        judge_llm_prompt="Score from 0 to 1.",
    )

    async def fake_inference(model, request, max_retries, client=None):  # noqa: ANN001
        return _judge_response({"score": 0.75, "reasoning": "good answer"})

    metric._inference_fn = fake_inference  # noqa: SLF001

    result = await compute_scores(
        metric,
        {"inputs": {"instruction": "2+2?"}, "reference": {"answer": "4"}},
        {"output_text": "4"},
    )

    validate_metric_result(result, metric.output_spec())
    values = {output.name: output.value for output in result.outputs}
    assert values["average_score"] == 0.75
    assert values["reasoning"] == "good answer"


@pytest.mark.asyncio
async def test_parse_failure_returns_zero_scores() -> None:
    metric = TunableRagEvaluatorMetric(model=_make_model(), default_scoring=True)

    async def fake_inference(model, request, max_retries, client=None):  # noqa: ANN001
        return _judge_response("not-json")

    metric._inference_fn = fake_inference  # noqa: SLF001

    result = await compute_scores(
        metric,
        {"inputs": {"instruction": "q"}, "reference": {"answer": "a"}},
        {"output_text": "bad"},
    )

    values = {output.name: output.value for output in result.outputs}
    assert values["average_score"] == 0.0
    assert "parsing" in str(values["reasoning"]).lower()


def test_output_spec_names() -> None:
    default_metric = TunableRagEvaluatorMetric(model=_make_model(), default_scoring=True)
    custom_metric = TunableRagEvaluatorMetric(model=_make_model(), default_scoring=False)
    assert output_names(default_metric) == [
        "average_score",
        "coverage_score",
        "correctness_score",
        "relevance_score",
        "reasoning",
    ]
    assert output_names(custom_metric) == ["average_score", "reasoning"]
    assert default_metric.type == MetricType.TUNABLE_RAG_EVALUATOR
