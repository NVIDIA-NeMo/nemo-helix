# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nhx_evals_sdk import Evaluator, LocalBackend
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.resolution import collect_model_refs, resolve_model_refs
from nhx_evals_sdk.resolver_protocols import ModelResolver
from nhx_evals_sdk.resolvers import LocalModelResolver
from nhx_evals_sdk.values.metrics import (
    LLMJudge,
    default_judge_prompt_template_completions,
)
from nhx_evals_sdk.values.models import Model, ModelRef
from nhx_evals_sdk.values.scores import JSONScoreParser, RangeScore
from pydantic import BaseModel, ValidationError


def _model(name: str = "judge") -> Model:
    return Model(url="https://models.example.test/v1/chat/completions", name=name)


def _completion_model(name: str = "judge") -> Model:
    return Model(url="https://models.example.test/v1/completions", name=name)


class ModelBackedMetric(BaseModel):
    type: str = "model-backed"
    model: Model | ModelRef

    def model_refs(self) -> dict[str, ModelRef]:
        return collect_model_refs(self)

    async def resolve_models(self, model_resolver: ModelResolver) -> None:
        await resolve_model_refs(self, model_resolver)

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.label("model")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        if isinstance(self.model, ModelRef):
            raise AssertionError("model was not resolved")
        return MetricResult(outputs=[MetricOutput(name="model", value=self.model.name)])


def test_local_model_resolver_registration_duplicate_and_missing_refs() -> None:
    resolver = LocalModelResolver()
    model_ref = ModelRef(root="workspace/judge")
    resolver.register_model(model_ref, _model())

    assert resolver.get_model(model_ref).name == "judge"
    with pytest.raises(ValueError, match="already registered"):
        resolver.register_model(model_ref, _model("other"))
    with pytest.raises(ValueError, match="not registered"):
        resolver.get_model(ModelRef(root="workspace/missing"))

    removed = resolver.unregister_model(model_ref)
    assert removed.name == "judge"
    with pytest.raises(ValueError, match="not registered"):
        resolver.unregister_model(model_ref)


def test_model_ref_uses_entity_name_pattern_per_segment() -> None:
    assert ModelRef(root="default/judge.v1+a100").root == "default/judge.v1+a100"

    with pytest.raises(ValidationError, match="String should match pattern"):
        ModelRef(root="Default/judge")
    with pytest.raises(ValidationError, match="String should match pattern"):
        ModelRef(root="default/j")
    with pytest.raises(ValidationError, match="String should match pattern"):
        ModelRef(root="default/judge--bad")


def test_metric_model_refs_declares_model_ref_fields() -> None:
    metric = ModelBackedMetric(model=ModelRef(root="workspace/judge"))

    assert metric.model_refs() == {"model": ModelRef(root="workspace/judge")}


async def test_metric_resolve_models_resolves_model_ref_fields() -> None:
    resolver = LocalModelResolver()
    resolver.register_model(ModelRef(root="workspace/judge"), _model("registered-judge"))
    metric = ModelBackedMetric(model=ModelRef(root="workspace/judge"))

    await metric.resolve_models(resolver)

    assert isinstance(metric.model, Model)
    assert metric.model.name == "registered-judge"


async def test_llm_judge_model_ref_defers_default_prompt_template_until_resolution() -> None:
    config = LLMJudge(
        model=ModelRef(root="workspace/judge"),
        scores=[
            RangeScore(
                name="quality",
                minimum=0,
                maximum=1,
                parser=JSONScoreParser(json_path="quality"),
            )
        ],
    )
    metric = LLMJudgeMetric(
        model=ModelRef(root="workspace/judge"),
        scores=[
            RangeScore(
                name="quality",
                minimum=0,
                maximum=1,
                parser=JSONScoreParser(json_path="quality"),
            )
        ],
    )
    resolver = LocalModelResolver()
    resolver.register_model(ModelRef(root="workspace/judge"), _completion_model())

    assert config.prompt_template is None
    assert metric.prompt_template is None

    await metric.resolve_models(resolver)

    assert metric.prompt_template == default_judge_prompt_template_completions()


def test_llm_judge_inline_completion_model_uses_completion_default_prompt_template() -> None:
    config = LLMJudge(
        model=_completion_model(),
        scores=[
            RangeScore(
                name="quality",
                minimum=0,
                maximum=1,
                parser=JSONScoreParser(json_path="quality"),
            )
        ],
    )

    assert config.prompt_template is None
    assert (
        config.input_schema()
        == LLMJudge(
            model=_completion_model(),
            scores=[
                RangeScore(
                    name="quality",
                    minimum=0,
                    maximum=1,
                    parser=JSONScoreParser(json_path="quality"),
                )
            ],
            prompt_template=default_judge_prompt_template_completions(),
        ).input_schema()
    )


def test_local_backend_execution_resolves_registered_model_ref() -> None:
    backend = LocalBackend()
    backend.model_resolver.register_model(ModelRef(root="workspace/judge"), _model("registered-judge"))
    evaluator = Evaluator(client=backend)
    metric = ModelBackedMetric(model=ModelRef(root="workspace/judge"))

    result = evaluator.run_sync(
        metrics=[metric],
        dataset=[{"output_text": "hello"}],
    )

    assert result.row_scores[0].metrics["model-backed"][0].value == "registered-judge"
    assert isinstance(metric.model, ModelRef)


def test_evaluator_local_execution_fails_for_unregistered_model_ref() -> None:
    evaluator = Evaluator()

    with pytest.raises(ValueError, match="workspace/missing.*not registered"):
        evaluator.run_sync(
            metrics=[ModelBackedMetric(model=ModelRef(root="workspace/missing"))],
            dataset=[{"output_text": "hello"}],
        )
