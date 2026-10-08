# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import cast

import pytest
from nemo_evals.api.schemas import MetricInline
from nemo_evals.shared.metric_bundles.bundles import (
    MetricBundle,
    MetricBundlingError,
    bundle_metric,
    unbundle_metric,
)
from nemo_evals.shared.metric_bundles.inline import (
    InlineMetricBundlePackager,
    InlineMetricPayload,
    inline_bundle_supported,
)
from nhx_evals_sdk.enums import ModelFormat
from nhx_evals_sdk.metrics.bleu import BLEUMetric
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.metrics.f1 import F1Metric
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.number_check import NumberCheckMetric
from nhx_evals_sdk.metrics.protocol import Metric, MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.ragas import (
    AgentGoalAccuracyMetric,
    AnswerAccuracyMetric,
    ContextEntityRecallMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextRelevanceMetric,
    FaithfulnessMetric,
    NoiseSensitivityMetric,
    ResponseGroundednessMetric,
    ResponseRelevancyMetric,
    ToolCallAccuracyMetric,
    TopicAdherenceMetric,
)
from nhx_evals_sdk.metrics.remote import NemoAgentToolkitRemoteMetric, RemoteMetric
from nhx_evals_sdk.metrics.rouge import ROUGEMetric
from nhx_evals_sdk.metrics.string_check import StringCheckMetric
from nhx_evals_sdk.metrics.tool_calling import ToolCallingMetric
from nhx_evals_sdk.values import Model, SecretRef
from nhx_evals_sdk.values.scores import JSONScoreParser, RangeScore, RemoteScore
from pydantic import BaseModel


class _CustomMetric:
    """A protocol-satisfying metric that is not part of MetricsUnion."""

    type = "custom-score"
    description = "custom metric"
    labels = {"source": "test"}

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


def _judge_model() -> Model:
    return Model(url="https://judge.example.test/v1/chat/completions", name="judge-model", format=ModelFormat.OPEN_AI)


def _embeddings_model() -> Model:
    return Model(url="https://judge.example.test/v1/embeddings", name="embedding-model", format=ModelFormat.OPEN_AI)


def _builtin_metric_cases() -> Sequence[tuple[str, Metric]]:
    judge_model = _judge_model()
    return [
        ("exact_match", ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")),
        ("f1", F1Metric(reference="{{item.expected}}", candidate="{{item.output}}")),
        ("bleu", BLEUMetric(references=["{{item.expected}}"], candidate="{{item.output}}")),
        ("rouge", ROUGEMetric(reference="{{item.expected}}", candidate="{{item.output}}")),
        (
            "string_check",
            StringCheckMetric(
                operation="contains", left_template="{{item.output}}", right_template="{{item.expected}}"
            ),
        ),
        (
            "number_check",
            NumberCheckMetric(operation="equals", left_template="{{item.left}}", right_template="{{item.right}}"),
        ),
        ("tool_calling", ToolCallingMetric(reference="{{item.expected_tool_calls}}")),
        (
            "llm_judge",
            LLMJudgeMetric(
                model=judge_model,
                scores=[
                    RangeScore(
                        name="helpfulness", minimum=1, maximum=5, parser=JSONScoreParser(json_path="helpfulness")
                    )
                ],
                prompt_template="Judge: {{item.expected}} -> {{item.output}}",
            ),
        ),
        (
            "remote",
            RemoteMetric(
                url="https://remote.example.test",
                body={"prompt": "{{item.prompt}}"},
                scores=[RemoteScore(name="quality", parser=JSONScoreParser(json_path="$.result.quality"))],
            ),
        ),
        (
            "nemo_agent_toolkit_remote",
            NemoAgentToolkitRemoteMetric(url="https://remote.example.test", evaluator_name="nat-quality"),
        ),
        ("topic_adherence", TopicAdherenceMetric(metric_mode="f1", judge_model=judge_model)),
        ("tool_call_accuracy", ToolCallAccuracyMetric()),
        ("agent_goal_accuracy", AgentGoalAccuracyMetric(judge_model=judge_model)),
        ("answer_accuracy", AnswerAccuracyMetric(judge_model=judge_model)),
        ("context_relevance", ContextRelevanceMetric(judge_model=judge_model)),
        ("response_groundedness", ResponseGroundednessMetric(judge_model=judge_model)),
        ("context_recall", ContextRecallMetric(judge_model=judge_model)),
        ("context_precision", ContextPrecisionMetric(judge_model=judge_model)),
        ("context_entity_recall", ContextEntityRecallMetric(judge_model=judge_model)),
        ("response_relevancy", ResponseRelevancyMetric(judge_model=judge_model, embeddings_model=_embeddings_model())),
        ("faithfulness", FaithfulnessMetric(judge_model=judge_model)),
        ("noise_sensitivity", NoiseSensitivityMetric(judge_model=judge_model)),
    ]


_CASES = _builtin_metric_cases()
_CASE_IDS = [name for name, _ in _CASES]


@pytest.mark.parametrize(("case_name", "metric"), _CASES, ids=_CASE_IDS)
def test_inline_packager_round_trips_every_builtin_metric(case_name: str, metric: Metric) -> None:
    """Every built-in metric serializes inline and reconstructs to an identical object."""
    bundle = bundle_metric(metric, InlineMetricBundlePackager())

    # Full wire round-trip: runtime bundle -> JSON -> runtime bundle.
    restored = MetricBundle.model_validate_json(bundle.model_dump_json())
    hydrated = unbundle_metric(restored)

    assert restored.payload.kind == "inline", case_name
    assert restored.metric_type == metric.type, case_name
    assert type(hydrated) is type(metric), case_name
    # Inline reconstruction must be lossless at the config level (not just the type).
    assert cast(BaseModel, hydrated).model_dump() == cast(BaseModel, metric).model_dump(), case_name
    assert [o.name for o in hydrated.output_spec()] == [o.name for o in metric.output_spec()], case_name


@pytest.mark.parametrize(("case_name", "metric"), _CASES, ids=_CASE_IDS)
def test_inline_payload_passes_through_wire_dto(case_name: str, metric: Metric) -> None:
    """The inline payload survives the MetricInline wire DTO (OpenAPI contract)."""
    bundle = bundle_metric(metric, InlineMetricBundlePackager())

    wire = MetricInline.model_validate_json(bundle.model_dump_json())

    assert wire.payload.kind == "inline", case_name
    assert wire.metric_type
    assert wire.outputs
    # Re-validating the wire DTO JSON back into a runtime bundle must still hydrate.
    runtime_again = MetricBundle.model_validate_json(wire.model_dump_json())
    assert type(unbundle_metric(runtime_again)) is type(metric), case_name


def test_inline_payload_digest_is_canonical_sha256() -> None:
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    bundle = bundle_metric(metric, InlineMetricBundlePackager())
    payload = InlineMetricPayload.model_validate(bundle.payload.model_dump(mode="python"))

    expected = hashlib.sha256(
        json.dumps(payload.metric, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert payload.digest == expected
    serialized = cast(dict[str, object], bundle.model_dump(mode="json")["payload"])
    assert serialized["kind"] == "inline"
    assert serialized["digest"] == expected


def test_inline_packager_rejects_custom_metric() -> None:
    with pytest.raises(MetricBundlingError, match="CloudpickleMetricBundlePackager"):
        bundle_metric(cast(Metric, _CustomMetric()), InlineMetricBundlePackager())


def test_inline_bundle_supported_classifies_metrics() -> None:
    assert inline_bundle_supported(ExactMatchMetric(reference="{{item.expected}}"))
    assert not inline_bundle_supported(cast(Metric, _CustomMetric()))


def test_inline_captures_metric_secrets() -> None:
    metric = LLMJudgeMetric(
        model=Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            api_key_secret=SecretRef(root="judge-secret"),
            format=ModelFormat.OPEN_AI,
        ),
        scores=[RangeScore(name="helpfulness", minimum=1, maximum=5, parser=JSONScoreParser(json_path="helpfulness"))],
    )

    bundle = bundle_metric(metric, InlineMetricBundlePackager())
    restored = MetricBundle.model_validate_json(bundle.model_dump_json())

    assert restored.secrets == {"judge_secret": SecretRef(root="judge-secret")}
