# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal, cast

import pytest
from nemo_evals.shared.metric_bundles.bundles import (
    MetricBundle,
    MetricBundlePackager,
    MetricBundlePayload,
    MetricBundlingError,
    bundle_metric,
    register_metric_bundle_kind,
    unbundle_metric,
)
from nemo_evals.shared.metric_bundles.cloudpickle import (
    MAX_CLOUDPICKLE_PAYLOAD_BYTES,
    CloudpickleMetricBundlePackager,
    CloudpickleMetricPayload,
)
from nhx_evals_sdk.enums import ModelFormat
from nhx_evals_sdk.metrics.bleu import BLEUMetric
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.metrics.f1 import F1Metric
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.number_check import NumberCheckMetric
from nhx_evals_sdk.metrics.protocol import (
    Metric,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
)
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

pytestmark = pytest.mark.usefixtures("allow_cloudpickle_metrics")


class _CustomMetric:
    type = "custom-score"
    description = "custom metric"
    labels = {"source": "test"}

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


class _NotMetric:
    pass


class _TestPayload(MetricBundlePayload):
    @property
    def kind(self) -> Literal["test-cloudpickle-registration"]:
        return "test-cloudpickle-registration"

    @property
    def digest(self) -> str:
        return "test-digest"


class _ConflictingPayload(MetricBundlePayload):
    @property
    def kind(self) -> Literal["test-cloudpickle-registration"]:
        return "test-cloudpickle-registration"

    @property
    def digest(self) -> str:
        return "test-conflicting-digest"


class _TestPackager(MetricBundlePackager):
    def package(self, metric: Metric) -> MetricBundlePayload:
        del metric
        return _TestPayload()

    def load(self, payload: MetricBundlePayload) -> Metric:
        del payload
        return _CustomMetric()


class _EmptyTypeMetric(_CustomMetric):
    type = ""


def _judge_model() -> Model:
    return Model(
        url="https://judge.example.test/v1/chat/completions",
        name="judge-model",
        format=ModelFormat.OPEN_AI,
    )


def _embeddings_model() -> Model:
    return Model(
        url="https://judge.example.test/v1/embeddings",
        name="embedding-model",
        format=ModelFormat.OPEN_AI,
    )


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
                        name="helpfulness",
                        minimum=1,
                        maximum=5,
                        parser=JSONScoreParser(json_path="helpfulness"),
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
        (
            "response_relevancy",
            ResponseRelevancyMetric(judge_model=judge_model, embeddings_model=_embeddings_model()),
        ),
        ("faithfulness", FaithfulnessMetric(judge_model=judge_model)),
        ("noise_sensitivity", NoiseSensitivityMetric(judge_model=judge_model)),
    ]


def test_cloudpickle_packager_round_trips_builtin_metric() -> None:
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    packager = CloudpickleMetricBundlePackager()

    bundle = bundle_metric(metric, packager)
    hydrated = unbundle_metric(bundle)

    assert bundle.metric_type == "exact-match"
    assert bundle.outputs[0].name == "exact-match"
    assert isinstance(hydrated, ExactMatchMetric)


@pytest.mark.parametrize(
    ("case_name", "metric"), _builtin_metric_cases(), ids=[case[0] for case in _builtin_metric_cases()]
)
def test_cloudpickle_packager_round_trips_every_builtin_metric(case_name: str, metric: Metric) -> None:
    packager = CloudpickleMetricBundlePackager()

    bundle = bundle_metric(metric, packager)
    restored = MetricBundle.model_validate_json(bundle.model_dump_json())
    hydrated = unbundle_metric(restored)

    assert restored.metric_type
    assert restored.outputs
    assert [output.name for output in hydrated.output_spec()] == [output.name for output in metric.output_spec()]
    assert type(hydrated) is type(metric), case_name


def test_cloudpickle_packager_round_trips_custom_protocol_metric() -> None:
    packager = CloudpickleMetricBundlePackager()

    bundle = bundle_metric(_CustomMetric(), packager)
    serialized = bundle.model_dump_json()
    restored = MetricBundle.model_validate_json(serialized)
    hydrated = unbundle_metric(restored)

    assert restored.bundle_kind == "metric-bundle"
    assert restored.metric_type == "custom-score"
    assert restored.metadata.description == "custom metric"
    assert restored.metadata.labels == {"source": "test"}
    assert restored.outputs[0].name == "score"
    assert isinstance(hydrated, _CustomMetric)


def test_cloudpickle_packager_captures_metric_secrets() -> None:
    metric = LLMJudgeMetric(
        model=Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            api_key_secret=SecretRef(root="judge-secret"),
            format=ModelFormat.OPEN_AI,
        ),
        scores=[
            RangeScore(
                name="helpfulness",
                minimum=1,
                maximum=5,
                parser=JSONScoreParser(json_path="helpfulness"),
            )
        ],
    )

    bundle = bundle_metric(metric, CloudpickleMetricBundlePackager())
    restored = MetricBundle.model_validate_json(bundle.model_dump_json())

    assert restored.secrets == {"judge_secret": SecretRef(root="judge-secret")}


def test_cloudpickle_packager_captures_digest_and_payload_metadata() -> None:
    bundle = bundle_metric(_CustomMetric(), CloudpickleMetricBundlePackager())
    payload = CloudpickleMetricPayload.model_validate(bundle.payload)
    serialized_payload = cast(dict[str, object], bundle.model_dump(mode="json")["payload"])

    assert payload.digest == hashlib.sha256(bytes(payload.blob)).hexdigest()
    assert serialized_payload["digest"] == payload.digest
    assert payload.kind == "cloudpickle"
    assert serialized_payload["kind"] == "cloudpickle"
    assert payload.python_version
    assert payload.cloudpickle_version
    assert payload.pickle_protocol > 0
    assert bundle.outputs[0].value_json_schema["title"] == "ContinuousScore"


def test_cloudpickle_packager_rejects_oversized_payload() -> None:
    oversized_blob = b"x" * (MAX_CLOUDPICKLE_PAYLOAD_BYTES + 1)

    with pytest.raises(MetricBundlingError, match="maximum allowed"):
        CloudpickleMetricPayload.from_blob(oversized_blob)


def test_cloudpickle_packager_rejects_python_version_mismatch() -> None:
    bundle = bundle_metric(_CustomMetric(), CloudpickleMetricBundlePackager())
    payload = CloudpickleMetricPayload.model_validate(bundle.payload)
    incompatible_payload = payload.model_copy(update={"python_version": "0.0.0"})
    incompatible_bundle = bundle.model_copy(update={"payload": incompatible_payload})

    with pytest.raises(MetricBundlingError, match="created with Python 0.0.0"):
        unbundle_metric(incompatible_bundle)


def test_unbundle_metric_rejects_output_contract_mismatch() -> None:
    bundle = bundle_metric(_CustomMetric(), CloudpickleMetricBundlePackager())
    incompatible_bundle = bundle.model_copy(
        update={"outputs": [bundle.outputs[0].model_copy(update={"description": "changed"})]}
    )

    with pytest.raises(MetricBundlingError, match="output spec"):
        unbundle_metric(incompatible_bundle)


def test_register_metric_bundle_kind_rejects_conflicting_registration() -> None:
    register_metric_bundle_kind(
        "test-cloudpickle-registration",
        payload_type=_TestPayload,
        packager_factory=_TestPackager,
    )
    register_metric_bundle_kind(
        "test-cloudpickle-registration",
        payload_type=_TestPayload,
        packager_factory=_TestPackager,
    )

    with pytest.raises(ValueError, match="already registered"):
        register_metric_bundle_kind(
            "test-cloudpickle-registration",
            payload_type=_ConflictingPayload,
            packager_factory=_TestPackager,
        )


def test_cloudpickle_packager_rejects_non_metric_object() -> None:
    with pytest.raises(MetricBundlingError, match="Metric protocol"):
        bundle_metric(cast(Metric, _NotMetric()), CloudpickleMetricBundlePackager())


def test_cloudpickle_packager_rejects_empty_metric_type() -> None:
    with pytest.raises(MetricBundlingError, match="metric type must not be empty"):
        bundle_metric(_EmptyTypeMetric(), CloudpickleMetricBundlePackager())


def test_cloudpickle_packager_hydrates_from_payload_without_bundle_envelope() -> None:
    packager = CloudpickleMetricBundlePackager()
    bundle = bundle_metric(_CustomMetric(), packager)

    hydrated = packager.load(bundle.payload)

    assert isinstance(hydrated, _CustomMetric)
