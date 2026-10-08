# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
from json import JSONDecodeError
from typing import Protocol, TypeGuard
from unittest.mock import MagicMock, patch

import httpx
import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.constants import PLACEHOLDER_INFERENCE_API_KEY
from nhx_evals_sdk.enums import MetricType
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
from nhx_evals_sdk.metrics.ragas.imports import get_evaluation_dataset_class
from nhx_evals_sdk.values import MetricResult, Model, SecretRef

MOCK_JUDGE_MODEL = Model(
    name="gpt-4",
    url="https://api.openai.com/v1",
)

MOCK_EMBEDDINGS_MODEL = Model(
    name="text-embedding-ada-002",
    url="https://api.openai.com/v1/embeddings",
)

MOCK_ITEM = {
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "contexts": ["Paris is the capital and largest city of France."],
    "ground_truth": "Paris",
}

MOCK_SAMPLE = {"output_text": "Based on the context, Paris is the capital of France."}


class _ChatHandler(Protocol):
    request_log: list[dict[str, object]]
    _current_request: dict[str, object] | None

    def on_chat_model_start(self, serialized: dict[str, object], messages: list[dict[str, str]]) -> None: ...
    def on_llm_end(self, response: dict[str, object]) -> None: ...
    def on_llm_error(self, error: BaseException) -> None: ...


class _ChatHandlerFactory(Protocol):
    def __call__(self, logger: logging.Logger) -> _ChatHandler: ...


def _is_chat_handler_factory(value: object) -> TypeGuard[_ChatHandlerFactory]:
    return callable(value)


def _get_score_value(result: MetricResult, score_name: str) -> float:
    """Helper to get output value from MetricResult."""
    for output in result.outputs:
        if output.name == score_name:
            return output.value
    raise KeyError(f"Output '{score_name}' not found in result")


class _SecretResolver:
    def __init__(self, secrets: dict[str, str] | None = None) -> None:
        self._secrets = secrets or {}

    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        return self._secrets.get(secret_ref.root)


class _UnexpectedSecretResolver:
    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        raise AssertionError(f"Resolver should not be called for {secret_ref.root}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_class,expected_type,params",
    [
        (TopicAdherenceMetric, MetricType.TOPIC_ADHERENCE, {"judge_model": MOCK_JUDGE_MODEL}),
        (ToolCallAccuracyMetric, MetricType.TOOL_CALL_ACCURACY, {}),
        (AgentGoalAccuracyMetric, MetricType.AGENT_GOAL_ACCURACY, {"judge_model": MOCK_JUDGE_MODEL}),
        (AnswerAccuracyMetric, MetricType.ANSWER_ACCURACY, {"judge_model": MOCK_JUDGE_MODEL}),
        (ContextRecallMetric, MetricType.CONTEXT_RECALL, {"judge_model": MOCK_JUDGE_MODEL}),
        (ContextPrecisionMetric, MetricType.CONTEXT_PRECISION, {"judge_model": MOCK_JUDGE_MODEL}),
        (ContextRelevanceMetric, MetricType.CONTEXT_RELEVANCE, {"judge_model": MOCK_JUDGE_MODEL}),
        (ContextEntityRecallMetric, MetricType.CONTEXT_ENTITY_RECALL, {"judge_model": MOCK_JUDGE_MODEL}),
        (ResponseGroundednessMetric, MetricType.RESPONSE_GROUNDEDNESS, {"judge_model": MOCK_JUDGE_MODEL}),
        (
            ResponseRelevancyMetric,
            MetricType.RESPONSE_RELEVANCY,
            {"judge_model": MOCK_JUDGE_MODEL, "embeddings_model": MOCK_EMBEDDINGS_MODEL},
        ),
        (FaithfulnessMetric, MetricType.FAITHFULNESS, {"judge_model": MOCK_JUDGE_MODEL}),
        (NoiseSensitivityMetric, MetricType.NOISE_SENSITIVITY, {"judge_model": MOCK_JUDGE_MODEL}),
    ],
)
async def test_metric_types(metric_class, expected_type, params):
    """Test that each metric class has the correct metric type."""
    metric = metric_class(**params)
    assert metric.type == expected_type


def test_score_names_defaults_to_metric_type_value():
    metric = TopicAdherenceMetric(metric_mode="f1", judge_model=MOCK_JUDGE_MODEL)
    assert output_names(metric) == [metric.type.value]


@pytest.mark.parametrize(
    "metric_class,params,ragas_score_name,sdk_score_name",
    [
        (
            AgentGoalAccuracyMetric,
            {"judge_model": MOCK_JUDGE_MODEL},
            "agent_goal_accuracy",
            MetricType.AGENT_GOAL_ACCURACY.value,
        ),
        (AnswerAccuracyMetric, {"judge_model": MOCK_JUDGE_MODEL}, "nv_accuracy", MetricType.ANSWER_ACCURACY.value),
        (
            ContextEntityRecallMetric,
            {"judge_model": MOCK_JUDGE_MODEL},
            "context_entity_recall",
            MetricType.CONTEXT_ENTITY_RECALL.value,
        ),
        (
            ContextPrecisionMetric,
            {"judge_model": MOCK_JUDGE_MODEL},
            "context_precision",
            MetricType.CONTEXT_PRECISION.value,
        ),
        (ContextRecallMetric, {"judge_model": MOCK_JUDGE_MODEL}, "context_recall", MetricType.CONTEXT_RECALL.value),
        (
            ContextRelevanceMetric,
            {"judge_model": MOCK_JUDGE_MODEL},
            "nv_context_relevance",
            MetricType.CONTEXT_RELEVANCE.value,
        ),
        (FaithfulnessMetric, {"judge_model": MOCK_JUDGE_MODEL}, "faithfulness", MetricType.FAITHFULNESS.value),
        (
            NoiseSensitivityMetric,
            {"judge_model": MOCK_JUDGE_MODEL},
            "noise_sensitivity",
            MetricType.NOISE_SENSITIVITY.value,
        ),
        (
            ResponseGroundednessMetric,
            {"judge_model": MOCK_JUDGE_MODEL},
            "nv_response_groundedness",
            MetricType.RESPONSE_GROUNDEDNESS.value,
        ),
        (
            ResponseRelevancyMetric,
            {"judge_model": MOCK_JUDGE_MODEL, "embeddings_model": MOCK_EMBEDDINGS_MODEL},
            "answer_relevancy",
            MetricType.RESPONSE_RELEVANCY.value,
        ),
        (ToolCallAccuracyMetric, {}, "tool_call_accuracy", MetricType.TOOL_CALL_ACCURACY.value),
        (
            TopicAdherenceMetric,
            {"metric_mode": "f1", "judge_model": MOCK_JUDGE_MODEL},
            "topic_adherence",
            MetricType.TOPIC_ADHERENCE.value,
        ),
    ],
)
def test_align_scores_maps_all_wrapped_ragas_metric_names_to_output_spec(
    metric_class,
    params,
    ragas_score_name,
    sdk_score_name,
):
    metric = metric_class(**params)
    aligned = metric._align_scores_to_output_spec({ragas_score_name: 0.75})
    assert aligned == {sdk_score_name: 0.75}


def test_align_scores_does_not_guess_unknown_ragas_metric_name():
    metric = AnswerAccuracyMetric(judge_model=MOCK_JUDGE_MODEL)
    aligned = metric._align_scores_to_output_spec({"unexpected_ragas_name": 0.75})
    assert aligned == {"unexpected_ragas_name": 0.75}


@pytest.mark.parametrize(
    "metric_factory,ragas_score_name,sdk_score_name",
    [
        # RAGAS keys mode-bearing metrics as "<name>(mode=<mode>)" (see ragas.evaluation).
        # NoiseSensitivity defaults to mode="relevant"; the SDK declares the bare
        # "noise_sensitivity" output, so the suffix must be stripped (NVBug 6369321).
        (
            lambda: NoiseSensitivityMetric(judge_model=MOCK_JUDGE_MODEL),
            "noise_sensitivity(mode=relevant)",
            MetricType.NOISE_SENSITIVITY.value,
        ),
        (
            lambda: NoiseSensitivityMetric(judge_model=MOCK_JUDGE_MODEL),
            "noise_sensitivity(mode=irrelevant)",
            MetricType.NOISE_SENSITIVITY.value,
        ),
        # The same mode-suffixing applies to TopicAdherence (mode=precision/recall/f1).
        (
            lambda: TopicAdherenceMetric(metric_mode="f1", judge_model=MOCK_JUDGE_MODEL),
            "topic_adherence(mode=f1)",
            MetricType.TOPIC_ADHERENCE.value,
        ),
    ],
)
def test_align_scores_strips_ragas_mode_suffix(metric_factory, ragas_score_name, sdk_score_name):
    metric = metric_factory()
    aligned = metric._align_scores_to_output_spec({ragas_score_name: 0.75})
    assert aligned == {sdk_score_name: 0.75}


@pytest.mark.asyncio
async def test_noise_sensitivity_metric_accepts_ragas_mode_qualified_score():
    """Regression for NVBug 6369321.

    RAGAS emits NoiseSensitivity scores under the mode-qualified key
    ``noise_sensitivity(mode=relevant)``. The metric must align that to the declared
    ``noise_sensitivity`` output instead of failing validation with
    "Missing declared metric outputs: ['noise_sensitivity']".
    """
    metric = NoiseSensitivityMetric(judge_model=MOCK_JUDGE_MODEL)

    mock_evaluate = MagicMock()
    mock_evaluate.return_value.scores = [{"noise_sensitivity(mode=relevant)": 0.42}]
    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        result = await compute_scores(
            metric,
            {
                "user_input": "What is the capital of France?",
                "retrieved_contexts": [
                    "Paris is the capital and largest city of France.",
                    "Berlin is the capital of Germany.",
                ],
                "response": "The capital of France is Paris.",
                "reference": "Paris is the capital of France.",
            },
            {},
        )
    assert isinstance(result, MetricResult)
    assert _get_score_value(result, "noise_sensitivity") == 0.42


def test_nan_scores_use_declared_output_spec_names():
    metric = ContextRelevanceMetric(judge_model=MOCK_JUDGE_MODEL)
    nan_scores = metric._nan_scores_for_metrics([])
    assert list(nan_scores) == ["context_relevance"]
    assert math.isnan(nan_scores["context_relevance"])


@pytest.mark.asyncio
async def test_compute_scores_accepts_ragas_nv_metric_names():
    metric = AnswerAccuracyMetric(judge_model=MOCK_JUDGE_MODEL)
    mock_evaluate = MagicMock()
    mock_evaluate.return_value.scores = [{"nv_accuracy": 0.75}]
    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        result = await compute_scores(
            metric,
            {
                "user_input": "What is Python?",
                "response": "A programming language.",
                "reference": "Python is a high-level programming language.",
            },
            {},
        )
    assert _get_score_value(result, "answer_accuracy") == 0.75


def test_llm_backed_ragas_metric_exposes_ignore_request_failure():
    schema_props = TopicAdherenceMetric.model_json_schema().get("properties", {})
    assert "ignore_request_failure" in schema_props


def test_tool_call_accuracy_metric_does_not_expose_ignore_request_failure():
    schema_props = ToolCallAccuracyMetric.model_json_schema().get("properties", {})
    assert "ignore_request_failure" not in schema_props


@pytest.mark.asyncio
async def test_topic_adherence_metric():
    """Test TopicAdherenceMetric with mocked RAGAS evaluation."""
    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=MOCK_JUDGE_MODEL,
    )

    mock_evaluate = MagicMock()
    mock_evaluate.return_value.scores = [{"topic_adherence": 0.85}]
    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        result = await compute_scores(metric, MOCK_ITEM, MOCK_SAMPLE)
        assert isinstance(result, MetricResult)
        assert _get_score_value(result, "topic_adherence") == 0.85


@pytest.mark.asyncio
async def test_tool_call_accuracy_metric():
    """Test ToolCallAccuracyMetric with mocked RAGAS evaluation."""
    metric = ToolCallAccuracyMetric()

    mock_evaluate = MagicMock()
    mock_evaluate.return_value.scores = [{"tool_call_accuracy": 0.9}]
    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        result = await compute_scores(metric, MOCK_ITEM, MOCK_SAMPLE)
        assert isinstance(result, MetricResult)
        assert _get_score_value(result, "tool_call_accuracy") == 0.9


@pytest.mark.asyncio
async def test_agent_goal_accuracy_metric():
    """Test AgentGoalAccuracyMetric with different reference configurations."""
    # Test with reference
    metric_with_ref = AgentGoalAccuracyMetric(
        use_reference=True,
        judge_model=MOCK_JUDGE_MODEL,
    )
    mock_evaluate = MagicMock()
    mock_evaluate.return_value.scores = [{"agent_goal_accuracy": 0.95}]
    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        result = await compute_scores(metric_with_ref, MOCK_ITEM, MOCK_SAMPLE)
        assert _get_score_value(result, "agent_goal_accuracy") == 0.95

    # Test without reference
    metric_without_ref = AgentGoalAccuracyMetric(
        use_reference=False,
        judge_model=MOCK_JUDGE_MODEL,
    )
    mock_evaluate2 = MagicMock()
    mock_evaluate2.return_value.scores = [{"agent_goal_accuracy": 0.85}]
    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate2):
        result = await compute_scores(metric_without_ref, MOCK_ITEM, MOCK_SAMPLE)
        assert _get_score_value(result, "agent_goal_accuracy") == 0.85


@pytest.mark.asyncio
async def test_evaluation_dataset_creation():
    """Test that evaluation dataset is created correctly with and without templates."""
    # Test without template
    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=MOCK_JUDGE_MODEL,
    )

    dataset = metric._create_evaluation_dataset(MOCK_ITEM, MOCK_SAMPLE)
    EvaluationDataset = get_evaluation_dataset_class()
    assert isinstance(dataset, EvaluationDataset)

    # Test with template
    metric_with_template = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=MOCK_JUDGE_MODEL,
        input_template={"question": "{{question}}", "answer": "{{answer}}"},
    )
    dataset = metric_with_template._create_evaluation_dataset(MOCK_ITEM, MOCK_SAMPLE)
    assert isinstance(dataset, EvaluationDataset)


@pytest.mark.asyncio
async def test_create_evaluation_dataset_uses_sample_output_text():
    """Test that sample['output_text'] is used as response when no template (online mode)."""
    metric = FaithfulnessMetric(judge_model=MOCK_JUDGE_MODEL)

    item = {
        "user_input": "What is the capital of France?",
        "retrieved_contexts": ["Paris is the capital of France."],
    }
    sample = {"output_text": "The capital of France is Paris."}

    dataset = metric._create_evaluation_dataset(item, sample)
    EvaluationDataset = get_evaluation_dataset_class()
    assert isinstance(dataset, EvaluationDataset)

    # Verify response was injected from sample
    payload = dataset.samples[0] if hasattr(dataset, "samples") else dataset.to_pandas().iloc[0].to_dict()
    if hasattr(payload, "response"):
        assert payload.response == "The capital of France is Paris."
    else:
        assert payload.get("response") == "The capital of France is Paris."


@pytest.mark.asyncio
async def test_create_evaluation_dataset_response_priority():
    """Test response priority for non-template flow."""
    metric = FaithfulnessMetric(judge_model=MOCK_JUDGE_MODEL)

    # sample['output_text'] takes precedence over sample['response'] and item['response'].
    item = {"user_input": "Question?", "response": "item response"}
    sample = {"output_text": "sample output_text", "response": "sample response"}
    dataset = metric._create_evaluation_dataset(item, sample)
    payload = dataset.samples[0] if hasattr(dataset, "samples") else dataset.to_pandas().iloc[0].to_dict()
    response_value = payload.response if hasattr(payload, "response") else payload.get("response")
    assert response_value == "sample output_text"

    # sample['response'] is used when payload has no response.
    item = {"user_input": "Question?"}
    sample = {"response": "sample response"}
    dataset = metric._create_evaluation_dataset(item, sample)
    payload = dataset.samples[0] if hasattr(dataset, "samples") else dataset.to_pandas().iloc[0].to_dict()
    response_value = payload.response if hasattr(payload, "response") else payload.get("response")
    assert response_value == "sample response"

    # item['response'] is used when sample does not provide a response.
    item = {"user_input": "Question?", "response": "item response"}
    sample = {}
    dataset = metric._create_evaluation_dataset(item, sample)
    payload = dataset.samples[0] if hasattr(dataset, "samples") else dataset.to_pandas().iloc[0].to_dict()
    response_value = payload.response if hasattr(payload, "response") else payload.get("response")
    assert response_value == "item response"


@pytest.mark.asyncio
async def test_create_evaluation_dataset_with_template():
    """Test that template-provided response is preserved over fallback injection."""
    metric = FaithfulnessMetric(
        judge_model=MOCK_JUDGE_MODEL,
        input_template={
            "user_input": "{{user_input}}",
            "retrieved_contexts": "{{retrieved_contexts}}",
            "response": "template response",
        },
    )

    item = {
        "user_input": "What is the capital?",
        "retrieved_contexts": ["Context here"],
    }
    sample = {"output_text": "sample output text"}

    dataset = metric._create_evaluation_dataset(item, sample)
    payload = dataset.samples[0] if hasattr(dataset, "samples") else dataset.to_pandas().iloc[0].to_dict()
    if hasattr(payload, "response"):
        assert payload.response == "template response"
        assert payload.user_input == "What is the capital?"
    else:
        assert payload.get("response") == "template response"
        assert payload.get("user_input") == "What is the capital?"


@pytest.mark.asyncio
async def test_create_evaluation_dataset_with_template_fallback_injects_response():
    """Test template path still injects response when template omits response."""
    metric = FaithfulnessMetric(
        judge_model=MOCK_JUDGE_MODEL,
        input_template={
            "user_input": "{{user_input}}",
            "retrieved_contexts": "{{retrieved_contexts}}",
        },
    )

    item = {
        "user_input": "What is the capital?",
        "retrieved_contexts": ["Context here"],
    }
    sample = {"output_text": "fallback response"}

    dataset = metric._create_evaluation_dataset(item, sample)
    payload = dataset.samples[0] if hasattr(dataset, "samples") else dataset.to_pandas().iloc[0].to_dict()
    response_value = payload.response if hasattr(payload, "response") else payload.get("response")
    assert response_value == "fallback response"


@pytest.mark.asyncio
async def test_create_evaluation_dataset_offline_mode():
    """Test offline mode: uses item['response'] when sample['output_text'] doesn't exist."""
    metric = FaithfulnessMetric(judge_model=MOCK_JUDGE_MODEL)

    item = {
        "user_input": "What is the capital of France?",
        "retrieved_contexts": ["Paris is the capital."],
        "response": "Paris is the capital of France.",  # Pre-generated response
    }
    sample = {}  # No output_text (offline mode)

    dataset = metric._create_evaluation_dataset(item, sample)
    payload = dataset.samples[0] if hasattr(dataset, "samples") else dataset.to_pandas().iloc[0].to_dict()
    response_value = payload.response if hasattr(payload, "response") else payload.get("response")
    assert response_value == "Paris is the capital of France."


@pytest.mark.asyncio
@patch.dict(os.environ, {"MY_SECRET": "secret-value"})
async def test_llm_judge_configuration():
    """Test that LLM judge is configured correctly with different parameter combinations."""
    # Test with minimal parameters
    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=Model(
            name="gpt-4",
            url="https://model.com",
        ),
    )
    llm_judge = metric._get_llm_judge(httpx.AsyncClient())
    assert llm_judge is not None

    # Test with full parameters including api_key
    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=Model(
            name="gpt-4",
            url="https://custom.api.com",
            api_key_secret="my-secret",
        ),
    )
    llm_judge = metric._get_llm_judge(httpx.AsyncClient())
    assert llm_judge is not None


@pytest.mark.asyncio
async def test_run_config_settings():
    """Test that run configuration is set correctly with default parameters."""
    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=MOCK_JUDGE_MODEL,
    )
    config = metric._get_run_config()
    # Default values from base_ragas.py
    assert config.timeout == 120  # DEFAULT_JUDGE_TIMEOUT
    assert config.max_retries == 3  # DEFAULT_JUDGE_MAX_RETRIES
    assert config.max_workers == 1  # DEFAULT_JUDGE_MAX_WORKER


def test_run_evaluate_returns_nan_on_empty_scores():
    metric = TopicAdherenceMetric(metric_mode="f1", judge_model=MOCK_JUDGE_MODEL)

    ragas_metric = MagicMock()
    ragas_metric.name = "topic_adherence"
    result = MagicMock()
    result.scores = []
    mock_evaluate = MagicMock(return_value=result)

    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        scores = metric._run_evaluate(dataset=MagicMock(), metrics=[ragas_metric])

    assert math.isnan(scores["topic_adherence"])
    assert mock_evaluate.call_count == 1
    assert all(call.kwargs["raise_exceptions"] is True for call in mock_evaluate.call_args_list)


def test_run_evaluate_strict_mode_returns_nan_on_json_parse_error():
    metric = TopicAdherenceMetric(metric_mode="f1", judge_model=MOCK_JUDGE_MODEL)

    ragas_metric = MagicMock()
    ragas_metric.name = "topic_adherence"

    mock_evaluate = MagicMock(side_effect=JSONDecodeError("invalid json", "", 0))
    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        scores = metric._run_evaluate(dataset=MagicMock(), metrics=[ragas_metric])

    assert math.isnan(scores["topic_adherence"])
    assert mock_evaluate.call_count == 1
    assert all(call.kwargs["raise_exceptions"] is True for call in mock_evaluate.call_args_list)


def test_run_evaluate_returns_nan_on_invalid_scores():
    metric = TopicAdherenceMetric(metric_mode="f1", judge_model=MOCK_JUDGE_MODEL)

    ragas_metric = MagicMock()
    ragas_metric.name = "topic_adherence"
    result = MagicMock()
    result.scores = [{"topic_adherence": float("nan")}]
    mock_evaluate = MagicMock(return_value=result)

    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        scores = metric._run_evaluate(dataset=MagicMock(), metrics=[ragas_metric])

    assert math.isnan(scores["topic_adherence"])
    assert mock_evaluate.call_count == 1
    assert all(call.kwargs["raise_exceptions"] is True for call in mock_evaluate.call_args_list)


def test_run_evaluate_handles_mixed_invalid_score_types_without_typeerror():
    metric = TopicAdherenceMetric(metric_mode="f1", judge_model=MOCK_JUDGE_MODEL)

    result = MagicMock()
    result.scores = [{"topic_adherence": True, "aux_score": 1, "aux_float": 0.25, "aux_string": "invalid"}]
    mock_evaluate = MagicMock(return_value=result)

    ragas_metric = MagicMock()
    ragas_metric.name = "topic_adherence"

    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        scores = metric._run_evaluate(dataset=MagicMock(), metrics=[ragas_metric])

    # Regression check: numeric-type validation should not raise and should produce NaN fallback.
    assert math.isnan(scores["topic_adherence"])


def test_run_evaluate_succeeds_on_valid_scores():
    metric = TopicAdherenceMetric(metric_mode="f1", judge_model=MOCK_JUDGE_MODEL)

    success = MagicMock()
    success.scores = [{"topic_adherence": 0.82}]
    mock_evaluate = MagicMock(return_value=success)

    ragas_metric = MagicMock()
    ragas_metric.name = "topic_adherence"

    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        scores = metric._run_evaluate(dataset=MagicMock(), metrics=[ragas_metric])

    assert scores["topic_adherence"] == pytest.approx(0.82)
    assert math.isfinite(scores["topic_adherence"])
    assert mock_evaluate.call_count == 1
    assert all(call.kwargs["raise_exceptions"] is True for call in mock_evaluate.call_args_list)


def test_run_evaluate_tolerant_mode_returns_nan_on_jsondecodeerror():
    metric = TopicAdherenceMetric(metric_mode="f1", judge_model=MOCK_JUDGE_MODEL)

    ragas_metric = MagicMock()
    ragas_metric.name = "topic_adherence"
    parse_error = JSONDecodeError("malformed JSON payload", "", 0)
    mock_evaluate = MagicMock(side_effect=parse_error)

    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        scores = metric._run_evaluate(dataset=MagicMock(), metrics=[ragas_metric])

    assert math.isnan(scores["topic_adherence"])
    assert mock_evaluate.call_count == 1


def test_run_evaluate_inference_error_raises_when_ignore_flag_disabled():
    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=MOCK_JUDGE_MODEL,
        ignore_request_failure=False,
    )

    inference_error = httpx.ConnectError("connection failed", request=httpx.Request("POST", "https://example.com"))
    mock_evaluate = MagicMock(side_effect=inference_error)
    ragas_metric = MagicMock()
    ragas_metric.name = "topic_adherence"

    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        with pytest.raises(httpx.ConnectError, match="connection failed"):
            metric._run_evaluate(dataset=MagicMock(), metrics=[ragas_metric])


def test_run_evaluate_inference_error_returns_nan_when_ignore_flag_enabled():
    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=MOCK_JUDGE_MODEL,
        ignore_request_failure=True,
    )

    inference_error = httpx.ConnectError("connection failed", request=httpx.Request("POST", "https://example.com"))
    mock_evaluate = MagicMock(side_effect=inference_error)
    ragas_metric = MagicMock()
    ragas_metric.name = "topic_adherence"

    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        scores = metric._run_evaluate(dataset=MagicMock(), metrics=[ragas_metric])

    assert math.isnan(scores["topic_adherence"])


@pytest.mark.asyncio
async def test_embeddings_client_validation():
    """Test that _require_embeddings_judge raises ValueError when required parameters are missing."""
    # Test with embeddings configured
    metric = ResponseRelevancyMetric(
        strictness=1,
        judge_model=MOCK_JUDGE_MODEL,
        embeddings_model=MOCK_EMBEDDINGS_MODEL,  # Required for ResponseRelevancy
    )

    # The metric should work when embeddings are configured
    mock_evaluate = MagicMock()
    mock_evaluate.return_value.scores = [{"answer_relevancy": 0.9}]
    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        result = await compute_scores(metric, MOCK_ITEM, MOCK_SAMPLE)
        assert isinstance(result, MetricResult)


@pytest.mark.asyncio
async def test_chat_callback_handler_request_response_flow():
    """Test ChatModelCallBackHandler request-response flow and logging."""
    from nhx_evals_sdk.inference import requests_log_var
    from nhx_evals_sdk.metrics.ragas.base import _get_chat_model_callback_handler_class

    mock_logger = logging.getLogger("test_logger")

    # Set up the context variable with an empty list
    token = requests_log_var.set([])
    try:
        handler_factory = _get_chat_model_callback_handler_class()
        assert _is_chat_handler_factory(handler_factory)
        handler = handler_factory(mock_logger)

        # Test complete request-response flow
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        test_response = {
            "choices": [{"message": {"content": "The capital of France is Paris."}, "finish_reason": "stop"}]
        }

        # Verify initial state
        assert len(handler.request_log) == 0
        assert handler._current_request is None

        # Start request and verify request state
        handler.on_chat_model_start({}, test_messages)
        assert handler._current_request is not None
        assert handler._current_request["request"] == test_messages
        assert len(handler.request_log) == 0  # Log should still be empty until response

        # Complete with response and verify final state
        handler.on_llm_end(test_response)
        assert len(handler.request_log) == 1
        assert handler._current_request is None  # Should be cleared

        # Verify log entry contains complete conversation
        log_entry = handler.request_log[0]
        assert log_entry["request"] == test_messages
        assert log_entry["response"] == test_response

        # Test error case in second conversation
        second_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the population of Mars?"},
        ]
        test_error = ValueError("API rate limit exceeded")

        # Handle second conversation that results in error
        handler.on_chat_model_start({}, second_messages)
        handler.on_llm_error(test_error)

        # Verify both conversations are logged (success and error)
        assert len(handler.request_log) == 2

        # Verify first successful conversation
        first_log = handler.request_log[0]
        assert first_log["request"] == test_messages
        assert first_log["response"] == test_response
        assert "error" not in first_log
        assert "error_type" not in first_log

        # Verify second conversation with error
        second_log = handler.request_log[1]
        assert second_log["request"] == second_messages
        assert second_log["error"] == "API rate limit exceeded"
        assert second_log["error_type"] == "ValueError"
        assert "response" not in second_log

        # Verify final state
        assert handler._current_request is None
    finally:
        requests_log_var.reset(token)  # Reset the context variable to its previous state


@pytest.mark.asyncio
async def test_metric_result_format():
    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=MOCK_JUDGE_MODEL,
    )

    mock_evaluate = MagicMock()
    mock_evaluate.return_value.scores = [{"topic_adherence": 0.85, "another_score": 0.9}]
    with patch("nhx_evals_sdk.metrics.ragas.base.get_evaluate_function", return_value=mock_evaluate):
        result = await compute_scores(metric, MOCK_ITEM, MOCK_SAMPLE)

        # MetricResult.outputs is a list of MetricOutput.
        assert isinstance(result, MetricResult)
        assert isinstance(result.outputs, list)
        assert len(result.outputs) == 1

        # Check that outputs have name and value.
        score_dict = {s.name: s.value for s in result.outputs}
        assert score_dict["topic_adherence"] == 0.85


@pytest.mark.asyncio
async def test_resolve_secrets_with_api_key_secret():
    """Test that resolve_secrets correctly resolves API key from secret resolver."""

    # Create model with api_key_secret
    judge_model_with_secret = Model(
        name="gpt-4",
        url="https://api.openai.com/v1",
        api_key_secret=SecretRef(root="my-secret"),
    )

    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=judge_model_with_secret,
    )

    # Before resolve_secrets, api_key should be None
    assert judge_model_with_secret.api_key is None
    assert metric._llm_model is not None
    assert metric._llm_model["api_key"] is None
    assert "my_secret" in metric.secrets()  # Underscores in env var name

    # Resolve secrets
    await metric.resolve_secrets(_SecretResolver({"my-secret": "resolved-api-key-12345"}))

    # After resolve_secrets, api_key should be populated
    assert metric._llm_model is not None
    assert metric._llm_model["api_key"] == "resolved-api-key-12345"


@pytest.mark.asyncio
async def test_resolve_secrets_raises_when_secret_not_found():
    """Test that resolve_secrets raises ValueError when secret is not found."""

    # Create model with api_key_secret
    judge_model_with_secret = Model(
        name="gpt-4",
        url="https://api.openai.com/v1",
        api_key_secret=SecretRef(root="missing-secret"),
    )

    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=judge_model_with_secret,
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="Missing secret 'missing-secret'"):
        await metric.resolve_secrets(_SecretResolver())


@pytest.mark.asyncio
async def test_resolve_secrets_skipped_when_no_api_key_secret():
    """Test that resolve_secrets does nothing when no api_key_secret is configured."""
    # Model without api_key_secret
    metric = TopicAdherenceMetric(
        metric_mode="f1",
        judge_model=MOCK_JUDGE_MODEL,  # No api_key_secret
    )

    # API key should already be the placeholder
    assert metric._llm_model is not None
    assert metric._llm_model["api_key"] == PLACEHOLDER_INFERENCE_API_KEY
    assert metric.secrets() == {}  # No secrets to resolve

    # Should complete without calling resolver
    await metric.resolve_secrets(_UnexpectedSecretResolver())

    # API key should still be the placeholder
    assert metric._llm_model is not None
    assert metric._llm_model["api_key"] == PLACEHOLDER_INFERENCE_API_KEY


@pytest.mark.asyncio
async def test_resolve_secrets_with_embeddings_model():
    """Test that resolve_secrets resolves both judge and embeddings API keys."""

    # Create models with api_key_secret
    judge_model_with_secret = Model(
        name="gpt-4",
        url="https://api.openai.com/v1",
        api_key_secret=SecretRef(root="judge-secret"),
    )
    embeddings_model_with_secret = Model(
        name="text-embedding-ada-002",
        url="https://api.openai.com/v1/embeddings",
        api_key_secret=SecretRef(root="embeddings-secret"),
    )

    metric = ResponseRelevancyMetric(
        strictness=1,
        judge_model=judge_model_with_secret,
        embeddings_model=embeddings_model_with_secret,
    )

    # Before resolve_secrets, both api_keys should be None
    assert metric._llm_model is not None
    assert metric._embed_params is not None
    assert metric._llm_model["api_key"] is None
    assert metric._embed_params["api_key"] is None
    assert "judge_secret" in metric.secrets()
    assert "embeddings_secret" in metric.secrets()

    # Resolve secrets
    await metric.resolve_secrets(
        _SecretResolver(
            {
                "judge-secret": "judge-key-123",
                "embeddings-secret": "embeddings-key-456",
            }
        )
    )

    # After resolve_secrets, both api_keys should be populated
    assert metric._llm_model is not None
    assert metric._embed_params is not None
    assert metric._llm_model["api_key"] == "judge-key-123"
    assert metric._embed_params["api_key"] == "embeddings-key-456"
