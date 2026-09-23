# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import math
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from jsonschema.exceptions import SchemaError
from metrics.helpers import compute_scores, output_names
from nemo_evaluator_sdk.enums import ModelFormat
from nemo_evaluator_sdk.inference import (
    AddInferenceParameter,
    ClientInferenceError,
    InjectSystemMessage,
    LogHook,
    TransformReasoningOutput,
    new_hooks,
)
from nemo_evaluator_sdk.metrics.llm_judge import (
    LLMJudgeMetric,
    ScoreParserRegex,
    default_judge_prompt_template_chat,
    default_judge_prompt_template_completions,
    generate_structured_output,
)
from nemo_evaluator_sdk.metrics.protocol import MetricOutput, MetricResult
from nemo_evaluator_sdk.structured_output import InferenceStructuredOutput, StructuredOutputMode
from nemo_evaluator_sdk.values.common import SecretRef, SupportedJobTypes
from nemo_evaluator_sdk.values.models import Model, ModelRef
from nemo_evaluator_sdk.values.params import InferenceParams, RunConfig
from nemo_evaluator_sdk.values.scores import (
    JSONScoreParser,
    RangeScore,
    RegexScoreParser,
    Rubric,
    RubricScore,
    ScoreParserJSON,
)
from pydantic import ValidationError
from pytest_mock import MockerFixture


def _make_metric_score(name: str = "helpfulness") -> RangeScore:
    return RangeScore(
        name=name,
        minimum=1,
        maximum=5,
        parser=JSONScoreParser(json_path=name),
    )


class _MissingSecretResolver:
    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        del secret_ref
        return None


class _RegisteredModelResolver:
    async def resolve_model(self, model_ref: ModelRef) -> Model:
        assert model_ref == ModelRef(root="workspace/judge")
        return _make_model().model_copy(update={"name": "resolved-judge"})


def _make_model(model_format: ModelFormat = ModelFormat.OPEN_AI) -> Model:
    return Model(
        url="https://judge.example.test/v1/chat/completions",
        name="judge-model",
        format=model_format,
    )


def _make_completion_model() -> Model:
    return Model(
        url="https://judge.example.test/v1/completions",
        name="judge-model",
        format=ModelFormat.OPEN_AI,
    )


def _new_rubric_score(parser: JSONScoreParser | RegexScoreParser | None) -> RubricScore:
    if parser is None:
        return RubricScore(
            name="length",
            rubric=[Rubric(label="short", value=0), Rubric(label="medium", value=1), Rubric(label="long", value=2)],
        )
    return RubricScore(
        name="length",
        rubric=[Rubric(label="short", value=0), Rubric(label="medium", value=1), Rubric(label="long", value=2)],
        parser=parser,
    )


def _new_range_score(parser: JSONScoreParser | RegexScoreParser | None) -> RangeScore:
    if parser is None:
        return RangeScore(name="truthfulness", minimum=0, maximum=1)
    return RangeScore(name="truthfulness", minimum=0, maximum=1, parser=parser)


def _empty_judge_response() -> dict:
    return {
        "id": "chatcmpl-9a4691b4bff51b26",
        "object": "chat.completion",
        "created": 1773007885,
        "model": "nvidia/nemotron-3-nano-30b-a3b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "refusal": None,
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [],
                    "reasoning": None,
                    "reasoning_content": "Need clarification.",
                },
            }
        ],
    }


def test_regex_score_parser_invalid():
    with pytest.raises(ValueError, match="invalid regex"):
        ScoreParserRegex(score=_new_range_score(RegexScoreParser(pattern=r"\invalid^pattern")))

    with pytest.raises(ValueError, match="incompatible score parser"):
        ScoreParserRegex(score=_new_range_score(JSONScoreParser(json_path="json-path")))


def test_regex_score_parser_nan():
    parser = ScoreParserRegex(score=_new_range_score(RegexScoreParser(pattern="SIMILARITY: (\\d+)")))
    score = parser.parse("no match")
    assert math.isnan(score.value)


@pytest.mark.parametrize(
    "desc,regex,text,expected_score",
    [
        (
            "integer",
            "SIMILARITY: (\\d+)",
            "SIMILARITY: 521",
            521,
        ),
        (
            "negative float",
            r"[^\d-]*(-?[0-9\.]+)",
            "my answer is -0.21",
            -0.21,
        ),
    ],
)
def test_regex_score_parser(desc, regex, text, expected_score):
    parser = ScoreParserRegex(score=_new_range_score(RegexScoreParser(pattern=regex)))
    score = parser.parse(text)
    assert score.value == expected_score, desc


@pytest.mark.parametrize(
    "desc,method,text,expected_nan",
    [
        # match only looks at the beginning - fails when pattern is mid-text
        ("match fails when pattern is mid-text", "match", "Some reasoning... SCORE: 5", True),
        # search finds pattern anywhere in the string
        ("search finds pattern mid-text", "search", "Some reasoning... SCORE: 5", False),
        # match works when pattern is at the beginning
        ("match works at beginning", "match", "SCORE: 5", False),
        # search also works at the beginning
        ("search works at beginning", "search", "SCORE: 5", False),
    ],
)
def test_regex_score_parser_method(desc, method, text, expected_nan):
    """Test that method='search' finds patterns anywhere while method='match' only matches from the beginning."""
    parser = ScoreParserRegex(score=_new_range_score(RegexScoreParser(pattern="SCORE: (\\d+)", method=method)))
    score = parser.parse(text)
    if expected_nan:
        assert math.isnan(score.value), desc
    else:
        assert score.value == 5, desc


def test_regex_score_parser_search_uses_first_match():
    """Test that method='search' uses the first match when multiple matches exist."""
    parser = ScoreParserRegex(score=_new_range_score(RegexScoreParser(pattern="SCORE: (\\d+)", method="search")))
    # Text contains two matches: SCORE: 5 (first) and SCORE: 4 (second)
    # search should return the first match (5), not the second (4)
    text = "abcdSCORE: 5defSCORE: 4ghi"
    score = parser.parse(text)
    assert score.value == 5, "search should use the first match found in the text"


@pytest.mark.parametrize(
    "desc,text",
    [
        ("empty", ""),
        ("no match", "no matching regex"),
        ("key not found", "QUALITY: not-a-label"),
        ("key not found with value mismatch", "QUALITY: 1"),
    ],
)
def test_regex_score_parser_rubric_nan(desc, text):
    metric_score = RubricScore(
        name="quality",
        rubric=[
            Rubric(label="good", value=1),
            Rubric(label="bad", value=0),
        ],
        parser=RegexScoreParser(pattern="(?s).*QUALITY: (\\S+)"),
    )
    parser = ScoreParserRegex(score=metric_score)
    score = parser.parse(text)
    assert math.isnan(score.value), desc


@pytest.mark.parametrize(
    "desc,regex,text,expected_score",
    [
        (
            "exact match",
            "QUALITY: (\\S+)",
            "QUALITY: good",
            1,
        ),
        (
            "extra content",
            "(?s).*QUALITY: (\\S+)",
            "reasoning content\nQUALITY: bad\nextra text",
            0,
        ),
    ],
)
def test_regex_score_parser_rubric(desc, regex, text, expected_score):
    metric_score = RubricScore(
        name="quality",
        rubric=[
            Rubric(label="good", value=1),
            Rubric(label="bad", value=0),
        ],
        parser=RegexScoreParser(pattern=regex),
    )
    parser = ScoreParserRegex(score=metric_score)
    score = parser.parse(text)
    assert score.value == expected_score, desc


def test_json_score_parser_invalid():
    with pytest.raises(ValueError, match="incompatible score parser to initialize"):
        ScoreParserJSON(score=_new_rubric_score(parser=RegexScoreParser(pattern=".*")))

    metric_score = _new_range_score(JSONScoreParser(json_path="key"))

    with pytest.raises(ValueError, match="missing schema"):
        ScoreParserJSON(score=metric_score, structured_output={"invalid-structured-output": "value"})

    with pytest.raises(ValueError, match="schema must be type 'object'"):
        ScoreParserJSON(score=metric_score, structured_output={"schema": {"type": "string"}})

    with pytest.raises(ValueError, match="schema must have .* defined as an object property"):
        ScoreParserJSON(
            score=metric_score,
            structured_output={"schema": {"type": "object", "properties": {"another-property": {"type": "string"}}}},
        )

    with pytest.raises(ValueError, match="schema property .* must be type number, integer, or boolean"):
        ScoreParserJSON(
            score=metric_score,
            structured_output={"schema": {"type": "object", "properties": {"key": {"type": "string"}}}},
        )

    with pytest.raises(ValueError, match="expected string type in schema for property .* when used with score rubric"):
        ScoreParserJSON(
            score=_new_rubric_score(JSONScoreParser(json_path="key")),
            structured_output={"schema": {"type": "object", "properties": {"key": {"type": "number"}}}},
        )

    with pytest.raises(ValueError, match="expected string type in schema for property .* when used with score rubric"):
        ScoreParserJSON(
            score=RubricScore(
                name="quality",
                rubric=[
                    Rubric(label="good", value=1),
                    Rubric(label="bad", value=0),
                ],
                parser=JSONScoreParser(json_path="quality"),
            ),
            structured_output={"schema": {"type": "object", "properties": {"quality": {"type": "integer"}}}},
        )


@pytest.mark.parametrize(
    "desc,text",
    [
        ("not json", "this is not a json response"),
        ("empty json", "{}"),
        ("doesn't match schema", '{"quality": "string not int"}'),
        ("key not found", '{"key": "string not int"}'),
    ],
)
def test_json_score_parser_nan(desc, text):
    parser = ScoreParserJSON(
        score=_new_range_score(JSONScoreParser(json_path="quality")),
        structured_output={"schema": {"type": "object", "properties": {"quality": {"type": "number"}}}},
    )
    score = parser.parse(text)
    assert math.isnan(score.value), f"{score} {desc}"


def test_json_score_parser():
    parser = ScoreParserJSON(
        score=_new_range_score(JSONScoreParser(json_path="quality")),
        structured_output={"schema": {"type": "object", "properties": {"quality": {"type": "number"}}}},
    )
    text = '{"quality": 7.2}'
    score = parser.parse(text)
    assert score.value == 7.2


def test_json_score_parser_bool():
    parser = ScoreParserJSON(
        score=_new_range_score(JSONScoreParser(json_path="quality")),
        structured_output={"schema": {"type": "object", "properties": {"quality": {"type": "boolean"}}}},
    )
    text = '{"quality": true}'
    score = parser.parse(text)
    assert score.value == 1.0


@pytest.mark.parametrize(
    "desc,text",
    [
        ("not json", "this is not a json response"),
        ("empty json", "{}"),
        ("key not found", '{"key": "string not int"}'),
        ("doesn't match label", '{"quality": "not-a-label"}'),
        ("value type mismatch", '{"quality": true}'),
    ],
)
def test_json_score_parser_rubric_nan(desc, text):
    parser = ScoreParserJSON(
        score=RubricScore(
            name="quality",
            rubric=[
                Rubric(label="good", value=1),
                Rubric(label="bad", value=0),
            ],
            parser=JSONScoreParser(json_path="quality"),
        ),
        structured_output={"schema": {"type": "object", "properties": {"quality": {"type": "string"}}}},
    )
    score = parser.parse(text)
    assert math.isnan(score.value), desc


def test_json_score_parser_rubric():
    parser = ScoreParserJSON(
        score=RubricScore(
            name="quality",
            rubric=[
                Rubric(label="good", value=1),
                Rubric(label="bad", value=0),
            ],
            parser=JSONScoreParser(json_path="quality"),
        ),
        structured_output={"schema": {"type": "object", "properties": {"quality": {"type": "string"}}}},
    )
    score = parser.parse('{"quality": "good"}')
    assert score.value == 1
    assert score.stats is not None
    assert score.stats.rubric_distribution is not None
    assert len(score.stats.rubric_distribution) == 2
    assert score.stats.rubric_distribution[0].label == "good"
    assert score.stats.rubric_distribution[0].count == 1
    assert score.stats.rubric_distribution[1].label == "bad"
    assert score.stats.rubric_distribution[1].count == 0

    score = parser.parse('{"quality": "bad"}')
    assert score.value == 0
    assert score.stats is not None
    assert score.stats.rubric_distribution is not None
    assert len(score.stats.rubric_distribution) == 2
    assert score.stats.rubric_distribution[0].label == "good"
    assert score.stats.rubric_distribution[0].count == 0
    assert score.stats.rubric_distribution[1].label == "bad"
    assert score.stats.rubric_distribution[1].count == 1


def test_json_parser_no_structured_output():
    parser = ScoreParserJSON(
        score=RubricScore(
            name="quality",
            rubric=[
                Rubric(label="good", value=1),
                Rubric(label="bad", value=0),
            ],
            parser=JSONScoreParser(json_path="quality"),
        ),
    )
    score = parser.parse('{"quality": "good"}')
    assert score.value == 1
    assert score.stats is not None
    assert score.stats.rubric_distribution is not None
    assert len(score.stats.rubric_distribution) == 2
    assert score.stats.rubric_distribution[0].label == "good"
    assert score.stats.rubric_distribution[0].count == 1
    assert score.stats.rubric_distribution[1].label == "bad"
    assert score.stats.rubric_distribution[1].count == 0

    score = parser.parse('{"quality": "bad"}')
    assert score.value == 0
    assert score.stats is not None
    assert score.stats.rubric_distribution is not None
    assert len(score.stats.rubric_distribution) == 2
    assert score.stats.rubric_distribution[0].label == "good"
    assert score.stats.rubric_distribution[0].count == 0
    assert score.stats.rubric_distribution[1].label == "bad"
    assert score.stats.rubric_distribution[1].count == 1


def _is_structured_output_probe(request: dict) -> bool:
    """Identify a preflight structured-output probe by its marker schema."""
    return "__nhx_probe_score" in json.dumps(request)


class TestLLMJudgeMetric:
    def test_rejects_duplicate_score_names(self):
        with pytest.raises(ValueError, match="score names must be unique"):
            LLMJudgeMetric(
                model=_make_model(),
                scores=[_make_metric_score("helpfulness"), _make_metric_score("helpfulness")],
            )

    def test_rejects_reserved_prompt_template_keys(self):
        with pytest.raises(ValueError, match="prompt_template cannot include system_prompt"):
            LLMJudgeMetric(
                model=_make_model(),
                scores=[_make_metric_score()],
                prompt_template={"system_prompt": "bad"},
            )

    def test_offline_default_prompt_template_uses_offline_default(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            job_type=SupportedJobTypes.OFFLINE,
        )
        metric.apply_evaluation_job_params(RunConfig())

        assert metric.prompt_template == default_judge_prompt_template_chat(SupportedJobTypes.OFFLINE)

    def test_offline_input_schema_uses_offline_default_prompt_template(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            job_type=SupportedJobTypes.OFFLINE,
        )

        assert metric.input_schema().model_dump()["schema"]["required"] == []

    def test_offline_completion_prompt_template_uses_completion_default(self):
        metric = LLMJudgeMetric(
            model=_make_completion_model(),
            scores=[_make_metric_score()],
            job_type=SupportedJobTypes.OFFLINE,
        )
        metric.apply_evaluation_job_params(RunConfig())

        assert metric.prompt_template == default_judge_prompt_template_completions(SupportedJobTypes.OFFLINE)

    def test_generates_structured_output_from_scores(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
        )

        assert metric.structured_output == {
            "schema": {
                "type": "object",
                "properties": {
                    "helpfulness": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    }
                },
                "required": ["helpfulness"],
            }
        }

    def test_initializes_json_parser_with_auto_generated_structured_output(self):
        """Ensure first construction validates JSON parsers against the derived schema."""
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
        )

        parser = metric._parsers["helpfulness"]
        structured_output = metric.structured_output

        assert isinstance(parser, ScoreParserJSON)
        assert structured_output is not None
        assert parser.structured_output == structured_output
        assert parser.json_schema == structured_output["schema"]

    def test_score_names_match_initialized_parsers(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        assert output_names(metric) == ["helpfulness"]

    def test_rubric_output_spec_includes_numeric_score_and_label(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_new_rubric_score(None)])

        assert output_names(metric) == ["length", "length.label"]

    @pytest.mark.asyncio
    async def test_rubric_compute_scores_emits_numeric_score_and_label(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_new_rubric_score(None)])
        metric.set_inference_fn(
            mocker.AsyncMock(return_value={"choices": [{"message": {"content": '{"length":"short"}'}}]})
        )

        result = await compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"})

        assert [(output.name, output.value) for output in result.outputs] == [("length", 0), ("length.label", "short")]

    def test_unique_scores_allows_empty_scores_for_constructed_model(self):
        metric = LLMJudgeMetric.model_construct(model=_make_model(), scores=[], _fields_set={"model", "scores"})
        assert metric.unique_scores() is metric  # ty: ignore[call-non-callable]  # Pydantic model_validator descriptor

    def test_string_prompt_template_bypasses_reserved_key_validation(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            prompt_template="Judge: {{sample.output_text}}",
        )
        assert metric.prompt_template == "Judge: {{sample.output_text}}"

    def test_input_schema_reflects_prompt_template_requirements(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            prompt_template={
                "messages": [
                    {
                        "role": "user",
                        "content": "Question: {{input}}\nAnswer: {{output}}\nReference: {{reference}}",
                    }
                ]
            },
        )

        schema = metric.input_schema().schema_

        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["properties"] == {"input": {}, "output": {}, "reference": {}}
        assert set(schema["required"]) == {"input", "output", "reference"}

    def test_input_schema_respects_optional_fields(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            prompt_template={
                "messages": [
                    {
                        "role": "user",
                        "content": "Question: {{input}}\nAnswer: {{output}}\nReference: {{reference}}",
                    }
                ]
            },
            optional_fields=["reference"],
        )

        schema = metric.input_schema().schema_

        assert schema["properties"] == {"input": {}, "output": {}, "reference": {}}
        assert set(schema["required"]) == {"input", "output"}

    def test_input_schema_ignores_runtime_scores_in_default_prompt_template(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
        )

        schema = metric.input_schema().schema_

        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["properties"] == {"output": {}}
        assert set(schema["required"]) == {"output"}

    def test_input_schema_ignores_scores_expression_with_function_call(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            prompt_template={
                "messages": [
                    {
                        "role": "system",
                        "content": "Evaluate across: {{ scores.keys() | join(', ') }}",
                    },
                    {
                        "role": "user",
                        "content": "Question: {{input}}\nAnswer: {{sample.output_text}}",
                    },
                ]
            },
        )

        schema = metric.input_schema().schema_

        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["properties"] == {"input": {}, "output": {}}
        assert set(schema["required"]) == {"input", "output"}

    def test_invalid_prompt_template_raises_when_input_schema_is_requested(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            prompt_template="{% for message in messages %}{{ message.content }}{% endfor %}",
        )

        with pytest.raises(ValueError, match="unsupported Jinja construct for dataset schema inference: For"):
            metric.input_schema()

    @pytest.mark.asyncio
    async def test_resolve_secrets_raises_when_secret_is_missing(self):
        metric = LLMJudgeMetric(
            model=_make_model().model_copy(update={"api_key_secret": SecretRef(root="judge-secret")}),
            scores=[_make_metric_score()],
        )

        with pytest.raises(ValueError, match="Missing secret 'judge-secret'"):
            await metric.resolve_secrets(_MissingSecretResolver())

    @pytest.mark.asyncio
    async def test_resolve_models_clears_cached_auth_and_client(self):
        metric = LLMJudgeMetric(model=ModelRef(root="workspace/judge"), scores=[_make_metric_score()])
        assert metric.__pydantic_private__ is not None
        metric.__pydantic_private__["_client"] = SimpleNamespace(copy=lambda: object())
        metric._api_key = "old-key"

        await metric.resolve_models(_RegisteredModelResolver())

        assert isinstance(metric.model, Model)
        assert metric.model.name == "resolved-judge"
        assert metric._client is None
        assert metric._api_key is None

    def test_deepcopy_clears_cached_auth_and_client(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        assert metric.__pydantic_private__ is not None
        metric.__pydantic_private__["_client"] = SimpleNamespace(copy=lambda: object())
        metric._api_key = "old-key"

        copied = deepcopy(metric)

        assert copied._client is None
        assert copied._api_key is None

    def test_secrets_returns_mapping_when_model_has_api_key_secret(self):
        metric = LLMJudgeMetric(
            model=_make_model().model_copy(update={"api_key_secret": SecretRef(root="judge-secret")}),
            scores=[_make_metric_score()],
        )
        assert metric.secrets() == {"judge_secret": SecretRef(root="judge-secret")}

    def test_secrets_returns_empty_mapping_without_api_key_secret(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        assert metric.secrets() == {}

    def test_initialize_score_parsers_requires_parser(self):
        with pytest.raises(ValueError, match="parser is required for LLM-as-a-Judge score helpfulness"):
            LLMJudgeMetric.model_construct(
                model=_make_model(),
                scores=[SimpleNamespace(name="helpfulness", parser=None)],
                prompt_template=default_judge_prompt_template_chat(),
                structured_output=None,
                _fields_set={"model", "scores", "prompt_template", "structured_output"},
            )

    def test_initialize_score_parsers_rejects_unknown_parser_type(self):
        with pytest.raises(ValueError, match="unknown parser type for LLM-as-a-Judge score helpfulness"):
            LLMJudgeMetric.model_construct(
                model=_make_model(),
                scores=[SimpleNamespace(name="helpfulness", parser=SimpleNamespace(type="unknown_parser"))],
                prompt_template=default_judge_prompt_template_chat(),
                structured_output=None,
                _fields_set={"model", "scores", "prompt_template", "structured_output"},
            )

    def test_handle_none_output_error_mentions_reasoning_only_responses(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        error = metric._handle_none_output_error(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "Need more tokens",
                        }
                    }
                ]
            }
        )
        assert "reasoning output but no final text content" in str(error)

    def test_handle_invalid_output_returns_fallback_when_ignore_request_failure_enabled(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()], ignore_request_failure=True)
        fallback = MetricResult(outputs=[MetricOutput(name="helpfulness", value=float("nan"))])
        assert metric._handle_invalid_output(ValueError("boom"), fallback, "ignored") is fallback

    def test_handle_invalid_output_raises_when_ignore_request_failure_disabled(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        fallback = MetricResult(outputs=[MetricOutput(name="helpfulness", value=float("nan"))])
        with pytest.raises(ValueError, match="boom"):
            metric._handle_invalid_output(ValueError("boom"), fallback, "ignored")

    def test_validate_output_text_raises_for_none_output(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        with pytest.raises(ValueError, match="LLM judge returned no usable textual content"):
            metric._validate_output_text(None, {"choices": [{"message": {"content": None}}]})

    def test_render_request_warns_on_overlapping_keys_and_includes_score_dumps(self, mocker: MockerFixture):
        warn = mocker.patch("nemo_evaluator_sdk.metrics.llm_judge._logger.warning")
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            prompt_template={"messages": [{"role": "user", "content": "{{prompt}} {{scores.helpfulness.minimum}}"}]},
        )

        request = metric._render_request({"prompt": "item"}, {"prompt": "sample"})

        warn.assert_called_once()
        assert request["messages"][0]["content"] == "sample 1"
        assert request["max_tokens"] == 1024

    def test_retry_with_max_completion_tokens_updates_request(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        request = {"max_tokens": 64, "messages": []}

        rewritten = metric._retry_with_max_completion_tokens(request)

        assert metric._use_max_completion_tokens is True
        assert rewritten == {"max_completion_tokens": 64, "messages": []}

    def test_render_request_uses_max_completion_tokens_when_flag_enabled(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        metric._use_max_completion_tokens = True

        request = metric._render_request({"prompt": "hello"}, {"output_text": "world"})

        assert "max_tokens" not in request
        assert request["max_completion_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_metric_retries_with_max_completion_tokens(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
        )
        error = ClientInferenceError(
            mocker.Mock(status_code=400, response=mocker.Mock(text="'max_tokens' is not supported with this model"))
        )
        captured_requests: list[dict] = []

        async def inference_fn(*args, **kwargs):
            request = kwargs.get("request", args[1])
            if _is_structured_output_probe(request):
                # Preflight probes share this inference_fn; they are not scoring requests.
                return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}
            captured_requests.append(deepcopy(request))
            if len(captured_requests) == 1:
                raise error
            return {"choices": [{"message": {"content": '{"helpfulness": 3}'}}]}

        metric.set_inference_fn(inference_fn)

        assert (await compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"})).outputs[0].value == 3
        assert len(captured_requests) == 2
        first_request = captured_requests[0]
        second_request = captured_requests[1]
        assert "max_tokens" in first_request
        assert "max_completion_tokens" in second_request

    @pytest.mark.asyncio
    async def test_metric_returns_score_when_trace_is_disabled(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
        )
        metric.set_inference_fn(
            mocker.AsyncMock(return_value={"choices": [{"message": {"content": '{"helpfulness": 3}'}}]})
        )

        assert (await compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"})).outputs[0].value == 3

    @pytest.mark.asyncio
    async def test_metric_invalid_output_returns_nan_when_ignore_enabled(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            ignore_request_failure=True,
        )
        metric.set_inference_fn(
            mocker.AsyncMock(return_value={"choices": [{"message": {"content": None, "reasoning_content": None}}]})
        )

        result = await compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"})
        assert len(result.outputs) == 1
        assert result.outputs[0].name == "helpfulness"
        assert math.isnan(result.outputs[0].value)

    @pytest.mark.asyncio
    async def test_compute_scores_returns_nan_when_inference_failure_is_ignored(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            ignore_request_failure=True,
        )
        error = ClientInferenceError(mocker.Mock(status_code=500, response=mocker.Mock(text="boom")))
        metric.set_inference_fn(mocker.AsyncMock(side_effect=error))

        result = await compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"})

        assert len(result.outputs) == 1
        assert result.outputs[0].name == "helpfulness"
        assert math.isnan(result.outputs[0].value)

    @pytest.mark.asyncio
    async def test_compute_scores_retries_with_max_completion_tokens(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        error = ClientInferenceError(
            mocker.Mock(status_code=400, response=mocker.Mock(text="'max_tokens' is not supported with this model"))
        )
        captured_requests: list[dict] = []

        async def inference_fn(*args, **kwargs):
            request = kwargs.get("request", args[1])
            if _is_structured_output_probe(request):
                # Preflight probes share this inference_fn; they are not scoring requests.
                return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}
            captured_requests.append(deepcopy(request))
            if len(captured_requests) == 1:
                raise error
            return {"choices": [{"message": {"content": '{"helpfulness": 4}'}}]}

        metric.set_inference_fn(inference_fn)

        result = await compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"})

        assert result.outputs[0].value == 4
        assert "max_tokens" in captured_requests[0]
        assert "max_completion_tokens" in captured_requests[1]

    @pytest.mark.asyncio
    async def test_compute_scores_raises_invalid_output_when_ignore_failure_disabled(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        metric.set_inference_fn(
            mocker.AsyncMock(return_value={"choices": [{"message": {"content": None, "reasoning_content": None}}]})
        )

        with pytest.raises(ValueError, match="LLM judge returned no usable textual content"):
            await compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"})

    @pytest.mark.asyncio
    async def test_preflight_selects_structured_output_mode_for_nim(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model()
            .with_default_headers({"X-NHX-Principal-Id": "service:evaluator"})
            .model_copy(update={"format": ModelFormat.NVIDIA_NIM}),
            scores=[_make_metric_score()],
        )
        detect = mocker.patch(
            "nemo_evaluator_sdk.metrics.llm_judge.detect_structured_output_mode",
            new_callable=mocker.AsyncMock,
            return_value=StructuredOutputMode.ROOT_GUIDED_JSON,
        )

        await metric.preflight()

        detect.assert_awaited_once()
        assert detect.await_args.kwargs["model"].default_headers == {"X-NHX-Principal-Id": "service:evaluator"}
        structured_hook = next(hook for hook in metric._preprocess_hooks if hasattr(hook, "mode"))
        assert structured_hook.mode == StructuredOutputMode.ROOT_GUIDED_JSON

    @pytest.mark.asyncio
    async def test_preflight_selects_structured_output_mode_for_non_nim_formats(self, mocker: MockerFixture):
        """Mode selection is driven by detection for every format, not just NIM."""
        metric = LLMJudgeMetric(
            model=_make_model().model_copy(update={"format": ModelFormat.OPEN_AI}),
            scores=[_make_metric_score()],
        )
        detect = mocker.patch(
            "nemo_evaluator_sdk.metrics.llm_judge.detect_structured_output_mode",
            new_callable=mocker.AsyncMock,
            return_value=StructuredOutputMode.OPENAI_RESPONSE_FORMAT,
        )

        await metric.preflight()

        detect.assert_awaited_once()
        structured_hook = next(hook for hook in metric._preprocess_hooks if hasattr(hook, "mode"))
        assert structured_hook.mode == StructuredOutputMode.OPENAI_RESPONSE_FORMAT

    @pytest.mark.asyncio
    async def test_preflight_is_noop_without_structured_output(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[RangeScore(name="helpfulness", minimum=1, maximum=5, parser=RegexScoreParser(pattern="(\\d+)"))],
        )
        detect = mocker.patch(
            "nemo_evaluator_sdk.metrics.llm_judge.detect_structured_output_mode",
            new_callable=mocker.AsyncMock,
        )

        await metric.preflight()

        detect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_preflight_is_noop_when_structured_hook_is_missing(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model().model_copy(update={"format": ModelFormat.NVIDIA_NIM}),
            scores=[_make_metric_score()],
        )
        metric._preprocess_hooks = []
        detect = mocker.patch(
            "nemo_evaluator_sdk.metrics.llm_judge.detect_structured_output_mode",
            new_callable=mocker.AsyncMock,
        )

        await metric.preflight()

        detect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_compute_scores_passes_model_default_headers_to_inference_fn(self):
        metric = LLMJudgeMetric(
            model=_make_model().with_default_headers({"special-header": "evaluator"}),
            scores=[_make_metric_score()],
        )
        captured: dict = {}

        async def inference_fn(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            captured["default_headers"] = model.default_headers
            return {"choices": [{"message": {"content": '{"helpfulness": 4}'}}]}

        metric.set_inference_fn(inference_fn)

        result = await compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"})

        assert result.outputs[0].value == 4
        assert captured["default_headers"] == {"special-header": "evaluator"}

    def test_render_request_default_prompt_includes_rubric_details(self):
        """Default judge prompt should render score and rubric details into system message."""
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[
                RubricScore(
                    name="quality",
                    description="Overall answer quality",
                    rubric=[
                        Rubric(label="poor", description="Contains major errors", value=0),
                        Rubric(label="good", description="Accurate and complete", value=1),
                    ],
                )
            ],
        )
        request = metric._render_request({"question": "What is AI?"}, {"output_text": "AI is artificial intelligence"})

        assert "messages" in request
        assert len(request["messages"]) >= 2

        system_message = request["messages"][0]["content"]
        assert "assess responses to user queries based on quality" in system_message
        assert "quality: Overall answer quality" in system_message
        assert "* poor: Contains major errors" in system_message
        assert "* good: Accurate and complete" in system_message

    @pytest.mark.asyncio
    async def test_ignore_request_failure_empty_output_returns_nan(self, mocker: MockerFixture):
        metric = LLMJudgeMetric.model_validate(llm_judge_param_dict({"ignore_request_failure": True}))
        metric.set_inference_fn(mocker.AsyncMock(return_value=_empty_judge_response()))

        result = await compute_scores(
            metric, {"question": "What is AI?"}, {"output_text": "AI is artificial intelligence"}
        )
        assert len(result.outputs) == 2
        assert result.outputs[0].name == "quality"
        assert math.isnan(result.outputs[0].value)
        assert result.outputs[1] == MetricOutput(name="quality.label", value="")

    @pytest.mark.asyncio
    async def test_empty_output_raises(self, mocker: MockerFixture):
        metric = LLMJudgeMetric.model_validate(llm_judge_param_dict())
        metric.set_inference_fn(mocker.AsyncMock(return_value=_empty_judge_response()))

        with pytest.raises(ValueError, match="LLM judge returned no usable textual content for score parsing"):
            await compute_scores(metric, {"question": "What is AI?"}, {"output_text": "AI is artificial intelligence"})

    @pytest.mark.asyncio
    async def test_none_output_raises(self, mocker: MockerFixture):
        metric = LLMJudgeMetric.model_validate(llm_judge_param_dict())
        metric.set_inference_fn(mocker.AsyncMock(return_value=_empty_judge_response()))

        with pytest.raises(
            ValueError,
            match="inference\\.extra_body\\.nvext\\.max_thinking_tokens",
        ):
            await compute_scores(metric, {"question": "What is AI?"}, {"output_text": "AI is artificial intelligence"})


class TestGenerateStructuredOutput:
    def test_returns_explicit_structured_output_when_provided(self):
        explicit = {"schema": {"type": "object", "properties": {"helpfulness": {"type": "integer"}}}}
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            structured_output=explicit,
        )
        assert generate_structured_output(metric) == explicit

    def test_returns_none_when_no_json_parsed_scores_exist(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[RangeScore(name="helpfulness", minimum=1, maximum=5, parser=RegexScoreParser(pattern="(\\d+)"))],
        )
        assert generate_structured_output(metric) is None

    def test_raises_for_conflicting_auto_generated_schemas(self):
        with pytest.raises(ValueError, match="conflicting auto-generated structured_output"):
            LLMJudgeMetric.model_construct(
                model=_make_model(),
                scores=[
                    RangeScore(name="helpfulness", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score")),
                    RubricScore(
                        name="quality",
                        rubric=[Rubric(label="good", value=1), Rubric(label="bad", value=0)],
                        parser=JSONScoreParser(json_path="score"),
                    ),
                ],
                prompt_template=default_judge_prompt_template_chat(),
                structured_output=None,
                _fields_set={"model", "scores", "prompt_template", "structured_output"},
            )

    def test_skips_json_scores_that_are_not_range_or_rubric(self):
        params = SimpleNamespace(
            structured_output=None,
            scores=[SimpleNamespace(name="custom", parser=JSONScoreParser(json_path="custom"))],
        )

        assert generate_structured_output(params) is None

    def test_roundtrip_uses_parser_json_path_for_auto_generated_schema(self):
        """Ensure direct construction and roundtrip reconstruction agree on parser json_path."""
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[
                RangeScore(
                    name="accuracy",
                    minimum=1,
                    maximum=5,
                    parser=JSONScoreParser(json_path="score"),
                )
            ],
        )

        assert metric.structured_output == {
            "schema": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    }
                },
                "required": ["score"],
            }
        }

        roundtrip_metric = LLMJudgeMetric.model_validate(metric.model_dump(mode="json"))
        parser = roundtrip_metric._parsers["accuracy"]

        assert isinstance(parser, ScoreParserJSON)
        assert parser.json_path == "score"
        assert parser.structured_output == roundtrip_metric.structured_output

    def test_uses_json_path_as_property_key(self):
        """Verify the generated schema keys on json_path, not the score name."""
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[RangeScore(name="accuracy", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score"))],
        )
        assert generate_structured_output(metric) == {
            "schema": {
                "type": "object",
                "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["score"],
            }
        }

    def test_deduplicates_shared_json_path(self):
        """Two scores sharing the same json_path and identical schema produce a single property."""
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[
                RangeScore(name="accuracy", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score")),
                RangeScore(name="relevance", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score")),
            ],
        )
        assert generate_structured_output(metric) == {
            "schema": {
                "type": "object",
                "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["score"],
            }
        }

    def test_generate_structured_output_none(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[RangeScore(name="accuracy", minimum=1, maximum=5, parser=RegexScoreParser(pattern="custom"))],
        )
        structured_output = generate_structured_output(metric)
        assert structured_output is None, "scores with regex parser does not generate structured output"

    def test_new_hooks_structured_output_validation_error(self):
        with pytest.raises(SchemaError):
            LLMJudgeMetric(
                model=_make_model(),
                scores=[RangeScore(name="accuracy", minimum=1, maximum=5)],
                structured_output={"schema": {"type": "non-object"}},
            )
        with pytest.raises(ValueError, match="structured output contains invalid JSON schema"):
            param_dict = llm_judge_param_dict({"structured_output": {"schema": {"type": "non-object"}}})
            LLMJudgeMetric.model_validate(param_dict)

    def test_rejects_conflicting_ranges_on_shared_json_path(self):
        """Two RangeScores sharing a json_path but with different ranges are rejected."""
        with pytest.raises(ValueError, match="conflicting auto-generated structured_output for json_path 'score'"):
            LLMJudgeMetric.model_construct(
                model=_make_model(),
                scores=[
                    RangeScore(name="accuracy", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score")),
                    RangeScore(name="relevance", minimum=0, maximum=10, parser=JSONScoreParser(json_path="score")),
                ],
                prompt_template=default_judge_prompt_template_chat(),
                structured_output=None,
                _fields_set={"model", "scores", "prompt_template", "structured_output"},
            )

    def test_accepts_preexisting_structured_output_with_custom_json_path(self):
        """Explicit structured_output is preserved even when scores use a custom json_path."""
        explicit = {
            "schema": {
                "type": "object",
                "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["score"],
            }
        }
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[RangeScore(name="accuracy", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score"))],
            structured_output=explicit,
        )
        assert metric.structured_output == explicit
        assert "accuracy" in metric._parsers

    def test_render_request_preserves_nvext_max_thinking_tokens_with_structured_output(self):
        """Structured output must merge into an existing extra_body.nvext, not replace it."""
        metric = LLMJudgeMetric.model_construct(
            model=_make_model(),
            scores=[RubricScore(name="quality", rubric=[Rubric(label="good", value=1), Rubric(label="bad", value=0)])],
            inference=InferenceParams.model_validate({"extra_body": {"nvext": {"max_thinking_tokens": 256}}}),
        )
        item = {"question": "What is AI?"}
        sample = {"output_text": "AI is artificial intelligence"}
        request = metric._render_request(item, sample)

        assert request["extra_body"]["nvext"]["max_thinking_tokens"] == 256
        assert "response_format" in request

    def test_render_request_merges_nvext_guided_json_beside_max_thinking_tokens(self):
        """The nvext placement still deep-merges once preflight selects it."""
        metric = LLMJudgeMetric.model_construct(
            model=_make_model(),
            scores=[RubricScore(name="quality", rubric=[Rubric(label="good", value=1), Rubric(label="bad", value=0)])],
            inference=InferenceParams.model_validate({"extra_body": {"nvext": {"max_thinking_tokens": 256}}}),
        )
        for hook in metric._preprocess_hooks:
            if isinstance(hook, InferenceStructuredOutput):
                hook.set_mode(StructuredOutputMode.NVEXT_GUIDED_JSON)
        request = metric._render_request({"question": "What is AI?"}, {"output_text": "AI is intelligence"})

        assert request["extra_body"]["nvext"]["max_thinking_tokens"] == 256
        assert "guided_json" in request["extra_body"]["nvext"]


# =============================================================================
# Hooks
# =============================================================================


def test_default_hooks():
    params = llm_judge_param_dict()
    metric = LLMJudgeMetric.model_validate(params)
    pre, post = metric._preprocess_hooks, metric._postprocess_hooks

    assert len(pre) == 1, "expected log and inference param hooks, no structured output with Regex score parser"
    assert isinstance(pre[0], LogHook), "log hook"

    assert len(post) == 1, "expect only log hook"
    assert isinstance(post[0], LogHook), "log hook"
    assert pre[0] == post[0], "pre and post should have the same instance of log hook"


def test_new_hooks_with_defaults():
    metric = LLMJudgeMetric.model_validate(
        {
            "prompt_template": {"messages": [{"role": "user", "content": "Evaluate this"}]},
            "scores": [
                {
                    "name": "quality",
                    "rubric": [{"label": "good", "value": 1}, {"label": "bad", "value": 0}],
                    # Test default score parser will generate structured output and its hook
                }
            ],
            "model": {"url": "http://nemo.test", "name": "model-id"},
            "inference": {
                "temperature": 0.9,
            },
        }
    )
    pre, post = metric._preprocess_hooks, metric._postprocess_hooks

    assert len(pre) == 3, "expected all hooks with generated structured output"
    assert isinstance(pre[0], AddInferenceParameter)
    assert pre[0].params == {"temperature": 0.9}
    assert isinstance(pre[1], InferenceStructuredOutput)
    assert pre[1].inference_param == {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": {
                    "type": "object",
                    "properties": {"quality": {"enum": ["good", "bad"], "type": "string"}},
                    "required": ["quality"],
                },
                "strict": False,
            },
        }
    }
    assert isinstance(pre[2], LogHook), "log hook"

    assert len(post) == 1, "expect only log hook"
    assert isinstance(post[0], LogHook), "log hook"
    assert pre[2] == post[0], "pre and post should have the same instance of log hook"


def test_new_hooks_defaults_to_response_format_before_probing():
    """new_hooks installs response_format as the PROVISIONAL default, not because of the format.

    Selection is endpoint-driven; preflight replaces this. Named away from "openai format uses
    response_format" so it cannot be read as licence to restore format-based selection.
    """
    metric = LLMJudgeMetric.model_validate(
        {
            "name": "test-judge",
            "workspace": "test-ws",
            "scores": [
                {
                    "name": "quality",
                    "rubric": [{"label": "good", "value": 1}, {"label": "bad", "value": 0}],
                }
            ],
            "model": {
                "url": "https://api.openai.com/v1",
                "name": "gpt-4",
                "format": "openai",
            },
        }
    )

    # Find the InferenceStructuredOutput hook
    structured_output_hook = None
    for hook in metric._preprocess_hooks:
        if isinstance(hook, InferenceStructuredOutput):
            structured_output_hook = hook
            break

    assert structured_output_hook is not None, "Expected InferenceStructuredOutput hook"
    assert "response_format" in structured_output_hook.inference_param
    assert structured_output_hook.inference_param["response_format"]["type"] == "json_schema"


def test_new_hooks_nim_format_no_longer_forces_nvext():
    """A `nim` label must not preselect nvext: preflight decides from the endpoint.

    Hardcoding nvext here is what made integrate.api.nvidia.com fall back to an unenforced prompt
    instruction, since that endpoint rejects nvext.guided_json outright.
    """
    metric = LLMJudgeMetric.model_validate(
        {
            "name": "test-judge",
            "workspace": "test-ws",
            "scores": [
                {
                    "name": "quality",
                    "rubric": [{"label": "good", "value": 1}, {"label": "bad", "value": 0}],
                }
            ],
            "model": {
                "url": "http://nim-endpoint",
                "name": "nim-model",
                "format": "nim",
            },
        }
    )

    # Find the InferenceStructuredOutput hook
    structured_output_hook = None
    for hook in metric._preprocess_hooks:
        if isinstance(hook, InferenceStructuredOutput):
            structured_output_hook = hook
            break

    assert structured_output_hook is not None, "Expected InferenceStructuredOutput hook"
    assert "extra_body" not in structured_output_hook.inference_param
    assert "response_format" in structured_output_hook.inference_param


# =============================================================================
# max_retries passed as None to OpenAI client
# =============================================================================


def llm_judge_param_dict(extra: dict | None = None) -> dict:
    if extra is None:
        extra = {}
    return {
        "prompt_template": {"messages": [{"role": "user", "content": "Evaluate this"}]},
        "scores": [
            {
                "name": "quality",
                "rubric": [{"label": "good", "value": 1}, {"label": "bad", "value": 0}],
                "parser": {"type": "regex", "pattern": "QUALITY: (\\S+)"},
            }
        ],
        "model": {"url": "http://nemo.test", "name": "model-id"},
        **extra,
    }


@pytest.mark.asyncio
async def test_llm_judge_metric_passes_max_retries():
    """
    Regression test for NVBug 5829809.

    Verify that the metric() method passes max_retries=3 to the inference
    function instead of None, which would cause the OpenAI client to raise:
    "max_retries cannot be None"
    """
    metric = LLMJudgeMetric.model_validate(llm_judge_param_dict())

    # Capture the max_retries argument passed to inference_fn
    captured_max_retries = None

    async def fake_inference_fn(model, request, max_retries, **kwargs):
        nonlocal captured_max_retries
        captured_max_retries = max_retries
        return {"choices": [{"message": {"content": "QUALITY: good"}}]}

    metric.set_inference_fn(fake_inference_fn)

    item = {"question": "What is AI?"}
    sample = {"output_text": "AI is artificial intelligence"}

    await compute_scores(metric, item, sample)

    # Verify max_retries was passed as 3, not None
    assert captured_max_retries == 3, f"Expected max_retries=3, got {captured_max_retries}"


@pytest.mark.asyncio
async def test_llm_judge_compute_scores_passes_max_retries():
    """
    Regression test for NVBug 5829809.

    Verify that the compute_scores() method passes max_retries=3 to the
    inference function instead of None, which would cause the OpenAI client to raise:
    "max_retries cannot be None"
    """
    metric = LLMJudgeMetric.model_validate(llm_judge_param_dict())

    # Capture the max_retries argument passed to inference_fn
    captured_max_retries = None

    async def fake_inference_fn(model, request, max_retries, **kwargs):
        nonlocal captured_max_retries
        captured_max_retries = max_retries
        return {"choices": [{"message": {"content": "QUALITY: good"}}]}

    metric.set_inference_fn(fake_inference_fn)

    item = {"question": "What is AI?"}
    sample = {"output_text": "AI is artificial intelligence"}

    await compute_scores(metric, item, sample)

    # Verify max_retries was passed as 3, not None
    assert captured_max_retries == 3, f"Expected max_retries=3, got {captured_max_retries}"


# =============================================================================
# system_prompt and reasoning support
# =============================================================================


def test_llm_judge_prompt_template_accepts_regular_dict_payload():
    metric = LLMJudgeMetric.model_validate(llm_judge_param_dict())
    assert isinstance(metric.prompt_template, dict)
    assert metric.prompt_template == {"messages": [{"role": "user", "content": "Evaluate this"}]}


def test_llm_judge_prompt_template_rejects_nested_system_prompt():
    param_dict = llm_judge_param_dict()
    param_dict["prompt_template"]["system_prompt"] = "Do detailed thinking"

    with pytest.raises(ValidationError, match="system_prompt"):
        LLMJudgeMetric.model_validate(param_dict)


def test_llm_judge_prompt_template_rejects_nested_reasoning():
    param_dict = llm_judge_param_dict()
    param_dict["prompt_template"]["reasoning"] = {"end_token": "</think>"}

    with pytest.raises(ValidationError, match="reasoning"):
        LLMJudgeMetric.model_validate(param_dict)


def test_new_hooks_with_system_prompt():
    """Test that new_hooks creates InjectSystemMessage hook when system_prompt is provided."""
    param_dict = llm_judge_param_dict({"system_prompt": "You are an expert evaluator. Be thorough and precise."})
    metric = LLMJudgeMetric.model_validate(param_dict)

    pre, post = new_hooks(metric)

    # InjectSystemMessage should be the first preprocess hook
    assert len(pre) >= 2
    assert isinstance(pre[0], InjectSystemMessage)
    assert pre[0].system_message == "You are an expert evaluator. Be thorough and precise."


def test_new_hooks_with_reasoning():
    """Test that new_hooks creates TransformReasoningOutput hook when reasoning is provided."""
    param_dict = llm_judge_param_dict({"reasoning": {"end_token": "</think>"}})
    metric = LLMJudgeMetric.model_validate(param_dict)

    pre, post = new_hooks(metric)

    # TransformReasoningOutput should be in postprocess hooks (after LogHook)
    assert len(post) == 2
    assert isinstance(post[0], LogHook)
    assert isinstance(post[1], TransformReasoningOutput)
    assert post[1].end_reasoning_token == "</think>"


def test_new_hooks_with_both_system_prompt_and_reasoning():
    """Test that new_hooks creates both hooks when system_prompt and reasoning are provided."""
    param_dict = llm_judge_param_dict(
        {"system_prompt": "Think carefully before answering.", "reasoning": {"end_token": "</reasoning>"}}
    )
    metric = LLMJudgeMetric.model_validate(param_dict)

    pre, post = new_hooks(metric)

    # Verify InjectSystemMessage is first in preprocess
    assert isinstance(pre[0], InjectSystemMessage)
    assert pre[0].system_message == "Think carefully before answering."

    # Verify TransformReasoningOutput is in postprocess
    assert len(post) == 2
    assert isinstance(post[1], TransformReasoningOutput)
    assert post[1].end_reasoning_token == "</reasoning>"


def test_new_hooks_reasoning_without_end_token():
    """Test that reasoning without end_token does not create TransformReasoningOutput hook."""
    # reasoning without end_token - should not create the transform hook
    param_dict = llm_judge_param_dict({"reasoning": {"include_if_not_finished": True}})
    metric = LLMJudgeMetric.model_validate(param_dict)

    pre, post = new_hooks(metric)

    # Should only have LogHook in postprocess (no TransformReasoningOutput)
    assert len(post) == 1
    assert isinstance(post[0], LogHook)


def test_llm_judge_metric_with_system_prompt_integration():
    """Integration test verifying system_prompt is applied to requests."""
    param_dict = llm_judge_param_dict({"system_prompt": "You are a precise evaluator."})
    metric = LLMJudgeMetric.model_validate(param_dict)

    with patch(
        "nemo_evaluator_sdk.metrics.llm_judge.inference.make_inference_request", new_callable=AsyncMock
    ) as mock_inference:
        mock_inference.return_value = {"choices": [{"message": {"content": "QUALITY: good"}}]}

        item = {"question": "What is AI?"}
        sample = {"output_text": "AI is artificial intelligence"}

        # Render the request to verify system message is injected
        request = metric._render_request(item, sample)

        # The request should have a system message prepended
        assert "messages" in request
        assert request["messages"][0]["role"] == "system"
        assert request["messages"][0]["content"] == "You are a precise evaluator."


def test_llm_judge_top_level_system_prompt_and_reasoning_still_work():
    param_dict = llm_judge_param_dict(
        {"system_prompt": "You are a precise evaluator.", "reasoning": {"end_token": "</think>"}}
    )
    metric = LLMJudgeMetric.model_validate(param_dict)

    request = metric._render_request({"question": "What is AI?"}, {"output_text": "AI is artificial intelligence"})
    assert request["messages"][0] == {"role": "system", "content": "You are a precise evaluator."}

    assert len(metric._postprocess_hooks) == 2
    assert isinstance(metric._postprocess_hooks[1], TransformReasoningOutput)
    assert metric._postprocess_hooks[1].end_reasoning_token == "</think>"


@pytest.mark.asyncio
async def test_compute_scores_probes_the_endpoint_when_called_directly(mocker: MockerFixture):
    """compute_scores is public API: a caller can score without any executor running preflight.

    Without self-preflight the judge would send the provisional encoding new_hooks guessed, and an
    endpoint that ignores it would silently drop the constraint.
    """
    metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
    seen: list[dict] = []

    async def inference_fn(*args, **kwargs):
        request = kwargs.get("request", args[1])
        seen.append(request)
        if _is_structured_output_probe(request):
            # Endpoint honours only nvext.guided_json.
            if "nvext" in request.get("extra_body", {}):
                return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}
            return {"choices": [{"message": {"content": "prose, not json"}}]}
        return {"choices": [{"message": {"content": '{"helpfulness": 3}'}}]}

    metric.set_inference_fn(inference_fn)

    await compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"})

    scoring = [r for r in seen if not _is_structured_output_probe(r)]
    assert [r for r in seen if _is_structured_output_probe(r)], "expected the endpoint to be probed"
    assert scoring[0]["extra_body"]["nvext"]["guided_json"] is not None


@pytest.mark.asyncio
async def test_compute_scores_self_preflight_keeps_the_unresolved_model_ref_error():
    """Self-preflight must not change the failure a caller sees for an unresolved ModelRef.

    preflight() calls _require_model(), which raises for a ModelRef. compute_scores already raised
    the same error at its own _require_model() call, so the guard only moves it earlier -- it must
    not turn into a different or less actionable exception.
    """
    metric = LLMJudgeMetric(model=ModelRef(root="workspace/judge"), scores=[_make_metric_score()])

    with pytest.raises(ValueError, match="has not been resolved"):
        await compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"})


@pytest.mark.asyncio
async def test_concurrent_direct_compute_scores_probe_the_endpoint_once():
    """Direct callers may gather many compute_scores() with no run session to cache detection.

    Without serialising the self-preflight, every concurrent call would see an unresolved hook and
    probe the same endpoint at once -- hundreds of extra requests before any scoring happens.
    """
    metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
    probes: list[dict] = []

    async def inference_fn(*args, **kwargs):
        request = kwargs.get("request", args[1])
        if _is_structured_output_probe(request):
            probes.append(request)
            await asyncio.sleep(0.01)  # widen the stampede window
            return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}
        return {"choices": [{"message": {"content": '{"helpfulness": 3}'}}]}

    metric.set_inference_fn(inference_fn)

    await asyncio.gather(*(compute_scores(metric, {"prompt": "hello"}, {"output_text": "world"}) for _ in range(8)))

    assert len(probes) == 1, f"expected a single probe for 8 concurrent callers, got {len(probes)}"


class TestPreflightLockLifecycle:
    """Pin where the preflight lock is and is not cleared.

    Clearing it in resolve_models() looks tidy and is actively harmful: that method rebuilds the
    hooks as unresolved, so dropping the lock mid-flight splits synchronisation and lets an
    in-flight scorer render through the rebuilt hook, sending a guessed encoding with only a
    warning. A lock retained from another event loop instead fails loudly on its next contended
    acquire. Deep copies are a different case -- a copy is a new object and must start fresh.
    """

    @pytest.mark.asyncio
    async def test_resolve_models_keeps_the_preflight_lock(self):
        metric = LLMJudgeMetric(model=ModelRef(root="workspace/judge"), scores=[_make_metric_score()])
        lock = metric._require_preflight_lock()

        await metric.resolve_models(_RegisteredModelResolver())

        assert metric._preflight_lock is lock, (
            "resolve_models must not clear the preflight lock: doing so splits preflight "
            "synchronisation while a concurrent compute_scores() is mid-probe"
        )

    def test_deepcopy_starts_with_a_fresh_preflight_lock(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        metric._require_preflight_lock()

        clone = deepcopy(metric)

        assert clone._preflight_lock is None
