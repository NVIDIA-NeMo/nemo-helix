# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_evaluator_sdk.execution.metric_execution."""

import asyncio
import json
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar, Literal, cast
from unittest.mock import AsyncMock, Mock

import nemo_evaluator_sdk.inference as inference
import pyarrow as pa
import pytest
from nemo_evaluator_sdk.agent_inference import AgentInvocationResult, AgentInvocationStatus
from nemo_evaluator_sdk.datasets.loader import discover_files, normalize_dataset, rows_from_dataset, split_glob_path
from nemo_evaluator_sdk.enums import AgentFormat, MetricType, ModelFormat
from nemo_evaluator_sdk.execution.backends.local.backend import LocalBackend
from nemo_evaluator_sdk.execution.metric_execution import (
    ComputeMetricPipeline,
    _default_online_request_template,
    _format_exception_summary,
    _is_completions_endpoint,
    _maybe_set_default_max_tokens,
    _merge_online_hooks,
    _resolve_online_prompt_template,
    _score_pipeline_samples,
    evaluate_metric,
    generate_online_sample,
    generate_online_sample_agent,
    run_generated_sample_scoring_pipeline,
    run_sync,
)
from nemo_evaluator_sdk.execution.pipeline import PipelineRuntime
from nemo_evaluator_sdk.execution.samples import build_metric_input
from nemo_evaluator_sdk.execution.scoring import empty_evaluation_result, finalize_evaluation_result
from nemo_evaluator_sdk.execution.values import EvaluationError, EvaluationPhase
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.hooks import HooksBase
from nemo_evaluator_sdk.metrics.llm_judge import LLMJudgeMetric as RuntimeLLMJudgeMetric
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.metrics.utils import metric_type_name
from nemo_evaluator_sdk.resolvers import LocalSecretResolver, _candidate_env_names
from nemo_evaluator_sdk.structured_output import StructuredOutputMode
from nemo_evaluator_sdk.values.agents import Agent, GenericAgent
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_evaluator_sdk.values.datasets import DatasetRows
from nemo_evaluator_sdk.values.models import Model, ReasoningParams
from nemo_evaluator_sdk.values.params import (
    InferenceParams,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)
from nemo_evaluator_sdk.values.results import (
    AggregatedMetricResult,
    EvaluationResult,
    RowScore,
)
from nemo_evaluator_sdk.values.scores import JSONScoreParser, RangeScore
from openai import AsyncOpenAI
from pydantic import BaseModel, PrivateAttr
from pytest_mock import MockerFixture

INFERENCE_FAILURE_HINT = (
    "To prevent failure of evaluation from inference request failures, check the model endpoint, "
    "credentials, request timeout, and retry settings, or set params.ignore_request_failure=true "
    "to mark failed rows as NaN."
)
GENERATION_FAILURE_MESSAGE = f"Row 0 failed inference: gen bad. {INFERENCE_FAILURE_HINT}"


class EmptyMessageError(Exception):
    """Exception whose string value is empty."""

    def __str__(self) -> str:
        """Return an empty error message."""
        return ""


def test_format_exception_summary_collapses_multiline_message() -> None:
    """Cause summaries should be concise single-line strings."""
    assert _format_exception_summary(RuntimeError("first line\nsecond\tline")) == "first line second line"


def test_format_exception_summary_uses_chained_cause_when_message_is_empty() -> None:
    """Empty wrapper exceptions should still surface the underlying cause."""
    try:
        try:
            raise RuntimeError("root\ncause")
        except RuntimeError as cause:
            raise EmptyMessageError() from cause
    except EmptyMessageError as error:
        assert _format_exception_summary(error) == "root cause"


def test_format_exception_summary_falls_back_to_exception_type() -> None:
    """Completely empty exceptions should still have a useful cause label."""
    assert _format_exception_summary(EmptyMessageError()) == "EmptyMessageError"


def _make_metric_result(*scores: tuple[str, float]) -> MetricResult:
    return MetricResult(outputs=[MetricOutput(name=n, value=v) for n, v in scores])


def _score_spec(*names: str) -> list[MetricOutputSpec]:
    return [MetricOutputSpec.continuous_score(name) for name in names]


def _make_mock_metric(mocker: MockerFixture, results: list[MetricResult] | None = None) -> Mock:
    """Create a mock Metric with compute_scores returning results sequentially.

    Also sets compute_corpus_scores to an AsyncMock returning None so that
    the @runtime_checkable CorpusMetric protocol check (which a Mock satisfies
    by auto-creating attributes) doesn't try to await a regular Mock.
    """
    metric = mocker.Mock()
    metric.type = None
    if results is not None:
        metric.compute_scores = mocker.AsyncMock(side_effect=results)
    else:
        default_result = _make_metric_result(("accuracy", 1.0))
        metric.compute_scores = mocker.AsyncMock(return_value=default_result)
    metric.compute_corpus_scores = mocker.AsyncMock(return_value=None)
    metric.output_spec = mocker.Mock(return_value=_score_spec("accuracy", "score", "corpus_score"))
    metric.corpus_output_spec = mocker.Mock(return_value=_score_spec("corpus_score"))
    return metric


def _make_model(
    *,
    url: str = "https://example.test/v1/chat/completions",
    format: Literal[ModelFormat.NVIDIA_NIM, ModelFormat.OPEN_AI, ModelFormat.LLAMA_STACK] = ModelFormat.OPEN_AI,
) -> Model:
    return Model(url=url, name="test-model", format=format)


def _is_structured_output_probe(request: dict) -> bool:
    """Identify a preflight structured-output probe by its marker schema.

    Detect on the probe's own schema marker rather than on message shape, since the probe is sent
    under whichever placement is being tried (response_format or either guided_json position).
    """
    return "__nhx_probe_score" in json.dumps(request)


def _make_agent() -> Agent:
    return GenericAgent(
        url="http://agent.test:8080",
        name="test-agent",
        format=AgentFormat.GENERIC,
        body={"input_message": "{{ messages[-1].content }}"},
        response_path="$.output",
    )


TEST_METRIC_KEY = MetricType.STRING_CHECK.value


class _TestMetric:
    type: ClassVar[MetricType] = MetricType.STRING_CHECK
    compute_scores: AsyncMock
    compute_corpus_scores: AsyncMock

    def __init__(self) -> None:
        self.compute_scores = AsyncMock(return_value=_make_metric_result(("score", 1.0)))
        self.compute_corpus_scores = AsyncMock(return_value=None)

    def __deepcopy__(self, memo: dict[int, object]) -> "_TestMetric":
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied.compute_scores = self.compute_scores
        copied.compute_corpus_scores = self.compute_corpus_scores
        return copied

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]


def _make_test_metric() -> Metric:
    return cast(Metric, _TestMetric())


class _DistinctScoreNameMetric(_TestMetric):
    """Metric stub whose public type differs from its emitted score name."""

    type: ClassVar[str] = "metric-a"

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return an output name that differs from the metric type."""
        return [MetricOutputSpec.continuous_score("score")]


def _make_distinct_score_name_metric() -> Metric:
    return cast(Metric, _DistinctScoreNameMetric())


class _AppendMessageHook(inference.PreprocessRequest):
    def __init__(self, content: str, expected_last_content: str | None = None):
        self._content = content
        self._expected_last_content = expected_last_content

    def preprocess(self, request: dict, id: str | None = None) -> dict:
        del id
        if self._expected_last_content is not None:
            assert request["messages"][-1]["content"] == self._expected_last_content
        request["messages"].append({"role": "user", "content": self._content})
        return request


class _AssertNoReasoningContentHook(inference.PostprocessResponse):
    def postprocess(self, response: dict, id: str | None = None) -> dict:
        del id
        message = response["choices"][0]["message"]
        assert "reasoning_content" not in message
        message["content"] = f"{message['content']} explicit"
        return response


class _AssertExplicitSuffixHook(inference.PostprocessResponse):
    def postprocess(self, response: dict, id: str | None = None) -> dict:
        del id
        message = response["choices"][0]["message"]
        assert message["content"].endswith(" explicit")
        return response


class _HookedMetric(HooksBase):
    type: ClassVar[MetricType] = MetricType.STRING_CHECK
    _result: MetricResult = PrivateAttr(default_factory=lambda: _make_metric_result(("score", 1.0)))

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return self._result

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]


class _PreparedMetric(BaseModel):
    type: ClassVar[MetricType] = MetricType.STRING_CHECK

    _events: list[tuple[str, str | None]] = PrivateAttr(default_factory=list)
    _resolved_secret: str | None = PrivateAttr(default=None)

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return _make_metric_result(("score", 1.0))

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    def secrets(self) -> dict[str, SecretRef]:
        return {"NVIDIA_BUILD_API_KEY": SecretRef(root="nvidia-build-api-key")}

    async def resolve_secrets(self, secret_resolver) -> None:
        resolved = await secret_resolver("nvidia-build-api-key")
        self._resolved_secret = resolved
        self._events.append(("resolve", resolved))

    async def preflight(self) -> None:
        self._events.append(("preflight", self._resolved_secret))


class TestMetricTypeName:
    def test_returns_metric_type_value_for_str_enum(self, mocker: MockerFixture):
        metric = mocker.Mock()
        metric.type = MetricType.BLEU

        assert metric_type_name(metric) == "bleu"

    def test_returns_string_type_when_present(self, mocker: MockerFixture):
        metric = mocker.Mock()
        metric.type = "my_metric"

        assert metric_type_name(metric) == "my_metric"

    def test_returns_class_name_for_non_string_type(self, mocker: MockerFixture):
        metric_type = mocker.Mock()
        metric_type.value = "my_metric"

        metric = mocker.Mock()
        metric.type = metric_type

        assert metric_type_name(metric) == "Mock"

    def test_returns_class_name_when_type_is_none(self, mocker: MockerFixture):
        metric = mocker.Mock()
        metric.type = None

        assert metric_type_name(metric) == "Mock"

    def test_returns_class_name_when_type_value_is_not_string(self, mocker: MockerFixture):
        metric_type = mocker.Mock()
        metric_type.value = 42

        metric = mocker.Mock()
        metric.type = metric_type

        assert metric_type_name(metric) == "Mock"

    def test_returns_class_name_when_type_has_no_value_attr(self, mocker: MockerFixture):
        metric_type = mocker.Mock(spec=[])
        metric = mocker.Mock()
        metric.type = metric_type

        assert metric_type_name(metric) == "Mock"


class TestRunSync:
    def test_runs_coroutine_and_returns_result(self):
        async def factory():
            return 42

        assert run_sync(factory) == 42

    def test_propagates_exception(self):
        async def factory():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            run_sync(factory)

    def test_propagates_exception_without_no_loop_context(self):
        async def factory():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom") as exc_info:
            run_sync(factory)

        assert exc_info.value.__context__ is None

    def test_works_when_event_loop_is_running(self):
        """When called from inside a running event loop, run_sync uses a thread."""

        async def outer():
            return run_sync(lambda: inner())

        async def inner():
            return "from-thread"

        result = asyncio.run(outer())
        assert result == "from-thread"

    def test_propagates_exception_from_thread(self):
        async def outer():
            return run_sync(lambda: inner())

        async def inner():
            raise RuntimeError("thread-error")

        with pytest.raises(RuntimeError, match="thread-error"):
            asyncio.run(outer())


class TestRowsFromDataset:
    def test_list_passthrough(self):
        rows = [{"a": 1}, {"a": 2}]
        assert rows_from_dataset(rows) is rows

    def test_dataset_rows(self):
        ds = DatasetRows(rows=[{"x": 10}])
        assert rows_from_dataset(ds) == [{"x": 10}]

    def test_pyarrow_table(self):
        table = pa.table({"col": [1, 2, 3]})
        result = rows_from_dataset(table)
        assert result == [{"col": 1}, {"col": 2}, {"col": 3}]

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported dataset type: str"):
            rows_from_dataset("not_a_dataset")

    def test_unsupported_type_int_raises(self):
        with pytest.raises(TypeError, match="Unsupported dataset type: int"):
            rows_from_dataset(123)


class TestNormalizeDataset:
    def test_list_passthrough(self):
        rows = [{"a": 1}]
        assert normalize_dataset(rows, pattern=None) is rows

    def test_dataset_rows_passthrough(self):
        ds = DatasetRows(rows=[{"a": 1}])
        assert normalize_dataset(ds, pattern=None) is ds

    def test_pyarrow_table_passthrough(self):
        table = pa.table({"col": [1]})
        assert normalize_dataset(table, pattern=None) is table

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported dataset type"):
            normalize_dataset(42, pattern=None)

    def test_nonexistent_path_raises(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.jsonl"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            normalize_dataset(missing, pattern=None)

    def test_nonexistent_str_path_raises(self, tmp_path: Path):
        missing = str(tmp_path / "nope.jsonl")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            normalize_dataset(missing, pattern=None)

    def test_glob_path_calls_loader_with_split_base_and_pattern(self, tmp_path: Path, mocker: MockerFixture):
        mock_load = mocker.patch(
            "nemo_evaluator_sdk.datasets.loader.load_dataset_as_dicts",
            return_value=[{"a": 1}],
        )

        result = normalize_dataset(tmp_path / "splits" / "**" / "*.jsonl", pattern=None)

        mock_load.assert_called_once_with(tmp_path / "splits", "**/*.jsonl")
        assert result == [{"a": 1}]

    def test_glob_string_path_calls_loader_with_split_base_and_pattern(self, tmp_path: Path, mocker: MockerFixture):
        mock_load = mocker.patch(
            "nemo_evaluator_sdk.datasets.loader.load_dataset_as_dicts",
            return_value=[{"a": 1}],
        )

        result = normalize_dataset(str(tmp_path / "splits" / "*.jsonl"), pattern=None)

        mock_load.assert_called_once_with(tmp_path / "splits", "*.jsonl")
        assert result == [{"a": 1}]

    def test_existing_exact_path_with_glob_metacharacters_calls_loader_as_file(
        self, tmp_path: Path, mocker: MockerFixture
    ):
        file_path = tmp_path / "eval[1].jsonl"
        file_path.touch()
        mock_load = mocker.patch(
            "nemo_evaluator_sdk.datasets.loader.load_dataset_as_dicts",
            return_value=[{"a": 1}],
        )

        result = normalize_dataset(file_path, pattern=None)

        mock_load.assert_called_once_with(tmp_path, "eval[1].jsonl")
        assert result == [{"a": 1}]

    def test_directory_calls_loader(self, tmp_path: Path, mocker: MockerFixture):
        mock_load = mocker.patch(
            "nemo_evaluator_sdk.datasets.loader.load_dataset_as_dicts",
            return_value=[{"a": 1}],
        )
        result = normalize_dataset(tmp_path, pattern="*.jsonl")
        mock_load.assert_called_once_with(tmp_path, "*.jsonl")
        assert result == [{"a": 1}]

    def test_directory_without_pattern(self, tmp_path: Path, mocker: MockerFixture):
        mock_load = mocker.patch(
            "nemo_evaluator_sdk.datasets.loader.load_dataset_as_dicts",
            return_value=[{"a": 1}],
        )
        result = normalize_dataset(tmp_path, pattern=None)
        mock_load.assert_called_once_with(tmp_path, None)
        assert result == [{"a": 1}]

    def test_file_path_calls_loader(self, tmp_path: Path, mocker: MockerFixture):
        file_path = tmp_path / "data.jsonl"
        file_path.touch()
        mock_load = mocker.patch(
            "nemo_evaluator_sdk.datasets.loader.load_dataset_as_dicts",
            return_value=[{"a": 1}],
        )
        result = normalize_dataset(file_path, pattern=None)
        mock_load.assert_called_once_with(tmp_path, "data.jsonl")
        assert result == [{"a": 1}]

    def test_file_path_with_pattern_raises(self, tmp_path: Path):
        file_path = tmp_path / "data.jsonl"
        file_path.touch()
        with pytest.raises(ValueError, match="pattern can only be used when dataset points to a directory"):
            normalize_dataset(file_path, pattern="*.jsonl")

    def test_str_file_path(self, tmp_path: Path, mocker: MockerFixture):
        file_path = tmp_path / "data.jsonl"
        file_path.touch()
        mock_load = mocker.patch(
            "nemo_evaluator_sdk.datasets.loader.load_dataset_as_dicts",
            return_value=[{"x": 1}],
        )
        result = normalize_dataset(str(file_path), pattern=None)
        mock_load.assert_called_once_with(tmp_path, "data.jsonl")
        assert result == [{"x": 1}]


class TestSplitGlobPath:
    def test_splits_relative_glob_path(self):
        base_path, pattern = split_glob_path(Path("datasets") / "splits" / "**" / "*.jsonl")

        assert base_path == Path("datasets") / "splits"
        assert pattern == "**/*.jsonl"

    def test_splits_absolute_glob_path(self, tmp_path: Path):
        base_path, pattern = split_glob_path(tmp_path / "splits" / "*.jsonl")

        assert base_path == tmp_path / "splits"
        assert pattern == "*.jsonl"

    def test_splits_glob_in_first_relative_segment(self):
        base_path, pattern = split_glob_path(Path("data*") / "eval.jsonl")

        assert base_path == Path(".")
        assert pattern == "data*/eval.jsonl"

    def test_rejects_non_glob_path(self):
        with pytest.raises(ValueError, match="does not contain a glob pattern"):
            split_glob_path(Path("data") / "eval.jsonl")


class TestDiscoverFiles:
    def test_exact_existing_file_takes_precedence_over_glob_metacharacters(self, tmp_path: Path):
        file_path = tmp_path / "eval[1].jsonl"
        file_path.touch()

        assert discover_files(tmp_path, "eval[1].jsonl") == [file_path]

    def test_glob_pattern_discovers_files(self, tmp_path: Path):
        train_path = tmp_path / "splits" / "train.jsonl"
        validation_path = tmp_path / "splits" / "nested" / "validation.jsonl"
        ignored_path = tmp_path / "splits" / "ignored.csv"
        for path in (train_path, validation_path, ignored_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        assert sorted(discover_files(tmp_path / "splits", "**/*.jsonl")) == sorted([train_path, validation_path])

    def test_glob_pattern_returns_files_in_sorted_order(self, tmp_path: Path):
        for name in ("part-c.json", "part-a.json", "part-b.json"):
            (tmp_path / name).touch()

        assert discover_files(tmp_path, "part-*.json") == [
            tmp_path / "part-a.json",
            tmp_path / "part-b.json",
            tmp_path / "part-c.json",
        ]

    def test_directory_without_a_pattern_returns_files_in_sorted_order(self, tmp_path: Path):
        for name in ("c.jsonl", "a.jsonl", "b.jsonl"):
            (tmp_path / name).touch()

        assert discover_files(tmp_path, None) == [
            tmp_path / "a.jsonl",
            tmp_path / "b.jsonl",
            tmp_path / "c.jsonl",
        ]


class TestIsCompletionsEndpoint:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://example.test/v1/completions", True),
            ("https://example.test/v1/chat/completions", False),
            ("https://example.test/v1/completions/", True),
            ("https://example.test/v1/chat/completions/", False),
            ("https://example.test/v1/completions?model=foo", True),
            ("https://example.test/v1/chat/completions?model=foo", False),
            ("https://example.test/v1/completions#frag", True),
            ("https://example.test/other", False),
            ("https://example.test/v1/completions///", True),
        ],
    )
    def test_detection(self, url: str, expected: bool):
        assert _is_completions_endpoint(url) is expected


class TestDefaultOnlineRequestTemplate:
    def test_returns_messages_template_when_row_contains_messages(self):
        result = _default_online_request_template(
            {"messages": [{"role": "user", "content": "hello"}]},
            _make_model(),
        )

        assert result == {"messages": "{{item.messages}}"}

    def test_raises_for_completions_endpoint_without_supported_fields(self):
        model = _make_model(url="https://example.test/v1/completions")

        with pytest.raises(ValueError, match="custom prompt_template"):
            _default_online_request_template({"text": "hello"}, model)

    def test_raises_for_chat_endpoint_without_supported_fields(self):
        with pytest.raises(ValueError, match="provide one of these row fields"):
            _default_online_request_template({"text": "hello"}, _make_model())


class TestResolveOnlinePromptTemplate:
    def test_logs_single_warning_with_inferred_template(self, caplog: pytest.LogCaptureFixture):
        model = _make_model()
        row = {"prompt": "hello"}

        with caplog.at_level(logging.WARNING):
            result = _resolve_online_prompt_template(None, model, row)

        assert result == {"messages": [{"role": "user", "content": "{{item.prompt}}"}]}
        warning_records = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warning_records) == 1
        assert "No prompt_template provided for online evaluation." in warning_records[0].message
        assert '"content": "{{item.prompt}}"' in warning_records[0].message
        assert getattr(warning_records[0], "prompt_template", None) == result


class TestMaybeSetDefaultMaxTokens:
    @pytest.mark.parametrize(
        ("request_payload", "params", "expected_request"),
        [
            ({}, RunConfigOnlineModel(inference=InferenceParams(max_tokens=32)), {}),
            ({"max_completion_tokens": 12}, None, {"max_completion_tokens": 12}),
            ({}, None, {"max_tokens": 4096}),
        ],
    )
    def test_applies_default_only_when_no_max_tokens_are_set(
        self,
        request_payload: dict[str, object],
        params: RunConfigOnlineModel | None,
        expected_request: dict[str, object],
    ):
        _maybe_set_default_max_tokens(request=request_payload, params=params)

        assert request_payload == expected_request

    @pytest.mark.asyncio
    @pytest.mark.parametrize("format", [ModelFormat.NVIDIA_NIM, ModelFormat.OPEN_AI, ModelFormat.LLAMA_STACK])
    async def test_applies_default_through_call_path_for_every_model_format(
        self, format: ModelFormat, mocker: MockerFixture
    ):
        """The cap used to be NIM-only, so flipping the default format would have silently removed it."""
        inference_fn = mocker.AsyncMock(return_value={})
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution._process_online_response",
            return_value=({}, None),
        )

        await generate_online_sample(
            target=_make_model(format=format),
            row={"prompt": "hello"},
            index=0,
            prompt_template={"messages": [{"role": "user", "content": "{{item.prompt}}"}]},
            inference_fn=inference_fn,
        )

        assert inference_fn.await_args.args[1]["max_tokens"] == 4096


class TestTargetStructuredOutputPreflight:
    """Target generation must probe the endpoint, not assume an encoding."""

    @pytest.mark.asyncio
    async def test_target_generation_falls_back_to_nvext_when_endpoint_only_honours_it(self, mocker: MockerFixture):
        """A target that only accepts nvext.guided_json must still get an enforced encoding.

        Before target-side preflight existed this path used the model's `format` label; with the
        label deprecated and ignored, an unprobed target would be sent response_format instead.
        """
        captured: list[dict] = []

        async def fake_inference(model, request, max_retries, **kwargs):
            captured.append(request)
            if "response_format" in request or "guided_json" in request.get("extra_body", {}):
                # Endpoint ignores both of these and answers in prose.
                return {"choices": [{"message": {"content": "sure thing"}}]}
            return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}

        mocker.patch.object(inference, "make_inference_request", side_effect=fake_inference)
        metric = ExactMatchMetric(reference="{{item.expected}}")

        await evaluate_metric(
            metric,
            rows=[{"prompt": "hi", "expected": "sure thing"}],
            target=_make_model(),
            prompt_template={"messages": [{"role": "user", "content": "{{item.prompt}}"}]},
            params=RunConfigOnlineModel(
                parallelism=1,
                structured_output={"schema": {"type": "object", "properties": {"a": {"type": "string"}}}},
            ),
        )

        probes = [r for r in captured if _is_structured_output_probe(r)]
        generation = [r for r in captured if not _is_structured_output_probe(r)]
        assert len(probes) == 3, "expected all three placements to be probed"
        assert generation, "expected a generation request"
        assert generation[0]["extra_body"]["nvext"]["guided_json"] is not None


class TestBenchmarkPathStructuredOutput:
    """The multi-metric LocalBackend path bypasses evaluate_metric and must still probe."""

    @pytest.mark.asyncio
    async def test_multi_metric_benchmark_path_resolves_the_target_encoding(self, mocker: MockerFixture):
        """Evaluator.run() with several metrics goes through evaluate_benchmark, not evaluate_metric.

        Without its own session and probe this path sent whichever encoding new_hooks guessed.
        """
        from nemo_evaluator_sdk.execution.backends.local.backend import LocalBackend

        captured: list[dict] = []

        async def fake_inference(model, request, max_retries, **kwargs):
            captured.append(request)
            if _is_structured_output_probe(request):
                if "response_format" in request or "guided_json" in request.get("extra_body", {}):
                    return {"choices": [{"message": {"content": "prose, not json"}}]}
                return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}
            return {"choices": [{"message": {"content": "sure thing"}}]}

        # Patch ONLY the benchmark module's binding. The probe must travel the same seam that
        # generation does, so patching one place has to be enough; if the probe used a different
        # binding it would escape this patch and attempt a real request.
        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.make_inference_request",
            side_effect=fake_inference,
        )

        await LocalBackend().evaluate_dataset(
            metrics=[ExactMatchMetric(reference="{{item.expected}}"), ExactMatchMetric(reference="{{item.expected}}")],
            dataset=[{"prompt": "hi", "expected": "sure thing"}],
            target=_make_model(),
            prompt_template={"messages": [{"role": "user", "content": "{{item.prompt}}"}]},
            params=RunConfigOnlineModel(
                parallelism=1,
                structured_output={"schema": {"type": "object", "properties": {"a": {"type": "string"}}}},
            ),
        )

        probes = [r for r in captured if _is_structured_output_probe(r)]
        generation = [r for r in captured if not _is_structured_output_probe(r)]
        assert probes, "the benchmark path must probe the target endpoint"
        assert generation, "expected a generation request"
        assert generation[0]["extra_body"]["nvext"]["guided_json"] is not None


class TestGenerateOnlineSample:
    @pytest.mark.asyncio
    async def test_returns_empty_sample_when_response_and_output_text_are_empty(self, mocker: MockerFixture):
        inference_fn = mocker.AsyncMock(return_value={})
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution._process_online_response",
            return_value=({}, None),
        )

        sample = await generate_online_sample(
            target=_make_model(),
            row={"prompt": "hello"},
            index=0,
            prompt_template={"messages": [{"role": "user", "content": "{{item.prompt}}"}]},
            inference_fn=inference_fn,
        )

        assert sample == {}

    @pytest.mark.asyncio
    async def test_copies_agent_trajectory_to_top_level_sample(self, mocker: MockerFixture):
        response = {
            "choices": [{"message": {"content": "done"}}],
            "trajectory": [{"tool": "search", "result": "ok"}],
        }
        inference_fn = mocker.AsyncMock(return_value=response)

        sample = await generate_online_sample(
            target=_make_agent(),
            row={"prompt": "hello"},
            index=0,
            prompt_template={"prompt": "{{item.prompt}}"},
            inference_fn=inference_fn,
        )

        assert sample == {
            "output_text": "done",
            "response": response,
            "trajectory": [{"tool": "search", "result": "ok"}],
        }

    @pytest.mark.asyncio
    async def test_copies_postprocessed_agent_trajectory_to_top_level_sample(self, mocker: MockerFixture):
        response = {
            "choices": [{"message": {"content": "raw"}}],
            "trajectory": [{"tool": "search", "result": "raw"}],
        }
        processed_response = {
            "choices": [{"message": {"content": "done"}}],
            "trajectory": [{"tool": "search", "result": "processed"}],
        }
        hook = mocker.Mock()
        hook.postprocess.return_value = processed_response
        inference_fn = mocker.AsyncMock(return_value=response)

        sample = await generate_online_sample(
            target=_make_agent(),
            row={"prompt": "hello"},
            index=0,
            prompt_template={"prompt": "{{item.prompt}}"},
            postprocess_hooks=[hook],
            inference_fn=inference_fn,
        )

        assert sample == {
            "output_text": "done",
            "response": processed_response,
            "trajectory": [{"tool": "search", "result": "processed"}],
        }
        hook.postprocess.assert_called_once_with(response, id="0")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invocation_output_text", ["raw", None])
    async def test_typed_agent_invocation_uses_postprocessed_output_text(
        self,
        mocker: MockerFixture,
        invocation_output_text: str | None,
    ) -> None:
        raw_response = {"choices": [{"message": {"content": "raw"}}]}
        processed_response = {"choices": [{"message": {"content": "processed"}}]}
        hook = mocker.Mock()
        hook.postprocess.return_value = processed_response
        inference_fn = mocker.AsyncMock(
            return_value=AgentInvocationResult(
                status=AgentInvocationStatus.COMPLETED,
                response=raw_response,
                output_text=invocation_output_text,
            )
        )

        sample = await generate_online_sample(
            target=_make_agent(),
            row={"prompt": "hello"},
            index=0,
            prompt_template={"prompt": "{{item.prompt}}"},
            postprocess_hooks=[hook],
            inference_fn=inference_fn,
        )

        assert sample["output_text"] == "processed"
        assert sample["response"] == processed_response
        assert sample["invocation_status"] == "completed"
        hook.postprocess.assert_called_once_with(raw_response, id="0")


class TestGenerateOnlineSampleAgent:
    @pytest.mark.asyncio
    async def test_defaults_to_typed_invoke_agent(self, mocker: MockerFixture):
        from nemo_evaluator_sdk.agent_inference import invoke_agent

        helper = mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.generate_online_sample",
            new_callable=AsyncMock,
            return_value={"invocation_status": "completed"},
        )

        result = await generate_online_sample_agent(
            agent=_make_agent(),
            row={"prompt": "hello"},
            index=0,
            prompt_template={"prompt": "{{item.prompt}}"},
        )

        assert result == {"invocation_status": "completed"}
        assert helper.await_args is not None
        assert helper.await_args.kwargs["inference_fn"] is invoke_agent

    @pytest.mark.asyncio
    async def test_uses_preconfigured_agent_inference_fn(self, mocker: MockerFixture):
        helper = mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.generate_online_sample",
            new_callable=AsyncMock,
            return_value={},
        )
        inference_fn = mocker.AsyncMock()

        await generate_online_sample_agent(
            agent=_make_agent(),
            row={"prompt": "hello"},
            index=0,
            prompt_template={"prompt": "{{item.prompt}}"},
            agent_inference_fn=inference_fn,
        )

        assert helper.await_args is not None
        assert helper.await_args.kwargs["inference_fn"] is inference_fn

    @pytest.mark.asyncio
    async def test_delegates_to_unified_online_sample_helper(self, mocker: MockerFixture):
        sample = {"output_text": "agent-response"}
        helper = mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.generate_online_sample",
            new_callable=AsyncMock,
            return_value=sample,
        )
        agent_inference_fn = mocker.AsyncMock()

        result = await generate_online_sample_agent(
            agent=_make_agent(),
            row={"prompt": "hello"},
            index=3,
            prompt_template={"prompt": "{{item.prompt}}"},
            params=RunConfigOnline(parallelism=1),
            agent_inference_fn=agent_inference_fn,
            default_headers={"X-Test": "1"},
        )

        assert result == sample
        helper.assert_awaited_once()
        helper_await_args = helper.await_args
        assert helper_await_args is not None
        assert helper_await_args.kwargs["inference_fn"] is agent_inference_fn
        assert helper_await_args.kwargs["default_headers"] == {"X-Test": "1"}


class TestMergeOnlineHooks:
    @pytest.mark.parametrize(
        ("preprocess_hooks", "postprocess_hooks"),
        [
            ([], [object()]),
            ([object()], []),
        ],
    )
    def test_requires_default_log_hooks_from_inference(
        self, mocker: MockerFixture, preprocess_hooks, postprocess_hooks
    ):
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.new_hooks",
            return_value=(preprocess_hooks, postprocess_hooks),
        )

        with pytest.raises(ValueError, match="must return at least the log hook"):
            _merge_online_hooks(
                params=RunConfig(),
                target=None,
                preprocess_hooks=None,
                postprocess_hooks=None,
            )


class TestFinalizeEvaluationResult:
    @pytest.mark.asyncio
    async def test_preserves_input_order_and_filters_none_from_aggregate(self, mocker: MockerFixture):
        metric = _make_mock_metric(mocker)
        success_result = _make_metric_result(("score", 1.0))
        aggregate_result = AggregatedMetricResult(scores=[])
        mock_agg = mocker.patch(
            "nemo_evaluator_sdk.execution.scoring.aggregate_metrics",
            return_value=aggregate_result,
        )

        completed = [
            (
                0,
                success_result,
                RowScore(
                    row_index=0,
                    item={"idx": 0},
                    sample={"value": "a"},
                    metrics={"mock": success_result.outputs},
                    requests=[],
                    metric_errors=None,
                ),
            ),
            (
                1,
                None,
                RowScore(
                    row_index=1,
                    item={"idx": 1},
                    sample={"value": "b"},
                    metrics={},
                    requests=[],
                    metric_errors={"mock": "bad row"},
                ),
            ),
        ]

        result = await finalize_evaluation_result(metric, completed)

        assert [row_score.row_index for row_score in result.row_scores] == [0, 1]
        mock_agg.assert_called_once_with([success_result], metric.output_spec())
        assert result.aggregate_scores is aggregate_result

    @pytest.mark.asyncio
    async def test_applies_corpus_scores(self, mocker: MockerFixture):
        metric = _make_mock_metric(mocker)
        row_result = _make_metric_result(("score", 1.0))
        corpus_result = _make_metric_result(("corpus_score", 0.5))
        metric.compute_corpus_scores = mocker.AsyncMock(return_value=corpus_result)

        aggregate_result = AggregatedMetricResult(scores=[])
        mocker.patch(
            "nemo_evaluator_sdk.execution.scoring.aggregate_metrics",
            return_value=aggregate_result,
        )
        mock_add = mocker.patch("nemo_evaluator_sdk.execution.scoring.add_corpus_scores")

        completed = [
            (
                0,
                row_result,
                RowScore(
                    row_index=0,
                    item={"idx": 0},
                    sample={"value": "a"},
                    metrics={"mock": row_result.outputs},
                    requests=[],
                    metric_errors=None,
                ),
            )
        ]

        await finalize_evaluation_result(metric, completed)

        metric.compute_corpus_scores.assert_awaited_once()
        assert metric.compute_corpus_scores.await_args.kwargs["inputs"] == [
            build_metric_input({"idx": 0}, {"value": "a"}, 0)
        ]
        mock_add.assert_called_once_with(aggregate_result, corpus_result, metric.corpus_output_spec())

    @pytest.mark.asyncio
    async def test_skip_errored_excludes_nan_placeholder_from_aggregate(self, mocker: MockerFixture):
        metric = _make_mock_metric(mocker)
        success_result = _make_metric_result(("score", 1.0))
        nan_result = _make_metric_result(("score", float("nan")))
        mock_agg = mocker.patch(
            "nemo_evaluator_sdk.execution.scoring.aggregate_metrics",
            return_value=AggregatedMetricResult(scores=[]),
        )

        completed = [
            (
                0,
                success_result,
                RowScore(
                    row_index=0,
                    item={},
                    sample={},
                    metrics={},
                    requests=[],
                    metric_errors=None,
                ),
            ),
            (
                1,
                nan_result,
                RowScore(
                    row_index=1,
                    item={},
                    sample={},
                    metrics={},
                    requests=[],
                    metric_errors={"mock": "ignored failure"},
                ),
            ),
        ]

        result = await finalize_evaluation_result(metric, completed, skip_errored=True)

        mock_agg.assert_called_once_with([success_result], metric.output_spec())
        assert [rs.row_index for rs in result.row_scores] == [0, 1]

    @pytest.mark.asyncio
    async def test_skip_errored_excludes_errored_rows_from_corpus_inputs(self, mocker: MockerFixture):
        """Corpus-level metrics must only receive non-errored rows' item/sample when skip_errored=True.

        Regression for CodeRabbit #34: errored rows carry empty/bogus samples, so
        passing them into ``compute_corpus_scores`` skews corpus-aware aggregation
        (e.g. BLEU-corpus, ROUGE-corpus).
        """
        metric = _make_mock_metric(mocker)
        success_result = _make_metric_result(("score", 1.0))
        nan_result = _make_metric_result(("score", float("nan")))
        corpus_result = _make_metric_result(("corpus_score", 0.5))
        metric.compute_corpus_scores = mocker.AsyncMock(return_value=corpus_result)

        mocker.patch(
            "nemo_evaluator_sdk.execution.scoring.aggregate_metrics",
            return_value=AggregatedMetricResult(scores=[]),
        )
        mocker.patch("nemo_evaluator_sdk.execution.scoring.add_corpus_scores")

        completed = [
            (
                0,
                success_result,
                RowScore(
                    row_index=0,
                    item={"idx": 0},
                    sample={"value": "a"},
                    metrics={"mock": success_result.outputs},
                    requests=[],
                    metric_errors=None,
                ),
            ),
            (
                1,
                nan_result,
                RowScore(
                    row_index=1,
                    item={"idx": 1},
                    sample={"value": "b"},
                    metrics={"mock": nan_result.outputs},
                    requests=[],
                    metric_errors={"mock": "ignored failure"},
                ),
            ),
        ]

        result = await finalize_evaluation_result(metric, completed, skip_errored=True)

        # Only row 0 feeds the corpus metric; row 1 is the errored/skipped row.
        metric.compute_corpus_scores.assert_awaited_once()
        assert metric.compute_corpus_scores.await_args.kwargs["inputs"] == [
            build_metric_input({"idx": 0}, {"value": "a"}, 0)
        ]
        # Both rows still surface in ``row_scores`` for reporting.
        assert [rs.row_index for rs in result.row_scores] == [0, 1]

    @pytest.mark.asyncio
    async def test_skip_errored_bypasses_corpus_scoring_when_all_rows_are_skipped(self, mocker: MockerFixture):
        metric = _make_mock_metric(mocker)
        nan_result = _make_metric_result(("score", float("nan")))
        mock_agg = mocker.patch(
            "nemo_evaluator_sdk.execution.scoring.aggregate_metrics",
            return_value=AggregatedMetricResult(scores=[]),
        )
        mock_add = mocker.patch("nemo_evaluator_sdk.execution.scoring.add_corpus_scores")

        completed = [
            (
                0,
                nan_result,
                RowScore(
                    row_index=0,
                    item={"idx": 0},
                    sample={"value": "a"},
                    metrics={"mock": nan_result.outputs},
                    requests=[],
                    metric_errors={"mock": "ignored failure"},
                ),
            )
        ]

        result = await finalize_evaluation_result(metric, completed, skip_errored=True)

        mock_agg.assert_called_once_with([], metric.output_spec())
        metric.compute_corpus_scores.assert_not_awaited()
        mock_add.assert_not_called()
        assert [rs.row_index for rs in result.row_scores] == [0]


class TestEvaluateMetric:
    @pytest.mark.asyncio
    async def test_returns_empty_result_when_rows_are_empty(
        self,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.WARNING, logger="nemo_evaluator_sdk.execution.metric_execution")
        metric = _make_mock_metric(mocker)

        result = await evaluate_metric(
            metric=metric,
            rows=[],
        )

        assert result == empty_evaluation_result()
        assert "No rows found in dataset, returning empty evaluation result" in caplog.text

    @pytest.mark.asyncio
    async def test_skips_prompt_inference_for_nonempty_offline_evaluation(self, mocker: MockerFixture):
        metric = _make_mock_metric(mocker)
        params = RunConfig(parallelism=2)
        completed = [
            (
                0,
                _make_metric_result(("score", 1.0)),
                RowScore(
                    row_index=0,
                    item={"input": "row"},
                    sample={},
                    metrics={},
                    requests=[],
                    metric_errors=None,
                ),
            )
        ]
        expected = EvaluationResult(row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]))
        mock_merge_hooks = mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution._merge_online_hooks",
            return_value=([], []),
        )
        mock_run_pipeline = mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.run_generated_sample_scoring_pipeline",
            new_callable=AsyncMock,
            return_value=completed,
        )
        mock_finalize = mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.finalize_evaluation_result",
            new_callable=AsyncMock,
            return_value=expected,
        )
        mock_default_template = mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution._default_online_request_template",
        )

        result = await evaluate_metric(
            metric=metric,
            rows=[{"input": "row"}],
            params=params,
        )

        assert result is expected
        mock_merge_hooks.assert_called_once_with(
            params=params,
            target=None,
            preprocess_hooks=None,
            postprocess_hooks=None,
        )
        assert mock_run_pipeline.await_args is not None
        pipeline = mock_run_pipeline.await_args.args[0]
        assert pipeline.target is None
        assert pipeline.prompt_template is None
        mock_default_template.assert_not_called()
        mock_finalize.assert_awaited_once_with(metric, completed)

    @pytest.mark.asyncio
    async def test_reraises_untyped_pipeline_failures(self, mocker: MockerFixture):
        metric = _make_mock_metric(mocker)
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.run_generated_sample_scoring_pipeline",
            new_callable=AsyncMock,
            side_effect=RuntimeError("pipeline broke"),
        )

        with pytest.raises(RuntimeError, match="pipeline broke"):
            await evaluate_metric(
                metric=metric,
                rows=[{"input": "row"}],
                params=RunConfig(parallelism=1),
            )

    @pytest.mark.asyncio
    async def test_builds_agent_pipeline_when_prompt_template_is_provided(self, mocker: MockerFixture):
        metric = _make_mock_metric(mocker)
        agent = _make_agent()
        expected = EvaluationResult(row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]))
        mock_run_pipeline = mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.run_generated_sample_scoring_pipeline",
            new_callable=AsyncMock,
            return_value=[],
        )
        mock_finalize = mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.finalize_evaluation_result",
            new_callable=AsyncMock,
            return_value=expected,
        )

        result = await evaluate_metric(
            metric=metric,
            rows=[{"prompt": "hello"}],
            target=agent,
            prompt_template={"prompt": "{{item.prompt}}"},
            params=RunConfigOnline(parallelism=1),
        )

        assert result is expected
        assert mock_run_pipeline.await_args is not None
        pipeline = mock_run_pipeline.await_args.args[0]
        assert pipeline.target is agent
        assert pipeline.prompt_template == {"prompt": "{{item.prompt}}"}
        mock_finalize.assert_awaited_once_with(metric, [])


class TestComputeMetricPipeline:
    @pytest.mark.asyncio
    async def test_generate_sample_runs_postprocess_hooks_for_offline_rows(self, mocker: MockerFixture):
        hook = mocker.Mock()
        hook.postprocess.return_value = {"offline": True}
        pipeline = ComputeMetricPipeline(
            rows=[{"input": "row"}],
            parallelism=1,
            metric=_make_mock_metric(mocker),
            target=None,
            metric_key="mock",
            params=RunConfig(),
            postprocess_hooks=[hook],
        )

        sample = await pipeline.generate_sample(0, {"input": "row"})

        assert sample == {"offline": True}
        hook.postprocess.assert_called_once_with({}, id="0")

    @pytest.mark.asyncio
    async def test_generate_sample_requires_prompt_template_for_online_model(self, mocker: MockerFixture):
        # Deliberately bypass the overload contract to exercise the runtime guard:
        # a Model target without prompt_template/inference_fn.
        pipeline = ComputeMetricPipeline(
            rows=[{"input": "row"}],
            parallelism=1,
            metric=_make_mock_metric(mocker),
            target=cast(Any, _make_model()),
            metric_key="mock",
            params=RunConfigOnlineModel(),
        )

        with pytest.raises(ValueError, match="prompt_template is required for service online evaluation"):
            await pipeline.generate_sample(0, {"input": "row"})

    @pytest.mark.asyncio
    async def test_generate_sample_raises_when_agent_pipeline_lacks_inference_fn(
        self,
        mocker: MockerFixture,
    ):
        # Deliberately bypass the overload contract to exercise the runtime guard.
        pipeline = ComputeMetricPipeline(
            rows=[{"input": "row"}],
            parallelism=1,
            metric=_make_mock_metric(mocker),
            target=_make_agent(),
            metric_key="mock",
            prompt_template={"messages": [{"role": "user", "content": "{{item.input}}"}]},
            inference_fn=cast(Any, None),
            params=RunConfigOnline(),
        )

        with pytest.raises(TypeError, match="expected AgentInferenceFn for Agent target"):
            await pipeline.generate_sample(0, {"input": "row"})

    @pytest.mark.asyncio
    async def test_generate_sample_forwards_default_headers_for_agent_target(
        self,
        mocker: MockerFixture,
    ):
        captured_headers = None

        async def _fake_agent_inference(
            agent: Agent,
            request: dict,
            max_retries: int | None,
            default_headers: dict[str, str] | None = None,
            **kwargs,
        ) -> dict:
            nonlocal captured_headers
            captured_headers = default_headers
            return {"choices": [{"message": {"content": "ok"}}]}

        pipeline = ComputeMetricPipeline(
            rows=[{"input": "row"}],
            parallelism=1,
            metric=_make_mock_metric(mocker),
            target=_make_agent(),
            metric_key="mock",
            prompt_template={"messages": [{"role": "user", "content": "{{item.input}}"}]},
            inference_fn=_fake_agent_inference,
            params=RunConfigOnline(ignore_request_failure=True),
            default_headers={"X-NHX-Principal-Id": "service:evaluator"},
        )

        await pipeline.generate_sample(0, {"input": "row"})

        assert captured_headers == {"X-NHX-Principal-Id": "service:evaluator"}

    @pytest.mark.parametrize(
        ("row", "expected_message"),
        [
            (
                {"messages": [{"role": "user", "content": ""}]},
                "Row 7 has empty message content and failed inference: bad input.",
            ),
            (
                {"prompt": ""},
                "Row 7 has empty prompt and failed inference: bad input.",
            ),
        ],
    )
    def test_handle_generation_error_identifies_empty_input_in_strict_mode(
        self,
        mocker: MockerFixture,
        row: dict[str, object],
        expected_message: str,
    ):
        pipeline = ComputeMetricPipeline(
            rows=[row],
            parallelism=1,
            metric=_make_mock_metric(mocker),
            target=None,
            metric_key="mock",
            params=RunConfig(),
        )

        with pytest.raises(EvaluationError, match=expected_message) as exc_info:
            pipeline.handle_generation_error(7, row, RuntimeError("bad input"), [])

        assert exc_info.value.phase is EvaluationPhase.SAMPLE_GENERATION
        assert exc_info.value.metric_key == "mock"


class TestScorePipelineSamples:
    @pytest.mark.asyncio
    async def test_raises_for_unexpected_queue_item(self, mocker: MockerFixture):
        runtime = PipelineRuntime(
            pipeline=mocker.Mock(),
            sample_queue=asyncio.Queue(),
            results=[],
        )
        await runtime.sample_queue.put(object())

        with pytest.raises(ValueError, match="Expected GeneratedSampleEvent, got: object"):
            await _score_pipeline_samples(runtime)


class TestRunGeneratedSampleScoringPipeline:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_pipeline_has_no_rows(self, mocker: MockerFixture):
        pipeline = mocker.Mock()
        pipeline.rows = []
        pipeline.parallelism = 4

        assert await run_generated_sample_scoring_pipeline(pipeline) == []

    @pytest.mark.asyncio
    async def test_raises_when_pipeline_finishes_with_missing_completed_rows(self, mocker: MockerFixture):
        @asynccontextmanager
        async def _noop_resilience_session():
            yield

        pipeline = mocker.Mock()
        pipeline.rows = [{"input": "row"}]
        pipeline.parallelism = 1
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.use_resilience_session",
            side_effect=_noop_resilience_session,
        )
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.run_indexed_tasks",
            new_callable=AsyncMock,
            return_value=None,
        )

        with pytest.raises(RuntimeError, match="missing row evaluation result after online execution"):
            await run_generated_sample_scoring_pipeline(pipeline)


class TestSecretResolution:
    def test_candidate_env_names_include_normalized_secret_name(self):
        assert _candidate_env_names("nvidia-build-api-key") == [
            "nvidia-build-api-key",
            "NVIDIA-BUILD-API-KEY",
            "nvidia_build_api_key",
            "NVIDIA_BUILD_API_KEY",
        ]

    def test_candidate_env_names_add_prefixed_variants_for_digit_prefix(self):
        assert _candidate_env_names("123-secret/key") == [
            "123-secret/key",
            "123-SECRET/KEY",
            "123_secret_key",
            "123_SECRET_KEY",
            "_123_secret_key",
            "_123_SECRET_KEY",
        ]

    @pytest.mark.asyncio
    async def test_resolve_secret_from_env_supports_prefixed_digit_variant(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("_123_secret_key", "secret-value")

        assert await LocalSecretResolver().resolve_secret(SecretRef(root="123-secret/key")) == "secret-value"

    @pytest.mark.asyncio
    async def test_resolve_secret_from_env_returns_none_when_missing(self):
        assert await LocalSecretResolver().resolve_secret(SecretRef(root="missing-secret")) is None


class TestEvaluateMetricOnline:
    @pytest.mark.asyncio
    async def test_success_combines_generation_and_metric_requests(self, mocker: MockerFixture):
        metric = _make_test_metric()
        model = _make_model()
        execution_logger = mocker.patch("nemo_evaluator_sdk.execution.metric_execution.log")
        generation_request = {"request": {"messages": [{"role": "user", "content": "hello"}]}, "response": {"id": "r1"}}
        metric_request = {"request": {"metric": "score"}, "response": {"value": 1.0}}

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            inference.requests_log_var.get([]).append(generation_request)
            return {"choices": [{"message": {"content": "world"}}]}

        async def _compute_scores(input: MetricInput) -> MetricResult:
            assert input.row.data == {"prompt": "hello"}
            assert input.candidate.as_sample() == {
                "output_text": "world",
                "response": {"choices": [{"message": {"content": "world"}}]},
            }
            inference.requests_log_var.get([]).append(metric_request)
            return _make_metric_result(("score", 1.0))

        metric.compute_scores = mocker.AsyncMock(side_effect=_compute_scores)
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )
        result = await evaluate_metric(
            metric,
            rows=[{"prompt": "hello"}],
            target=model,
            params=RunConfigOnlineModel(parallelism=1),
        )

        row_score = result.row_scores[0]
        assert row_score.sample == {
            "output_text": "world",
            "response": {"choices": [{"message": {"content": "world"}}]},
        }
        assert row_score.requests == [generation_request, metric_request]
        assert row_score.error is None
        assert row_score.metric_errors is None
        execution_logger.debug.assert_called()
        _, debug_kwargs = execution_logger.debug.call_args
        assert debug_kwargs["extra"]["item_index"] == 0
        assert debug_kwargs["extra"]["metric_type"] == MetricType.STRING_CHECK.value
        assert debug_kwargs["extra"]["outputs"][0]["name"] == "score"
        assert debug_kwargs["extra"]["outputs"][0]["value"] == 1.0

    @pytest.mark.asyncio
    async def test_metric_failure_preserves_generation_and_metric_requests(self, mocker: MockerFixture):
        metric = _make_distinct_score_name_metric()
        model = _make_model()
        generation_request = {"request": {"messages": [{"role": "user", "content": "hello"}]}, "response": {"id": "r1"}}
        metric_request = {"request": {"metric": "score"}, "error": "bad score"}

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            inference.requests_log_var.get([]).append(generation_request)
            return {"choices": [{"message": {"content": "world"}}]}

        async def _compute_scores(input: MetricInput) -> MetricResult:
            del input
            inference.requests_log_var.get([]).append(metric_request)
            raise ValueError("bad score")

        metric.compute_scores = mocker.AsyncMock(side_effect=_compute_scores)
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )
        result = await evaluate_metric(
            metric,
            rows=[{"prompt": "hello"}],
            target=model,
            params=RunConfigOnlineModel(parallelism=1, ignore_request_failure=True),
        )

        row_score = result.row_scores[0]
        assert row_score.requests == [generation_request, metric_request]
        assert row_score.error == "metric-a: bad score"
        assert row_score.metric_errors == {"metric-a": "bad score"}
        assert row_score.metrics["metric-a"][0].name == "score"
        assert math.isnan(row_score.metrics["metric-a"][0].value)
        assert result.aggregate_scores.scores[0].name == "score"
        assert result.aggregate_scores.scores[0].count == 0
        assert result.aggregate_scores.scores[0].nan_count == 1

    @pytest.mark.asyncio
    async def test_generation_failure_preserves_partial_generation_requests(self, mocker: MockerFixture):
        metric = _make_test_metric()
        model = _make_model()
        execution_logger = mocker.patch("nemo_evaluator_sdk.execution.metric_execution.log")
        generation_request = {"request": {"messages": [{"role": "user", "content": "hello"}]}, "error": "gen bad"}

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            inference.requests_log_var.get([]).append(generation_request)
            raise RuntimeError("gen bad")

        metric.compute_scores = mocker.AsyncMock()
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )
        result = await evaluate_metric(
            metric,
            rows=[{"prompt": "hello"}],
            target=model,
            params=RunConfigOnlineModel(parallelism=1, ignore_request_failure=True),
        )

        row_score = result.row_scores[0]
        assert row_score.requests == [generation_request]
        assert row_score.sample == {
            "output_text": None,
            "response": {},
            "inference_error": GENERATION_FAILURE_MESSAGE,
        }
        assert row_score.error == f"{TEST_METRIC_KEY}: {GENERATION_FAILURE_MESSAGE}"
        assert row_score.metric_errors == {TEST_METRIC_KEY: GENERATION_FAILURE_MESSAGE}
        assert row_score.metrics[TEST_METRIC_KEY][0].name == "score"
        assert math.isnan(row_score.metrics[TEST_METRIC_KEY][0].value)
        assert result.aggregate_scores.scores[0].count == 0
        assert result.aggregate_scores.scores[0].nan_count == 1
        execution_logger.warning.assert_any_call(
            "Inference failed, marking as NaN",
            extra={
                "item_index": 0,
                "error": GENERATION_FAILURE_MESSAGE,
            },
        )
        metric.compute_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_template_generation_failure_maps_to_nan_with_ignore_request_failure(self, mocker: MockerFixture):
        metric = _make_test_metric()
        model = _make_model()
        mock_inference = mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
        )
        metric.compute_scores = mocker.AsyncMock()

        result = await evaluate_metric(
            metric,
            rows=[{"prompt": "hello"}],
            target=model,
            params=RunConfigOnlineModel(parallelism=1, ignore_request_failure=True),
            prompt_template={"messages": [{"role": "user", "content": "{{item.missing}}"}]},
        )

        row_score = result.row_scores[0]
        assert row_score.metrics[TEST_METRIC_KEY][0].name == "score"
        assert math.isnan(row_score.metrics[TEST_METRIC_KEY][0].value)
        assert row_score.metric_errors is not None
        error_message = row_score.metric_errors[TEST_METRIC_KEY]
        assert "'dict object' has no attribute 'missing'" in error_message
        assert "Row 0 failed inference: 'dict object' has no attribute 'missing'." in error_message
        assert result.aggregate_scores.scores[0].count == 0
        assert result.aggregate_scores.scores[0].nan_count == 1
        mock_inference.assert_not_awaited()
        metric.compute_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generation_failure_raises_runtime_error_with_strict_mode(self, mocker: MockerFixture):
        metric = _make_test_metric()
        model = _make_model()

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            inference.requests_log_var.get([]).append({"request": request, "error": "gen bad"})
            raise RuntimeError("gen bad")

        metric.compute_scores = mocker.AsyncMock()
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )

        with pytest.raises(EvaluationError, match="sample generation") as exc_info:
            await evaluate_metric(
                metric,
                rows=[{"prompt": "hello"}],
                target=model,
                params=RunConfigOnlineModel(parallelism=1),
            )

        metric.compute_scores.assert_not_awaited()
        assert exc_info.value.index == 0
        assert exc_info.value.phase is EvaluationPhase.SAMPLE_GENERATION
        assert exc_info.value.metric_key == "string-check"
        assert exc_info.value.message == GENERATION_FAILURE_MESSAGE
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_generation_failure_maps_to_nan_with_ignore_request_failure(self, mocker: MockerFixture):
        metric = _make_distinct_score_name_metric()
        model = _make_model()

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            inference.requests_log_var.get([]).append({"request": request, "error": "gen bad"})
            raise RuntimeError("gen bad")

        metric.compute_scores = mocker.AsyncMock()
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )

        result = await evaluate_metric(
            metric,
            rows=[{"prompt": "hello"}],
            target=model,
            params=RunConfigOnlineModel(parallelism=1, ignore_request_failure=True),
        )

        row_score = result.row_scores[0]
        assert row_score.metrics["metric-a"][0].name == "score"
        assert math.isnan(row_score.metrics["metric-a"][0].value)
        assert row_score.metric_errors == {"metric-a": GENERATION_FAILURE_MESSAGE}
        assert result.aggregate_scores.scores[0].name == "score"
        assert result.aggregate_scores.scores[0].count == 0
        assert result.aggregate_scores.scores[0].nan_count == 1
        metric.compute_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_metric_failure_raises_evaluation_error_with_strict_mode(self, mocker: MockerFixture):
        metric = _make_test_metric()
        model = _make_model()

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            return {"choices": [{"message": {"content": "world"}}]}

        metric.compute_scores = mocker.AsyncMock(side_effect=ValueError("bad score"))
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )

        with pytest.raises(EvaluationError, match="bad score") as exc_info:
            await evaluate_metric(
                metric,
                rows=[{"prompt": "hello"}],
                target=model,
                params=RunConfigOnlineModel(parallelism=1),
            )

        assert exc_info.value.index == 0
        assert exc_info.value.phase is EvaluationPhase.METRIC_SCORING
        assert exc_info.value.metric_key == "string-check"
        assert exc_info.value.message == "bad score"
        assert isinstance(exc_info.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_empty_output_text_is_omitted_from_sdk_sample(self, mocker: MockerFixture):
        metric = _make_test_metric()
        model = _make_model()

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            return {"choices": [{"message": {"content": ""}}]}

        async def _compute_scores(input: MetricInput) -> MetricResult:
            assert input.row.data == {"prompt": "hello"}
            assert input.candidate.as_sample() == {"response": {"choices": [{"message": {"content": ""}}]}}
            return _make_metric_result(("score", 1.0))

        metric.compute_scores = mocker.AsyncMock(side_effect=_compute_scores)
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )

        await evaluate_metric(
            metric,
            rows=[{"prompt": "hello"}],
            target=model,
            params=RunConfigOnlineModel(parallelism=1),
        )

    @pytest.mark.asyncio
    async def test_explicit_prompt_template_and_params_drive_online_request(self, mocker: MockerFixture):
        metric = _make_test_metric()
        model = _make_model()
        captured: dict[str, Any] = {}

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            default_headers: dict[str, str] | None = None,
            timeout: float | None = None,
            **kwargs,
        ) -> dict:
            captured["request"] = request
            captured["max_retries"] = max_retries
            captured["default_headers"] = default_headers
            captured["timeout"] = timeout
            return {"choices": [{"message": {"content": "done"}}]}

        metric.compute_scores = mocker.AsyncMock(return_value=_make_metric_result(("score", 1.0)))
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )

        await evaluate_metric(
            metric,
            rows=[{"prompt": "ignored", "question": "What is 2 + 2?"}],
            target=model,
            prompt_template={"messages": [{"role": "user", "content": "{{item.question}}"}]},
            params=RunConfigOnlineModel(
                parallelism=1,
                inference=InferenceParams(max_completion_tokens=12, temperature=0.25),
                max_retries=7,
                request_timeout=19,
            ),
        )

        assert captured["request"] == {
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "temperature": 0.25,
            "max_completion_tokens": 12,
        }
        assert "max_tokens" not in captured["request"]
        assert captured["max_retries"] == 7
        assert captured["default_headers"] is None
        assert captured["timeout"] == 19

    @pytest.mark.asyncio
    async def test_agent_evaluation_requires_prompt_template(self, mocker: MockerFixture):
        metric = _make_test_metric()
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.run_generated_sample_scoring_pipeline",
            new_callable=AsyncMock,
        )

        with pytest.raises(ValueError, match="prompt_template is required for agent online evaluation"):
            await evaluate_metric(
                metric,
                rows=[{"prompt": "hello"}],
                target=_make_agent(),
                params=RunConfigOnline(parallelism=1),
            )

    @pytest.mark.asyncio
    async def test_question_field_is_inferred_for_chat_requests(self, mocker: MockerFixture):
        metric = _make_test_metric()
        model = _make_model()
        captured: dict[str, object] = {}

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            captured["request"] = request
            return {"choices": [{"message": {"content": "4"}}]}

        metric.compute_scores = mocker.AsyncMock(return_value=_make_metric_result(("score", 1.0)))
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )

        await evaluate_metric(
            metric,
            rows=[{"question": "What is 2 + 2?"}],
            target=model,
            params=RunConfigOnlineModel(parallelism=1),
        )

        assert captured["request"] == {
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "max_tokens": 4096,
        }

    @pytest.mark.asyncio
    async def test_query_field_is_inferred_for_completions_requests(self, mocker: MockerFixture):
        metric = _make_test_metric()
        model = _make_model(url="https://example.test/v1/completions")
        captured: dict[str, object] = {}

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            captured["request"] = request
            return {"choices": [{"text": "Paris"}]}

        metric.compute_scores = mocker.AsyncMock(return_value=_make_metric_result(("score", 1.0)))
        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )

        await evaluate_metric(
            metric,
            rows=[{"query": "What is the capital of France?"}],
            target=model,
            params=RunConfigOnlineModel(parallelism=1),
        )

        assert captured["request"] == {
            "prompt": "What is the capital of France?",
            "max_tokens": 4096,
        }

    @pytest.mark.asyncio
    async def test_metric_hooks_do_not_leak_into_sample_generation(self, mocker: MockerFixture):
        metric = _HookedMetric().with_hooks(
            preprocess=[_AppendMessageHook("metric", expected_last_content="explicit")],
            postprocess=[_AssertExplicitSuffixHook()],
        )
        model = _make_model()
        captured_requests: list[dict] = []

        async def _fake_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            captured_requests.append(request)
            return {"choices": [{"message": {"role": "assistant", "content": "reason</think>answer"}}]}

        mocker.patch(
            "nemo_evaluator_sdk.execution.metric_execution.inference.make_inference_request",
            new_callable=AsyncMock,
            side_effect=_fake_inference,
        )

        result = await evaluate_metric(
            metric,
            rows=[{"prompt": "hello"}],
            target=model,
            prompt_template={"messages": [{"role": "user", "content": "{{item.prompt}}"}]},
            params=RunConfigOnlineModel(
                parallelism=1,
                system_prompt="SYS",
                reasoning=ReasoningParams(end_token="</think>"),
            ),
            preprocess_hooks=[_AppendMessageHook("explicit", expected_last_content="hello")],
            postprocess_hooks=[_AssertNoReasoningContentHook()],
        )

        assert captured_requests[0]["messages"] == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "explicit"},
        ]
        assert result.row_scores[0].sample["output_text"] == "answer explicit"
        assert result.row_scores[0].sample["response"]["choices"][0]["message"]["reasoning_content"] == "reason</think>"

    @pytest.mark.asyncio
    async def test_llm_judge_metric_hooks_apply_only_to_judge_scoring(self, mocker: MockerFixture):
        candidate_model = _make_model()
        judge_model = Model(
            url="https://judge.example.test/v1/chat/completions", name="judge-model", format=ModelFormat.OPEN_AI
        )
        metric = RuntimeLLMJudgeMetric(
            model=judge_model,
            scores=[
                RangeScore(
                    name="helpfulness",
                    minimum=1,
                    maximum=5,
                    parser=JSONScoreParser(json_path="helpfulness"),
                )
            ],
            prompt_template={
                "messages": [
                    {"role": "system", "content": "Rate helpfulness."},
                    {
                        "role": "user",
                        "content": "Question: {{item.prompt}}\n\nAssistant response: {{sample.output_text}}",
                    },
                ]
            },
        )
        captured_generation_requests: list[dict] = []
        captured_judge_requests: list[dict] = []

        async def fake_generation_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            captured_generation_requests.append(request)
            return {"choices": [{"message": {"role": "assistant", "content": "Paris"}}]}

        async def fake_judge_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            captured_judge_requests.append(request)
            if _is_structured_output_probe(request):
                # Behave like an endpoint that honours response_format, so preflight resolves to it.
                return {"choices": [{"message": {"content": '{"__nhx_probe_score": 1}'}}]}
            return {"choices": [{"message": {"role": "assistant", "content": '{"helpfulness": 4}'}}]}

        metric.set_inference_fn(fake_judge_inference)
        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.make_inference_request",
            new_callable=AsyncMock,
            side_effect=fake_generation_inference,
        )

        result = await LocalBackend().evaluate_dataset(
            metrics=[metric],
            dataset=[{"prompt": "What is the capital of France?"}],
            target=candidate_model,
            prompt_template={"messages": [{"role": "user", "content": "{{item.prompt}}"}]},
            params=RunConfigOnlineModel(parallelism=1),
        )

        assert captured_generation_requests == [
            {"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 4096}
        ]
        assert "response_format" not in captured_generation_requests[0]
        # Preflight probes the endpoint before scoring, so the judge sees a probe request first.
        probe_requests = [r for r in captured_judge_requests if _is_structured_output_probe(r)]
        scoring_requests = [r for r in captured_judge_requests if not _is_structured_output_probe(r)]
        assert len(probe_requests) == 1
        assert scoring_requests[0]["response_format"]["type"] == "json_schema"
        assert "Assistant response: Paris" in scoring_requests[0]["messages"][-1]["content"]
        assert result.row_scores[0].sample["output_text"] == "Paris"
        assert result.row_scores[0].metrics["llm-judge"][0].value == 4.0

    @pytest.mark.asyncio
    async def test_nim_llm_judge_metric_runs_preflight_before_local_evaluation(
        self,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        candidate_model = _make_model()
        judge_model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            format=ModelFormat.NVIDIA_NIM,
            api_key_secret=SecretRef(root="nvidia-build-api-key"),
        )
        metric = RuntimeLLMJudgeMetric(
            model=judge_model,
            scores=[
                RangeScore(
                    name="helpfulness",
                    minimum=1,
                    maximum=5,
                    parser=JSONScoreParser(json_path="helpfulness"),
                )
            ],
            prompt_template={
                "messages": [
                    {"role": "system", "content": "Rate helpfulness."},
                    {
                        "role": "user",
                        "content": "Question: {{item.prompt}}\n\nAssistant response: {{sample.output_text}}",
                    },
                ]
            },
        )
        monkeypatch.setenv("nvidia_build_api_key", "secret-value")
        detect_mode = mocker.patch(
            "nemo_evaluator_sdk.metrics.llm_judge.detect_structured_output_mode",
            new_callable=AsyncMock,
            return_value=StructuredOutputMode.ROOT_GUIDED_JSON,
        )
        captured_generation_requests: list[dict] = []
        captured_judge_requests: list[dict] = []

        async def fake_generation_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            **kwargs,
        ) -> dict:
            captured_generation_requests.append(request)
            return {"choices": [{"message": {"role": "assistant", "content": "Paris"}}]}

        async def fake_judge_inference(
            model: Model,
            request: dict,
            max_retries: int | None,
            client: AsyncOpenAI | None = None,
            api_key: str | None = None,
            **kwargs,
        ) -> dict:
            captured_judge_requests.append(
                {"request": request, "api_key": api_key or client.api_key if client else None}
            )
            return {"choices": [{"message": {"role": "assistant", "content": '{"helpfulness": 4}'}}]}

        metric.set_inference_fn(fake_judge_inference)
        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.make_inference_request",
            new_callable=AsyncMock,
            side_effect=fake_generation_inference,
        )

        result = await LocalBackend().evaluate_dataset(
            metrics=[metric],
            dataset=[{"prompt": "What is the capital of France?"}],
            target=candidate_model,
            prompt_template={"messages": [{"role": "user", "content": "{{item.prompt}}"}]},
            params=RunConfigOnlineModel(parallelism=1),
        )

        detect_mode.assert_awaited_once()
        assert detect_mode.await_args is not None
        detect_kwargs = detect_mode.await_args.kwargs
        # Detection is endpoint-driven; the deprecated model format is deliberately not passed.
        assert "format" not in detect_kwargs
        assert detect_kwargs["api_key"] == "secret-value"
        assert captured_generation_requests == [
            {"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 4096}
        ]
        assert captured_judge_requests[0]["api_key"] == "secret-value"
        assert captured_judge_requests[0]["request"]["extra_body"]["guided_json"]["type"] == "object"
        assert "response_format" not in captured_judge_requests[0]["request"]
        assert result.row_scores[0].metrics["llm-judge"][0].value == 4.0

    def test_apply_postprocess_hooks_runs_in_order(self):
        metric = _HookedMetric().with_hooks(
            postprocess=[_AssertNoReasoningContentHook(), _AssertExplicitSuffixHook()],
        )

        response = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

        assert metric._apply_postprocess_hooks(response, id="row-1") == {
            "choices": [{"message": {"role": "assistant", "content": "done explicit"}}]
        }
