# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_evaluator_sdk.execution.benchmark_execution."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter
from typing import Any

import pytest
from nemo_evaluator_sdk import inference
from nemo_evaluator_sdk.enums import AgentFormat
from nemo_evaluator_sdk.execution.benchmark_execution import (
    ProgressReporter,
    _benchmark_error_from_exception,
    _build_metric_pipelines,
    _metric_worker,
    _MetricPipeline,
    _normalize_metric_result,
    _put_pipeline_sentinels,
    evaluate_benchmark,
)
from nemo_evaluator_sdk.execution.values import EvaluationError, EvaluationPhase
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.values import Agent, GenericAgent
from nemo_evaluator_sdk.values.models import Model
from nemo_evaluator_sdk.values.params import RunConfig, RunConfigOnlineModel
from nemo_evaluator_sdk.values.results import RowScore
from pytest_mock import MockerFixture


class _ScriptedMetric:
    """Metric stub returning a configurable score per (item, sample) call.

    ``score_fn`` receives the item and sample dicts and returns a float; this
    lets tests assert row-identity alignment without threading shared state.
    """

    def __init__(self, name: str, score_fn):
        """Configure the public metric type name and per-row scoring callable."""
        self._name = name
        self._score_fn = score_fn
        self.calls: list[tuple[dict, dict]] = []

    @property
    def type(self) -> str:
        """Return the public metric type identifier."""
        return self._name

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return the outputs exposed by this metric."""
        return [MetricOutputSpec.continuous_score("score")]

    def metric(self, item: dict, sample: dict, trace=None) -> float:
        """Return a raw score for protocol conformance."""
        del trace
        return float(self._score_fn(item, sample))

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Record the call and produce a single-score metric result."""
        item = input.row.data
        sample = input.candidate.as_sample()
        self.calls.append((dict(item), dict(sample)))
        return MetricResult(outputs=[MetricOutput(name="score", value=float(self._score_fn(item, sample)))])


class _RaisingMetric:
    """Metric stub that raises on every ``compute_scores`` call."""

    def __init__(self, name: str, exc: BaseException):
        """Capture the exception to raise on each scoring call."""
        self._name = name
        self._exc = exc

    @property
    def type(self) -> str:
        """Return the public metric type identifier."""
        return self._name

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return the outputs exposed by this metric."""
        return [MetricOutputSpec.continuous_score("score")]

    def metric(self, item: dict, sample: dict, trace=None) -> float:
        """Raise the configured exception for protocol conformance."""
        del item, sample, trace
        raise self._exc

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Raise the configured exception to exercise failure handling."""
        del input
        raise self._exc


class _NoOutputsMetric(_ScriptedMetric):
    """Metric stub that deliberately declares no outputs."""

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return no outputs to exercise pipeline validation."""
        return []


class _NoOpProgressReporter:
    """Progress reporter stub for structural protocol checks."""

    def increment_work(self, _increment: int = 1, /) -> None:
        """Accept progress increments without side effects."""


class _CorpusMetric(_ScriptedMetric):
    """Metric stub that emits a corpus-level score during finalization."""

    def __init__(self, name: str, score_fn):
        """Configure row scoring and call tracking for corpus scoring."""
        super().__init__(name, score_fn)
        self.corpus_calls: list[list[MetricInput]] = []

    def corpus_output_spec(self) -> list[MetricOutputSpec]:
        """Return corpus-level outputs exposed by this metric."""
        return [MetricOutputSpec.continuous_score("corpus")]

    async def compute_corpus_scores(self, inputs: list[MetricInput]) -> MetricResult:
        """Record corpus inputs and return one corpus score."""
        self.corpus_calls.append(inputs)
        return MetricResult(outputs=[MetricOutput(name="corpus", value=42.0)])


def _make_model() -> Model:
    """Build a minimal Model instance suitable for online benchmark tests."""
    return Model(url="http://example.test/v1", name="test-model")


def _make_agent() -> Agent:
    """Build a minimal generic Agent instance suitable for online benchmark tests."""
    return GenericAgent(
        url="http://agent.test",
        name="test-agent",
        format=AgentFormat.GENERIC,
        body={"prompt": "{{ prompt }}"},
        response_path="$.answer",
    )


@pytest.fixture
def online_params() -> RunConfigOnlineModel:
    """Default online params with ``parallelism=2`` to exercise concurrency."""
    return RunConfigOnlineModel(parallelism=2)


class TestEvaluateBenchmarkOnline:
    """Coverage for online benchmark execution across multiple metrics."""

    @pytest.mark.asyncio
    async def test_generates_each_sample_once_across_metrics(
        self, mocker: MockerFixture, online_params: RunConfigOnlineModel
    ) -> None:
        """Multi-metric online benchmarks must call inference once per row, not per metric."""
        rows = [{"prompt": f"row-{idx}"} for idx in range(3)]
        model = _make_model()

        async def _fake_sample(*, row: dict, **_kwargs) -> dict:
            return {"output_text": f"echo-{row['prompt']}", "response": {"raw": row["prompt"]}}

        sample_mock = mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=_fake_sample,
        )

        metric_a = _ScriptedMetric("a", lambda item, sample: 1.0)
        metric_b = _ScriptedMetric("b", lambda item, sample: 2.0)

        result = await evaluate_benchmark(
            metrics=[("a", metric_a), ("b", metric_b)],
            rows=rows,
            target=model,
            params=online_params,
            prompt_template="{{prompt}}",
        )

        assert sample_mock.await_count == len(rows)
        observed_prompts = Counter(call.kwargs["row"]["prompt"] for call in sample_mock.call_args_list)
        assert observed_prompts == Counter(row["prompt"] for row in rows)
        assert len(metric_a.calls) == len(rows)
        assert len(metric_b.calls) == len(rows)
        assert set(result.per_metric) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_row_identity_alignment_across_metrics(
        self, mocker: MockerFixture, online_params: RunConfigOnlineModel
    ) -> None:
        """Each metric should see the same sample for a given row index."""
        rows = [{"prompt": f"row-{idx}", "tag": idx} for idx in range(4)]
        model = _make_model()

        async def _fake_sample(*, row: dict, index: int, **_kwargs) -> dict:
            return {"output_text": str(index), "response": {"tag": row["tag"]}}

        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=_fake_sample,
        )
        metric_a = _ScriptedMetric("a", lambda item, sample: item["tag"])
        metric_b = _ScriptedMetric("b", lambda item, sample: item["tag"] * 2)

        result = await evaluate_benchmark(
            metrics=[("a", metric_a), ("b", metric_b)],
            rows=rows,
            target=model,
            params=online_params,
            prompt_template="{{prompt}}",
        )

        # Row scores keep input order because _initialize_row_scores is
        # deterministic and each metric worker writes into row_index slots.
        assert [row.item["tag"] for row in result.row_scores] == [r["tag"] for r in rows]
        for row in result.row_scores:
            sample_output = row.sample.get("output_text")
            assert sample_output == str(row.row_index)
            a_scores = row.metrics["a"]
            b_scores = row.metrics["b"]
            assert a_scores[0].value == float(row.item["tag"])
            assert b_scores[0].value == float(row.item["tag"] * 2)

    @pytest.mark.asyncio
    async def test_ignore_request_failure_maps_row_to_nan(
        self, mocker: MockerFixture, online_params: RunConfigOnlineModel
    ) -> None:
        """Online inference failures should propagate as NaN-eligible samples to every metric."""
        rows = [{"prompt": "ok"}, {"prompt": "bad"}]
        model = _make_model()

        async def _maybe_failing(*, row: dict, **_kwargs) -> dict:
            if row["prompt"] == "bad":
                raise RuntimeError("simulated inference failure")
            return {"output_text": "ok", "response": {}}

        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=_maybe_failing,
        )

        def _score(item: dict, sample: dict) -> float:
            if sample.get("inference_error"):
                return float("nan")
            return 1.0

        metric_a = _ScriptedMetric("a", _score)
        metric_b = _ScriptedMetric("b", _score)

        params = RunConfigOnlineModel(parallelism=2, ignore_request_failure=True)

        result = await evaluate_benchmark(
            metrics=[("a", metric_a), ("b", metric_b)],
            rows=rows,
            target=model,
            params=params,
            prompt_template="{{prompt}}",
        )

        # Both metric workers see two rows — the failing row just has a
        # NaN-eligible sample rather than being skipped.
        assert len(metric_a.calls) == 2
        assert len(metric_b.calls) == 2

        bad_row = next(r for r in result.row_scores if r.item["prompt"] == "bad")
        assert bad_row.sample.get("inference_error")
        for metric_ref in ("a", "b"):
            assert math.isnan(bad_row.metrics[metric_ref][0].value)

    @pytest.mark.asyncio
    async def test_metric_failure_is_raised_without_ignore_flag(self, mocker: MockerFixture) -> None:
        """Metric scoring failures should surface benchmark row context in strict mode."""
        rows = [{"prompt": "a"}]
        model = _make_model()

        async def _fake_sample(**_kwargs) -> dict:
            return {"output_text": "ok", "response": {}}

        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=_fake_sample,
        )

        metric_good = _ScriptedMetric("good", lambda item, sample: 1.0)
        metric_bad = _RaisingMetric("bad", RuntimeError("boom"))

        params = RunConfigOnlineModel(parallelism=1)

        with pytest.raises(EvaluationError, match="metric scoring") as exc_info:
            await evaluate_benchmark(
                metrics=[("good", metric_good), ("bad", metric_bad)],
                rows=rows,
                target=model,
                params=params,
                prompt_template="{{prompt}}",
            )
        assert exc_info.value.index == 0
        assert exc_info.value.metric_key == "bad"
        assert exc_info.value.phase is EvaluationPhase.METRIC_SCORING
        assert exc_info.value.message == "boom"
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_sample_generation_failure_is_raised_with_row_context(self, mocker: MockerFixture) -> None:
        """Strict online sample-generation failures should identify the failed row."""
        rows = [{"prompt": "bad"}]
        model = _make_model()

        async def _failing_sample(**_kwargs) -> dict:
            raise RuntimeError("inference exploded")

        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=_failing_sample,
        )
        metric = _ScriptedMetric("good", lambda item, sample: 1.0)

        with pytest.raises(EvaluationError, match="sample generation") as exc_info:
            await evaluate_benchmark(
                metrics=[("good", metric)],
                rows=rows,
                target=model,
                params=RunConfigOnlineModel(parallelism=1),
                prompt_template="{{prompt}}",
            )
        assert exc_info.value.index == 0
        assert exc_info.value.metric_key is None
        assert exc_info.value.phase is EvaluationPhase.SAMPLE_GENERATION
        assert exc_info.value.message == "inference exploded"
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_metric_failure_maps_to_nan_with_ignore_flag(self, mocker: MockerFixture) -> None:
        """Metric scoring failures should NaN-fallback only when ignore_request_failure=True."""
        rows = [{"prompt": "a"}, {"prompt": "b"}]
        model = _make_model()

        async def _fake_sample(**_kwargs) -> dict:
            return {"output_text": "ok", "response": {}}

        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=_fake_sample,
        )

        metric_good = _ScriptedMetric("good", lambda item, sample: 1.0)
        metric_bad = _RaisingMetric("bad", RuntimeError("boom"))
        params = RunConfigOnlineModel(parallelism=2, ignore_request_failure=True)

        result = await evaluate_benchmark(
            metrics=[("good", metric_good), ("bad", metric_bad)],
            rows=rows,
            target=model,
            params=params,
            prompt_template="{{prompt}}",
        )

        for row in result.row_scores:
            assert row.metrics["good"][0].value == 1.0
            assert math.isnan(row.metrics["bad"][0].value)

    @pytest.mark.asyncio
    async def test_default_headers_plumbed_to_generate_online_sample(
        self, mocker: MockerFixture, online_params: RunConfigOnlineModel
    ) -> None:
        """``default_headers`` must flow into each online sample generation call."""
        rows = [{"prompt": "x"}]
        model = _make_model()

        async def _fake_sample(**_kwargs) -> dict:
            return {"output_text": "ok", "response": {}}

        sample_mock = mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=_fake_sample,
        )
        metric_a = _ScriptedMetric("a", lambda item, sample: 1.0)

        headers = {"X-Test-Header": "abc"}
        await evaluate_benchmark(
            metrics=[("a", metric_a)],
            rows=rows,
            target=model,
            params=online_params,
            prompt_template="{{prompt}}",
            default_headers=headers,
        )
        assert sample_mock.call_args.kwargs["default_headers"] == headers

    @pytest.mark.asyncio
    async def test_default_headers_plumbed_to_generate_online_sample_agent(
        self, mocker: MockerFixture, online_params: RunConfigOnlineModel
    ) -> None:
        """``default_headers`` must flow into agent sample generation calls."""
        rows = [{"prompt": "agent-row"}]
        agent = _make_agent()

        async def _fake_agent_sample(*, row: dict, **_kwargs) -> dict:
            """Return a deterministic generated sample for the patched agent helper."""
            return {"output_text": f"agent-{row['prompt']}", "response": {"answer": row["prompt"]}}

        agent_mock = mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample_agent",
            side_effect=_fake_agent_sample,
        )
        metric = _ScriptedMetric("a", lambda item, sample: 1.0)
        headers = {"X-NMP-Principal-Id": "service:evaluator"}

        await evaluate_benchmark(
            metrics=[("a", metric)],
            rows=rows,
            target=agent,
            params=online_params,
            prompt_template="{{prompt}}",
            default_headers=headers,
        )

        await_args = agent_mock.await_args
        assert await_args is not None
        assert await_args.kwargs["default_headers"] == headers

    @pytest.mark.asyncio
    async def test_agent_targets_use_agent_sample_generation(
        self, mocker: MockerFixture, online_params: RunConfigOnlineModel
    ) -> None:
        """Agent targets must route through ``generate_online_sample_agent`` only."""
        rows = [{"prompt": "agent-row"}]
        agent = _make_agent()

        sample_mock = mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=AssertionError("model sample generation must not be called for agent targets"),
        )

        async def _fake_agent_sample(*, row: dict, **_kwargs) -> dict:
            return {"output_text": f"agent-{row['prompt']}", "response": {"answer": row["prompt"]}}

        agent_mock = mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample_agent",
            side_effect=_fake_agent_sample,
        )

        metric_a = _ScriptedMetric("a", lambda item, sample: 1.0)

        result = await evaluate_benchmark(
            metrics=[("a", metric_a)],
            rows=rows,
            target=agent,
            params=online_params,
            prompt_template="{{prompt}}",
        )

        assert sample_mock.await_count == 0
        assert agent_mock.await_count == 1
        assert result.row_scores[0].sample["output_text"] == "agent-agent-row"
        assert result.row_scores[0].metrics["a"][0].value == 1.0

    @pytest.mark.asyncio
    async def test_missing_online_prompt_template_raises_generation_error(self, online_params: RunConfigOnlineModel):
        """Online benchmark execution should fail before inference when no prompt template is provided."""
        metric = _ScriptedMetric("a", lambda item, sample: 1.0)

        with pytest.raises(EvaluationError, match="prompt_template is required") as exc_info:
            await evaluate_benchmark(
                metrics=[("a", metric)],
                rows=[{"prompt": "hello"}],
                target=_make_model(),
                params=online_params,
            )

        assert exc_info.value.phase is EvaluationPhase.SAMPLE_GENERATION


class TestEvaluateBenchmarkOffline:
    """Coverage for offline benchmark execution (no ``target``)."""

    @pytest.mark.asyncio
    async def test_offline_skips_sample_generation_and_passes_empty_sample(self, mocker: MockerFixture) -> None:
        """Offline benchmarks must not invoke any sample generation function."""
        rows = [{"prompt": "a"}, {"prompt": "b"}]

        sample_mock = mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=AssertionError("must not be called for offline benchmarks"),
        )
        agent_mock = mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample_agent",
            side_effect=AssertionError("must not be called for offline benchmarks"),
        )

        observed_samples: list[dict] = []

        def _score(item: dict, sample: dict) -> float:
            observed_samples.append(sample)
            return 1.0

        metric_a = _ScriptedMetric("a", _score)

        result = await evaluate_benchmark(
            metrics=[("a", metric_a)],
            rows=rows,
            target=None,
            params=RunConfig(parallelism=2),
        )

        assert sample_mock.await_count == 0
        assert agent_mock.await_count == 0
        assert observed_samples == [{}, {}]
        for row in result.row_scores:
            assert row.sample == {}
            assert row.metrics["a"][0].value == 1.0

    @pytest.mark.asyncio
    async def test_progress_increments_once_per_metric_row(self, mocker: MockerFixture) -> None:
        """Progress callbacks fire after each metric worker stores a row result."""
        rows = [{"prompt": "a"}, {"prompt": "b"}]
        progress = mocker.Mock()
        metric_a = _ScriptedMetric("a", lambda item, sample: 1.0)

        await evaluate_benchmark(
            metrics=[("a", metric_a)],
            rows=rows,
            target=None,
            params=RunConfig(parallelism=1),
            progress=progress,
        )

        assert progress.increment_work.call_count == len(rows)


class TestEvaluateBenchmarkFailurePolicy:
    """Coverage for params-derived benchmark failure policy.

    Online params with ``ignore_request_failure=True`` should map row failures
    to NaN, while offline params should fail fast by default.
    """

    @pytest.mark.asyncio
    async def test_online_metric_failure_maps_to_nan_with_ignore_request_failure(self, mocker: MockerFixture) -> None:
        """Online metric failure on one row should NaN only that row when ignore_request_failure=True."""
        rows = [{"prompt": "ok"}, {"prompt": "bad"}, {"prompt": "ok2"}]
        model = _make_model()
        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            return_value={"output_text": "ok", "response": {}},
        )

        def _score(item: dict, sample: dict) -> float:
            if item["prompt"] == "bad":
                raise RuntimeError("metric blew up on row 1")
            return 2.0

        metric = _ScriptedMetric("m", _score)

        result = await evaluate_benchmark(
            metrics=[("m", metric)],
            rows=rows,
            target=model,
            params=RunConfigOnlineModel(parallelism=1, ignore_request_failure=True),
            prompt_template="{{prompt}}",
        )

        assert result.row_scores[0].metrics["m"][0].value == 2.0
        assert math.isnan(result.row_scores[1].metrics["m"][0].value)
        assert result.row_scores[1].metric_errors == {"m": "metric blew up on row 1"}
        assert result.row_scores[2].metrics["m"][0].value == 2.0
        aggregate_score = result.per_metric["m"].aggregate_scores.scores[0]
        assert aggregate_score.name == "m.score"
        assert aggregate_score.count == 2
        assert aggregate_score.nan_count == 1
        assert [record["status"] for record in result.to_records("rows")] == ["ok", "error", "ok"]
        assert [record["status"] for record in result.metric_result("m").to_records("rows")] == [
            "ok",
            "error",
            "ok",
        ]

    @pytest.mark.asyncio
    async def test_sibling_metric_survives_when_other_metric_always_fails_with_ignore_request_failure(
        self, mocker: MockerFixture
    ) -> None:
        """ignore_request_failure=True must keep sibling-metric results intact when another metric raises."""
        rows = [{"prompt": f"row-{i}"} for i in range(3)]
        model = _make_model()
        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            return_value={"output_text": "ok", "response": {}},
        )

        metric_good = _ScriptedMetric("good", lambda item, sample: 1.0)
        metric_bad = _RaisingMetric("bad", RuntimeError("always blows up"))

        result = await evaluate_benchmark(
            metrics=[("good", metric_good), ("bad", metric_bad)],
            rows=rows,
            target=model,
            params=RunConfigOnlineModel(parallelism=1, ignore_request_failure=True),
            prompt_template="{{prompt}}",
        )

        for row in result.row_scores:
            assert row.metrics["good"][0].value == 1.0
            assert math.isnan(row.metrics["bad"][0].value)
            assert row.metric_errors == {"bad": "always blows up"}
        for row in result.metric_result("bad").row_scores:
            assert row.metric_errors == {"bad": "always blows up"}
        bad_aggregate_score = result.per_metric["bad"].aggregate_scores.scores[0]
        assert bad_aggregate_score.name == "bad.score"
        assert bad_aggregate_score.count == 0
        assert bad_aggregate_score.nan_count == len(rows)

    @pytest.mark.asyncio
    async def test_per_metric_rows_include_generation_and_own_metric_requests(
        self, mocker: MockerFixture, online_params: RunConfigOnlineModel
    ) -> None:
        """Per-metric row logs should include generation logs and exclude sibling metric logs."""
        rows = [{"prompt": "row-0"}]
        model = _make_model()

        async def _fake_sample(**_kwargs) -> dict:
            """Record one generation request and return a generated sample."""
            inference.requests_log_var.get().append({"phase": "generation", "row": "row-0"})
            return {"output_text": "ok", "response": {}}

        class _RequestLoggingMetric:
            """Metric test double that records a request log entry while scoring."""

            def __init__(self, name: str, value: float) -> None:
                """Configure the metric name and score value."""
                self._name = name
                self._value = value

            @property
            def type(self) -> str:
                """Return the public metric type identifier."""
                return self._name

            def output_spec(self) -> list[MetricOutputSpec]:
                """Return the outputs exposed by this metric."""
                return [MetricOutputSpec.continuous_score("score")]

            async def compute_scores(self, input: MetricInput) -> MetricResult:
                """Record a metric-specific request and return the configured score."""
                del input
                inference.requests_log_var.get().append({"phase": "metric", "metric": self._name})
                return MetricResult(outputs=[MetricOutput(name="score", value=self._value)])

        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=_fake_sample,
        )

        result = await evaluate_benchmark(
            metrics=[("a", _RequestLoggingMetric("a", 1.0)), ("b", _RequestLoggingMetric("b", 2.0))],
            rows=rows,
            target=model,
            params=online_params,
            prompt_template="{{prompt}}",
        )

        assert result.row_scores[0].requests == [
            {"phase": "generation", "row": "row-0"},
            {"phase": "metric", "metric": "a"},
            {"phase": "metric", "metric": "b"},
        ]
        assert result.metric_result("a").row_scores[0].requests == [
            {"phase": "generation", "row": "row-0"},
            {"phase": "metric", "metric": "a"},
        ]
        assert result.metric_result("b").row_scores[0].requests == [
            {"phase": "generation", "row": "row-0"},
            {"phase": "metric", "metric": "b"},
        ]

    @pytest.mark.asyncio
    async def test_fail_fast_true_propagates_metric_failure(self) -> None:
        """fail_fast=True should abort with a deterministic typed benchmark error."""
        rows = [{"prompt": f"row-{i}"} for i in range(2)]

        metric_good = _ScriptedMetric("good", lambda item, sample: 1.0)
        metric_bad = _RaisingMetric("bad", RuntimeError("boom"))

        with pytest.raises(EvaluationError, match="metric scoring") as exc_info:
            await evaluate_benchmark(
                metrics=[("good", metric_good), ("bad", metric_bad)],
                rows=rows,
                target=None,
                params=RunConfig(parallelism=2),
            )
        assert exc_info.value.index == 0
        assert exc_info.value.metric_key == "bad"
        assert exc_info.value.phase is EvaluationPhase.METRIC_SCORING
        assert exc_info.value.message == "boom"
        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestEvaluateBenchmarkEdgeCases:
    """Coverage for input validation and result shape guarantees."""

    @pytest.mark.asyncio
    async def test_rejects_empty_metrics(self) -> None:
        """``evaluate_benchmark`` must reject empty metric sequences."""
        with pytest.raises(ValueError, match="at least one"):
            await evaluate_benchmark(
                metrics=[],
                rows=[{"prompt": "a"}],
                target=None,
                params=RunConfig(parallelism=1),
            )

    @pytest.mark.asyncio
    async def test_per_metric_scores_namespaced_at_aggregate_level(self, mocker: MockerFixture) -> None:
        """Aggregate score names must be prefixed with the metric_ref for disambiguation."""
        rows = [{"prompt": "a"}]
        metric_a = _ScriptedMetric("a", lambda item, sample: 1.0)

        result = await evaluate_benchmark(
            metrics=[("custom.ref", metric_a)],
            rows=rows,
            target=None,
            params=RunConfig(parallelism=1),
        )

        per_metric_scores = result.per_metric["custom.ref"].aggregate_scores.scores
        assert [score.name for score in per_metric_scores] == ["custom.ref.score"]
        # Top-level aggregate_scores mirrors per-metric aggregate names.
        assert [score.name for score in result.aggregate_scores.scores] == ["custom.ref.score"]

    @pytest.mark.asyncio
    async def test_corpus_scores_are_namespaced_at_aggregate_level(self) -> None:
        """Corpus metric scores should be finalized and namespaced in benchmark results."""
        rows = [{"prompt": "a"}, {"prompt": "b"}]
        metric = _CorpusMetric("corpus-metric", lambda item, sample: 1.0)

        result = await evaluate_benchmark(
            metrics=[("custom.ref", metric)],
            rows=rows,
            target=None,
            params=RunConfig(parallelism=1),
        )

        assert [[input.row.data for input in call] for call in metric.corpus_calls] == [
            [{"prompt": "a"}, {"prompt": "b"}]
        ]
        assert [[input.candidate.as_sample() for input in call] for call in metric.corpus_calls] == [[{}, {}]]
        assert [score.name for score in result.per_metric["custom.ref"].aggregate_scores.scores] == [
            "custom.ref.score",
            "custom.ref.corpus",
        ]
        assert [score.name for score in result.aggregate_scores.scores] == [
            "custom.ref.score",
            "custom.ref.corpus",
        ]

    @pytest.mark.asyncio
    async def test_corpus_scores_skip_failed_rows_but_aggregates_keep_nan_placeholders(
        self, mocker: MockerFixture
    ) -> None:
        """Corpus inputs skip failed rows while aggregate statistics include NaN placeholders."""
        rows = [{"prompt": "ok"}, {"prompt": "bad"}]
        model = _make_model()
        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            return_value={"output_text": "ok", "response": {}},
        )

        def _score(item: dict, sample: dict) -> float:
            """Raise on the bad row and score successful rows."""
            del sample
            if item["prompt"] == "bad":
                raise RuntimeError("boom")
            return 1.0

        metric = _CorpusMetric("corpus-metric", _score)

        result = await evaluate_benchmark(
            metrics=[("custom.ref", metric)],
            rows=rows,
            target=model,
            params=RunConfigOnlineModel(parallelism=1, ignore_request_failure=True),
            prompt_template="{{prompt}}",
        )

        assert [[input.row.data for input in call] for call in metric.corpus_calls] == [[{"prompt": "ok"}]]
        assert [[input.candidate.as_sample() for input in call] for call in metric.corpus_calls] == [
            [{"output_text": "ok", "response": {}}]
        ]
        aggregate_scores = result.per_metric["custom.ref"].aggregate_scores.scores
        assert [(score.name, score.count, score.nan_count) for score in aggregate_scores] == [
            ("custom.ref.score", 1, 1),
            ("custom.ref.corpus", 1, 0),
        ]

    @pytest.mark.asyncio
    async def test_corpus_scores_skip_generation_failure_placeholders(self, mocker: MockerFixture) -> None:
        """Corpus inputs skip ignored generation failures without changing row error output."""
        rows = [{"prompt": "ok"}, {"prompt": "bad"}]
        model = _make_model()

        async def _maybe_failing_sample(*, row: dict, **_kwargs) -> dict:
            """Return one generated sample and fail the other row."""
            if row["prompt"] == "bad":
                raise RuntimeError("generation boom")
            return {"output_text": "ok", "response": {}}

        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.generate_online_sample",
            side_effect=_maybe_failing_sample,
        )

        def _score(_item: dict, sample: dict) -> float:
            """Return NaN for ignored generation failure placeholders."""
            if sample.get("inference_error"):
                return float("nan")
            return 1.0

        metric = _CorpusMetric("corpus-metric", _score)

        result = await evaluate_benchmark(
            metrics=[("custom.ref", metric)],
            rows=rows,
            target=model,
            params=RunConfigOnlineModel(parallelism=1, ignore_request_failure=True),
            prompt_template="{{prompt}}",
        )

        failed_row = result.row_scores[1]
        assert failed_row.sample == {
            "output_text": None,
            "response": {},
            "inference_error": "generation boom",
        }
        assert failed_row.metric_errors is None
        assert [[input.row.data for input in call] for call in metric.corpus_calls] == [[{"prompt": "ok"}]]
        assert [[input.candidate.as_sample() for input in call] for call in metric.corpus_calls] == [
            [{"output_text": "ok", "response": {}}]
        ]
        aggregate_scores = result.per_metric["custom.ref"].aggregate_scores.scores
        assert [(score.name, score.count, score.nan_count) for score in aggregate_scores] == [
            ("custom.ref.score", 1, 1),
            ("custom.ref.corpus", 1, 0),
        ]

    @pytest.mark.asyncio
    async def test_untyped_streaming_failure_is_reraised(self, mocker: MockerFixture) -> None:
        """Only typed benchmark errors are unwrapped from the streaming pipeline."""
        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution._run_streaming_pipeline",
            side_effect=ValueError("plain failure"),
        )
        metric = _ScriptedMetric("a", lambda item, sample: 1.0)

        with pytest.raises(ValueError, match="plain failure"):
            await evaluate_benchmark(
                metrics=[("a", metric)],
                rows=[{"prompt": "a"}],
                target=None,
                params=RunConfig(parallelism=1),
            )

    @pytest.mark.asyncio
    async def test_raises_when_pipeline_finishes_with_missing_metric_results(self, mocker: MockerFixture) -> None:
        """Missing metric result slots should fail loudly after pipeline completion."""
        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution._run_streaming_pipeline",
            return_value=None,
        )
        metric = _ScriptedMetric("a", lambda item, sample: 1.0)

        with pytest.raises(RuntimeError, match="missing metric results for 'a'"):
            await evaluate_benchmark(
                metrics=[("a", metric)],
                rows=[{"prompt": "a"}],
                target=None,
                params=RunConfig(parallelism=1),
            )


class TestBenchmarkHelpers:
    """Coverage for small helper branches used by benchmark orchestration."""

    def test_protocol_increment_work_body_is_callable(self) -> None:
        reporter: ProgressReporter = _NoOpProgressReporter()
        assert reporter.increment_work() is None

    def test_normalize_metric_result_orders_declared_outputs(self) -> None:
        result = _normalize_metric_result(
            MetricResult(
                outputs=[
                    MetricOutput(name="second", value=2.0),
                    MetricOutput(name="first", value=1.0),
                ]
            ),
            [MetricOutputSpec.continuous_score("first"), MetricOutputSpec.continuous_score("second")],
        )

        assert [output.name for output in result.outputs] == ["first", "second"]
        assert [output.value for output in result.outputs] == [1.0, 2.0]

    def test_normalize_metric_result_preserves_declared_order_while_omitting_optional_outputs(self) -> None:
        result = _normalize_metric_result(
            MetricResult(
                outputs=[
                    MetricOutput(name="third", value=3.0),
                    MetricOutput(name="first", value=1.0),
                ]
            ),
            [
                MetricOutputSpec.continuous_score("first"),
                MetricOutputSpec.continuous_score("second", required=False),
                MetricOutputSpec.continuous_score("third"),
            ],
        )

        assert [output.name for output in result.outputs] == ["first", "third"]
        assert [output.value for output in result.outputs] == [1.0, 3.0]

    def test_benchmark_error_from_exception_returns_none_without_typed_leaf(self) -> None:
        assert _benchmark_error_from_exception(ValueError("plain failure")) is None

    def test_build_metric_pipelines_rejects_metric_without_outputs(self) -> None:
        metric = _NoOutputsMetric("empty", lambda item, sample: 1.0)

        with pytest.raises(RuntimeError, match="does not declare any outputs"):
            _build_metric_pipelines([("empty", metric)], item_count=1, queue_capacity=1)


class TestMetricWorker:
    """Coverage for direct metric worker validation branches."""

    @pytest.mark.asyncio
    async def test_raises_for_unexpected_queue_item(self) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        queue.put_nowait(object())
        pipeline = _MetricPipeline(
            metric_ref="a",
            metric=_ScriptedMetric("a", lambda item, sample: 1.0),
            output_spec=[MetricOutputSpec.continuous_score("score")],
            queue=queue,
            results=[None],
        )

        with pytest.raises(ValueError, match="Expected _SampleEvent, got: object"):
            await _metric_worker(
                params=RunConfig(parallelism=1),
                pipeline=pipeline,
                row_scores=[
                    RowScore(row_index=0, item={"prompt": "a"}, sample={}, metrics={}, requests=[]),
                ],
                row_metric_requests=[{}],
                logger=logging.getLogger(__name__),
            )


class TestPutPipelineSentinelsCancellation:
    """Verify cancellation-safe sentinel signaling does not deadlock."""

    @pytest.mark.asyncio
    async def test_put_pipeline_sentinels_uses_put_nowait_when_cancelling(self, mocker: MockerFixture) -> None:
        """When the running task has pending cancellations, sentinel insertion must not await."""
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        queue.put_nowait("filler")  # Saturate the queue so a blocking put would deadlock.
        pipeline = _MetricPipeline(
            metric_ref="a",
            metric=_ScriptedMetric("a", lambda item, sample: 1.0),
            output_spec=[MetricOutputSpec.continuous_score("score")],
            queue=queue,
            results=[None],
        )

        # Stub current_task so cancelling() > 0 is observable without actually
        # sending a cancellation through the event loop (which would re-raise
        # at the next await point and mask the behavior under test).
        fake_task = mocker.Mock()
        fake_task.cancelling.return_value = 1
        mocker.patch(
            "nemo_evaluator_sdk.execution.benchmark_execution.asyncio.current_task",
            return_value=fake_task,
        )

        await asyncio.wait_for(
            _put_pipeline_sentinels(pipelines=[pipeline], worker_count=1),
            timeout=0.5,
        )

        # Queue was already full; put_nowait raised QueueFull (swallowed), so
        # the filler item stays put and no sentinel ends up enqueued.
        assert queue.qsize() == 1
        assert queue.get_nowait() == "filler"


@pytest.mark.asyncio
async def test_cancelling_resilient_inference_propagates_under_ignore_request_failure(
    online_params: RunConfigOnlineModel,
) -> None:
    """Task cancellation must not become an ignored NaN benchmark row."""
    from nemo_evaluator_sdk.resilience.api import run_with_resilience

    rows = [{"prompt": "cancel-me"}]
    model = _make_model()
    started = asyncio.Event()

    async def _never_finishes() -> dict:
        started.set()
        await asyncio.sleep(60)
        return {"choices": [{"message": {"content": "ok"}}]}

    async def _resilient_inference(
        model: Model,
        request: dict,
        max_retries: int | None,
        **kwargs,
    ) -> dict:
        return await run_with_resilience("cancel-endpoint", _never_finishes, max_attempts=1)

    online_params.ignore_request_failure = True
    task = asyncio.create_task(
        evaluate_benchmark(
            metrics=[("a", _ScriptedMetric("a", lambda item, sample: 1.0))],
            rows=rows,
            target=model,
            inference_fn=_resilient_inference,
            params=online_params,
            prompt_template="{{prompt}}",
        )
    )

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
