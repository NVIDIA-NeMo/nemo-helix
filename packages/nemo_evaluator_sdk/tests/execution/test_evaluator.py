# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import builtins
import importlib
import sys
from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest
from nemo_evaluator_sdk.enums import MetricType
from nemo_evaluator_sdk.execution.config import RunConfig, RunConfigOnlineModel
from nemo_evaluator_sdk.execution.evaluator import Evaluator
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.values import FieldMapping, Model
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.results import (
    AggregatedMetricResult,
    EvaluationResult,
)
from pydantic import ValidationError
from pytest_mock import MockerFixture


class _CustomMetric:
    @property
    def type(self) -> str:
        return MetricType.STRING_CHECK.value

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        score = 1.0 if input.row.data["expected"] == input.row.data["model_output"] else 0.0
        return MetricResult(outputs=[MetricOutput(name="string-check", value=score)])

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("string-check")]


class _CandidateOutputMetric:
    @property
    def type(self) -> str:
        return "candidate-output-check"

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        score = 1.0 if input.candidate.output_text == input.row.data["reference"] else 0.0
        return MetricResult(outputs=[MetricOutput(name="match", value=score)])

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("match")]


_DATASET = [
    {"expected": "blue", "model_output": "Blue"},
    {"expected": "Jupiter", "model_output": "Saturn"},
]

_MAPPED_DATASET = [
    {"expected": "blue", "prediction": "blue"},
    {"expected": "Jupiter", "prediction": "Saturn"},
]


def _empty_evaluation_result() -> EvaluationResult:
    return EvaluationResult(row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]))


def _empty_benchmark_result() -> BenchmarkEvaluationResult:
    return BenchmarkEvaluationResult(
        row_scores=[],
        aggregate_scores=AggregatedMetricResult(scores=[]),
        per_metric={},
    )


class _FakeDirectBackend:
    """Test backend that satisfies the evaluator protocol."""

    def __init__(self, result: BenchmarkEvaluationResult):
        self.result = result
        self.dataset_calls: list[dict[str, Any]] = []

    async def evaluate_dataset(
        self,
        *,
        metrics: Sequence[Metric],
        **kwargs: Any,
    ) -> BenchmarkEvaluationResult:
        self.dataset_calls.append({"metrics": metrics, **kwargs})
        return self.result


class _FakeSyncBackend:
    """Test backend that satisfies the sync evaluator protocol."""

    def __init__(self, result: BenchmarkEvaluationResult):
        self.result = result
        self.dataset_calls: list[dict[str, Any]] = []

    def evaluate_dataset(
        self,
        *,
        metrics: Sequence[Metric],
        **kwargs: Any,
    ) -> BenchmarkEvaluationResult:
        self.dataset_calls.append({"metrics": metrics, **kwargs})
        return self.result


class _LoopSensitiveSyncBackend(_FakeSyncBackend):
    """Sync backend that fails if called on a thread with an active event loop."""

    def _raise_if_running_on_active_loop(self) -> None:
        """Raise when the sync backend is executing on an active event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError("sync backend ran on an active event loop")

    def evaluate_dataset(
        self,
        *,
        metrics: Sequence[Metric],
        **kwargs: Any,
    ) -> BenchmarkEvaluationResult:
        self._raise_if_running_on_active_loop()
        return super().evaluate_dataset(metrics=metrics, **kwargs)


class _MissingEvaluateBackend:
    """Invalid backend: it does not implement the one contract method at all."""


class TestEvaluator:
    @pytest.mark.parametrize("flag_name", ["soft_fail", "fail_fast"])
    def test_run_config_rejects_run_level_failure_flags(self, flag_name: str) -> None:
        with pytest.raises(ValidationError):
            RunConfig.model_validate({"parallelism": 1, flag_name: True})

    def test_run_config_rejects_aggregate_fields(self) -> None:
        with pytest.raises(ValidationError):
            RunConfig.model_validate({"aggregate_fields": ["mean"]})

    def test_rejects_legacy_backend_argument(self):
        backend = _FakeDirectBackend(result=_empty_benchmark_result())

        legacy_kwargs: dict = {"backend": backend}
        with pytest.raises(TypeError, match="backend"):
            Evaluator(**legacy_kwargs)

    @pytest.mark.asyncio
    async def test_run_uses_offline_params_without_request_fail_fast(self):
        backend = _FakeDirectBackend(result=_empty_benchmark_result())
        evaluator = Evaluator(client=backend)

        await evaluator.run(
            metrics=[_CustomMetric()],
            dataset=_DATASET,
            config=RunConfig(parallelism=1),
        )

        call = backend.dataset_calls[0]
        assert call["params"] == RunConfig(parallelism=1)
        assert call["aggregate_fields"] is None
        assert "fail_fast" not in call

    @pytest.mark.asyncio
    async def test_run_preserves_aggregate_fields_on_request(self):
        backend = _FakeDirectBackend(result=_empty_benchmark_result())
        evaluator = Evaluator(client=backend)

        await evaluator.run(
            metrics=[_CustomMetric()],
            dataset=_DATASET,
            config=RunConfig(parallelism=1),
            aggregate_fields=("mean",),
        )

        call = backend.dataset_calls[0]
        assert call["params"] == RunConfig(parallelism=1)
        assert call["aggregate_fields"] == ("mean",)

    @pytest.mark.asyncio
    async def test_run_preserves_field_mapping_on_request(self):
        backend = _FakeDirectBackend(result=_empty_benchmark_result())
        evaluator = Evaluator(client=backend)
        field_mapping = FieldMapping(output="prediction", reference="expected")

        await evaluator.run(
            metrics=[_CustomMetric()],
            dataset=_MAPPED_DATASET,
            field_mapping=field_mapping,
        )

        call = backend.dataset_calls[0]
        assert call["field_mapping"] == field_mapping

    @pytest.mark.asyncio
    async def test_run_preserves_ignored_online_request_failure_params(self):
        backend = _FakeDirectBackend(result=_empty_benchmark_result())
        evaluator = Evaluator(client=backend)
        params = RunConfigOnlineModel(parallelism=1, ignore_request_failure=True)

        await evaluator.run(
            metrics=[_CustomMetric()],
            dataset=_DATASET,
            config=params,
            target=Model(url="http://model.test/v1", name="test-model"),
        )

        call = backend.dataset_calls[0]
        assert call["params"] is params
        assert "fail_fast" not in call

    @pytest.mark.asyncio
    async def test_run_accepts_sdk_metric_instance(self):
        evaluator = Evaluator()

        result = await evaluator.run(
            metrics=[ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}")],
            dataset=_DATASET,
            config=RunConfig(parallelism=2),
        )

        assert len(result.row_scores) == 2
        assert result.aggregate_scores.scores[0].name == "exact-match.exact-match"
        assert result.aggregate_scores.scores[0].mean == 0.5

    @pytest.mark.asyncio
    async def test_run_filters_aggregate_fields(self):
        evaluator = Evaluator()

        result = await evaluator.run(
            metrics=[ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}")],
            dataset=_DATASET,
            config=RunConfig(parallelism=2),
            aggregate_fields=("mean",),
        )

        assert result.aggregate_scores.model_dump(mode="json") == {
            "scores": [{"name": "exact-match.exact-match", "count": 2, "mean": 0.5}]
        }

    @pytest.mark.asyncio
    async def test_run_accepts_mixed_sdk_and_custom_metrics(self):
        evaluator = Evaluator()

        result = await evaluator.run(
            metrics=[
                ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}"),
                _CustomMetric(),
            ],
            dataset=_DATASET,
            config=RunConfig(parallelism=2),
        )

        assert list(result.per_metric) == ["exact-match", "string-check"]
        assert result.metric_result("exact-match").aggregate_scores.scores[0].name == "exact-match.exact-match"
        assert result.metric_result("string-check").aggregate_scores.scores[0].name == "string-check.string-check"

    @pytest.mark.asyncio
    async def test_run_filters_benchmark_aggregate_fields(self):
        evaluator = Evaluator()

        result = await evaluator.run(
            metrics=[
                ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}"),
                _CustomMetric(),
            ],
            dataset=_DATASET,
            config=RunConfig(parallelism=2),
            aggregate_fields=("mean",),
        )

        assert result.aggregate_scores.model_dump(mode="json") == {
            "scores": [
                {"name": "exact-match.exact-match", "count": 2, "mean": 0.5},
                {"name": "string-check.string-check", "count": 2, "mean": 0.0},
            ]
        }
        assert result.metric_result("exact-match").aggregate_scores.model_dump(mode="json") == {
            "scores": [{"name": "exact-match.exact-match", "count": 2, "mean": 0.5}]
        }

    @pytest.mark.asyncio
    async def test_run_sync_matches_async_run(self):
        evaluator = Evaluator()
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}")

        async_result = await evaluator.run(metrics=[metric], dataset=_DATASET, config=RunConfig(parallelism=2))
        sync_result = evaluator.run_sync(metrics=[metric], dataset=_DATASET, config=RunConfig(parallelism=2))

        assert async_result.model_dump(mode="python") == sync_result.model_dump(mode="python")
        assert isinstance(sync_result, BenchmarkEvaluationResult)

    def test_run_sync_custom_metric(self):
        evaluator = Evaluator()

        result = evaluator.run_sync(
            metrics=[_CustomMetric()],
            dataset=_DATASET,
        )

        assert isinstance(result, BenchmarkEvaluationResult)
        assert result.aggregate_scores.scores[0].name == "string-check.string-check"
        assert result.row_scores[0].metrics["string-check"][0].value == 0.0
        assert result.row_scores[1].metrics["string-check"][0].value == 0.0

    def test_run_sync_field_mapping_populates_offline_candidate_output(self):
        evaluator = Evaluator()

        result = evaluator.run_sync(
            metrics=[_CandidateOutputMetric()],
            dataset=_MAPPED_DATASET,
            field_mapping=FieldMapping(output="prediction", reference="expected"),
        )

        assert result.aggregate_scores.scores[0].mean == 0.5
        assert result.row_scores[0].item["output"] == "blue"
        assert result.row_scores[0].sample["output_text"] == "blue"
        assert result.row_scores[0].metrics["candidate-output-check"][0].value == 1.0

    def test_run_sync_field_mapping_populates_benchmark_offline_candidate_output(self):
        evaluator = Evaluator()

        result = evaluator.run_sync(
            metrics=[ExactMatchMetric(reference="{{reference}}")],
            dataset=_MAPPED_DATASET,
            field_mapping=FieldMapping(output="prediction", reference="expected"),
        )

        assert result.aggregate_scores.scores[0].mean == 0.5
        assert result.row_scores[0].sample["output_text"] == "blue"

    @pytest.mark.asyncio
    async def test_run_uses_sync_backend_adapter_thread_bridge(self, mocker: MockerFixture):
        expected = _empty_benchmark_result()
        backend = _FakeSyncBackend(result=expected)
        evaluator = Evaluator(client=backend)

        async def run_in_thread(func: object, *args: object, **kwargs: object) -> object:
            """Execute the submitted sync callable while recording the thread boundary."""
            return cast(Callable[..., object], func)(*args, **kwargs)

        to_thread = mocker.patch(
            "nemo_evaluator_sdk.execution.evaluator.asyncio.to_thread",
            new=mocker.AsyncMock(side_effect=run_in_thread),
        )

        result = await evaluator.run(
            metrics=[_CustomMetric()],
            dataset=_DATASET,
            config=RunConfig(parallelism=1),
        )

        assert result is expected
        to_thread.assert_awaited_once()
        assert len(backend.dataset_calls) == 1
        call = backend.dataset_calls[0]
        assert call["params"] == RunConfig(parallelism=1)

    def test_run_sync_uses_sync_backend_adapter(self, mocker: MockerFixture):
        expected = _empty_benchmark_result()
        backend = _FakeSyncBackend(result=expected)
        evaluator = Evaluator(client=backend)

        async def run_in_thread(func: object, *args: object, **kwargs: object) -> object:
            """Execute the submitted sync callable while recording the thread boundary."""
            return cast(Callable[..., object], func)(*args, **kwargs)

        to_thread = mocker.patch(
            "nemo_evaluator_sdk.execution.evaluator.asyncio.to_thread",
            new=mocker.AsyncMock(side_effect=run_in_thread),
        )

        result = evaluator.run_sync(
            metrics=[_CustomMetric()],
            dataset=_DATASET,
            config=RunConfig(parallelism=1),
        )

        assert result is expected
        to_thread.assert_awaited_once()
        assert len(backend.dataset_calls) == 1
        call = backend.dataset_calls[0]
        assert call["params"] == RunConfig(parallelism=1)

    @pytest.mark.asyncio
    async def test_run_sync_uses_thread_bridge_for_sync_backend_when_loop_is_running(self):
        expected = _empty_benchmark_result()
        backend = _LoopSensitiveSyncBackend(result=expected)
        evaluator = Evaluator(client=backend)

        result = evaluator.run_sync(
            metrics=[_CustomMetric()],
            dataset=_DATASET,
            config=RunConfig(parallelism=1),
        )

        assert result is expected
        assert len(backend.dataset_calls) == 1
        call = backend.dataset_calls[0]
        assert call["params"] == RunConfig(parallelism=1)

    def test_rejects_client_with_missing_backend_method(self):
        with pytest.raises(TypeError, match="callable evaluate_dataset"):
            Evaluator(client=cast(Any, _MissingEvaluateBackend()))

    def test_does_not_expose_submit_api(self):
        assert not hasattr(Evaluator(), "submit")
        assert not hasattr(Evaluator(), "submit_sync")

    def test_does_not_export_evaluatorv2(self):
        import nemo_evaluator_sdk.execution.evaluator as evaluator_module

        assert not hasattr(evaluator_module, "Evaluatorv2")

    def test_evaluator_module_import_does_not_require_nemo_helix(self, mocker: MockerFixture):
        evaluator_module = sys.modules.pop("nemo_evaluator_sdk.execution.evaluator", None)
        real_import = builtins.__import__

        def import_without_nemo_helix(name: str, *args: Any, **kwargs: Any) -> object:
            if name == "nemo_helix" or name.startswith("nemo_helix."):
                raise ModuleNotFoundError("No module named 'nemo_helix'", name="nemo_helix")
            if name == "nemo_evaluator" or name.startswith("nemo_evaluator."):
                raise ModuleNotFoundError("No module named 'nemo_evaluator'", name="nemo_evaluator")
            return real_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=import_without_nemo_helix)
        try:
            imported = importlib.import_module("nemo_evaluator_sdk.execution.evaluator")
            assert imported.Evaluator is not None
        finally:
            if evaluator_module is not None:
                sys.modules["nemo_evaluator_sdk.execution.evaluator"] = evaluator_module
                # import_module also rebinds `evaluator` on the parent package, and that binding is
                # what monkeypatch's string form resolves. Leaving the duplicate there sends later
                # patches to a module object nobody holds a reference to, so the patch silently
                # misses and the real call runs.
                setattr(sys.modules["nemo_evaluator_sdk.execution"], "evaluator", evaluator_module)

    def test_run_sync_uses_async_backend_through_run_bridge(self):
        expected = _empty_benchmark_result()
        backend = _FakeDirectBackend(result=expected)
        evaluator = Evaluator(client=backend)

        result = evaluator.run_sync(
            metrics=[ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}")],
            dataset=_DATASET,
        )

        assert result is expected
        assert len(backend.dataset_calls) == 1
