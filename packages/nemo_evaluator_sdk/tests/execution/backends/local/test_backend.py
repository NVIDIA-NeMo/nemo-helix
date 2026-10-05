# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_evaluator_sdk.execution.backends.local.backend."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from nemo_evaluator_sdk.execution.backends.local.backend import LocalBackend, _prepare_rows
from nemo_evaluator_sdk.values import FieldMapping, Model, RunConfig, RunConfigOnline, RunConfigOnlineModel
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.results import AggregatedMetricResult
from pytest_mock import MockerFixture

from packages.nemo_evaluator_sdk.tests.execution.backends.local._stubs import (
    DuplicateMetric,
    IdentityPostprocessHook,
    IdentityPreprocessHook,
    PreparedBenchmarkMetric,
)


class TestLocalBackendEvaluateBenchmark:
    """Coverage for multi-metric local backend delegation to the SDK pipeline."""

    @pytest.mark.asyncio
    async def test_delegates_to_sdk_evaluate_benchmark_with_unique_metric_keys(self, mocker: MockerFixture) -> None:
        """The backend must pass unique metric keys and prepared rows to the SDK pipeline."""
        dataset = [{"prompt": "a"}, {"prompt": "b"}]
        params = RunConfig(parallelism=2)
        backend = LocalBackend()
        expected_result = BenchmarkEvaluationResult(
            row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]), per_metric={}
        )
        prepared_rows = [{"prompt": "a"}, {"prompt": "b"}]
        mock_prepare = mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=prepared_rows,
        )
        mock_sdk = mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
            new=AsyncMock(return_value=expected_result),
        )
        metrics = [DuplicateMetric(), DuplicateMetric()]

        result = await backend.evaluate_dataset(metrics=metrics, dataset=dataset, params=params)

        assert result is expected_result
        mock_prepare.assert_called_once_with(dataset, None, None)
        assert mock_sdk.await_args is not None
        sdk_kwargs = mock_sdk.await_args.kwargs
        assert [ref for ref, _ in sdk_kwargs["metrics"]] == ["duplicate", "duplicate_2"]
        assert [type(metric) for _, metric in sdk_kwargs["metrics"]] == [DuplicateMetric, DuplicateMetric]
        assert [metric for _, metric in sdk_kwargs["metrics"]] != metrics
        assert sdk_kwargs["rows"] is prepared_rows
        assert sdk_kwargs["target"] is None
        assert sdk_kwargs["params"] is params

    @pytest.mark.asyncio
    async def test_delegates_without_explicit_fail_fast(self, mocker: MockerFixture) -> None:
        """LocalBackend must let sdk_evaluate_benchmark derive fail-fast from params."""
        dataset = [{"prompt": "a"}]
        params = RunConfigOnline(parallelism=1, ignore_request_failure=True)
        backend = LocalBackend()
        expected_result = BenchmarkEvaluationResult(
            row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]), per_metric={}
        )
        mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[{"prompt": "a"}],
        )
        mock_sdk = mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
            new=AsyncMock(return_value=expected_result),
        )

        await backend.evaluate_dataset(metrics=[DuplicateMetric()], dataset=dataset, params=params)

        assert mock_sdk.await_args is not None
        assert "fail_fast" not in mock_sdk.await_args.kwargs
        assert mock_sdk.await_args.kwargs["params"] is params

    @pytest.mark.asyncio
    async def test_prepare_rows_failure_is_raised_without_sdk_call(self, mocker: MockerFixture) -> None:
        """Dataset preparation failures must abort before delegating to the SDK pipeline."""
        dataset = [{"prompt": "a"}]
        params = RunConfig(parallelism=1)
        backend = LocalBackend()
        mock_sdk = mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
            new=AsyncMock(),
        )
        mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.prepare_dataset_rows",
            side_effect=RuntimeError("bad dataset"),
        )

        with pytest.raises(RuntimeError, match="bad dataset"):
            await backend.evaluate_dataset(metrics=[DuplicateMetric()], dataset=dataset, params=params)

        mock_sdk.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_explicit_default_params(self, mocker: MockerFixture) -> None:
        """The backend should receive concrete default params from the evaluator."""
        dataset = [{"prompt": "a"}]
        params = RunConfig()
        backend = LocalBackend()
        expected_result = BenchmarkEvaluationResult(
            row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]), per_metric={}
        )
        mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[{"prompt": "a"}],
        )
        mock_sdk = mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
            new=AsyncMock(return_value=expected_result),
        )

        await backend.evaluate_dataset(metrics=[DuplicateMetric()], dataset=dataset, params=params)

        assert mock_sdk.await_args is not None
        assert mock_sdk.await_args.kwargs["params"] is params

    @pytest.mark.asyncio
    async def test_prepares_metrics_before_sdk_benchmark_execution(self, mocker: MockerFixture) -> None:
        """Local benchmark execution should prepare copied metrics before SDK delegation."""
        dataset = [{"prompt": "a"}]
        params = RunConfig(parallelism=3)
        backend = LocalBackend()
        expected_result = BenchmarkEvaluationResult(
            row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]), per_metric={}
        )
        mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[{"prompt": "a"}],
        )
        mock_sdk = mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
            new=AsyncMock(return_value=expected_result),
        )
        original = PreparedBenchmarkMetric()

        await backend.evaluate_dataset(metrics=[original], dataset=dataset, params=params)

        assert mock_sdk.await_args is not None
        prepared = mock_sdk.await_args.kwargs["metrics"][0][1]
        assert prepared == PreparedBenchmarkMetric(
            applied_parallelism=3,
            secrets_resolved=True,
            preflight_ran=True,
        )
        assert prepared is not original
        assert original == PreparedBenchmarkMetric()

    @pytest.mark.asyncio
    async def test_online_benchmark_merges_default_generation_hooks(self, mocker: MockerFixture) -> None:
        """Online local benchmarks should preserve default generation hooks."""
        explicit_preprocess = IdentityPreprocessHook()
        explicit_postprocess = IdentityPostprocessHook()
        dataset = [{"prompt": "a"}]
        target = Model(url="http://example.test/v1", name="test-model")
        params = RunConfigOnlineModel(parallelism=1)
        backend = LocalBackend()
        expected_result = BenchmarkEvaluationResult(
            row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]), per_metric={}
        )
        mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[{"prompt": "a"}],
        )
        mock_sdk = mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
            new=AsyncMock(return_value=expected_result),
        )

        await backend.evaluate_dataset(
            metrics=[DuplicateMetric()],
            dataset=dataset,
            params=params,
            target=target,
            preprocess_hooks=(explicit_preprocess,),
            postprocess_hooks=(explicit_postprocess,),
        )

        assert mock_sdk.await_args is not None
        sdk_kwargs = mock_sdk.await_args.kwargs
        assert explicit_preprocess in sdk_kwargs["preprocess_hooks"]
        assert explicit_postprocess in sdk_kwargs["postprocess_hooks"]
        assert len(sdk_kwargs["preprocess_hooks"]) > 1
        assert len(sdk_kwargs["postprocess_hooks"]) > 1
        assert tuple(sdk_kwargs["preprocess_hooks"]) != (explicit_preprocess,)
        assert tuple(sdk_kwargs["postprocess_hooks"]) != (explicit_postprocess,)

    @pytest.mark.asyncio
    async def test_offline_benchmark_does_not_merge_default_generation_hooks(self, mocker: MockerFixture) -> None:
        """Offline local benchmarks should keep only explicitly supplied hooks."""
        explicit_preprocess = IdentityPreprocessHook()
        explicit_postprocess = IdentityPostprocessHook()
        dataset = [{"prompt": "a"}]
        params = RunConfig(parallelism=1)
        backend = LocalBackend()
        expected_result = BenchmarkEvaluationResult(
            row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]), per_metric={}
        )
        mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[{"prompt": "a"}],
        )
        mock_sdk = mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
            new=AsyncMock(return_value=expected_result),
        )

        await backend.evaluate_dataset(
            metrics=[DuplicateMetric()],
            dataset=dataset,
            params=params,
            preprocess_hooks=(explicit_preprocess,),
            postprocess_hooks=(explicit_postprocess,),
        )

        assert mock_sdk.await_args is not None
        sdk_kwargs = mock_sdk.await_args.kwargs
        assert sdk_kwargs["preprocess_hooks"] == (explicit_preprocess,)
        assert sdk_kwargs["postprocess_hooks"] == (explicit_postprocess,)


class TestPrepareRowsUnresolvedMapping:
    """A binding that resolves nowhere must be reported, not left to fail obscurely later."""

    _ROWS = [
        {"messages": [{"role": "user", "content": "q"}], "reference": "stale raw column"},
        {"messages": [{"role": "user", "content": "q2"}], "reference": "stale raw column 2"},
    ]

    def test_warns_when_a_binding_resolves_on_no_row(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The raw column survives a miss, so presence of the canonical name proves nothing.

        `apply_column_mapping_to_row` copies the row before mapping. A path that never resolves
        therefore leaves a same-named raw column in place and that value gets scored -- a wrong
        answer rather than a missing one. A diagnostic asking `"reference" in mapped` would see the
        stale column and stay silent, so this test fails under that implementation.
        """
        mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[dict(row) for row in self._ROWS],
        )

        with caplog.at_level("WARNING"):
            rows = _prepare_rows(
                dataset=[],
                params=RunConfig(),
                field_mapping=FieldMapping(reference="messages[role=assistant].content"),
            )

        assert rows[0]["reference"] == "stale raw column"
        assert "reference" in caplog.text
        assert "resolved no value" in caplog.text

    def test_stays_quiet_when_a_binding_resolves_somewhere(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Resolving on only some rows is legitimate, so only a total miss is worth a warning."""
        rows = [dict(row) for row in self._ROWS]
        rows[0]["messages"].append({"role": "assistant", "content": "answer"})
        mocker.patch(
            "nemo_evaluator_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=rows,
        )

        with caplog.at_level("WARNING"):
            mapped = _prepare_rows(
                dataset=[],
                params=RunConfig(),
                field_mapping=FieldMapping(reference="messages[role=assistant].content"),
            )

        assert mapped[0]["reference"] == "answer"
        assert "resolved no value" not in caplog.text
