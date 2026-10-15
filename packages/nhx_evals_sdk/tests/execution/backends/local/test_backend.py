# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nhx_evals_sdk.execution.backends.local.backend."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from nhx_evals_sdk.execution.backends.local.backend import LocalBackend, _prepare_rows
from nhx_evals_sdk.values import FieldMapping, Model, RunConfig, RunConfigOnline, RunConfigOnlineModel
from nhx_evals_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nhx_evals_sdk.values.results import AggregatedMetricResult
from pytest_mock import MockerFixture

from packages.nhx_evals_sdk.tests.execution.backends.local._stubs import (
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
            "nhx_evals_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=prepared_rows,
        )
        mock_sdk = mocker.patch(
            "nhx_evals_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
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
            "nhx_evals_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[{"prompt": "a"}],
        )
        mock_sdk = mocker.patch(
            "nhx_evals_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
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
            "nhx_evals_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
            new=AsyncMock(),
        )
        mocker.patch(
            "nhx_evals_sdk.execution.backends.local.backend.prepare_dataset_rows",
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
            "nhx_evals_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[{"prompt": "a"}],
        )
        mock_sdk = mocker.patch(
            "nhx_evals_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
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
            "nhx_evals_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[{"prompt": "a"}],
        )
        mock_sdk = mocker.patch(
            "nhx_evals_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
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
            "nhx_evals_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[{"prompt": "a"}],
        )
        mock_sdk = mocker.patch(
            "nhx_evals_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
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
            "nhx_evals_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=[{"prompt": "a"}],
        )
        mock_sdk = mocker.patch(
            "nhx_evals_sdk.execution.backends.local.backend.sdk_evaluate_benchmark",
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
    """An unresolved mapping must be visible, and must never leave a same-named column in its place."""

    _MAPPING = FieldMapping(reference="messages[role=assistant].content")

    @staticmethod
    def _rows(*, with_assistant: tuple[int, ...] = ()) -> list[dict]:
        """Rows that each carry their own `reference` column, the case that used to hide a miss."""
        rows = []
        for index in range(3):
            messages = [{"role": "user", "content": f"q{index}"}]
            if index in with_assistant:
                messages.append({"role": "assistant", "content": f"answer{index}"})
            rows.append({"messages": messages, "reference": "FROM THE FILE"})
        return rows

    def _prepare(self, mocker: MockerFixture, rows: list[dict]) -> list[dict]:
        mocker.patch(
            "nhx_evals_sdk.execution.backends.local.backend.prepare_dataset_rows",
            return_value=rows,
        )
        return _prepare_rows(dataset=[], params=RunConfig(), field_mapping=self._MAPPING)

    def test_an_unresolved_mapping_leaves_the_field_unset(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The mapping owns the field: a miss clears it rather than yielding to the file's column.

        Without this the file's `reference` survives and is scored, which is a wrong answer rather
        than a missing one. A diagnostic asking `"reference" in mapped` cannot see that, so the
        warning is checked against the raw rows instead.
        """
        with caplog.at_level("WARNING"):
            mapped = self._prepare(mocker, self._rows())

        assert all("reference" not in row for row in mapped)
        assert "resolved nothing on 3 of 3 rows" in caplog.text

    def test_the_warning_names_the_mapped_path_not_just_the_field(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Naming only `reference` is ambiguous when the dataset also has a column by that name."""
        with caplog.at_level("WARNING"):
            self._prepare(mocker, self._rows())

        assert "messages[role=assistant].content" in caplog.text

    def test_a_partial_miss_is_reported(self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture) -> None:
        """Rows resolving unevenly is the common case for a predicate, and the silent one before."""
        with caplog.at_level("WARNING"):
            mapped = self._prepare(mocker, self._rows(with_assistant=(1,)))

        assert mapped[1]["reference"] == "answer1"
        assert "reference" not in mapped[0]
        assert "resolved nothing on 2 of 3 rows (first at index 0)" in caplog.text

    def test_no_warning_when_every_row_resolves(self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            mapped = self._prepare(mocker, self._rows(with_assistant=(0, 1, 2)))

        assert [row["reference"] for row in mapped] == ["answer0", "answer1", "answer2"]
        assert "resolved nothing" not in caplog.text
