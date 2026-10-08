# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for MetricInput/MetricOutput execution plumbing."""

from __future__ import annotations

import json
import math

import pytest
from nhx_evals_sdk.execution.benchmark_execution import evaluate_benchmark
from nhx_evals_sdk.execution.scoring import finalize_evaluation_result, score_row
from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.values import (
    MetricDiagnostic,
    RunConfig,
)


class _OutputMetric:
    @property
    def type(self) -> str:
        return "test-output-metric"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.continuous_score("score"),
            MetricOutputSpec.boolean("passed"),
            MetricOutputSpec.label("reason"),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        assert input.row.row_index is not None
        assert input.row.data["expected"] == "yes"
        assert input.candidate.output_text in {"yes", "no", None}
        passed = input.candidate.output_text == input.row.data["expected"]
        return MetricResult(
            outputs=[
                MetricOutput(name="score", value=1.0 if passed else 0.0),
                MetricOutput(name="passed", value=passed),
                MetricOutput(name="reason", value="matched" if passed else "mismatched"),
            ]
        )


class _RaisingMetric(_OutputMetric):
    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        raise RuntimeError("metric failed")


class _DiagnosticMetric:
    """Metric that attaches an expected-vs-actual diagnostic when it mismatches."""

    @property
    def type(self) -> str:
        return "diagnostic-metric"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        expected = input.row.data["expected"]
        actual = input.candidate.output_text
        matched = actual == expected
        outputs = [MetricOutput(name="score", value=1.0 if matched else 0.0)]
        if matched:
            return MetricResult(outputs=outputs)
        return MetricResult(
            outputs=outputs,
            diagnostics=[
                MetricDiagnostic(
                    message="mismatch",
                    details={"expected": expected, "actual": actual},
                )
            ],
        )


@pytest.mark.asyncio
async def test_score_row_stores_outputs_and_aggregates_score_like_specs() -> None:
    metric = _OutputMetric()
    completed = [
        await score_row(
            metric=metric,
            row={"expected": "yes"},
            sample={"output_text": "yes", "response": {"id": "ok"}},
            index=0,
            metric_key="metric",
            fail_fast=True,
            generation_requests=[],
        )
    ]

    result = await finalize_evaluation_result(metric, completed)

    row = result.row_scores[0]
    assert [output.name for output in row.metrics["metric"]] == ["score", "passed", "reason"]
    assert result.to_records()[0]["output.score"] == 1.0
    assert result.to_records()[0]["output.reason"] == "matched"
    assert [score.name for score in result.aggregate_scores.scores] == ["score", "passed"]
    assert [score.mean for score in result.aggregate_scores.scores] == [1.0, 1.0]


@pytest.mark.asyncio
async def test_lenient_score_row_creates_nan_outputs_for_aggregateable_specs() -> None:
    metric = _RaisingMetric()

    _, metric_result, row = await score_row(
        metric=metric,
        row={"expected": "yes"},
        sample={"output_text": "no"},
        index=0,
        metric_key="metric",
        fail_fast=False,
        generation_requests=[],
    )

    assert metric_result is not None
    assert [output.name for output in metric_result.outputs] == ["score", "passed"]
    assert all(math.isnan(output.value) for output in metric_result.outputs)
    assert [output.name for output in row.metrics["metric"]] == ["score", "passed"]
    assert row.metric_errors == {"metric": "metric failed"}

    result = await finalize_evaluation_result(metric, [(0, metric_result, row)])

    assert [(score.name, score.count, score.nan_count) for score in result.aggregate_scores.scores] == [
        ("score", 0, 1),
        ("passed", 0, 1),
    ]


@pytest.mark.asyncio
async def test_benchmark_namespaces_output_rows_and_aggregates() -> None:
    rows = [{"expected": "yes"}, {"expected": "yes"}]

    result = await evaluate_benchmark(
        metrics=[("custom", _OutputMetric())],
        rows=rows,
        target=None,
        params=RunConfig(parallelism=1),
    )

    assert result.row_scores[0].metrics["custom"][0] == MetricOutput(name="score", value=0.0)
    assert result.to_records()[0]["output.custom.reason"] == "mismatched"
    assert [score.name for score in result.aggregate_scores.scores] == ["custom.score", "custom.passed"]


def test_metric_result_diagnostics_default_is_empty_list() -> None:
    # Existing metrics that only return outputs keep working unchanged.
    result = MetricResult(outputs=[MetricOutput(name="score", value=1.0)])
    assert result.diagnostics == []
    assert result.model_dump()["diagnostics"] == []

    enriched = MetricResult(
        outputs=[MetricOutput(name="score", value=0.0)],
        diagnostics=[MetricDiagnostic(message="mismatch", details={"diff": "- a\n+ b"})],
    )
    assert enriched.diagnostics == [MetricDiagnostic(message="mismatch", details={"diff": "- a\n+ b"})]


@pytest.mark.asyncio
async def test_score_row_preserves_metric_diagnostics_on_mismatch() -> None:
    metric = _DiagnosticMetric()
    expected_diagnostics = [
        MetricDiagnostic(
            message="mismatch",
            details={"expected": "yes", "actual": "no"},
        )
    ]

    _, metric_result, row = await score_row(
        metric=metric,
        row={"expected": "yes"},
        sample={"output_text": "no"},
        index=0,
        metric_key="metric",
        fail_fast=True,
        generation_requests=[],
    )

    assert metric_result is not None
    assert metric_result.diagnostics == expected_diagnostics
    assert row.metric_diagnostics == {"metric": expected_diagnostics}

    result = await finalize_evaluation_result(metric, [(0, metric_result, row)])
    record = result.to_records()[0]
    assert json.loads(record["diagnostics.metric"]) == [
        {
            "message": "mismatch",
            "details": {"expected": "yes", "actual": "no"},
        }
    ]


@pytest.mark.asyncio
async def test_score_row_omits_diagnostics_when_metric_attaches_none() -> None:
    metric = _DiagnosticMetric()

    _, metric_result, row = await score_row(
        metric=metric,
        row={"expected": "yes"},
        sample={"output_text": "yes"},
        index=0,
        metric_key="metric",
        fail_fast=True,
        generation_requests=[],
    )

    assert metric_result is not None
    assert metric_result.diagnostics == []
    assert row.metric_diagnostics is None

    result = await finalize_evaluation_result(metric, [(0, metric_result, row)])
    assert "diagnostics.metric" not in result.to_records()[0]


@pytest.mark.asyncio
async def test_benchmark_namespaces_metric_diagnostics() -> None:
    rows = [{"expected": "yes"}]

    result = await evaluate_benchmark(
        metrics=[("custom", _DiagnosticMetric())],
        rows=rows,
        target=None,
        params=RunConfig(parallelism=1),
    )

    assert result.row_scores[0].metric_diagnostics == {
        "custom": [
            MetricDiagnostic(
                message="mismatch",
                details={"expected": "yes", "actual": None},
            )
        ]
    }
    assert json.loads(result.to_records()[0]["diagnostics.custom"]) == [
        {
            "message": "mismatch",
            "details": {"expected": "yes", "actual": None},
        }
    ]
