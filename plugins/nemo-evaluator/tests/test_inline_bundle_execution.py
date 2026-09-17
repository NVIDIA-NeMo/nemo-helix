# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end execution tests for inline-bundled metrics.

These tests run real metric scoring — no mocks. The end-to-end cases drive the
full evaluator job (`EvaluateJob`) through its sync job entrypoint, exercising the
complete inline path: bundle -> MetricInline wire DTO -> job spec -> unbundle
(reconstruct from config) -> execute -> aggregate scores. The reconstruction
tests round-trip each metric through the bundle and then actually invoke the
hydrated metric's `compute_scores`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from nemo_evaluator.jobs.evaluate import EvaluateJob
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric, unbundle_metric
from nemo_evaluator.shared.metric_bundles.hybrid import HybridMetricBundlePackager
from nemo_evaluator.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evaluator_sdk.execution.samples import build_metric_input
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.f1 import F1Metric
from nemo_evaluator_sdk.metrics.number_check import NumberCheckMetric
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.metrics.string_check import StringCheckMetric
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults


class _CustomConstantMetric:
    """Module-level custom metric (cloudpicklable) that always scores 1.0."""

    type = "custom-constant"
    description = "custom constant metric"
    labels: dict[str, str] = {}

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("constant")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[MetricOutput(name="constant", value=1.0)])


def _inline_payload(metric: Metric) -> dict[str, Any]:
    """Bundle a metric inline and project it to the job-spec wire shape."""
    return bundle_metric(metric, InlineMetricBundlePackager()).model_dump(mode="json")


def _load_artifact_payload(run_result: dict[str, Any]) -> dict[str, Any]:
    artifact_path = Path(run_result["artifact"]["artifact_url"].removeprefix("file://"))
    return cast(dict[str, Any], json.loads(artifact_path.read_text(encoding="utf-8")))


def _aggregate_scores(run_result: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], _load_artifact_payload(run_result)["aggregate_scores"]["scores"])


def _sync_client() -> NemoClient:
    return NemoClient(
        base_url="http://platform.test",
        workspace="dev",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500, request=request))),
    )


def _job_context(tmp_path: Path) -> JobContext:
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    return JobContext(
        workspace="dev",
        storage=storage,
        results=LocalJobResults(root=storage.persistent / "results"),
    )


def _run_evaluate_job(spec: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    return EvaluateJob().run(spec, ctx=_job_context(tmp_path), client=_sync_client())


def test_evaluate_job_runs_inline_bundled_exact_match_metric(tmp_path: Path) -> None:
    """Full job run with an inline-bundled metric produces real aggregate scores."""
    spec = {
        "metrics": [
            _inline_payload(ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}"))
        ],
        "dataset": [
            {"expected": "blue", "model_output": "Blue"},  # normalizes equal -> 1.0
            {"expected": "Jupiter", "model_output": "Saturn"},  # -> 0.0
        ],
        "params": {"parallelism": 2},
    }

    result = _run_evaluate_job(spec, tmp_path)

    scores = _aggregate_scores(result)
    assert scores[0]["name"] == "exact-match.exact-match"
    assert scores[0]["mean"] == 0.5


def test_evaluate_job_runs_multiple_inline_metrics(tmp_path: Path) -> None:
    """Multiple inline-bundled metrics in one job each execute and aggregate."""
    spec = {
        "metrics": [
            _inline_payload(ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}")),
            _inline_payload(
                StringCheckMetric(
                    operation="contains",
                    left_template="{{item.model_output}}",
                    right_template="{{item.expected}}",
                )
            ),
        ],
        "dataset": [
            {"expected": "Paris", "model_output": "Paris"},
            {"expected": "Paris", "model_output": "London"},
        ],
        "params": {"parallelism": 2},
    }

    result = _run_evaluate_job(spec, tmp_path)

    by_name = {score["name"]: score for score in _aggregate_scores(result)}
    assert by_name["exact-match.exact-match"]["mean"] == 0.5
    assert by_name["string-check.string-check"]["mean"] == 0.5


def test_evaluate_job_runs_hybrid_bundled_mixed_metrics(tmp_path: Path) -> None:
    """Hybrid bundling: built-in goes inline, custom is cloudpickled, and both execute in one job."""
    packager = HybridMetricBundlePackager()
    builtin_payload = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}"), packager
    ).model_dump(mode="json")
    custom_payload = bundle_metric(cast(Metric, _CustomConstantMetric()), packager).model_dump(mode="json")

    # The built-in stays inline (no Python-version coupling); only the custom metric is cloudpickled.
    assert builtin_payload["payload"]["kind"] == "inline"
    assert custom_payload["payload"]["kind"] == "cloudpickle"

    spec = {
        "metrics": [builtin_payload, custom_payload],
        "dataset": [
            {"expected": "blue", "model_output": "Blue"},  # exact-match -> 1.0
            {"expected": "Jupiter", "model_output": "Saturn"},  # exact-match -> 0.0
        ],
        "params": {"parallelism": 2},
    }

    result = _run_evaluate_job(spec, tmp_path)

    by_name = {score["name"]: score for score in _aggregate_scores(result)}
    assert by_name["exact-match.exact-match"]["mean"] == 0.5
    assert by_name["custom-constant.constant"]["mean"] == 1.0


@pytest.mark.asyncio
async def test_round_tripped_deterministic_metrics_execute_identically() -> None:
    """After an inline bundle round-trip, hydrated metrics score identically to the originals."""
    item = {"expected": "the answer is 42", "left": "42", "right": "42.0"}
    sample = {"output_text": "The answer is 42!"}

    metrics: list[Metric] = [
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{sample.output_text}}"),
        F1Metric(reference="{{item.expected}}", candidate="{{sample.output_text}}"),
        NumberCheckMetric(operation="equals", left_template="{{item.left}}", right_template="{{item.right}}"),
        StringCheckMetric(
            operation="contains", left_template="{{sample.output_text}}", right_template="{{item.expected}}"
        ),
    ]

    for metric in metrics:
        hydrated = unbundle_metric(bundle_metric(metric, InlineMetricBundlePackager()))

        original_result = await metric.compute_scores(build_metric_input(item, sample))
        hydrated_result = await hydrated.compute_scores(build_metric_input(item, sample))

        original_values = [(o.name, o.value) for o in original_result.outputs]
        hydrated_values = [(o.name, o.value) for o in hydrated_result.outputs]
        assert hydrated_values == original_values, type(metric).__name__
        # Sanity: the deterministic scorers actually produced a score.
        assert hydrated_result.outputs, type(metric).__name__
