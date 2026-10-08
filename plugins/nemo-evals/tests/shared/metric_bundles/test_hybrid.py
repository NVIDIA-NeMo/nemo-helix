# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

from nemo_evals.shared.metric_bundles.bundles import MetricBundle, bundle_metric, unbundle_metric
from nemo_evals.shared.metric_bundles.hybrid import HybridMetricBundlePackager
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.metrics.protocol import Metric, MetricInput, MetricOutput, MetricOutputSpec, MetricResult


class _CustomMetric:
    """A protocol-satisfying metric that is not inline-bundleable (module-level so it cloudpickles)."""

    type = "custom-score"
    description = "custom metric"
    labels: dict[str, str] = {}

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


def _roundtrip(bundle: MetricBundle) -> Metric:
    return unbundle_metric(MetricBundle.model_validate_json(bundle.model_dump_json()))


def test_hybrid_inlines_builtin_metric() -> None:
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")

    bundle = bundle_metric(metric, HybridMetricBundlePackager())

    assert bundle.payload.kind == "inline"
    assert type(_roundtrip(bundle)) is ExactMatchMetric


def test_hybrid_cloudpickles_custom_metric() -> None:
    bundle = bundle_metric(cast(Metric, _CustomMetric()), HybridMetricBundlePackager())

    assert bundle.payload.kind == "cloudpickle"
    assert isinstance(_roundtrip(bundle), _CustomMetric)


def test_hybrid_routes_each_metric_independently() -> None:
    """A mixed set bundles each metric with the lightest representation that supports it."""
    packager = HybridMetricBundlePackager()
    metrics: list[Metric] = [
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
        cast(Metric, _CustomMetric()),
    ]

    kinds = [bundle_metric(metric, packager).payload.kind for metric in metrics]

    assert kinds == ["inline", "cloudpickle"]


def test_hybrid_load_dispatches_by_payload_kind() -> None:
    packager = HybridMetricBundlePackager()
    inline_bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"), packager
    )
    cloudpickle_bundle = bundle_metric(cast(Metric, _CustomMetric()), packager)

    assert type(packager.load(inline_bundle.payload)) is ExactMatchMetric
    assert isinstance(packager.load(cloudpickle_bundle.payload), _CustomMetric)
