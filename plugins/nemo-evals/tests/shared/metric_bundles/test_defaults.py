# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import pytest
from nemo_evals.shared.metric_bundles.bundles import MetricBundlePackagerPolicyError
from nemo_evals.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evals.shared.metric_bundles.defaults import resolve_default_metric_bundle_packager
from nemo_evals.shared.metric_bundles.hybrid import HybridMetricBundlePackager
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.metrics.protocol import Metric, MetricInput, MetricOutput, MetricOutputSpec, MetricResult


class _CustomMetric:
    """A protocol-satisfying metric that is not part of MetricsUnion (not inline-bundleable)."""

    type = "custom-score"
    description = "custom metric"
    labels: dict[str, str] = {}

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


def _builtin() -> Metric:
    return ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")


def test_defaults_inline_for_builtin_and_raises_for_custom_submit() -> None:
    assert isinstance(
        resolve_default_metric_bundle_packager(_builtin(), None, allow_cloudpickle_fallback=False, action="Submitting"),
        InlineMetricBundlePackager,
    )
    with pytest.raises(MetricBundlePackagerPolicyError, match="CloudpickleMetricBundlePackager"):
        resolve_default_metric_bundle_packager(
            cast(Metric, _CustomMetric()), None, allow_cloudpickle_fallback=False, action="Submitting"
        )


def test_uses_hybrid_for_custom_local_run() -> None:
    assert isinstance(
        resolve_default_metric_bundle_packager(
            cast(Metric, _CustomMetric()), None, allow_cloudpickle_fallback=True, action="Running"
        ),
        HybridMetricBundlePackager,
    )


def test_uses_hybrid_for_mixed_local_run_but_inline_when_all_builtin() -> None:
    mixed = [_builtin(), cast(Metric, _CustomMetric())]
    assert isinstance(
        resolve_default_metric_bundle_packager(mixed, None, allow_cloudpickle_fallback=True, action="Running"),
        HybridMetricBundlePackager,
    )
    assert isinstance(
        resolve_default_metric_bundle_packager(
            [_builtin()], None, allow_cloudpickle_fallback=False, action="Submitting"
        ),
        InlineMetricBundlePackager,
    )


def test_honors_explicit_packager() -> None:
    explicit = CloudpickleMetricBundlePackager()
    assert (
        resolve_default_metric_bundle_packager(
            _builtin(), explicit, allow_cloudpickle_fallback=False, action="Submitting"
        )
        is explicit
    )
