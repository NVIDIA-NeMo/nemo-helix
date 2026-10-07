# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid metric bundle packager.

Packages each metric with the lightest representation that supports it: built-in
metric types are bundled inline (config-serialized, reconstructed from the metric
type union), and only metrics that cannot be reconstructed from configuration are
cloudpickled. This minimizes cloudpickle usage so built-in metrics avoid the
Python-version coupling of cloudpickle payloads — relevant when a remote service
runs a different interpreter than the submitter.
"""

from __future__ import annotations

from nemo_evals.shared.metric_bundles.bundles import (
    MetricBundlePackager,
    MetricBundlePayload,
    metric_bundle_packager_for_payload,
)
from nemo_evals.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager, inline_bundle_supported
from nhx_evals_sdk.metrics.protocol import Metric


class HybridMetricBundlePackager(MetricBundlePackager):
    """Inline built-in metrics; cloudpickle only metrics that require it.

    Applied per metric, so a mixed set bundles each metric independently: inline
    where the metric type is reconstructable, cloudpickle otherwise. Loading is
    dispatched by the stored payload kind, so hydration works regardless of which
    representation each metric used.
    """

    def __init__(self) -> None:
        """Build the delegate inline and cloudpickle packagers."""
        self._inline = InlineMetricBundlePackager()
        self._cloudpickle = CloudpickleMetricBundlePackager()

    def package(self, metric: Metric) -> MetricBundlePayload:
        """Inline the metric when its type is reconstructable; cloudpickle otherwise."""
        if inline_bundle_supported(metric):
            return self._inline.package(metric)
        return self._cloudpickle.package(metric)

    def load(self, payload: MetricBundlePayload) -> Metric:
        """Hydrate by dispatching to the packager registered for the payload kind."""
        return metric_bundle_packager_for_payload(payload).load(payload)
