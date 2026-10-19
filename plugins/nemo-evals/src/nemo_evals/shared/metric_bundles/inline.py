# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inline metric bundle implementation.

The inline packager stores a built-in metric as its own JSON configuration
instead of a serialized code blob. The runtime reconstructs the metric from the
``MetricsUnion`` discriminated union (keyed on the metric ``type``), so no
arbitrary code is executed on load. This is the default bundler for metric types
the platform already recognizes; custom metric classes that are not part of
``MetricsUnion`` cannot be reconstructed from config and require the
``CloudpickleMetricBundlePackager`` instead.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, get_args

from nemo_evals.shared.metric_bundles.bundles import (
    MetricBundlePackager,
    MetricBundlePayload,
    MetricBundlingError,
    register_metric_bundle_kind,
)
from nhx_evals_sdk.metrics.protocol import Metric
from nhx_evals_sdk.metrics.types import MetricsUnion, MetricVariants
from pydantic import ConfigDict, TypeAdapter, computed_field, field_validator

INLINE_KIND = "inline"

# Discriminated union (keyed on ``type``) used to serialize and reconstruct the
# concrete built-in metric. Reconstruction is pure data validation — no code is
# executed — so inline bundles are safe to hydrate.
_METRIC_ADAPTER: TypeAdapter[Any] = TypeAdapter(MetricsUnion)

# Concrete metric classes that participate in ``MetricsUnion``. A metric must be
# an instance of one of these to be inline-bundleable.
_INLINE_SUPPORTED_TYPES: tuple[type, ...] = tuple(get_args(MetricVariants))


def inline_bundle_supported(metric: object) -> bool:
    """Return whether a metric can be bundled inline (reconstructed from config)."""
    return isinstance(metric, _INLINE_SUPPORTED_TYPES)


class InlineMetricPayload(MetricBundlePayload):
    """Payload storing a built-in metric as its JSON configuration."""

    model_config = ConfigDict(extra="ignore")

    metric: dict[str, Any]

    @field_validator("metric")
    @classmethod
    def _metric_must_declare_type(cls, value: dict[str, Any]) -> dict[str, Any]:
        metric_type = value.get("type")
        if not isinstance(metric_type, str) or not metric_type:
            raise MetricBundlingError("inline metric payload must include a non-empty 'type'")
        return value

    @property
    def kind(self) -> Literal["inline"]:
        """Payload discriminator used by the metric bundle registry."""
        return INLINE_KIND

    @computed_field
    @property
    def digest(self) -> str:
        """Digest of the canonical serialized metric configuration."""
        canonical = json.dumps(self.metric, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class InlineMetricBundlePackager(MetricBundlePackager):
    """Inline metric bundle packager.

    Serializes a built-in metric as JSON and reconstructs it from the metric
    type union on load. No arbitrary code is executed when hydrating, so this is
    the preferred default for platform-recognized metric types.
    """

    def package(self, metric: Metric) -> MetricBundlePayload:
        """Package a built-in metric object as its JSON configuration."""
        if not isinstance(metric, Metric):
            raise MetricBundlingError("object does not satisfy the Metric protocol")
        if not inline_bundle_supported(metric):
            raise MetricBundlingError(
                "inline metric bundling supports only built-in metric types; "
                "pass CloudpickleMetricBundlePackager() to bundle a custom metric."
            )
        data: dict[str, Any] = _METRIC_ADAPTER.dump_python(metric, mode="json")
        return InlineMetricPayload(metric=data)

    def load(self, payload: MetricBundlePayload) -> Metric:
        """Hydrate a metric from an inline payload by validating its configuration."""
        inline_payload = InlineMetricPayload.model_validate(payload.model_dump(mode="python"))
        hydrated_metric = _METRIC_ADAPTER.validate_python(inline_payload.metric)
        if not isinstance(hydrated_metric, Metric):
            raise MetricBundlingError("unbundled object does not satisfy the Metric protocol")
        return hydrated_metric


register_metric_bundle_kind(
    INLINE_KIND,
    payload_type=InlineMetricPayload,
    packager_factory=InlineMetricBundlePackager,
)
