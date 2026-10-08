# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cloudpickle-backed metric bundle implementation."""

from __future__ import annotations

import hashlib
import pickle
import platform
import sys
from typing import Annotated, Literal

import cloudpickle
from nemo_evals.config import get_config
from nemo_evals.shared.metric_bundles.bundles import (
    MetricBundlePackager,
    MetricBundlePayload,
    MetricBundlingError,
    register_metric_bundle_kind,
)
from nhx_evals_sdk.metrics.protocol import Metric
from pydantic import ConfigDict, Field, computed_field, field_validator

CLOUDPICKLE_KIND = "cloudpickle"
ALLOW_CLOUDPICKLE_METRICS_ENV_VAR = "NEMO_EVALS_ALLOW_INSECURE_CLOUDPICKLE_METRICS"
MAX_CLOUDPICKLE_PAYLOAD_BYTES = 10 * 1024 * 1024
CloudpickleBlob = Annotated[bytes, Field(min_length=1, max_length=MAX_CLOUDPICKLE_PAYLOAD_BYTES)]


def _format_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


def _validate_payload_size(blob: bytes) -> bytes:
    if len(blob) > MAX_CLOUDPICKLE_PAYLOAD_BYTES:
        raise MetricBundlingError(
            "cloudpickle metric payload is "
            f"{_format_bytes(len(blob))}; maximum allowed is {_format_bytes(MAX_CLOUDPICKLE_PAYLOAD_BYTES)}"
        )
    return blob


class CloudpickleMetricsDisabledError(MetricBundlingError):
    """Raised when a cloudpickle metric reaches a deployment that has not enabled them."""


def require_cloudpickle_metrics_allowed() -> None:
    """Refuse cloudpickle metrics unless the deployment has opted in.

    Raises:
        CloudpickleMetricsDisabledError: ``allow_insecure_cloudpickle_metrics`` is not enabled.
    """
    if not get_config().allow_insecure_cloudpickle_metrics:
        raise CloudpickleMetricsDisabledError(
            "cloudpickle metrics are disabled on this deployment because they run caller-supplied Python "
            "with the evaluator's service credentials. Bundle built-in metrics inline instead "
            "(HybridMetricBundlePackager keeps them inline), or ask an operator to enable "
            "`evaluator.allow_insecure_cloudpickle_metrics` and set "
            f"{ALLOW_CLOUDPICKLE_METRICS_ENV_VAR}=true in the job executor environment."
        )


def _python_major_minor(version: str) -> tuple[int, int]:
    try:
        major, minor, *_ = version.split(".")
        return int(major), int(minor)
    except ValueError as exc:
        raise MetricBundlingError(f"invalid cloudpickle payload python_version: {version!r}") from exc


def _validate_python_version(payload: CloudpickleMetricPayload) -> None:
    payload_version = _python_major_minor(payload.python_version)
    runtime_version = sys.version_info[:2]
    if payload_version != runtime_version:
        raise MetricBundlingError(
            "cloudpickle metric payload was created with "
            f"Python {payload.python_version}, but this runtime is Python {platform.python_version()}; "
            "recreate the metric bundle with the runtime Python version."
        )


class CloudpickleMetricPayload(MetricBundlePayload):
    """Cloudpickle payload for an executable metric object."""

    model_config = ConfigDict(extra="ignore", ser_json_bytes="base64", val_json_bytes="base64")

    python_version: str
    cloudpickle_version: str
    pickle_protocol: int
    blob: CloudpickleBlob

    @field_validator("blob")
    @classmethod
    def _blob_must_fit_step_config(cls, blob: bytes) -> bytes:
        return _validate_payload_size(blob)

    @property
    def kind(self) -> Literal["cloudpickle"]:
        """Payload discriminator used by the metric bundle registry."""
        return CLOUDPICKLE_KIND

    @computed_field
    @property
    def digest(self) -> str:
        """Digest of the serialized metric payload."""
        return hashlib.sha256(bytes(self.blob)).hexdigest()

    @classmethod
    def from_blob(cls, blob: bytes) -> CloudpickleMetricPayload:
        """Create a JSON-safe cloudpickle payload from raw bytes."""
        _validate_payload_size(blob)
        return cls(
            python_version=platform.python_version(),
            cloudpickle_version=cloudpickle.__version__,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
            blob=blob,
        )


class CloudpickleMetricBundlePackager(MetricBundlePackager):
    """Cloudpickle-backed metric bundle packager.

    Cloudpickle bundles execute arbitrary Python code when hydrated. This
    implementation is intended for explicit opt-in development/MVP use.
    """

    def package(self, metric: Metric) -> MetricBundlePayload:
        """Package a runtime metric object as a cloudpickle payload."""
        if not isinstance(metric, Metric):
            raise MetricBundlingError("object does not satisfy the Metric protocol")

        blob = cloudpickle.dumps(metric, protocol=pickle.HIGHEST_PROTOCOL)
        return CloudpickleMetricPayload.from_blob(blob)

    def load(self, payload: MetricBundlePayload) -> Metric:
        """Hydrate a metric from a cloudpickle payload, if the deployment allows cloudpickle metrics."""
        require_cloudpickle_metrics_allowed()
        cloudpickle_payload = CloudpickleMetricPayload.model_validate(payload.model_dump(mode="python"))
        _validate_python_version(cloudpickle_payload)
        hydrated_metric = cloudpickle.loads(cloudpickle_payload.blob)
        if not isinstance(hydrated_metric, Metric):
            raise MetricBundlingError("unbundled object does not satisfy the Metric protocol")
        return hydrated_metric


register_metric_bundle_kind(
    CLOUDPICKLE_KIND,
    payload_type=CloudpickleMetricPayload,
    packager_factory=CloudpickleMetricBundlePackager,
)
