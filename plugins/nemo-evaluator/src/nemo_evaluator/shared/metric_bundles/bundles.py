# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral metric bundle models and protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, cast

from nemo_evaluator_sdk.metrics.protocol import (
    Metric,
    MetricOutputSpec,
    MetricWithSecrets,
)
from nemo_evaluator_sdk.values.common import SecretRef
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    StrictBool,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

BundleMetricTypeName = Annotated[str, StringConstraints(min_length=1)]


class MetricBundlingError(ValueError):
    """Raised when a metric cannot be bundled or hydrated."""


class MetricBundlePackagerPolicyError(RuntimeError):
    """Raised when metric packaging is not configured for an operation."""


class MetricMetadata(BaseModel):
    """User-facing metadata captured with a bundled metric."""

    model_config = ConfigDict(extra="allow", revalidate_instances="never")

    description: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class BundledMetricOutputSpec(BaseModel):
    """JSON-safe projection of a runtime metric output spec."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    value_json_schema: dict[str, Any]
    required: StrictBool = Field(default=True, exclude_if=lambda value: value is True)

    @classmethod
    def from_output_spec(cls, output: MetricOutputSpec) -> BundledMetricOutputSpec:
        """Capture the serializable contract for one runtime output."""
        return cls(
            name=output.name,
            description=output.description,
            value_json_schema=output.value_json_schema(),
            required=output.required,
        )


class MetricBundlePayload(BaseModel, ABC):
    """Base class for concrete Pydantic metric bundle payloads."""

    @property
    @abstractmethod
    def kind(self) -> str:
        """Payload discriminator used to select the packager implementation."""
        ...

    @property
    @abstractmethod
    def digest(self) -> str:
        """Format-specific digest for the payload contents."""
        ...


class MetricBundlePackager(Protocol):
    """Strategy for packaging a runtime metric into a bundle payload and loading it later."""

    def package(self, metric: Metric) -> MetricBundlePayload:
        """Package a runtime metric object into a format-specific payload."""
        ...

    def load(self, payload: MetricBundlePayload) -> Metric:
        """Hydrate an executable metric from a bundle payload."""
        ...


@dataclass(frozen=True)
class _MetricBundleRegistration:
    payload_type: type[MetricBundlePayload]
    packager_factory: Callable[[], MetricBundlePackager]


_BUNDLE_REGISTRY: dict[str, _MetricBundleRegistration] = {}


def _payload_kind(payload: MetricBundlePayload) -> str:
    kind = payload.kind
    if not kind:
        raise MetricBundlingError("metric bundle payload kind must not be empty")
    return kind


def register_metric_bundle_kind(
    kind: str,
    *,
    payload_type: type[MetricBundlePayload],
    packager_factory: Callable[[], MetricBundlePackager],
) -> None:
    """Register the payload model and packager factory for a bundle kind."""
    if not kind:
        raise ValueError("metric bundle payload kind must not be empty")
    registration = _MetricBundleRegistration(
        payload_type=payload_type,
        packager_factory=packager_factory,
    )
    existing = _BUNDLE_REGISTRY.get(kind)
    if existing is not None:
        if existing == registration:
            return
        raise ValueError(f"metric bundle payload kind already registered: {kind}")
    _BUNDLE_REGISTRY[kind] = registration


class MetricBundle(BaseModel):
    """Standalone executable metric bundle entity used by backend execution."""

    model_config = ConfigDict(extra="forbid")

    bundle_kind: Literal["metric-bundle"] = "metric-bundle"
    bundle_format_version: Literal["v1"] = "v1"
    metric_type: BundleMetricTypeName
    metadata: MetricMetadata = Field(default_factory=MetricMetadata)
    outputs: list[BundledMetricOutputSpec] = Field(min_length=1)
    secrets: dict[str, SecretRef] = Field(default_factory=dict)
    payload: SerializeAsAny[MetricBundlePayload]

    @field_serializer("payload")
    def _serialize_payload(self, payload: MetricBundlePayload) -> dict[str, Any]:
        value = payload.model_dump(mode="json")
        value["kind"] = _payload_kind(payload)
        return value

    @field_validator("payload", mode="before")
    @classmethod
    def _payload_must_have_kind(cls, value: object) -> object:
        if isinstance(value, MetricBundlePayload):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("metric bundle payload must be an object")
        payload_data = cast(Mapping[str, object], value)
        kind = payload_data.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("metric bundle payload must include a non-empty kind")
        registration = _BUNDLE_REGISTRY.get(kind)
        if registration is None:
            raise ValueError(f"unsupported metric bundle payload kind: {kind}")
        payload_fields = {
            field_name: field_value for field_name, field_value in payload_data.items() if field_name != "kind"
        }
        return registration.payload_type.model_validate(payload_fields)

    @model_validator(mode="after")
    def _output_names_must_be_unique(self) -> MetricBundle:
        names = [output.name for output in self.outputs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate metric output names: {duplicates}")
        return self


def metric_bundle_packager_for_payload(payload: MetricBundlePayload) -> MetricBundlePackager:
    """Create the packager registered for a metric bundle payload."""
    kind = _payload_kind(payload)
    registration = _BUNDLE_REGISTRY.get(kind)
    if registration is None:
        raise MetricBundlingError(f"unsupported metric bundle payload kind: {kind}")
    return registration.packager_factory()


def bundle_metric(metric: Metric, packager: MetricBundlePackager) -> MetricBundle:
    """Build a standard metric bundle envelope around a format-specific payload."""
    if not isinstance(metric, Metric):
        raise MetricBundlingError("object does not satisfy the Metric protocol")
    payload = packager.package(metric)
    return MetricBundle(
        metric_type=validate_metric_type(metric),
        metadata=metric_metadata(metric),
        outputs=[BundledMetricOutputSpec.from_output_spec(output) for output in metric.output_spec()],
        secrets=metric_secrets(metric),
        payload=payload,
    )


def unbundle_metric(bundle: MetricBundle) -> Metric:
    """Hydrate a runtime metric from a standard metric bundle envelope."""
    packager = metric_bundle_packager_for_payload(bundle.payload)
    hydrated_metric = packager.load(bundle.payload)
    _validate_metric_matches_bundle(hydrated_metric, bundle)
    return hydrated_metric


def _validate_metric_matches_bundle(metric: object, bundle: MetricBundle) -> None:
    """Validate that bundle metadata still describes the hydrated metric."""
    if not isinstance(metric, Metric):
        raise MetricBundlingError("unbundled object does not satisfy the Metric protocol")

    hydrated_outputs = [BundledMetricOutputSpec.from_output_spec(output) for output in metric.output_spec()]
    if hydrated_outputs != bundle.outputs:
        mismatches = {
            field
            for expected, actual in zip(bundle.outputs, hydrated_outputs, strict=False)
            for field in BundledMetricOutputSpec.model_fields
            if getattr(expected, field) != getattr(actual, field)
        }
        suffix = f" (mismatched fields: {sorted(mismatches)})" if mismatches else ""
        raise MetricBundlingError(f"unbundled metric output spec does not match bundle metadata{suffix}")
    if validate_metric_type(metric) != bundle.metric_type:
        raise MetricBundlingError("unbundled metric type does not match bundle metadata")


def validate_metric_type(metric: Metric) -> str:
    """Return the runtime metric type after validating the protocol contract."""
    value = metric.type
    if not isinstance(value, str):
        raise MetricBundlingError("metric type must be a string")
    if not value:
        raise MetricBundlingError("metric type must not be empty")
    return value


def metric_metadata(metric: Metric) -> MetricMetadata:
    """Capture optional runtime metric metadata."""
    description = getattr(metric, "description", None)
    if description is not None and not isinstance(description, str):
        raise MetricBundlingError("metric description must be a string when provided")

    raw_labels = getattr(metric, "labels", None) or {}
    if not isinstance(raw_labels, Mapping):
        raise MetricBundlingError("metric labels must be a mapping when provided")
    labels = dict(raw_labels)
    return MetricMetadata(description=description, labels=labels)


def metric_secrets(metric: Metric) -> dict[str, SecretRef]:
    """Capture secret environment mappings needed to execute one metric."""
    if not isinstance(metric, MetricWithSecrets):
        return {}
    return metric.secrets()
