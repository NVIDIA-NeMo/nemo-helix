# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from typing import Any

import nemo_evaluator.shared.metric_bundles.inline  # noqa: F401
import pytest
from jsonschema import Draft202012Validator
from nemo_evaluator.api.fields import MetricInline
from nemo_evaluator.shared.metric_bundles.bundles import (
    BundledMetricOutputSpec,
    MetricBundle,
    MetricBundlingError,
    bundle_metric,
    unbundle_metric,
)
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult


class _OptionalMetric:
    type = "optional-score"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score", required=False)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[])


class _RequiredMetric:
    type = "required-score"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        del input
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


def _bundle_data(*, version: str, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bundle_kind": "metric-bundle",
        "bundle_format_version": version,
        "metric_type": "required-score",
        "outputs": outputs,
        "payload": {
            "kind": "inline",
            "metric": {"type": "exact-match", "reference": "x", "candidate": "y"},
        },
    }


def _required_output(*, required: object = None) -> dict[str, Any]:
    output: dict[str, Any] = {
        "name": "score",
        "description": None,
        "value_json_schema": {"title": "ContinuousScore", "type": "number"},
    }
    if required is not None:
        output["required"] = required
    return output


def test_legacy_v1_output_omission_hydrates_as_required() -> None:
    bundle = MetricBundle.model_validate(_bundle_data(version="v1", outputs=[_required_output()]))

    assert bundle.outputs[0].required is True


def test_required_true_is_omitted_from_every_bundle_representation() -> None:
    output = BundledMetricOutputSpec.model_validate(_required_output(required=True))
    bundle = MetricBundle.model_validate(_bundle_data(version="v1", outputs=[_required_output(required=True)]))
    wire = MetricInline.model_validate_json(bundle.model_dump_json())

    assert "required" not in output.model_dump(mode="json")
    assert "required" not in bundle.model_dump(mode="json")["outputs"][0]
    assert "required" not in wire.model_dump(mode="json")["outputs"][0]
    assert '"required"' not in bundle.model_dump_json()
    assert '"required"' not in wire.model_dump_json()


def test_optional_output_round_trips_in_v1() -> None:
    bundle = MetricBundle.model_validate(_bundle_data(version="v1", outputs=[_required_output(required=False)]))
    wire = MetricInline.model_validate_json(bundle.model_dump_json())
    restored = MetricBundle.model_validate_json(wire.model_dump_json())

    assert bundle.bundle_format_version == "v1"
    assert bundle.outputs[0].required is False
    assert wire.outputs[0].required is False
    assert restored.outputs[0].required is False
    assert restored.model_dump(mode="json")["outputs"][0]["required"] is False


@pytest.mark.parametrize("malformed_required", ["false", "no", 0, 1])
def test_bundle_rejects_non_boolean_required_values(malformed_required: object) -> None:
    with pytest.raises(ValueError, match="required"):
        MetricBundle.model_validate(_bundle_data(version="v1", outputs=[_required_output(required=malformed_required)]))


@pytest.mark.parametrize("malformed_required", ["false", "no", 0, 1])
def test_metric_inline_rejects_non_boolean_required_values(malformed_required: object) -> None:
    with pytest.raises(ValueError, match="required"):
        MetricInline.model_validate(_bundle_data(version="v1", outputs=[_required_output(required=malformed_required)]))


@pytest.mark.parametrize("model", [MetricBundle, MetricInline])
def test_v2_bundle_format_is_rejected(model: type[MetricBundle] | type[MetricInline]) -> None:
    with pytest.raises(ValueError, match="bundle_format_version"):
        model.model_validate(_bundle_data(version="v2", outputs=[_required_output(required=False)]))


def test_runtime_optional_metric_bundle_is_v1_and_hydrates() -> None:
    bundle = bundle_metric(_OptionalMetric(), CloudpickleMetricBundlePackager())

    assert bundle.bundle_format_version == "v1"
    assert bundle.outputs[0].required is False
    restored = MetricBundle.model_validate_json(bundle.model_dump_json())
    assert unbundle_metric(restored).__class__ is _OptionalMetric


def _sorted_json_digest(bundle: MetricBundle) -> str:
    """The derived-metric identity digest, as ``MetricService.store_derived_metric`` computes it."""
    canonical = json.dumps(bundle.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_explicit_required_true_has_same_identity_as_omission() -> None:
    omitted = MetricBundle.model_validate(_bundle_data(version="v1", outputs=[_required_output()]))
    explicit = MetricBundle.model_validate(_bundle_data(version="v1", outputs=[_required_output(required=True)]))
    optional = MetricBundle.model_validate(_bundle_data(version="v1", outputs=[_required_output(required=False)]))

    assert explicit.model_dump_json() == omitted.model_dump_json()
    assert _sorted_json_digest(explicit) == _sorted_json_digest(omitted)
    assert _sorted_json_digest(optional) != _sorted_json_digest(omitted)


def test_hydration_output_contract_mismatch_reports_required() -> None:
    bundle = bundle_metric(_RequiredMetric(), CloudpickleMetricBundlePackager())
    mismatched = bundle.model_copy(
        update={
            "outputs": [BundledMetricOutputSpec.model_validate({**bundle.outputs[0].model_dump(), "required": False})],
        }
    )

    with pytest.raises(MetricBundlingError, match="required"):
        unbundle_metric(mismatched)


def test_metric_inline_serialization_schema_keeps_typed_output_contract() -> None:
    outputs_schema = MetricInline.model_json_schema(mode="serialization")["properties"]["outputs"]

    assert outputs_schema["items"] == {"$ref": "#/$defs/BundledMetricOutputSpec"}
    assert outputs_schema["minItems"] == 1


def test_metric_inline_json_schema_matches_required_contract() -> None:
    schema = MetricInline.model_json_schema(mode="validation")
    validator = Draft202012Validator(schema)
    output_schema = schema["$defs"]["BundledMetricOutputSpec"]

    assert validator.is_valid(_bundle_data(version="v1", outputs=[_required_output()]))
    assert validator.is_valid(_bundle_data(version="v1", outputs=[_required_output(required=True)]))
    assert validator.is_valid(_bundle_data(version="v1", outputs=[_required_output(required=False)]))
    assert not validator.is_valid(_bundle_data(version="v1", outputs=[_required_output(required="false")]))
    assert not validator.is_valid(_bundle_data(version="v2", outputs=[_required_output(required=False)]))
    assert "oneOf" not in schema
    assert output_schema["properties"]["required"]["type"] == "boolean"
    assert output_schema["properties"]["required"]["default"] is True
    assert "required" not in output_schema["required"]
