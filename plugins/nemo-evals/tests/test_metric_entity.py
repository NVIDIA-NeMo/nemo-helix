# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialization round-trip tests for MetricBundleEntity.

The entity store persists an entity's custom fields with
``model_dump(exclude=base, mode="json")`` into a JSON column and rebuilds it with
``model_validate``. These tests exercise that exact round-trip (which the
in-memory fakes elsewhere bypass) to guard the complex fields — ``secrets``,
``outputs``, and ``labels`` — that carry non-trivial nested types.
"""

from __future__ import annotations

import json

from nemo_evals.entities import MetricBundleEntity
from nemo_evals.shared.metric_bundles.bundles import MetricBundle, bundle_metric
from nemo_evals.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nhx_evals_sdk.enums import ModelFormat
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.values import Model, SecretRef
from nhx_evals_sdk.values.scores import JSONScoreParser, RangeScore


def _entity(bundle: MetricBundle) -> MetricBundleEntity:
    return MetricBundleEntity(
        name="m",
        workspace="default",
        metric_type=bundle.metric_type,
        description=bundle.metadata.description,
        labels=bundle.metadata.labels,
        outputs=bundle.outputs,
        secrets=bundle.secrets,
        payload_kind=bundle.payload.kind,
        payload_digest=bundle.payload.digest,
        bundle_ref="default/metric-bundle.abc#bundle.json",
    )


def _roundtrip(entity: MetricBundleEntity) -> MetricBundleEntity:
    """Mirror the entity store: dump custom fields to JSON, then rebuild."""
    data = entity.model_dump(exclude=MetricBundleEntity.__base_fields__, exclude_computed_fields=True, mode="json")
    # Prove the data is genuinely JSON-serializable (the entity store uses a JSON column).
    data = json.loads(json.dumps(data))
    return MetricBundleEntity.model_validate({"name": entity.name, "workspace": entity.workspace, **data})


def test_roundtrip_preserves_simple_metric() -> None:
    bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
        CloudpickleMetricBundlePackager(),
    )
    entity = _entity(bundle)

    restored = _roundtrip(entity)

    assert restored.metric_type == entity.metric_type
    assert restored.outputs == entity.outputs
    assert restored.labels == entity.labels
    assert restored.payload_kind == entity.payload_kind
    assert restored.payload_digest == entity.payload_digest
    assert restored.bundle_ref == entity.bundle_ref


def test_roundtrip_preserves_secrets_and_metadata() -> None:
    metric = LLMJudgeMetric(
        model=Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            api_key_secret=SecretRef(root="judge-secret"),
            format=ModelFormat.OPEN_AI,
        ),
        scores=[RangeScore(name="helpfulness", minimum=1, maximum=5, parser=JSONScoreParser(json_path="helpfulness"))],
    )
    bundle = bundle_metric(metric, CloudpickleMetricBundlePackager())
    entity = _entity(bundle)
    assert entity.secrets  # sanity: this metric captures a secret reference

    restored = _roundtrip(entity)

    # dict[str, SecretRef] and the JSON-schema-bearing outputs must survive the column.
    assert restored.secrets == entity.secrets
    assert restored.outputs == entity.outputs
