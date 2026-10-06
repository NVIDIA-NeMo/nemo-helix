# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest
from nemo_evaluator.shared.metric_bundles.bundles import MetricBundlingError, bundle_metric, unbundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import (
    CloudpickleMetricBundlePackager,
    CloudpickleMetricPayload,
    allow_cloudpickle_loading,
    cloudpickle_loading_allowed,
)
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric


class _Detonator:
    def __reduce__(self):
        return (pytest.fail, ("cloudpickle payload was deserialized",))


def test_load_refuses_cloudpickle_payload_by_default() -> None:
    payload = CloudpickleMetricPayload.from_blob(pickle.dumps(_Detonator()))

    with pytest.raises(MetricBundlingError, match="only loaded inside an evaluation job"):
        CloudpickleMetricBundlePackager().load(payload)


def test_unbundle_refuses_cloudpickle_bundle_by_default() -> None:
    bundle = bundle_metric(ExactMatchMetric(reference="a", candidate="a"), CloudpickleMetricBundlePackager())

    with pytest.raises(MetricBundlingError, match="only loaded inside an evaluation job"):
        unbundle_metric(bundle)


def test_allow_cloudpickle_loading_is_scoped() -> None:
    bundle = bundle_metric(ExactMatchMetric(reference="a", candidate="a"), CloudpickleMetricBundlePackager())

    with allow_cloudpickle_loading():
        assert unbundle_metric(bundle).type == "exact-match"

    assert not cloudpickle_loading_allowed()
    with pytest.raises(MetricBundlingError):
        unbundle_metric(bundle)
