# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_evals.api.schemas import MetricInline
from nemo_evals.config import EvaluatorConfig
from nemo_evals.jobs.metric_resolution import to_runtime_bundle
from nemo_evals.shared.metric_bundles.bundles import bundle_metric, unbundle_metric
from nemo_evals.shared.metric_bundles.cloudpickle import (
    ALLOW_CLOUDPICKLE_METRICS_ENV_VAR,
    CloudpickleMetricBundlePackager,
    CloudpickleMetricsDisabledError,
)
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric


def _exact_match_bundle(packager):
    return bundle_metric(ExactMatchMetric(reference="a", candidate="a"), packager)


def test_cloudpickle_metrics_are_disabled_by_default() -> None:
    assert EvaluatorConfig().allow_insecure_cloudpickle_metrics is False


def test_load_refuses_cloudpickle_payload_without_deserializing_it(detonating_cloudpickle_metric: dict) -> None:
    bundle = to_runtime_bundle(MetricInline.model_validate(detonating_cloudpickle_metric))

    with pytest.raises(CloudpickleMetricsDisabledError, match="disabled on this deployment"):
        CloudpickleMetricBundlePackager().load(bundle.payload)


def test_unbundle_refuses_cloudpickle_bundle_by_default() -> None:
    with pytest.raises(CloudpickleMetricsDisabledError, match=ALLOW_CLOUDPICKLE_METRICS_ENV_VAR):
        unbundle_metric(_exact_match_bundle(CloudpickleMetricBundlePackager()))


@pytest.mark.usefixtures("allow_cloudpickle_metrics")
def test_unbundle_hydrates_cloudpickle_bundle_when_enabled() -> None:
    assert unbundle_metric(_exact_match_bundle(CloudpickleMetricBundlePackager())).type == "exact-match"


def test_inline_bundles_need_no_opt_in() -> None:
    assert unbundle_metric(_exact_match_bundle(InlineMetricBundlePackager())).type == "exact-match"


def test_job_workers_enable_cloudpickle_through_the_executor_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workers never see the platform config file, so the documented env var must reach this setting."""
    monkeypatch.setenv(ALLOW_CLOUDPICKLE_METRICS_ENV_VAR, "true")

    assert EvaluatorConfig().allow_insecure_cloudpickle_metrics is True
