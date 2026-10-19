# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for restart policy → docker run kwargs."""

from __future__ import annotations

import pytest
from nemo_deployments_plugin.backends.docker.containers import DeploymentConfigError, restart_policy_kwargs
from nemo_deployments_plugin.types import RestartPolicy


@pytest.mark.parametrize(
    ("policy", "backoff", "expected"),
    [
        ("Always", 6, {"restart_policy": {"Name": "always"}}),
        ("OnFailure", 3, {"restart_policy": {"Name": "on-failure", "MaximumRetryCount": 3}}),
        ("Never", 6, {}),
    ],
)
def test_restart_policy_kwargs(policy: RestartPolicy, backoff: int, expected: dict) -> None:
    assert restart_policy_kwargs(policy, backoff) == expected


def test_restart_policy_kwargs_rejects_zero_on_failure_backoff() -> None:
    with pytest.raises(DeploymentConfigError, match="backoff_limit"):
        restart_policy_kwargs("OnFailure", 0)
