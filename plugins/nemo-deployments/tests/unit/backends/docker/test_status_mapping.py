# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for docker status mapping helpers."""

from __future__ import annotations

import pytest
from nemo_deployments_plugin.backends.docker.status import (
    map_exited_status,
    missing_container_status,
)
from nemo_deployments_plugin.types import RestartPolicy


@pytest.mark.parametrize(
    ("exit_code", "restart_policy", "expected"),
    [
        (0, "Never", "SUCCEEDED"),
        (0, "Always", "SUCCEEDED"),
        (1, "Never", "FAILED"),
        (1, "Always", "FAILED"),
    ],
)
def test_map_exited_status(exit_code: int, restart_policy: RestartPolicy, expected: str) -> None:
    assert map_exited_status(exit_code, restart_policy) == expected


def test_missing_container_lost_for_always() -> None:
    update = missing_container_status("Always", container_name="dep-default-srv")
    assert update.status == "LOST"


def test_missing_container_failed_for_never() -> None:
    update = missing_container_status("Never", container_name="dep-default-job")
    assert update.status == "FAILED"
