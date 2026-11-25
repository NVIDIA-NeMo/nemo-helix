# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator's copies of the Gym host's quantity checks answer as the Gym host's own do.

Evaluator plans with its copies because the ``nemo-helix`` wheel installs a released ``nemo-sandboxed-gym``
that lacks them; the Gym host applies the originals. A difference would let a job through that the host
then refuses, or refuse one it would run.
"""

from typing import Any

import pytest
from nemo_evals import sandbox_quantity
from sandboxed_gym import quantity
from sandboxed_gym.host import opensandbox

QUANTITIES = ["1", "30Gi", "512Mi", "32G", "1.5Gi", "1.5", "0", "0Gi", "-1Gi", "1e9", "30gi", "Gi", "", "30 Gi"]


def _outcome(function: Any, *args: Any) -> tuple[str, Any]:
    try:
        return ("value", function(*args))
    except ValueError as exc:
        return ("error", str(exc))


@pytest.mark.parametrize("value", QUANTITIES)
def test_quantity_bytes_matches_the_gym_hosts(value: str) -> None:
    assert _outcome(sandbox_quantity.quantity_bytes, value) == _outcome(quantity.quantity_bytes, value)


@pytest.mark.parametrize(
    ("resources", "create"),
    [
        (None, {}),
        ({"cpu": "2"}, {}),
        ({"ephemeral-storage": ""}, {}),
        ({"ephemeral-storage": "45Gi"}, {}),
        ({"ephemeral-storage": "45G"}, {}),
        ({"ephemeral-storage": "45Gi"}, {"resource": {"ephemeral-storage": "50Gi"}}),
        ({"ephemeral-storage": "45Gi"}, {"resource": {"ephemeral-storage": "40Gi"}}),
        ({"ephemeral-storage": "45.5Gi"}, {"resource": {"ephemeral-storage": "46Gi"}}),
        ({"ephemeral-storage": "lots"}, {}),
    ],
)
def test_host_disk_request_matches_the_gym_hosts(resources: dict[str, str] | None, create: dict[str, Any]) -> None:
    assert _outcome(sandbox_quantity.host_disk_request_gib, resources, create) == _outcome(
        opensandbox.host_disk_request_gib, resources, create
    )
