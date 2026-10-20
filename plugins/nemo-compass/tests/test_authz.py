# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for Compass plugin.

Compass job persists its change-set through a task client that calls as
``service:compass`` on behalf of the run's submitter. The policy denies a
service principal on any route whose ``callers`` omit ``service_principal``, so
every Insight route the job touches must admit both caller kinds.
"""

from __future__ import annotations

import pytest
from nemo_compass_plugin.service import CompassService
from nemo_helix_plugin.authz import AuthzContribution
from nemo_helix_plugin.authz_discovery import _derive_service_contribution

_BASE = "/apis/compass/v2/workspaces/{workspace}"


def _contribution() -> AuthzContribution:
    contrib, problems, _warnings = _derive_service_contribution(CompassService())
    assert problems == [], problems
    return contrib


@pytest.mark.parametrize(
    ("path", "method"),
    [
        (f"{_BASE}/insights", "get"),
        (f"{_BASE}/insights", "post"),
        (f"{_BASE}/insights/{{insight_id}}", "get"),
        (f"{_BASE}/insights/{{insight_id}}", "patch"),
    ],
)
def test_compass_job_routes_admit_service_principal(path: str, method: str) -> None:
    binding = _contribution().endpoints[path][method]
    assert binding.callers == ["principal", "service_principal"]


def test_insight_delete_stays_principal_only() -> None:
    binding = _contribution().endpoints[f"{_BASE}/insights/{{insight_id}}"]["delete"]
    assert binding.callers == ["principal"]
