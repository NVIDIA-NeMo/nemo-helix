# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for metric-type filtering through the SDK against a real platform.

Verifies the route's custom-field filter actually works end-to-end: ``metric_type`` is a ``data.*``
entity field, so without the ``DataFilter`` translation the entity store 500s. Pure CRUD (create +
list) — no online target or IGW — so it only needs the host subprocess backend. Shares the evaluator-plugin
integration opt-in (``RUN_AGENT_EVAL_INTEGRATION``) and the session-scoped ``subprocess_platform``.
"""

from __future__ import annotations

import os
import uuid

import pytest
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.string_check import StringCheckMetric
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.sdk import NeMoPlatform
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RUN_AGENT_EVAL_INTEGRATION"),
        reason="opt-in; set RUN_AGENT_EVAL_INTEGRATION=1 to run (spins real nemo services platforms)",
    ),
]

WORKSPACE = "default"


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.timeout(300)
def test_metric_type_filter_narrows_listing(subprocess_platform: str) -> None:
    client = NeMoPlatform(base_url=subprocess_platform, max_retries=2)
    client_from_platform(client, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()

    exact = _unique("exact")
    strcheck = _unique("strcheck")
    client.evaluator.metrics.create(
        exact, metric=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    )
    client.evaluator.metrics.create(
        strcheck,
        metric=StringCheckMetric(
            operation="contains", left_template="{{sample.output_text}}", right_template="{{item.phrase}}"
        ),
    )

    # Server-side filter on metric_type (a data.* field) must narrow the listing — the whole point of
    # the DataFilter translation. Robust to other metrics the shared platform may hold.
    exact_only = client.evaluator.metrics.list(workspace=WORKSPACE, metric_type="exact-match")
    names = {m.name for m in exact_only.data}
    assert exact in names
    assert strcheck not in names
    assert all(m.metric_type == "exact-match" for m in exact_only.data)

    assert client.evaluator.metrics.list(workspace=WORKSPACE, metric_type="no-such-type").data == []
