# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``client.insights.insights`` SDK sub-resource."""

from nemo_insights_plugin.entities import InsightStatus
from nemo_insights_plugin.sdk_resources.insights import _build_update_body


def test_update_insight_sdk_body_omits_none_fields() -> None:
    empty = _build_update_body(agent=None, description=None, status=None, trace_refs=None)
    partial = _build_update_body(
        agent=None,
        description="Updated description",
        status=InsightStatus.RESOLVED,
        trace_refs=[],
    )

    assert empty.model_dump(mode="json", exclude_unset=True) == {}
    assert partial.model_dump(mode="json", exclude_unset=True) == {
        "description": "Updated description",
        "status": "resolved",
        "trace_refs": [],
    }
