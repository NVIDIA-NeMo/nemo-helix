#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite audit coverage for capability items."""

from __future__ import annotations

from typing import Any, TypeAlias

from measurements._composite import CompositeSpec, ToolGateSpec, measure_composite

try:
    from harbor.models.trajectories import Trajectory  # ty: ignore[unresolved-import]
except ImportError:
    Trajectory = Any  # type: ignore[assignment,misc]

JsonObject: TypeAlias = dict[str, Any]

_SPEC = CompositeSpec(
    method_name="capabilities",
    item_kind="capability",
    details_schema="nemo.eval_author.audit_capabilities_details.v1",
    judgments_schema="nemo.eval_author.audit_capability_judgments.v1",
    tool_gate=ToolGateSpec(
        item_field="required_tools",
        result_field="required_tool_results",
        should_be_observed=True,
        failure_reason="missing_required_tool",
    ),
)

METHOD_NAME = _SPEC.method_name
ITEM_KIND = _SPEC.item_kind
DETAILS_SCHEMA = _SPEC.details_schema
JUDGMENTS_SCHEMA = _SPEC.judgments_schema


def measure(audit: JsonObject, trajectory: Trajectory, *, judgments: JsonObject | None = None) -> JsonObject:
    """Measure capability items from deterministic trace evidence plus optional judgments."""
    return measure_composite(audit, trajectory, judgments=judgments, spec=_SPEC)
