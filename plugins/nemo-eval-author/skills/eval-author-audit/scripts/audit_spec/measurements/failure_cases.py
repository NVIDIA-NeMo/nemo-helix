#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite audit coverage for failure-case items."""

from __future__ import annotations

from typing import Any, TypeAlias

from measurements._composite import CompositeSpec, ToolGateSpec, measure_composite

try:
    from harbor.models.trajectories import Trajectory  # ty: ignore[unresolved-import]
except ImportError:
    Trajectory = Any  # type: ignore[assignment,misc]

JsonObject: TypeAlias = dict[str, Any]

_SPEC = CompositeSpec(
    method_name="failure_cases",
    item_kind="failure_case",
    details_schema="nemo.eval_author.audit_failure_cases_details.v1",
    judgments_schema="nemo.eval_author.audit_failure_case_judgments.v1",
    tool_gate=ToolGateSpec(
        item_field="prohibited_tools",
        result_field="prohibited_tool_results",
        should_be_observed=False,
        failure_reason="prohibited_tool_observed",
    ),
)

METHOD_NAME = _SPEC.method_name
ITEM_KIND = _SPEC.item_kind
DETAILS_SCHEMA = _SPEC.details_schema
JUDGMENTS_SCHEMA = _SPEC.judgments_schema


def measure(audit: JsonObject, trajectory: Trajectory, *, judgments: JsonObject | None = None) -> JsonObject:
    """Measure failure cases from prohibited tools, trace evidence, and optional judgments."""
    return measure_composite(audit, trajectory, judgments=judgments, spec=_SPEC)
