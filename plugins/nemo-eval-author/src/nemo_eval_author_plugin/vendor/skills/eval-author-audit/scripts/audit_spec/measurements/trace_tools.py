#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ATIF trace helpers for deterministic audit measurement."""

from __future__ import annotations

from typing import Any, TypeAlias

from _values import optional_string

try:
    from harbor.models.trajectories import Trajectory
except ImportError:
    Trajectory = Any  # type: ignore[assignment,misc]

JsonObject: TypeAlias = dict[str, Any]


def collect_tool_calls(trajectory: Trajectory, *, trajectory_path: str = "$") -> list[JsonObject]:
    """Return all ATIF tool calls, including embedded subagent trajectories."""
    calls: list[JsonObject] = []
    trajectory_id = optional_string(trajectory.trajectory_id)
    for step_index, step in enumerate(trajectory.steps or [], start=1):
        step_id = step.step_id or step_index
        for tool_call in step.tool_calls or []:
            function_name = tool_call.function_name
            if not isinstance(function_name, str) or not function_name.strip():
                continue
            call: JsonObject = {
                "tool": function_name.strip(),
                "step_id": step_id,
                "trajectory_path": trajectory_path,
            }
            tool_call_id = optional_string(getattr(tool_call, "tool_call_id", None))
            if tool_call_id is not None:
                call["tool_call_id"] = tool_call_id
            if trajectory_id is not None:
                call["trajectory_id"] = trajectory_id
            calls.append(call)

    for index, subagent in enumerate(trajectory.subagent_trajectories or []):
        calls.extend(collect_tool_calls(subagent, trajectory_path=f"{trajectory_path}.subagent_trajectories[{index}]"))
    return calls


def tool_call_counts(tool_calls: list[JsonObject]) -> dict[str, int]:
    """Return observed tool call counts keyed by tool name."""
    counts: dict[str, int] = {}
    for call in tool_calls:
        tool = call["tool"]
        counts[tool] = counts.get(tool, 0) + 1
    return counts
