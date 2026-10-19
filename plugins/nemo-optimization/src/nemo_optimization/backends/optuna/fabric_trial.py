# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for the shared Fabric candidate evaluator."""

from __future__ import annotations

from nemo_optimization.fabric_evaluator import (
    FabricCandidateEvaluator,
    FabricTrialEvaluator,
    _build_metrics,
    _model_from_fabric,
    _runtime_agent_config,
    build_agent_eval_tasks,
    reduce_agent_eval_scores,
    resolve_mcp_server_paths,
)

__all__ = [
    "FabricCandidateEvaluator",
    "FabricTrialEvaluator",
    "_build_metrics",
    "_model_from_fabric",
    "_runtime_agent_config",
    "build_agent_eval_tasks",
    "reduce_agent_eval_scores",
    "resolve_mcp_server_paths",
]
