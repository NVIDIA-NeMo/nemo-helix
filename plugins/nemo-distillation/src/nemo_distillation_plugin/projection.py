# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derived ``stages`` and ``results`` on a workflow.

``results`` is filled once the three eval groups have succeeded. It is not a node.
"""

from __future__ import annotations

from typing import Any

from nemo_distillation_plugin.models import RuntimeNode, Workflow

_STAGES: tuple[tuple[str, str, str], ...] = (
    ("prepare", "Prepare", "prepare"),
    ("customization", "Customization", "train"),
    ("eval-original", "Original eval", "eval_original"),
    ("eval-base", "Base eval", "eval_base"),
    ("eval-distilled", "Distilled eval", "eval_distilled"),
)


def project(workflow: Workflow) -> None:
    """Write the derived views onto the entity. Tick and the read routes both call this."""
    workflow.stages = stages(workflow)
    workflow.results = results(workflow)


def stages(workflow: Workflow) -> list[dict[str, Any]]:
    view: list[dict[str, Any]] = []
    for stage_id, title, prefix in _STAGES:
        keys = _keys(workflow, prefix)
        nodes = [workflow.nodes[key] for key in keys]
        status, progress = _stage_status(nodes)
        job_ref = next((node.external.get("job") for node in nodes if node.external.get("job")), None)
        metrics = next((dict(node.outputs) for node in nodes if node.outputs), {})
        view.append(
            {
                "id": stage_id,
                "name": title,
                "status": status,
                "progress": progress,
                "metrics": metrics,
                "job_ref": job_ref,
            }
        )
    return view


def results(workflow: Workflow) -> dict[str, Any] | None:
    original = _nodes(workflow, "eval_original")
    base = _nodes(workflow, "eval_base")
    distilled = _nodes(workflow, "eval_distilled")
    groups = (original, base, distilled)
    if any(not group for group in groups):
        return None
    if any(node.status != "succeeded" for group in groups for node in group):
        return None
    prices = workflow.inputs.get("prices") if isinstance(workflow.inputs, dict) else None
    return {
        "original": _arm(original[0]),
        "base": _arm(base[0]),
        "distilled": _arm(distilled[0]),
        "prices": prices or {},
    }


def descendants(graph: dict, key: str) -> set[str]:
    """Node keys that depend on ``key``, directly or further down the graph."""
    found: set[str] = set()
    pending = [key]
    while pending:
        current = pending.pop()
        for child, node in graph.items():
            if current in node.depends_on and child not in found:
                found.add(child)
                pending.append(child)
    return found


def _keys(workflow: Workflow, prefix: str) -> list[str]:
    return [key for key in workflow.graph if key == prefix or key.startswith(prefix + "[")]


def _nodes(workflow: Workflow, prefix: str) -> list[RuntimeNode]:
    return [workflow.nodes[key] for key in _keys(workflow, prefix)]


def _stage_status(nodes: list[RuntimeNode]) -> tuple[str, int]:
    statuses = [node.status for node in nodes]
    if not statuses:
        return "pending", 0
    if any(status == "failed" for status in statuses) or any(status == "skipped" for status in statuses):
        return "failed", 100
    if any(status == "running" for status in statuses):
        return "active", 50
    if all(status == "succeeded" for status in statuses):
        return "completed", 100
    if any(status == "cancelled" for status in statuses):
        return "failed", 100
    return "pending", 0


def _arm(node: RuntimeNode) -> dict[str, Any]:
    outputs = node.outputs
    return {
        "score": outputs.get("score"),
        "input_tokens_per_task": outputs.get("input_tokens_per_task"),
        "output_tokens_per_task": outputs.get("output_tokens_per_task"),
        "median_latency_sec": outputs.get("median_latency_sec"),
        "estimated_cost_per_1k_tasks": outputs.get("estimated_cost_per_1k_tasks"),
        "eval_job": node.external.get("job"),
    }
