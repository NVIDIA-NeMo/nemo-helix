# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assign resource names once, at workflow creation.

The base is the workflow name. Walking the expanded graph in template order, every
resource a node will create takes the next free suffix: ``{workflow}-t1``, ``-t2``.
A retry reuses the suffixes stored on the node. Nothing calls ``sanitize_name`` and
nothing reads a generated name back off a job.

A deployment name is stored with a ``-v{version}`` suffix and the deployments plugin
derives ``{name}-weights``, so the assigned name stays at or under
:data:`~nemo_distillation_plugin.models.MAX_RESOURCE_NAME_LEN`.
"""

from __future__ import annotations

import re

from nemo_distillation_plugin.models import MAX_RESOURCE_NAME_LEN, GraphError, GraphNode

#: Short enough that ``{name}-t99`` and ``{name}-data`` fit the resource and fileset limits.
WORKFLOW_NAME = re.compile(r"^[a-z](?!.*--)[a-z0-9-]{0,30}[a-z0-9]$")
CANDIDATE_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$")


def check_workflow_name(name: str) -> None:
    if WORKFLOW_NAME.fullmatch(name) is None:
        raise GraphError(f"workflow name {name!r} must be 2-32 characters of lowercase letters, digits, and hyphens")


def check_candidate_name(name: str) -> None:
    if CANDIDATE_NAME.fullmatch(name) is None or "--" in name:
        raise GraphError(f"candidate name {name!r} must be a lowercase identifier")


def allocate(workflow_name: str, graph: dict[str, GraphNode]) -> dict[str, dict[str, str]]:
    """Map each node key to the resource names it will create. The key order is the graph order."""
    assigned: dict[str, dict[str, str]] = {}
    number = 1
    for key, node in graph.items():
        external: dict[str, str] = {}
        for resource in _resources(key, node):
            name = f"{workflow_name}-t{number}"
            if len(name) > MAX_RESOURCE_NAME_LEN:
                raise GraphError(
                    f"{name} is {len(name)} characters; assigned names must be at most {MAX_RESOURCE_NAME_LEN}"
                )
            external[resource] = name
            number += 1
        assigned[key] = external
    return assigned


def _resources(key: str, node: GraphNode) -> list[str]:
    if node.kind == "convert_atif":
        return ["job"]
    if node.kind == "customization_job":
        return ["job", "model"]
    if node.kind == "model_deployment":
        return ["config", "deployment"]
    if node.kind == "agent_eval":
        return ["agent", "job"]
    if node.kind == "delete_deployment":
        return []
    if node.kind == "experiment":
        arms = node.inputs.get("arms")
        if not isinstance(arms, list) or not arms or not all(isinstance(arm, str) and arm for arm in arms):
            raise GraphError(f"{key} needs a non-empty arms list")
        return ["experiment", *[f"evaluation_{arm}" for arm in arms]]
    raise GraphError(f"unknown kind {node.kind!r} on {key}")
