# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse, validate, and expand a workflow graph.

Expansion walks the template in insertion order, which is also dependency order.
A ``matrix`` of candidates becomes ``key[candidate]``. ``matrix: inherit`` fans out
over the candidate of an already-expanded dependency. A node without a matrix that
depends on an expanded node depends on every expanded key.
"""

from __future__ import annotations

import re
from typing import Any

from nemo_distillation_plugin.models import NODE_KINDS, GraphError, GraphNode

#: A binding is the entire string. The executor substitutes the object, not a rendering of it.
BINDING = re.compile(r"^\$\{nodes\.([^.]+)\.(outputs|external)\.([^}]+)\}$")


def expand(nodes: dict[str, GraphNode]) -> dict[str, GraphNode]:
    """Return the graph with matrix fan-out applied and bindings rewritten onto expanded keys."""
    produced: dict[str, list[tuple[str | None, str]]] = {}
    expanded: dict[str, GraphNode] = {}
    for key, node in nodes.items():
        choices = _choices(key, node, produced)
        group: list[tuple[str | None, str]] = []
        for choice in choices:
            new_key = key if choice is None else f"{key}[{choice}]"
            if new_key in expanded:
                raise GraphError(f"duplicate node {new_key}")
            expanded[new_key] = node.model_copy(
                update={
                    "matrix": None,
                    "depends_on": _remap(key, node.depends_on, choice, produced),
                    "inputs": _rewrite(node.inputs, choice, produced),
                }
            )
            group.append((choice, new_key))
        produced[key] = group
    validate(expanded)
    return expanded


def validate(nodes: dict[str, GraphNode]) -> None:
    """Reject unknown kinds, dangling dependencies, bad bindings, and cycles."""
    if not nodes:
        raise GraphError("the graph is empty")
    for key, node in nodes.items():
        if node.kind not in NODE_KINDS:
            raise GraphError(f"unknown kind {node.kind!r} on {key}")
        for dep in node.depends_on:
            if dep not in nodes:
                raise GraphError(f"{key} depends on unknown node {dep}")
        _check_bindings(node.inputs, nodes, key)
    _cycles(nodes)


def resolve(value: Any, workflow: Any) -> Any:
    """Replace whole-string bindings with the live entity value."""
    if isinstance(value, str):
        match = BINDING.fullmatch(value)
        if value.startswith("${") and match is None:
            raise GraphError(f"bad binding {value!r}")
        if match is None:
            return value
        node_key, field, leaf = match.groups()
        try:
            runtime = workflow.nodes[node_key]
        except KeyError as exc:
            raise GraphError(f"binding {value!r} names an unknown node") from exc
        bag = runtime.outputs if field == "outputs" else runtime.external
        if leaf not in bag:
            raise GraphError(f"binding {value!r} is not available yet")
        return bag[leaf]
    if isinstance(value, dict):
        return {key: resolve(item, workflow) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, workflow) for item in value]
    return value


def candidate_of(key: str) -> str | None:
    """The candidate a matrix key was expanded for, if any."""
    if "[" not in key or not key.endswith("]"):
        return None
    return key[key.index("[") + 1 : -1]


def _choices(key: str, node: GraphNode, produced: dict[str, list[tuple[str | None, str]]]) -> list[str | None]:
    matrix = node.matrix
    if matrix is None:
        return [None]
    if matrix == "inherit":
        names: list[str] | None = None
        for dep in node.depends_on:
            entries = produced.get(dep)
            if entries is None:
                raise GraphError(f"{key} inherits from {dep}, which is not expanded yet")
            found = [cand for cand, _ in entries if cand is not None]
            if not found:
                continue
            names = found if names is None else [cand for cand in names if cand in found]
        if not names:
            raise GraphError(f"{key} has matrix inherit but no expanded dependency")
        inherited: list[str | None] = []
        inherited.extend(names)
        return inherited
    raw = matrix.get("candidate") if isinstance(matrix, dict) else None
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) and item for item in raw):
        raise GraphError(f"{key} matrix.candidate must be a non-empty list of names")
    return list(raw)


def _remap(
    key: str,
    deps: list[str],
    choice: str | None,
    produced: dict[str, list[tuple[str | None, str]]],
) -> list[str]:
    remapped: list[str] = []
    for dep in deps:
        entries = produced.get(dep)
        if entries is None:
            raise GraphError(f"{key} depends on {dep}, which is not expanded yet")
        if choice is None:
            remapped.extend(expanded_key for _, expanded_key in entries)
            continue
        matched = [expanded_key for cand, expanded_key in entries if cand == choice]
        remapped.extend(matched or [expanded_key for _, expanded_key in entries])
    return remapped


def _rewrite(value: Any, choice: str | None, produced: dict[str, list[tuple[str | None, str]]]) -> Any:
    if isinstance(value, str):
        match = BINDING.fullmatch(value)
        if match is None:
            return value
        ref, field, leaf = match.groups()
        entries = produced.get(ref)
        if not entries:
            return value
        if choice is not None:
            matched = [expanded_key for cand, expanded_key in entries if cand == choice]
            if len(matched) == 1:
                return f"${{nodes.{matched[0]}.{field}.{leaf}}}"
        if len(entries) == 1:
            return f"${{nodes.{entries[0][1]}.{field}.{leaf}}}"
        return value
    if isinstance(value, dict):
        return {key: _rewrite(item, choice, produced) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, choice, produced) for item in value]
    return value


def _check_bindings(value: Any, nodes: dict[str, GraphNode], owner: str) -> None:
    if isinstance(value, str):
        if not value.startswith("${"):
            return
        match = BINDING.fullmatch(value)
        if match is None:
            raise GraphError(f"bad binding {value!r} on {owner}")
        if match.group(1) not in nodes:
            raise GraphError(f"binding {value!r} on {owner} names an unknown node")
        return
    if isinstance(value, dict):
        for item in value.values():
            _check_bindings(item, nodes, owner)
    elif isinstance(value, list):
        for item in value:
            _check_bindings(item, nodes, owner)


def _cycles(nodes: dict[str, GraphNode]) -> None:
    white, gray, black = 0, 1, 2
    color = {key: white for key in nodes}

    def visit(key: str) -> None:
        color[key] = gray
        for dep in nodes[key].depends_on:
            if color[dep] == gray:
                raise GraphError(f"cycle through {key} and {dep}")
            if color[dep] == white:
                visit(dep)
        color[key] = black

    for key in nodes:
        if color[key] == white:
            visit(key)
