# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve service load order from a dependency graph."""


class CircularDependencyError(Exception):
    """Raised when the dependency graph contains a cycle and no valid order exists."""

    def __init__(self, message: str, cycle: list[str] | None = None, nodes: set[str] | None = None):
        super().__init__(message)
        self.cycle = cycle
        self.nodes = nodes or (set(cycle) if cycle else None)


def _find_cycle(remaining: set[str], dependency_tree: dict[str, list[str]]) -> list[str] | None:
    """Return one cycle as a list [a, b, ..., a] if the subgraph induced by remaining has a cycle."""
    path_order: list[str] = []
    path_set: set[str] = set()

    def dfs(node: str) -> list[str] | None:
        if node in path_set:
            return path_order[path_order.index(node) :] + [node]
        path_set.add(node)
        path_order.append(node)
        for dep in dependency_tree.get(node, []):
            if dep not in remaining:
                continue
            cycle = dfs(dep)
            if cycle is not None:
                return cycle
        path_order.pop()
        path_set.discard(node)
        return None

    for start in remaining:
        cycle = dfs(start)
        if cycle is not None:
            return cycle
    return None


def resolve_service_loading_order(
    services: list[str],
    dependency_tree: dict[str, list[str]],
) -> list[str]:
    """Resolve the order in which services should be started.

    Given a list of services and a dependency graph, returns a list in startup
    order: every dependency appears before any service that depends on it. Uses
    topological sort so shared dependencies appear once. Services not in the
    dependency tree are appended at the end in input order.

    Args:
        services: List of service names to order.
        dependency_tree: Map of service name -> list of dependency names.
            A service not present is treated as having no dependencies.

    Returns:
        List of service names in startup order (dependencies before dependents).
    """
    # Services with no dependency info are appended at the end in input order
    unknown = [s for s in services if s not in dependency_tree]
    known = [s for s in services if s in dependency_tree]
    if not known:
        return list(unknown)

    # Collect all nodes: known services plus their transitive dependencies
    all_nodes: set[str] = set(known)
    stack = list(known)
    while stack:
        node = stack.pop()
        for dep in dependency_tree.get(node, []):
            if dep not in all_nodes:
                all_nodes.add(dep)
                stack.append(dep)

    # Build graph: edge (dep -> service) means dep must start before service
    graph: dict[str, list[str]] = {n: [] for n in all_nodes}
    in_degree: dict[str, int] = {n: 0 for n in all_nodes}
    for node in all_nodes:
        for dep in dependency_tree.get(node, []):
            if dep in all_nodes:
                graph[dep].append(node)
                in_degree[node] += 1

    # Kahn's algorithm: nodes with no dependencies first
    order: list[str] = []
    queue = [n for n in all_nodes if in_degree[n] == 0]
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Detect circular dependency: any remaining node has in_degree > 0
    remaining = {n for n in all_nodes if n not in order}
    if remaining:
        cycle = _find_cycle(remaining, dependency_tree)
        cycle_msg = " -> ".join(cycle) if cycle else ""
        node_list = ", ".join(sorted(remaining))
        msg = f"Circular dependency among services: {node_list}."
        if cycle_msg:
            msg += f" Cycle: {cycle_msg}"
        raise CircularDependencyError(msg, cycle=cycle, nodes=remaining)

    return order + unknown
