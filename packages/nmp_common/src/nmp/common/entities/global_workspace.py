# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolution order for the global workspace.

The ``default`` workspace doubles as the installation-wide GLOBAL workspace: entities of
a shared type that live there are visible from every workspace, so teams do not duplicate
expensive resources per workspace.

A bare name can therefore match in two places. The precedence rule is **local wins**: a
name is resolved in the request workspace first and only then in the global workspace, so
adding a global entity can never silently redirect a workspace's existing reference. Every
lookup that participates in global sharing must iterate :func:`workspace_lookup_order`
rather than rolling its own fallback, otherwise the services disagree about which entity a
name refers to.
"""

from nmp.common.entities.constants import DEFAULT_WORKSPACE

GLOBAL_WORKSPACE = DEFAULT_WORKSPACE


def is_global_workspace(workspace: str) -> bool:
    return workspace == GLOBAL_WORKSPACE


def workspace_lookup_order(workspace: str) -> tuple[str, ...]:
    """Workspaces to try for *workspace*, in precedence order (local first, then global)."""
    if is_global_workspace(workspace):
        return (GLOBAL_WORKSPACE,)
    return (workspace, GLOBAL_WORKSPACE)
