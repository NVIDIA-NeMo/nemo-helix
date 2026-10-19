# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permission utilities for authorization logic."""

from typing import List, Literal, Set, Union

from .principal_identifier import is_service_principal

# Type alias for full access indicator
ALL_WORKSPACES: Literal["-"] = "-"


def compute_accessible_workspaces(
    principal_id: str,
    role_bindings: List[dict],
) -> Union[Set[str], Literal["-"]]:
    """Compute workspaces accessible to a principal from role bindings.

    This is a pure helper function that takes role binding data and computes
    which workspaces the principal has access to. The caller is responsible
    for fetching the role bindings.

    Args:
        principal_id: The principal identifier (e.g., "user@example.com", "service:auth")
        role_bindings: List of role binding dicts, each containing at least:
            - workspace: The workspace this binding grants access to
            - role: The role granted (e.g., "Editor", "Viewer", "HelixAdmin")

    Returns:
        Set of workspace names the principal can access, or "*" for unrestricted access.
        "*" is returned for:
        - Service principals (prefix "service:")
        - Platform admins (any binding with workspace="system" and role="HelixAdmin")

    Example:
        ```python
        from nhx.common.auth import compute_accessible_workspaces

        # Role bindings fetched from entity store (caller's responsibility)
        role_bindings = [
            {"workspace": "team-a", "role": "Editor"},
            {"workspace": "team-b", "role": "Viewer"},
        ]

        accessible = compute_accessible_workspaces("user@example.com", role_bindings)
        # Returns: {"team-a", "team-b"}

        # Service principals get full access
        accessible = compute_accessible_workspaces("service:auth", [])
        # Returns: "*"
        ```
    """
    # Service principals have full access
    if is_service_principal(principal_id):
        return ALL_WORKSPACES

    # Check for platform admin role
    for binding in role_bindings:
        workspace = binding.get("workspace")
        role = binding.get("role")
        if workspace == "system" and role == "HelixAdmin":
            return ALL_WORKSPACES

    # Extract unique workspace names from bindings
    workspaces = {
        workspace for binding in role_bindings if isinstance((workspace := binding.get("workspace")), str) and workspace
    }

    return workspaces
