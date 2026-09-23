# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``default`` workspace doubles as the global workspace; lookups try the local workspace first."""

from nmp.common.entities.constants import DEFAULT_WORKSPACE

GLOBAL_WORKSPACE = DEFAULT_WORKSPACE


def is_global_workspace(workspace: str) -> bool:
    return workspace == GLOBAL_WORKSPACE


def workspace_lookup_order(workspace: str) -> tuple[str, ...]:
    """Workspaces to try for *workspace*, in precedence order (local first, then global)."""
    if is_global_workspace(workspace):
        return (GLOBAL_WORKSPACE,)
    return (workspace, GLOBAL_WORKSPACE)
