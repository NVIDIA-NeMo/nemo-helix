# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for v2 API endpoints."""

from contextvars import ContextVar
from typing import Optional, Set

from fastapi import HTTPException, status
from nemo_helix_plugin.auth.decisions import (
    CollectionTarget,
    DecisionRequest,
    RestrictedWorkspaces,
    UnrestrictedWorkspaces,
    WorkspaceDecision,
)
from nhx.common.api.filter import (
    ComparisonOperation,
    FilterOperation,
    FilterOperator,
    LogicalOperation,
)
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.config import get_auth_config

# IAM entities keep their domain type name; grants are resolved only in Rego.
ROLE_BINDING_ENTITY_TYPE = "role_binding"

workspace_decision_context: ContextVar[WorkspaceDecision | None] = ContextVar(
    "entities_workspace_decision", default=None
)


def workspaces_from_decision(decision: WorkspaceDecision) -> set[str] | None:
    """Translate an approved scope; never recompute role bindings or grants."""
    access = decision.access
    if isinstance(access, UnrestrictedWorkspaces):
        return None
    if isinstance(access, RestrictedWorkspaces):
        return set(access.workspaces)
    raise HTTPException(status_code=403, detail="Workspace access denied")


async def get_accessible_workspaces() -> Optional[Set[str]]:
    """Use the request-bound PDP scope without querying role bindings."""
    auth_client = auth_client_context.get()
    if auth_client is None:
        if get_auth_config().enabled:
            raise HTTPException(status_code=503, detail="Missing authenticated workspace context")
        return None
    decision = workspace_decision_context.get()
    if decision is None:
        target = decision.target if decision is not None else CollectionTarget(workspace="-", resource_type="entity")
        result = await auth_client.decide(DecisionRequest(operation="entities.workspaces.visible", target=target))
        if not isinstance(result, WorkspaceDecision):
            raise ValueError("Workspace visibility requires a typed workspace decision")
        decision = result
    return workspaces_from_decision(decision)


def raise_if_workspace_inaccessible(
    accessible_workspaces: Optional[Set[str]],
    workspace: str,
    detail: str | None = None,
    status_code: int = status.HTTP_403_FORBIDDEN,
) -> None:
    """For principals scoped to a set of workspaces, 403 if ``workspace`` is not in that set.

    When ``accessible_workspaces`` is ``None`` (unscoped / full access), the check is skipped.
    """
    if accessible_workspaces is not None and workspace not in accessible_workspaces:
        raise HTTPException(
            status_code=status_code,
            detail=detail or f"Not allowed to list entities in workspace '{workspace}'",
        )


async def require_workspace_access(
    workspace: str,
    detail: str | None = None,
    status_code: int = status.HTTP_403_FORBIDDEN,
) -> Optional[Set[str]]:
    """Resolve accessible workspaces, enforce a named path workspace for scoped users, and return the set.

    The returned set is the same as :func:`get_accessible_workspaces` and may be ``None`` when
    no per-workspace filtering applies.
    """
    accessible = await get_accessible_workspaces()
    raise_if_workspace_inaccessible(
        accessible,
        workspace,
        detail=detail,
        status_code=status_code,
    )
    return accessible


def require_accessible_workspaces(
    accessible: Optional[Set[str]],
    required_workspaces: Optional[Set[str]],
) -> None:
    """Raise 403 if auth scopes the principal to a workspace set and any required
    workspace is missing from that set.

    If ``accessible`` is ``None`` (auth disabled, service / platform-wide access), no
    check is applied. If ``required_workspaces`` is ``None`` or empty, no check.
    """
    if accessible is None or required_workspaces is None or not required_workspaces:
        return
    for ws in required_workspaces:
        if ws not in accessible:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to a workspace required for this operation.",
            )


def add_workspace_filtering(
    accessible_workspaces: Optional[Set[str]],
    user_filter: Optional[FilterOperation],
    field: str = "workspace",
) -> Optional[FilterOperation]:
    """Build a combined filter for workspace access and user filter.

    Args:
        accessible_workspaces: Set of workspace IDs, or None for full access
        user_filter: User's filter, or None
        field: The field name to filter on (e.g., "workspace" for entities, "id" for workspaces)

    Returns:
        Combined filter operation, or None if no filtering needed
    """
    # If full access (None), just return user's filter
    if accessible_workspaces is None:
        return user_filter

    # Build workspace filter
    workspace_filter = ComparisonOperation(
        operator=FilterOperator.IN,
        field=field,
        value=list(accessible_workspaces),
    )

    # Combine with user's filter if present
    if user_filter is not None:
        return LogicalOperation(
            operator=FilterOperator.AND,
            operations=[workspace_filter, user_filter],
        )

    return workspace_filter
