# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for v2 API endpoints."""

import asyncio
from typing import List, Optional, Set

from fastapi import HTTPException, status
from nhx.common.api.filter import (
    ComparisonOperation,
    FilterOperation,
    FilterOperator,
    LogicalOperation,
)
from nhx.common.auth import ALL_WORKSPACES, compute_accessible_workspaces
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.models import Principal
from nhx.core.entities.app.repository.entity import EntityRepositoryInterface
from nhx.core.entities.cache import TTLCache
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.entities import Entity

# Entity type for role bindings (used for access control queries)
ROLE_BINDING_ENTITY_TYPE = "role_binding"

# Wildcard principal that grants access to all authenticated users
WILDCARD_PRINCIPAL = "*"

# In-process LRU + TTL cache for _fetch_bindings_for_principal.
# Configure with NHX_ENTITIES_PRINCIPAL_BINDINGS_CACHE_{ENABLED,TTL_SEC,MAX_SIZE} (see EntitiesConfig).
_bindings_cache_lock = asyncio.Lock()
_bindings_store: Optional[TTLCache] = None


def _bindings_cache_store() -> Optional[TTLCache]:
    """Return the shared :class:`TTLCache` or ``None`` if caching is disabled.

    Call only while holding :data:`_bindings_cache_lock`. Reads
    :class:`nhx.core.entities.config.EntitiesConfig` (env: ``NHX_ENTITIES_*``).
    """
    global _bindings_store
    cfg: EntitiesConfig = EntitiesConfig.get()
    if not cfg.principal_bindings_cache_enabled:
        _bindings_store = None
        return None
    if _bindings_store is None:
        _bindings_store = TTLCache(
            maxsize=int(cfg.principal_bindings_cache_max_size),
            ttl=float(cfg.principal_bindings_cache_ttl_sec),
        )
    return _bindings_store


async def clear_principal_bindings_cache() -> None:
    """Drop the role-bindings cache and future lookups create a new store (e.g. in tests)."""
    async with _bindings_cache_lock:
        store = _bindings_cache_store()
        if store is not None:
            store.clear()


async def _bindings_cache_get(principal: str) -> Optional[List]:
    """Return cached role bindings for ``principal``, or ``None`` if miss or cache off.

    Async-safe. When caching is disabled or the key is missing/expired, returns
    ``None`` and the caller should load from the repository.
    """
    async with _bindings_cache_lock:
        store = _bindings_cache_store()
        if store is None:
            return None
        return store.get(principal)


async def _bindings_cache_set(principal: str, value: List) -> None:
    """Store a shallow copy of ``value`` under ``principal`` in the shared TTL cache.

    No-op if caching is disabled. Async-safe.
    """
    to_store: List[Entity] = list(value)
    async with _bindings_cache_lock:
        store = _bindings_cache_store()
        if store is not None:
            store[principal] = to_store


async def bindings_cache_delete(principal: str) -> None:
    """Delete the cached role bindings for ``principal``."""
    async with _bindings_cache_lock:
        store = _bindings_cache_store()
        if store is not None:
            store.pop(principal, None)


async def _fetch_bindings_for_principal(
    entity_repository: EntityRepositoryInterface,
    principal: str,
) -> List[Entity]:
    """Fetch all role bindings for a specific principal.

    Results are cached in-process (LRU, TTL) when enabled in
    :class:`nhx.core.entities.config.EntitiesConfig` (e.g. env
    ``NHX_ENTITIES_PRINCIPAL_BINDINGS_CACHE_ENABLED``).

    Args:
        entity_repository: Repository for querying role binding entities
        principal: The principal identifier to fetch bindings for

    Returns:
        List of role binding entities
    """
    cache_cfg: EntitiesConfig = EntitiesConfig.get()
    if cache_cfg.principal_bindings_cache_enabled:
        cached = await _bindings_cache_get(principal)
        if cached is not None:
            return list(cached)

    principal_filter = ComparisonOperation(
        operator=FilterOperator.EQ,
        field="data.principal",
        value=principal,
    )

    all_bindings: List[Entity] = []
    page = 1
    page_size = 1000

    while True:
        bindings, total = await entity_repository.list_entities(
            workspace=ALL_WORKSPACES,
            entity_type=ROLE_BINDING_ENTITY_TYPE,
            filter_op=principal_filter,
            page=page,
            page_size=page_size,
        )
        all_bindings.extend(bindings)
        if len(all_bindings) >= total or len(bindings) < page_size:
            break
        page += 1

    if cache_cfg.principal_bindings_cache_enabled:
        await _bindings_cache_set(principal, all_bindings)
    return all_bindings


def _applicable_principal_strings(principal: Principal) -> List[str]:
    """Principal identifiers that may appear on role bindings (aligned with OPA get_applicable_principals).

    Bindings may be keyed by stable account id, JWT subject/oid, aliases, email, or group id;
    all must be queried.
    """
    identifiers: List[str] = []
    if principal.id:
        pid = principal.id.strip()
        if pid:
            identifiers.append(pid)
    if principal.account_id:
        account_id = principal.account_id.strip()
        if account_id and account_id not in identifiers:
            identifiers.append(account_id)
    for alias in principal.authz_aliases:
        a = alias.strip() if isinstance(alias, str) else ""
        if a and a != WILDCARD_PRINCIPAL and a not in identifiers:
            identifiers.append(a)
    if principal.email:
        email = principal.email.strip()
        if email and email not in identifiers:
            identifiers.append(email)
    for group in principal.groups:
        g = group.strip() if isinstance(group, str) else ""
        if g and g not in identifiers:
            identifiers.append(g)
    return identifiers


async def get_accessible_workspaces(
    entity_repository: EntityRepositoryInterface,
) -> Optional[Set[str]]:
    """Get accessible workspaces for the current principal.

    Role bindings are stored in the workspace they grant access to. The entity's
    `workspace` field indicates which workspace the binding applies to. The
    `workspace` field in the binding data is used for system-level bindings
    (e.g., HelixAdmin with workspace="system").

    This function also checks for wildcard principal "*" bindings which grant
    access to all authenticated users.

    Role bindings may be keyed by subject id, email, or group (same as OPA). We
    query bindings for every applicable identifier and merge the results.

    Args:
        entity_repository: Repository for querying role binding entities

    Returns:
        Set of workspace names, or None if all workspaces are accessible
        (auth disabled, or service principal with no on-behalf-of, or platform admin).
        When the caller is a service principal (``X-NHX-Principal-Id``) with
        ``X-NHX-Principal-On-Behalf-Of`` set — the pattern used by downstream
        services to call the entity store — access is computed for the **on-behalf-of**
        user (id, email, groups), not the service id. This matches the embedded PDP
        and avoids treating every OBO call as unscoped ``service:*`` access.
    """
    auth_client = auth_client_context.get()

    # If no auth context or auth disabled, allow all
    if auth_client is None or not auth_client.auth_enabled:
        return None

    # Act as the end user when the platform calls us as service:… with on-behalf-of
    # (e.g. EntityClient from a microservice). Otherwise the raw principal id is
    # "service:platform" and compute_accessible_workspaces would return all workspaces.
    principal = auth_client.principal
    if principal.is_privileged():
        if not principal.is_delegated:
            return None

        effective_principal = principal.effective_principal
    else:
        effective_principal = principal

    principal_id = effective_principal.id

    # Fetch role bindings for each identifier (id, email, groups) — same union as PDP/OPA
    seen_binding_ids: Set[str] = set()
    principal_bindings_entities: List[Entity] = []
    for ident in _applicable_principal_strings(effective_principal):
        for binding in await _fetch_bindings_for_principal(entity_repository, ident):
            if binding.id not in seen_binding_ids:
                seen_binding_ids.add(binding.id)
                principal_bindings_entities.append(binding)

    # Also fetch wildcard principal "*" bindings (grants access to all authenticated users)
    for binding in await _fetch_bindings_for_principal(entity_repository, WILDCARD_PRINCIPAL):
        if binding.id not in seen_binding_ids:
            seen_binding_ids.add(binding.id)
            principal_bindings_entities.append(binding)

    # Extract workspace info from bindings
    # Use workspace from data if present, otherwise use the entity's workspace
    principal_bindings = [
        {
            "workspace": b.data.get("workspace") or b.workspace,
            "role": b.data.get("role"),
        }
        for b in principal_bindings_entities
    ]

    # Compute accessible workspaces (principal_id is the acting user when OBO is set)
    accessible = compute_accessible_workspaces(principal_id, principal_bindings)

    if accessible == ALL_WORKSPACES:
        return None  # No filtering needed

    return accessible


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
    entity_repository: EntityRepositoryInterface,
    workspace: str,
    detail: str | None = None,
    status_code: int = status.HTTP_403_FORBIDDEN,
) -> Optional[Set[str]]:
    """Resolve accessible workspaces, enforce a named path workspace for scoped users, and return the set.

    The returned set is the same as :func:`get_accessible_workspaces` and may be ``None`` when
    no per-workspace filtering applies.
    """
    accessible = await get_accessible_workspaces(entity_repository)
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
