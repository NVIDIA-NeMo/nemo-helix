# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge plugin-contributed authorization data into static policy data."""

from __future__ import annotations

import copy
from typing import Any


def _deep_merge_permission_registry(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge nested permission registry trees (leaf nodes have ``description``)."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key not in merged:
            merged[key] = copy.deepcopy(value)
            continue
        if isinstance(merged[key], dict) and isinstance(value, dict):
            if "description" in value or "description" in merged[key]:
                # Leaf or partial leaf — overlay wins at this key when overlay is a leaf
                if "description" in value:
                    merged[key] = copy.deepcopy(value)
                else:
                    merged[key] = _deep_merge_permission_registry(merged[key], value)
            else:
                merged[key] = _deep_merge_permission_registry(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _permission_id_to_nested(permission_id: str, description: str) -> dict[str, Any]:
    """Turn ``customization.automodel.jobs.create`` into nested registry dict."""
    parts = permission_id.split(".")
    node: dict[str, Any] = {}
    cursor = node
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = {"description": description}
    return node


def _merge_flat_permissions(
    registry: dict[str, Any],
    flat_permissions: dict[str, str],
) -> dict[str, Any]:
    merged = registry
    for perm_id, description in flat_permissions.items():
        nested = _permission_id_to_nested(perm_id, description)
        merged = _deep_merge_permission_registry(merged, nested)
    return merged


def _default_roles_for_permission(permission_id: str) -> list[str]:
    """Mirror ``auth-tools update`` role assignment heuristics."""
    suffix = permission_id.rsplit(".", 1)[-1]
    if suffix in {"list", "read"}:
        return ["Viewer", "Editor"]
    return ["Editor"]


def merge_authz_contributions(
    base_data: dict[str, Any],
    contributions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge plugin :class:`AuthzContribution` payloads into loaded static authz data.

    Each contribution dict may contain:

    - ``permissions``: flat ``permission_id -> description`` for the registry
    - ``endpoints``: ``path -> {method: {permissions, scopes?}}``
    - ``role_permissions``: optional ``role -> [permission_id, ...]`` extra grants
      (defaults: ``.list``/``.read`` → Viewer+Editor, else Editor only)

    Later contributions override endpoint methods for the same path+method.
    """
    if not contributions:
        return base_data

    merged = copy.deepcopy(base_data)
    authz = merged.setdefault("authz", {})
    registry = authz.setdefault("permissions", {})
    endpoints = authz.setdefault("endpoints", {})
    roles = authz.setdefault("roles", {})

    auto_role_grants: dict[str, set[str]] = {}

    for contribution in contributions:
        flat_permissions = contribution.get("permissions") or {}
        if isinstance(flat_permissions, dict):
            registry = _merge_flat_permissions(registry, flat_permissions)

        contrib_endpoints = contribution.get("endpoints") or {}
        if isinstance(contrib_endpoints, dict):
            for path, methods in contrib_endpoints.items():
                if not isinstance(methods, dict):
                    continue
                endpoints.setdefault(path, {})
                for method, spec in methods.items():
                    if isinstance(spec, dict):
                        endpoints[path][method.lower()] = copy.deepcopy(spec)

        explicit_roles = contribution.get("role_permissions") or {}
        if isinstance(explicit_roles, dict):
            for role_name, perms in explicit_roles.items():
                if not isinstance(perms, list):
                    continue
                auto_role_grants.setdefault(role_name, set()).update(str(p) for p in perms)

        for perm_id in flat_permissions:
            for role_name in _default_roles_for_permission(perm_id):
                auto_role_grants.setdefault(role_name, set()).add(perm_id)

    authz["permissions"] = registry

    for role_name, perm_ids in auto_role_grants.items():
        role_cfg = roles.setdefault(role_name, {"permissions": []})
        existing = role_cfg.setdefault("permissions", [])
        if not isinstance(existing, list):
            continue
        for perm_id in sorted(perm_ids):
            if perm_id not in existing:
                existing.append(perm_id)

    return merged
