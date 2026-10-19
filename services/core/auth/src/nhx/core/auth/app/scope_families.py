# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive ``data.authz.scope_families`` so scopes.rego can check a scope family with one lookup
instead of scanning ``data.authz.endpoints`` on every evaluation."""

import copy
from typing import Any


def scope_family(scope: Any) -> str | None:
    """The ``area`` of an ``area:action`` scope, else None. Mirrors ``scope_family`` in scopes.rego."""
    if not isinstance(scope, str):
        return None
    parts = scope.split(":")
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0]
    return None


def derive_scope_families(authz: dict[str, Any]) -> dict[str, bool]:
    """Every scope family configured on any endpoint/method. Skips malformed entries rather than raising."""
    families: set[str] = set()
    endpoints = authz.get("endpoints")
    if not isinstance(endpoints, dict):
        return {}
    for methods in endpoints.values():
        if not isinstance(methods, dict):
            continue
        for rule in methods.values():
            scopes = rule.get("scopes") if isinstance(rule, dict) else None
            if not isinstance(scopes, list):
                continue
            for scope in scopes:
                family = scope_family(scope)
                if family is not None:
                    families.add(family)
    return {family: True for family in sorted(families)}


def with_scope_families(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with ``authz.scope_families`` derived from ``authz.endpoints``."""
    authz = data.get("authz")
    if not isinstance(authz, dict):
        return data
    result = copy.copy(data)
    result["authz"] = {**authz, "scope_families": derive_scope_families(authz)}
    return result
