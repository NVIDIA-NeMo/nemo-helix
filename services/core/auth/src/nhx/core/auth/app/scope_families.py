# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive ``data.authz.scope_families`` so scopes.rego can check a scope family with one lookup
instead of scanning ``data.authz.endpoints`` on every evaluation."""

import copy
import hashlib
import json
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
    endpoints = authz.get("endpoints", {})
    operations = authz.get("operations", {})
    rules: list[Any] = list(operations.values()) if isinstance(operations, dict) else []
    if isinstance(endpoints, dict):
        for methods in endpoints.values():
            if isinstance(methods, dict):
                rules.extend(methods.values())
    for rule in rules:
        scopes = rule.get("scopes") if isinstance(rule, dict) else None
        if isinstance(scopes, list):
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


def with_policy_revision(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the evaluator contract and identify the exact normalized policy snapshot."""
    authz = data.get("authz")
    if (
        not isinstance(authz, dict)
        or type(authz.get("decision_contract_version")) is not int
        or authz["decision_contract_version"] != 2
    ):
        raise ValueError("Policy data requires decision contract version 2")
    document = with_scope_families(data)
    config = {key: value for key, value in authz.get("config", {}).items() if key != "policy_revision"}
    document = {**document, "authz": {**document["authz"], "config": config}}
    revision = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**document, "authz": {**document["authz"], "config": {**config, "policy_revision": revision}}}
