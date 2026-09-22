# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate NeMo Platform permission vs OAuth scope string formats.

Permissions use dot-separated segments (e.g. ``secrets.read``, ``models.create``).
Scopes use colon-separated area/action pairs (e.g. ``secrets:read``, ``platform:write``).

Mixing formats fails silently in policy checks; this module rejects invalid inputs early.
"""

from __future__ import annotations

import re
from typing import Any


class InvalidPermissionFormatError(ValueError):
    """Raised when a string is not valid NeMo Platform permission syntax (dot-separated segments)."""


class InvalidScopeFormatError(ValueError):
    """Raised when a string is not valid NeMo Platform scope syntax, or a permission was used as a scope."""


# One segment: lowercase alphanumerics with optional internal hyphens (e.g. data-designer).
_SEGMENT = r"[a-z0-9]+(?:-[a-z0-9]+)*"
# Permission IDs: one or more dot-separated segments.
PERMISSION_ID_PATTERN = re.compile(rf"^{_SEGMENT}(\.{_SEGMENT})+$")
# NeMo Platform OAuth scopes in static-authz: ``area:verb`` (e.g. ``platform:read``, ``data-designer:write``).
NMP_SCOPE_PATTERN = re.compile(rf"^{_SEGMENT}:{_SEGMENT}$")


def is_valid_permission_id(value: str) -> bool:
    """Return True if *value* matches NeMo Platform permission syntax (dot-separated)."""
    if not isinstance(value, str) or not value:
        return False
    return PERMISSION_ID_PATTERN.fullmatch(value) is not None


def is_valid_nmp_scope_id(value: str) -> bool:
    """Return True if *value* matches NeMo Platform scope syntax used in ``static-authz.yaml`` (colon-separated)."""
    if not isinstance(value, str) or not value:
        return False
    return NMP_SCOPE_PATTERN.fullmatch(value) is not None


def looks_like_mistaken_scope_for_permission(value: str) -> bool:
    """True if *value* uses scope syntax (contains ``:``) where a permission was expected."""
    return ":" in value


def is_wildcard_permission(value: str) -> bool:
    """True for the role-level ``*`` entry meaning *all* API permissions in OPA (see ``common.rego``)."""
    return value == "*"


def looks_like_mistaken_permission_for_scope(value: str) -> bool:
    """True if *value* looks like a permission (valid permission id) but was passed as a scope."""
    return is_valid_permission_id(value)


def validate_permission_strings(permissions: list[str], *, context: str) -> None:
    """Ensure each string is a valid NeMo Platform permission id.

    Raises:
        InvalidPermissionFormatError: If any entry is not dot-separated permission syntax,
            or appears to use scope syntax (colon).
    """
    for p in permissions:
        if looks_like_mistaken_scope_for_permission(p):
            raise InvalidPermissionFormatError(
                f"Invalid permission {p!r} in {context}: permission strings use dots "
                f"(e.g. 'secrets.read'), not colons. Did you mean scope syntax like 'secrets:read'?"
            )
        if is_wildcard_permission(p):
            continue
        if not is_valid_permission_id(p):
            raise InvalidPermissionFormatError(
                f"Invalid permission {p!r} in {context}: expected dot-separated segments "
                f"(e.g. 'models.create', 'audit.configs.read'), or '*' for all permissions."
            )


def validate_nmp_scope_strings_for_config(scopes: list[str], *, context: str) -> None:
    """Validate scope strings from ``static-authz.yaml`` (strict NeMo Platform ``area:verb`` form).

    Raises:
        InvalidScopeFormatError: If any entry is not valid NeMo Platform scope syntax, or looks like a permission.
    """
    for s in scopes:
        if is_valid_permission_id(s):
            raise InvalidScopeFormatError(
                f"Invalid scope {s!r} in {context}: scope strings use colons "
                f"(e.g. 'secrets:read', 'platform:write'), not dots. Put dot-separated "
                f"identifiers only under 'permissions', not 'scopes'."
            )
        if not is_valid_nmp_scope_id(s):
            raise InvalidScopeFormatError(
                f"Invalid scope {s!r} in {context}: expected 'area:action' with a single colon "
                f"(e.g. 'jobs:read', 'platform:write')."
            )


def validate_runtime_authorize_scopes(scopes: list[str] | None) -> None:
    """Validate scopes passed to :meth:`AuthClient.authorize_request`.

    OIDC may send scopes without colons (e.g. ``openid``); those are allowed.
    Reject values that are valid *permission* ids — the usual mix-up when calling
    the PDP with token/scopes.

    Raises:
        InvalidScopeFormatError: If a scope string matches NeMo Platform permission syntax.
    """
    if not scopes:
        return

    for s in scopes:
        if looks_like_mistaken_permission_for_scope(s):
            raise InvalidScopeFormatError(
                f"Invalid scope {s!r}: this value uses permission syntax (dots). "
                f"Scopes use colons (e.g. 'secrets:read'). Did you pass a permission by mistake?"
            )


def validate_static_authz_data(data: dict[str, Any]) -> None:
    """Validate permission, scope, and caller-kind string formats in loaded static authz data.

    Call after parsing ``static-authz.yaml``. Validates ``roles[*].permissions``,
    ``endpoints[*][*].{permissions,scopes,callers}``. Raises the same exceptions as the granular
    validators above (and ``ValueError`` for an unknown caller kind) — catching a hand-edited
    bad value at load/build time rather than letting it fail silently in policy checks.
    """
    authz = data.get("authz")
    if not isinstance(authz, dict):
        return

    roles = authz.get("roles")
    if isinstance(roles, dict):
        for role_name, role_cfg in roles.items():
            if not isinstance(role_cfg, dict):
                continue
            perms = role_cfg.get("permissions")
            if isinstance(perms, list):
                validate_permission_strings(
                    [p for p in perms if isinstance(p, str)],
                    context=f"roles[{role_name!r}].permissions",
                )

    endpoints = authz.get("endpoints")
    if isinstance(endpoints, dict):
        for path, methods in endpoints.items():
            if not isinstance(methods, dict):
                continue
            for method_name, op in methods.items():
                if not isinstance(op, dict):
                    continue
                perms = op.get("permissions")
                if isinstance(perms, list):
                    validate_permission_strings(
                        [p for p in perms if isinstance(p, str)],
                        context=f"endpoints[{path!r}].{method_name}.permissions",
                    )
                sc_list = op.get("scopes")
                if isinstance(sc_list, list):
                    validate_nmp_scope_strings_for_config(
                        [s for s in sc_list if isinstance(s, str)],
                        context=f"endpoints[{path!r}].{method_name}.scopes",
                    )
                callers_list = op.get("callers")
                if isinstance(callers_list, list):
                    # Validate caller-kind strings (e.g. hand-edited static-authz.yaml). Local
                    # import keeps this format module light and avoids importing the authoring
                    # surface unless callers are actually present.
                    from nemo_platform_plugin.authz import validate_caller_strings

                    validate_caller_strings(
                        [c for c in callers_list if isinstance(c, str)],
                        context=f"endpoints[{path!r}].{method_name}.callers",
                    )
