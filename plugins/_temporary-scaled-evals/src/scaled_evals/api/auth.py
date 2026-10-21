# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ownership identity every ``/v1`` route resolves a caller through.

The platform authenticates the request before a plugin route runs and installs
the caller in its auth context. This module maps that principal onto the
ownership model the repositories filter on (``owner_id``); it never verifies
credentials itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException
from nemo_helix_plugin.auth import current_auth_context, is_service_principal_id, platform_auth_enabled


@dataclass(frozen=True)
class CurrentPrincipal:
    """Who a request acts as, for ownership and attribution.

    ``owner_id`` is the only field the repositories filter on. The rest is
    attribution that ``GET /users/me`` echoes.
    """

    owner_type: str
    owner_id: str
    # Load-bearing, not descriptive: several call sites treat "disabled" as
    # "skip owner filtering" so single-owner development can still reach rows
    # written before ownership existed (owner_id IS NULL). Renaming this string
    # silently hides those rows.
    source: str = "disabled"
    email: str | None = None
    username: str | None = None
    display_name: str | None = None
    groups: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()


def current_principal() -> CurrentPrincipal:
    """Resolve the caller's ownership identity from platform auth.

    With platform auth disabled every caller is the shared development owner.
    With it enabled the owner is the effective principal: the user a service
    acts on behalf of, otherwise the authenticated caller.

    Raises:
        HTTPException: 401 when platform auth is enabled but the request carries
            no authenticated principal.
    """
    if not platform_auth_enabled():
        return CurrentPrincipal(owner_type="DEV", owner_id="dev", source="disabled")
    context = current_auth_context()
    if context is None:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "principal_unavailable", "message": "request has no platform principal"}},
        )
    if context.principal_on_behalf_of:
        owner_id = context.principal_on_behalf_of
        email = context.principal_on_behalf_of_email
        groups = context.principal_on_behalf_of_groups or []
    else:
        owner_id = context.principal_id
        email = context.principal_email
        groups = context.principal_groups
    return CurrentPrincipal(
        owner_type="SERVICE" if is_service_principal_id(owner_id) else "USER",
        owner_id=owner_id,
        source="platform",
        email=email,
        groups=tuple(groups),
    )
