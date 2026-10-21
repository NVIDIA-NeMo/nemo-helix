# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authz helpers for the scaled-evals plugin surface."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.routing import APIRoute
from nemo_helix_plugin.authz import AuthzScope, CallerKind, path_rule

scope = AuthzScope("scaled-evals")
# No platform role grants this, so only HelixAdmin (which bypasses permission
# checks) holds it unless an operator binds a custom role that does.
ADMIN = scope.permission("admin", description="Operate scaled-evals: analytics, qualification, promotion")
admin_only = path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ADMIN])

_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def stamp_router_authz(router: APIRouter) -> APIRouter:
    """Stamp platform ``@path_rule`` + scope onto every route on *router*.

    Routes without their own rule are authenticated-but-permissionless, so any
    valid principal reaches them. Admin routes declare ``@admin_only`` on the
    handler, which this leaves in place.
    """
    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue
        endpoint = route.endpoint
        methods = route.methods or set()
        if methods & _WRITE_METHODS:
            scope.write(endpoint)
        else:
            scope.read(endpoint)
        if not getattr(endpoint, "__nemo_path_rules__", None):
            path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])(endpoint)
    return router
