# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authz helpers for the scaled-evals plugin surface."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.routing import APIRoute
from nemo_helix_plugin.authz import AuthzScope, CallerKind, path_rule

scope = AuthzScope("scaled-evals")

_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def stamp_router_authz(router: APIRouter) -> APIRouter:
    """Stamp platform ``@path_rule`` + scope onto every route on *router*.

    The rules are authenticated-but-permissionless so the vendored scaled-evals
    routers mount without rewriting every handler. Any valid principal therefore
    reaches every route, including the admin router; per-route permissions have
    to be declared on the handlers themselves to narrow that.
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
