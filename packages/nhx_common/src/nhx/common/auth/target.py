# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypedDict

from fastapi import Request
from fastapi.routing import APIRoute, iter_route_contexts
from starlette.routing import Match
from starlette.types import Scope


class AuthorizationTarget(TypedDict):
    service: str
    entity_type: str


class AuthorizationTargetRoute(APIRoute, ABC):
    @abstractmethod
    def pdp_target(self, path_params: Mapping[str, object]) -> AuthorizationTarget | None:
        """Build the PDP input from the bound route parameters."""


def match_authorization_target(request: Request) -> AuthorizationTarget | None:
    """Read authorization metadata from the first fully matched route."""
    for context in iter_route_contexts(request.app.routes):
        matched: tuple[Match, Scope] = context.matches(request.scope)
        match, scope = matched
        if match != Match.FULL:
            continue
        route = context.original_route
        if isinstance(route, AuthorizationTargetRoute):
            return route.pdp_target(scope["path_params"])
        return None
    return None
