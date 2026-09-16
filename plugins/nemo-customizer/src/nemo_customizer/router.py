# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Customization router service — merges contributor HTTP routes."""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.customization_contributor import CustomizationContributorDiscoveryError
from nemo_platform_plugin.discovery import (
    CUSTOMIZATION_CONTRIBUTORS_GROUP,
    discover_customization_contributors,
)
from nemo_platform_plugin.service import NemoService, RouterSpec

from nemo_customizer.api.v2 import templates


class CustomizationRouterError(CustomizationContributorDiscoveryError):
    """Raised when the customization router cannot start."""


_ROUTER_BASE_DEPENDENCIES = ("entities", "auth", "jobs", "secrets", "files", "models")
_JOB_TEMPLATE_ROUTER_SPEC = RouterSpec(
    router=templates.router,
    tag="Customization Job Templates",
    description="Saved customization job inputs.",
    prefix="/v2/workspaces/{workspace}",
)


def merge_router_dependencies(contributors: dict[str, object]) -> list[str]:
    """Union platform router deps with each contributor's ``dependencies``."""
    deps = set(_ROUTER_BASE_DEPENDENCIES)
    for contributor in contributors.values():
        contrib_deps = getattr(type(contributor), "dependencies", None) or []
        deps.update(contrib_deps)
    return sorted(deps)


def _record_router_spec_routes(
    seen: dict[tuple[str, str], str],
    *,
    owner: str,
    specs: list[RouterSpec],
) -> None:
    for spec in specs:
        prefix = spec.prefix.rstrip("/")
        for route in spec.router.routes:
            methods = getattr(route, "methods", None) or {"*"}
            path = getattr(route, "path", "")
            full_path = f"{prefix}{path}"
            for method in methods:
                op = (method, full_path)
                if op in seen:
                    raise CustomizationRouterError(
                        f"Route collision: routers {seen[op]!r} and {owner!r} both handle {method} {full_path}",
                    )
                seen[op] = owner


def _assert_no_route_collisions(contributors: dict[str, object]) -> None:
    """Catch contributors that would handle the same ``(METHOD, PATH)`` pair.

    Contributors are free to share a parent mount prefix — e.g. every backend's
    jobs router is mounted under ``/v2/workspaces/{workspace}`` and adds its
    own ``/{backend}/jobs/...`` paths via ``job_collection_path_for``. We only
    error when two contributors would respond to the same HTTP method on the
    same fully-qualified path.
    """
    # Map (method, full_path) -> contributor key
    seen: dict[tuple[str, str], str] = {}
    _record_router_spec_routes(seen, owner="customization job templates", specs=[_JOB_TEMPLATE_ROUTER_SPEC])
    for key, contributor in contributors.items():
        _record_router_spec_routes(seen, owner=key, specs=contributor.get_routers())


class CustomizationRouterService(NemoService):
    """Sole ``nemo.services`` owner for ``/apis/customization``."""

    name: ClassVar[str] = "customization"
    dependencies: ClassVar[list[str]] = list(_ROUTER_BASE_DEPENDENCIES)

    def __init__(self) -> None:
        self._contributors = discover_customization_contributors()
        if not self._contributors:
            raise CustomizationRouterError(
                "Customization router is enabled but no contributors were discovered. "
                "Install a backend plugin (e.g. nemo-automodel) and ensure "
                f"'{CUSTOMIZATION_CONTRIBUTORS_GROUP}' entry points are registered.",
            )
        _assert_no_route_collisions(self._contributors)
        type(self).dependencies = merge_router_dependencies(self._contributors)

    def get_routers(self) -> list[RouterSpec]:
        router = APIRouter()

        @router.get("/healthz")
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def healthz() -> dict[str, object]:
            return {
                "plugin": self.name,
                "status": "ok",
                "contributors": sorted(self._contributors.keys()),
            }

        specs: list[RouterSpec] = [
            RouterSpec(
                router=router,
                tag="Customization",
                description="Customization router health.",
                prefix="/v2",
            ),
            _JOB_TEMPLATE_ROUTER_SPEC,
        ]

        for key in sorted(self._contributors.keys()):
            contributor = self._contributors[key]
            contributor_specs = contributor.get_routers()
            for spec in contributor_specs:
                specs.append(
                    RouterSpec(
                        router=spec.router,
                        tag=spec.tag or f"Customization {key}",
                        description=spec.description,
                        prefix=spec.prefix,
                    ),
                )
        return specs
