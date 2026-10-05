# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi.routing import APIRoute
from nemo_deployments_plugin.service import DeploymentsService
from nemo_helix_plugin.authz import get_path_rules


def _mounted_paths() -> set[str]:
    service = DeploymentsService()
    paths: set[str] = set()
    for spec in service.get_routers():
        for route in spec.router.routes:
            if isinstance(route, APIRoute):
                paths.add(f"/apis/deployments{spec.prefix}{route.path}")
    return paths


def test_service_mounts_core_routes() -> None:
    paths = _mounted_paths()
    assert "/apis/deployments/v2/workspaces/{workspace}/deployment-configs" in paths
    assert "/apis/deployments/v2/workspaces/{workspace}/deployments" in paths
    assert "/apis/deployments/v2/workspaces/{workspace}/volumes" in paths
    assert "/apis/deployments/v2/workspaces/{workspace}/deployments/{name}/status" in paths
    assert "/apis/deployments/v2/workspaces/{workspace}/volumes/{name}/status" in paths


def test_service_name_matches_entry_point() -> None:
    assert DeploymentsService.name == "deployments"


def test_service_authz_covers_mounted_routes() -> None:
    """Every mounted route carries at least one ``@path_rule``.

    Authz is derived from the routes (no separate contribution); an unruled route would be
    treated as invalid and fenced/denied by the PDP, so coverage is the property to assert.
    """
    service = DeploymentsService()
    for spec in service.get_routers():
        for route in spec.router.routes:
            if isinstance(route, APIRoute):
                full = f"/apis/deployments{spec.prefix}{route.path}"
                assert get_path_rules(route.endpoint), f"missing @path_rule for {full}"


def test_controller_entry_point() -> None:
    from nemo_deployments_plugin.controller import DeploymentsController

    assert DeploymentsController.name == "deployments"
