# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Phase 1 scaled-evals ephemeral plugin surface."""

from __future__ import annotations

import pytest

# This plugin is absent from `enabled-plugins`, so a default sync leaves it and its database
# driver uninstalled and the repo-wide test run still sweeps this directory. Skip rather than
# error there; the job that owns these tests installs the `scaled-evals` group first.
try:
    from fastapi.testclient import TestClient
    from nemo_helix_plugin.authz import get_path_rules, get_path_scope
    from nemo_scaled_evals_plugin.service import ScaledEvalsService
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_scaled_evals_service_mounts_health_and_v1_routers() -> None:
    service = ScaledEvalsService()
    assert service.name == "scaled-evals"
    specs = service.get_routers()
    assert specs
    prefixes = {spec.prefix for spec in specs}
    assert "" in prefixes or any(spec.prefix == "" for spec in specs)
    assert "/v1" in prefixes

    # Every mounted APIRoute must carry platform authz (hard_fail otherwise).
    for spec in specs:
        for route in spec.router.routes:
            endpoint = getattr(route, "endpoint", None)
            if endpoint is None or not hasattr(route, "methods"):
                continue
            assert get_path_rules(endpoint), f"missing path_rule on {route.path}"
            assert get_path_scope(endpoint), f"missing scope on {route.path}"


def test_scaled_evals_healthz_via_adapter() -> None:
    from nhx.platform_runner.plugin_adapter import NemoServiceAdapter

    adapter = NemoServiceAdapter(ScaledEvalsService())
    app = adapter.create_app()
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["plugin"] == "scaled-evals"
    assert body["status"] == "ok"


def test_admin_routes_require_the_admin_permission() -> None:
    from nemo_scaled_evals_plugin.authz import ADMIN
    from scaled_evals.api.tenancy import _ADMIN_PROBE

    gated = []
    for spec in ScaledEvalsService().get_routers():
        for route in spec.router.routes:
            path = getattr(route, "path", "")
            if not (path.startswith("/admin/") or path.endswith(("/qualification", "/promote"))):
                continue
            rules = get_path_rules(route.endpoint)
            # Any rule without ADMIN would OR-open the route to non-admins.
            assert rules and all(ADMIN in rule.permissions for rule in rules), path
            gated.append(f"{spec.prefix}{path}")
    assert "/v1/admin/capacity" in gated
    assert _ADMIN_PROBE[1].endswith("/v1/admin/capacity")
    assert {"/v1/admin/usage", "/v1/admin/compute", "/v1/admin/failures"} <= set(gated)
    assert len([path for path in gated if path.endswith(("/qualification", "/promote"))]) == 4
