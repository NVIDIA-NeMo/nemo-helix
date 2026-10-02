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
