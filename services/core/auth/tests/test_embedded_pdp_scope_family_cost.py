# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for AIRCORE-1373: scope-family checking must not scale with the route table.

Runs the real embedded PDP with the original pre-#3012 caps (200M fuel / 32MB memory) against the shipped
static-authz.yaml padded with synthetic plugin routes, for platform-scoped and unscoped requests.

Run: uv run --frozen pytest services/core/auth/tests/test_embedded_pdp_scope_family_cost.py -v -s
"""

import copy
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from nhx.common.config import Configuration
from nhx.core.auth.app.embedded_pdp import PolicyEngineError, evaluate, set_policy_data
from nhx.core.auth.app.embedded_pdp import engine as embedded_pdp_engine
from nhx.core.auth.config import AuthServiceConfig

# Defaults before PR #3012 raised them to 1500 / 256.
ORIGINAL_FUEL_LIMIT_M = 200
ORIGINAL_MEMORY_LIMIT_MB = 32

# Synthetic routes added on top of the static table. The ticket measured 112 static endpoints on prod
# plus enabled plugins; each synthetic plugin route carries a few scopes like real routes do.
EXTRA_ROUTE_COUNTS = [0, 250, 500, 1000, 2000]
METHODS = ["get", "post", "put", "delete"]

PRINCIPAL = "svc-dispatcher"
SECRETS_PATH = "/apis/secrets/v2/workspaces/internal-agents/secrets"


def _set_caps(fuel_m: int, memory_mb: int) -> None:
    cfg = AuthServiceConfig(embedded_pdp_cpu_limit=fuel_m, embedded_pdp_memory_limit_mb=memory_mb)  # type: ignore[misc]
    Configuration.set_override(cfg)
    embedded_pdp_engine._reset_policy_state_for_testing()


@pytest.fixture(autouse=True)
def original_caps():
    _set_caps(ORIGINAL_FUEL_LIMIT_M, ORIGINAL_MEMORY_LIMIT_MB)
    yield
    Configuration.clear_override(AuthServiceConfig)
    embedded_pdp_engine._reset_policy_state_for_testing()


@pytest.fixture(scope="module")
def static_authz_data() -> dict[str, Any]:
    path = Path(__file__).parent.parent / "src/nhx/core/auth/assets/static-authz.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _padded_data(static_authz_data: dict[str, Any], extra_routes: int) -> dict[str, Any]:
    """Static authz data + `extra_routes` synthetic plugin routes + a service principal with a role."""
    data = copy.deepcopy(static_authz_data)
    endpoints = data["authz"]["endpoints"]
    for i in range(extra_routes):
        endpoints[f"/apis/plugin-{i % 50}/v2/workspaces/{{workspace}}/resource-{i}"] = {
            method: {
                "permissions": [f"plugin{i % 50}.resource.{method}"],
                "scopes": [
                    f"plugin{i % 50}:{'read' if method == 'get' else 'write'}",
                    "platform:read",
                    "platform:write",
                ],
            }
            for method in METHODS
        }
    data["authz"]["principals"] = {PRINCIPAL: {"workspaces": {"internal-agents": ["Editor"]}}}
    data["authz"]["workspaces"] = {"internal-agents": {}}
    return data


def _platform_scoped_request() -> dict[str, Any]:
    """What a Scoped Access Key / service principal carrying platform scopes sends."""
    return {
        "principal_id": PRINCIPAL,
        "method": "POST",
        "path": SECRETS_PATH,
        "scopes": ["platform:read", "platform:write"],
    }


def _unscoped_request() -> dict[str, Any]:
    """What internal service-to-service callers send (no platform scopes -> cheap branch)."""
    return {"principal_id": PRINCIPAL, "method": "POST", "path": SECRETS_PATH}


@pytest.mark.parametrize("extra_routes", EXTRA_ROUTE_COUNTS)
def test_platform_scoped_check_stays_within_original_caps(static_authz_data, extra_routes):
    """AIRCORE-1373: must not hit the fuel/memory cap regardless of route-table size."""
    set_policy_data(_padded_data(static_authz_data, extra_routes))

    # Before the O(1) lookup this raised PolicyEngineError ("exceeded fuel limit" / "exceeded memory limit" /
    # "opa_abort <- opa_malloc") once the table was large enough; the auth service maps that to a 502.
    result = evaluate("allow", _platform_scoped_request())

    # Not asserting `allowed` is True: the point is that a real decision is returned, not an abort.
    assert "allowed" in result


@pytest.mark.parametrize("extra_routes", EXTRA_ROUTE_COUNTS)
def test_unscoped_check_is_independent_of_table_size(static_authz_data, extra_routes):
    """Control: the same data with no scopes never runs the scope-family scan and always completes."""
    set_policy_data(_padded_data(static_authz_data, extra_routes))

    result = evaluate("allow", _unscoped_request())

    assert result["allowed"] is True


def test_report_cost_by_table_size(static_authz_data, capsys):
    """Informational: print the outcome and latency of scoped vs unscoped evals as the table grows."""
    rows = []
    for extra_routes in EXTRA_ROUTE_COUNTS:
        set_policy_data(_padded_data(static_authz_data, extra_routes))
        outcome = {}
        for label, request in (("scoped", _platform_scoped_request()), ("unscoped", _unscoped_request())):
            start = time.perf_counter()
            try:
                evaluate("allow", request)
                status = "ok"
            except PolicyEngineError as exc:
                status = str(exc).split(" for entrypoint")[0][:60]
            outcome[label] = f"{status} ({(time.perf_counter() - start) * 1000:.0f}ms)"
        rows.append((len(_padded_data(static_authz_data, extra_routes)["authz"]["endpoints"]), outcome))

    with capsys.disabled():
        print(f"\ncaps: {ORIGINAL_FUEL_LIMIT_M}M fuel / {ORIGINAL_MEMORY_LIMIT_MB}MB memory")
        print(f"{'endpoints':>10}  {'platform-scoped':<70}  unscoped")
        for endpoints, outcome in rows:
            print(f"{endpoints:>10}  {outcome['scoped']:<70}  {outcome['unscoped']}")
