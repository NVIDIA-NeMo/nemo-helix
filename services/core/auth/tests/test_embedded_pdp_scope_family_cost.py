# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope-family correctness against large route tables, including unrelated plugin scopes."""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from nhx.core.auth.app.embedded_pdp import engine as embedded_pdp_engine
from nhx.core.auth.app.embedded_pdp import evaluate, set_policy_data

EXTRA_ROUTE_COUNTS = [0, 250, 500, 1000, 2000]
METHODS = ["get", "post", "put", "delete"]
PRINCIPAL = "svc-dispatcher"
SECRETS_PATH = "/apis/secrets/v2/workspaces/internal-agents/secrets"


@pytest.fixture(autouse=True)
def reset_policy():
    embedded_pdp_engine._reset_policy_state_for_testing()
    yield
    embedded_pdp_engine._reset_policy_state_for_testing()


@pytest.fixture(scope="module")
def static_authz_data() -> dict[str, Any]:
    path = Path(__file__).parent.parent / "src/nhx/core/auth/assets/static-authz.yaml"
    return yaml.safe_load(path.read_text())


def _padded_data(static_authz_data: dict[str, Any], extra_routes: int) -> dict[str, Any]:
    data = copy.deepcopy(static_authz_data)
    for i in range(extra_routes):
        data["authz"]["endpoints"][f"/apis/plugin-{i % 50}/v2/workspaces/{{workspace}}/resource-{i}"] = {
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


@pytest.mark.parametrize("extra_routes", EXTRA_ROUTE_COUNTS)
@pytest.mark.parametrize(
    ("scopes", "allowed"),
    [
        (None, True),
        (["platform:read", "platform:write"], True),
        (["platform:write"], True),
        (["platform:read"], False),
        (["secrets:write"], True),
        (["nonexistent:write"], True),
        ([], True),
    ],
)
def test_scope_validation_with_large_route_table(static_authz_data, extra_routes, scopes, allowed):
    """Valid scopes stay valid; empty and external-only scopes retain optional-scope behavior."""
    set_policy_data(_padded_data(static_authz_data, extra_routes))
    request = {"principal_id": PRINCIPAL, "method": "POST", "path": SECRETS_PATH}
    if scopes is not None:
        request["scopes"] = scopes
    assert evaluate("allow", request)["allowed"] is allowed


@pytest.mark.parametrize("extra_routes", [250, 2000])
def test_synthetic_route_at_end_of_large_table(static_authz_data, extra_routes):
    """Scope-family discovery includes routes beyond the shipped static table."""
    data = _padded_data(static_authz_data, extra_routes)
    index = extra_routes - 1
    family = f"plugin{index % 50}"
    data["authz"]["roles"]["Editor"]["permissions"].append(f"{family}.resource.post")
    set_policy_data(data)
    request = {
        "principal_id": PRINCIPAL,
        "method": "POST",
        "path": f"/apis/plugin-{index % 50}/v2/workspaces/internal-agents/resource-{index}",
    }
    assert evaluate("allow", {**request, "scopes": [f"{family}:write"]})["allowed"] is True
    assert evaluate("allow", {**request, "scopes": [f"{family}:read"]})["allowed"] is False
    assert evaluate("allow", {**request, "scopes": ["secrets:write"]})["allowed"] is False
    assert (
        evaluate(
            "allow",
            {"principal_id": PRINCIPAL, "method": "POST", "path": SECRETS_PATH, "scopes": [f"{family}:write"]},
        )["allowed"]
        is False
    )
