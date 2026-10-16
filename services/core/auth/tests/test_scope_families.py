# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the precomputed ``data.authz.scope_families`` lookup consumed by scopes.rego."""

from pathlib import Path

import pytest
import yaml
from nhx.core.auth.app.bundle import build_authorization_data
from nhx.core.auth.app.embedded_pdp import evaluate, set_policy_data
from nhx.core.auth.app.scope_families import derive_scope_families, scope_family, with_scope_families

STATIC_AUTHZ = Path(__file__).parent.parent / "src/nhx/core/auth/assets/static-authz.yaml"


def test_scope_family_matches_rego_rules():
    assert scope_family("platform:read") == "platform"
    for scope in ("openid", "a:b:c", ":read", "platform:", "urn:x:y"):
        assert scope_family(scope) is None


def test_derive_scope_families_collects_all_endpoints_and_methods():
    authz = {
        "endpoints": {
            "/a": {"get": {"scopes": ["models:read", "platform:read"]}, "post": {"scopes": ["models:write"]}},
            "/b": {"get": {"scopes": ["intake:read", "not-a-family"]}, "put": {"permissions": []}},
        }
    }
    assert derive_scope_families(authz) == {"intake": True, "models": True, "platform": True}


def test_derive_scope_families_skips_malformed_entries():
    authz = {
        "endpoints": {
            "/none-methods": None,
            "/str-methods": "oops",
            "/none-rule": {"get": None},
            "/str-scopes": {"get": {"scopes": "platform:read"}},
            "/int-scope": {"get": {"scopes": [1, None, "models:read"]}},
        }
    }
    assert derive_scope_families(authz) == {"models": True}
    assert derive_scope_families({"endpoints": ["not", "a", "dict"]}) == {}


def test_derive_scope_families_empty():
    assert derive_scope_families({}) == {}
    assert derive_scope_families({"endpoints": {}}) == {}


def test_with_scope_families_does_not_mutate_input():
    data = {"authz": {"endpoints": {"/a": {"get": {"scopes": ["x:read"]}}}}}
    result = with_scope_families(data)
    assert result["authz"]["scope_families"] == {"x": True}
    assert "scope_families" not in data["authz"]


def test_with_scope_families_rederives_stale_value():
    data = {"authz": {"endpoints": {}, "scope_families": {"stale": True}}}
    assert with_scope_families(data)["authz"]["scope_families"] == {}


def test_static_authz_includes_platform_family():
    with open(STATIC_AUTHZ) as f:
        data = yaml.safe_load(f)
    assert "platform" in with_scope_families(data)["authz"]["scope_families"]


@pytest.mark.asyncio
async def test_built_authorization_data_carries_scope_families():
    """Fail-open guard: scopes.rego ignores scopes when ``scope_families`` is missing, so the built document must have it."""
    data = await build_authorization_data(None)
    families = data["authz"]["scope_families"]
    assert families.get("platform") is True
    assert families == derive_scope_families(data["authz"])


def test_embedded_pdp_enforces_platform_scopes_from_derived_families():
    """End to end: raw data without ``scope_families`` still enforces scopes via ``set_policy_data``."""
    with open(STATIC_AUTHZ) as f:
        data = yaml.safe_load(f)
    assert "scope_families" not in data["authz"]
    data["authz"]["principals"] = {"user1": {"workspaces": {"ns1": ["Editor"]}}}
    data["authz"]["workspaces"] = {"ns1": {}}
    set_policy_data(data)

    path = "/apis/secrets/v2/workspaces/ns1/secrets"
    request = {"principal_id": "user1", "method": "POST", "path": path}
    assert evaluate("allow", request)["allowed"] is True
    assert evaluate("allow", {**request, "scopes": ["platform:write"]})["allowed"] is True
    # A configured platform-family scope that does not satisfy the route is denied, not ignored.
    assert evaluate("allow", {**request, "scopes": ["platform:read"]})["allowed"] is False
    # Unconfigured provider scopes are still ignored.
    assert evaluate("allow", {**request, "scopes": ["openid", "external:read"]})["allowed"] is True
