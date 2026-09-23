# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization derivation for the agents plugin.

Asserts that every mounted route carries a valid ``@path_rule`` whose permissions all
share the ``agents`` namespace (so ``_derive_service_contribution`` reports no problems
and derives the catalog from the routes), and spot-checks the shapes that matter: a CRUD
binding, the gateway proxy binding (PRINCIPAL + ``agents.gateway.invoke`` across the
wildcard path and every proxied method), and a job-factory binding.
"""

from __future__ import annotations

from nemo_agents_plugin.service import AgentsService
from nemo_helix_plugin.authz import AuthzContribution
from nemo_helix_plugin.authz_discovery import _derive_service_contribution

_BASE = "/apis/agents/v2/workspaces/{workspace}"
_GATEWAY_AGENT = f"{_BASE}/agents/{{name}}/-/{{trailing_uri:path}}"
# Methods the gateway forwards, split by scope (see gateway._PROXY_READ_METHODS /
# _PROXY_WRITE_METHODS), lower-cased for the wire format.
_PROXY_READ_METHODS = {"get", "head", "options"}
_PROXY_WRITE_METHODS = {"post", "put", "patch", "delete"}
_PROXY_METHODS = _PROXY_READ_METHODS | _PROXY_WRITE_METHODS


def _contribution() -> AuthzContribution:
    contrib, problems, _warnings = _derive_service_contribution(AgentsService())
    # No problems is the load-bearing assertion: every route is ruled and every
    # referenced permission lives under the service's own ``agents`` namespace.
    assert problems == [], problems
    return contrib


def test_agents_service_derivation_has_no_problems() -> None:
    contrib = _contribution()
    # All derived permissions live under the agents namespace.
    assert contrib.permissions
    assert all(perm_id.startswith("agents.") for perm_id in contrib.permissions)
    # Every derived permission carries a non-empty description.
    assert all(desc for desc in contrib.permissions.values())


def test_crud_binding_agent_create() -> None:
    contrib = _contribution()
    binding = contrib.endpoints[f"{_BASE}/agents"]["post"]
    assert binding.permissions == ["agents.agents.create"]
    assert binding.scopes == ["agents:write", "platform:write"]
    assert binding.callers == ["principal"]
    assert not binding.deny
    # The corresponding permission id is declared with a description.
    assert "agents.agents.create" in contrib.permissions


def test_crud_binding_deployment_read_covers_logs() -> None:
    contrib = _contribution()
    # The two log routes are read-only and share the deployments.read permission.
    for path in (f"{_BASE}/deployments/{{name}}/logs", f"{_BASE}/deployments/{{name}}/logs/stream"):
        binding = contrib.endpoints[path]["get"]
        assert binding.permissions == ["agents.deployments.read"]
        assert binding.scopes == ["agents:read", "platform:read"]
        assert binding.callers == ["principal"]


def test_session_lifecycle_bindings() -> None:
    expected_bindings = [
        (f"{_BASE}/sessions", "post", "agents.sessions.create", ["agents:write", "platform:write"]),
        (f"{_BASE}/sessions", "get", "agents.sessions.list", ["agents:read", "platform:read"]),
        (f"{_BASE}/sessions/{{name}}", "get", "agents.sessions.read", ["agents:read", "platform:read"]),
        (f"{_BASE}/sessions/{{name}}", "delete", "agents.sessions.delete", ["agents:write", "platform:write"]),
        (
            f"{_BASE}/sessions/{{name}}/close",
            "post",
            "agents.sessions.close",
            ["agents:write", "platform:write"],
        ),
    ]
    contrib = _contribution()

    for path, method, permission, scopes in expected_bindings:
        binding = contrib.endpoints[path][method]
        assert binding.permissions == [permission], (path, method)
        assert binding.scopes == scopes, (path, method)
        assert binding.callers == ["principal"], (path, method)
        assert not binding.deny, (path, method)
        assert permission in contrib.permissions


def test_gateway_proxy_binding() -> None:
    contrib = _contribution()
    methods = contrib.endpoints[_GATEWAY_AGENT]
    # The proxy spans the wildcard ``{trailing_uri:path}`` route across every forwarded method.
    # All methods require agents.gateway.invoke, but read-like methods are agents:read-scoped
    # and mutating methods agents:write-scoped, so a read-scoped token isn't denied on read-only
    # proxy calls (mirrors the Inference Gateway's per-method proxy scopes).
    assert set(methods) == _PROXY_METHODS
    for method, binding in methods.items():
        assert binding.permissions == ["agents.gateway.invoke"], method
        assert binding.callers == ["principal"], method
        assert not binding.deny, method
        expected_scopes = (
            ["agents:write", "platform:write"] if method in _PROXY_WRITE_METHODS else ["agents:read", "platform:read"]
        )
        assert binding.scopes == expected_scopes, method
    # The deployment-name proxy route is split identically.
    deployment_gw = f"{_BASE}/deployments/{{name}}/-/{{trailing_uri:path}}"
    assert set(contrib.endpoints[deployment_gw]) == _PROXY_METHODS
    assert contrib.endpoints[deployment_gw]["post"].scopes == ["agents:write", "platform:write"]
    assert contrib.endpoints[deployment_gw]["get"].scopes == ["agents:read", "platform:read"]
    assert contrib.endpoints[deployment_gw]["post"].permissions == ["agents.gateway.invoke"]
    # The coarse permission is declared.
    assert "agents.gateway.invoke" in contrib.permissions


def test_gateway_invoke_granted_to_viewer() -> None:
    # Regression (P1): pre-derivation authz granted the gateway permission to Viewer + Editor
    # explicitly. Its suffix (`invoke`) isn't read/list, so the default role heuristic assigns it
    # to Editor only; AgentsService.extra_role_permissions() restores the Viewer grant so a Viewer
    # can still invoke a deployed agent through the proxy.
    contrib = _contribution()
    assert contrib.role_permissions == {"Viewer": ["agents.gateway.invoke"]}


def test_gateway_invoke_reaches_both_roles_after_merge() -> None:
    # End-to-end: merging the derived contribution into static authz lands the gateway permission
    # on BOTH roles — Viewer via the explicit extra_role_permissions grant, Editor via the default
    # suffix heuristic. This is the actual user-facing guarantee the P1 regression broke.
    from nemo_helix_plugin.authz_merge import merge_authz_contributions

    base = {
        "authz": {
            "permissions": {},
            "roles": {"Viewer": {"permissions": []}, "Editor": {"permissions": []}},
            "endpoints": {},
        }
    }
    merged = merge_authz_contributions(base, [_contribution().to_dict()])
    roles = merged["authz"]["roles"]
    assert "agents.gateway.invoke" in roles["Viewer"]["permissions"]
    assert "agents.gateway.invoke" in roles["Editor"]["permissions"]


def test_job_factory_binding() -> None:
    contrib = _contribution()
    # evaluate-suite maps to the ``agents.suite`` sub-namespace; its collection
    # POST is a create and its item DELETE is a delete. Factory-generated routes carry
    # both caller kinds (see authz.GENERATED_ROUTE_CALLERS), unlike the hand-written
    # routes above, which stay principal-only.
    collection = f"{_BASE}/jobs/evaluate-suite"
    create = contrib.endpoints[collection]["post"]
    assert create.permissions == ["agents.suite.create"]
    assert create.scopes == ["agents:write", "platform:write"]
    assert create.callers == ["principal", "service_principal"]

    delete = contrib.endpoints[f"{collection}/{{name}}"]["delete"]
    assert delete.permissions == ["agents.suite.delete"]

    # Every job-factory permission for all five collections is declared.
    expected_job_perms = {
        f"agents.{sub}.{verb}"
        for sub in ("evaluate", "suite", "optimize-skills", "analyze", "optimize")
        for verb in ("create", "list", "read", "delete", "cancel")
    }
    assert expected_job_perms <= set(contrib.permissions)
