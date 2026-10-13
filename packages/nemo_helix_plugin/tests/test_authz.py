# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
from fastapi import APIRouter
from nemo_helix_plugin.authz import (
    AuthzScope,
    CallerKind,
    Permission,
    path_rule,
    scopes_for,
)
from nemo_helix_plugin.authz_discovery import (
    _derive_service_contribution,
    clear_plugin_authz_cache,
    discover_authz_contributions,
)
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.scheduler import NemoJobScheduler
from nemo_helix_plugin.service import NemoService, RouterSpec


class _ExampleSubmitJob(NemoJob):
    name = "example-submit"
    description = "Job used to verify authenticated remote submit."

    def run(self, config: dict) -> dict:
        return config


_ExampleSubmitJob.__module__ = "example_plugin.jobs.example_submit"


class _FakeEntryPoint:
    """Minimal EntryPoint stand-in: discover_plugin_authz only calls ``.load()`` / reads ``.name``."""

    def __init__(self, name: str, loader) -> None:
        self.name = name
        self.value = f"test:{name}"
        self._loader = loader

    def load(self):
        return self._loader()


def test_authz_scope_mints_scopes_from_oauth_area() -> None:
    """Scope helpers mirror scopes_for(self.scope, ...); .child() keeps the parent area."""
    agents = AuthzScope("agents")
    assert agents.read_scopes() == scopes_for("agents", write=False) == ["agents:read", "platform:read"]
    assert agents.write_scopes() == scopes_for("agents", write=True) == ["agents:write", "platform:write"]
    # child() deepens the permission namespace but the scope area (hence the scopes) is unchanged.
    nested = agents.child("deployments")
    assert nested.namespace == "agents.deployments"
    assert nested.scope == "agents"
    assert nested.write_scopes() == scopes_for("agents", write=True)


def test_derive_contribution_composes_mounted_path(monkeypatch) -> None:
    """A service's @path_rule routes derive to the final /apis/<name>/<prefix> paths.

    The permission catalog (id -> description) is derived from the Permission objects on the
    routes — there is no separate declaration.
    """
    router = APIRouter()
    scope = AuthzScope("example")

    @router.get("/v2/workspaces/{workspace}/items/{name}")
    @scope.read
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[Permission("example", "items", "read", "Read example items")],
    )
    async def get_item(workspace: str, name: str) -> dict[str, str]:
        return {"name": name}

    class _Svc(NemoService):
        name = "example"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    def _fake_discover_entry_points(group: str) -> dict[str, _FakeEntryPoint]:
        assert group == "nemo.services"
        return {"example": _FakeEntryPoint("example", lambda: _Svc)}

    monkeypatch.setattr(
        "nemo_helix_plugin.discovery.discover_entry_points",
        _fake_discover_entry_points,
    )
    clear_plugin_authz_cache()
    try:
        contribs = discover_authz_contributions()
    finally:
        clear_plugin_authz_cache()

    assert len(contribs) == 1
    contrib = contribs[0]
    assert contrib.permissions == {"example.items.read": "Read example items"}

    path = "/apis/example/v2/workspaces/{workspace}/items/{name}"
    assert set(contrib.endpoints[path]) == {"get"}
    binding = contrib.endpoints[path]["get"]
    assert binding.permissions == ["example.items.read"]
    assert binding.scopes == ["example:read", "platform:read"]
    assert binding.callers == ["principal"]


def test_derive_service_only_route_emits_service_principal_callers() -> None:
    router = APIRouter()

    @router.post("/v2/internal/sync")
    @path_rule(callers=[CallerKind.SERVICE_PRINCIPAL])
    async def sync() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, problems, _warnings = _derive_service_contribution(_Svc())
    assert problems == []
    binding = contrib.endpoints["/apis/svc/v2/internal/sync"]["post"]
    assert binding.callers == ["service_principal"]
    assert binding.permissions == []


def test_derive_unions_callers_across_rules_with_shared_permissions() -> None:
    router = APIRouter()
    svc_read = Permission("svc", "", "read", "Read")

    @router.get("/v2/y")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[svc_read])
    @path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[svc_read])
    async def y() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, problems, _warnings = _derive_service_contribution(_Svc())
    assert problems == []
    binding = contrib.endpoints["/apis/svc/v2/y"]["get"]
    assert binding.callers == ["principal", "service_principal"]
    assert binding.permissions == ["svc.read"]


def test_derive_denies_route_with_or_of_distinct_permission_sets() -> None:
    """v1 cannot represent (principal & permA) OR (service & permB): the route is denied
    (fail-closed) with a recorded problem, without crashing the rest of the plugin."""
    router = APIRouter()

    @router.get("/v2/z")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "", "read", "Read svc")])
    @path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[Permission("svc", "", "internal", "Internal svc")])
    async def z() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, problems, _warnings = _derive_service_contribution(_Svc())
    assert contrib.endpoints["/apis/svc/v2/z"]["get"].deny is True
    assert any("distinct permission sets" in p for p in problems)


def test_derive_emits_deny_for_unruled_route() -> None:
    router = APIRouter()

    @router.get("/v2/unruled")
    async def unruled() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, problems, _warnings = _derive_service_contribution(_Svc())
    # Unruled routes are explicit-deny (fail-closed), never omitted.
    assert contrib.endpoints["/apis/svc/v2/unruled"]["get"].deny is True
    assert any("no @path_rule" in p for p in problems)


def test_extra_role_permissions_threaded_into_contribution() -> None:
    """A service can grant a role a permission the suffix heuristic wouldn't (e.g. ``.invoke`` to
    Viewer). The grant is registered in the catalog and threaded into ``role_permissions``."""

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return []

        def extra_role_permissions(self) -> dict[str, list[Permission]]:
            return {"Viewer": [Permission("svc", "gateway", "invoke", "Invoke")]}

    contrib, problems, _warnings = _derive_service_contribution(_Svc())
    assert problems == []
    assert contrib.role_permissions == {"Viewer": ["svc.gateway.invoke"]}
    # Registered in the catalog with its description even though no route references it.
    assert contrib.permissions == {"svc.gateway.invoke": "Invoke"}


def test_extra_role_permissions_outside_namespace_fails_closed() -> None:
    """Granting a role another service's permission is namespace squatting: the whole plugin
    fails closed (no permissions, no role grants) — the same fence applied to the route catalog."""

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return []

        def extra_role_permissions(self) -> dict[str, list[Permission]]:
            return {"Viewer": [Permission("other", "", "read", "Read other")]}

    contrib, problems, _warnings = _derive_service_contribution(_Svc())
    assert any("outside the service namespace" in p for p in problems)
    assert contrib.permissions == {}
    assert contrib.role_permissions == {}


def test_extra_role_permissions_hook_failure_recorded_not_raised() -> None:
    """A broken ``extra_role_permissions()`` is recorded as a problem, never raised — raising would
    drop the route bindings and let them fall through the ``service:`` no-match bypass."""

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return []

        def extra_role_permissions(self) -> dict[str, list[Permission]]:
            raise RuntimeError("boom")

    _contrib, problems, _warnings = _derive_service_contribution(_Svc())
    assert any("extra_role_permissions() raised" in p for p in problems)


def test_submit_remote_forwards_authorization_header() -> None:
    """Authenticated CLI submit passes Authorization to the protected job route."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.headers)
        return httpx.Response(200, json={"id": "job-123", "status": "queued"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    scheduler = NemoJobScheduler()

    result = scheduler.submit_remote(
        _ExampleSubmitJob,
        {"foo": "bar"},
        base_url="https://nhx.test",
        workspace="ws-a",
        headers={"Authorization": "Bearer test-token"},
        http_client=client,
    )

    assert result == {"id": "job-123", "status": "queued"}
    assert captured.get("authorization") == "Bearer test-token"
