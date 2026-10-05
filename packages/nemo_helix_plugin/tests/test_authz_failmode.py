# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed derivation: invalid plugin routes derive to explicit DENY + reported errors.

``_derive_service_contribution`` returns ``(contribution, errors, warnings)``. *Errors* are
deny-worthy (unruled routes, OR of distinct permission sets, duplicate ``(path, method)``,
malformed / cross-namespace permission ids, load/derivation failures). *Warnings* are
metadata-only (missing / conflicting permission descriptions) and never deny a route.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import APIRouter
from nemo_helix_plugin.authz import (
    AuthzContribution,
    AuthzEndpointMethod,
    CallerKind,
    Permission,
    path_rule,
)
from nemo_helix_plugin.authz_discovery import (
    _derive_service_contribution,
    _method_from_dict,
    clear_plugin_authz_cache,
    discover_plugin_authz,
)
from nemo_helix_plugin.service import NemoService, RouterSpec


class _FakeEntryPoint:
    """Minimal ``importlib.metadata.EntryPoint`` stand-in for discovery tests.

    ``discover_plugin_authz`` enumerates ``discover_entry_points("nemo.services")`` and calls
    ``ep.load()`` per entry in its own try/except, so a fake only needs ``name`` and ``load``.
    """

    def __init__(self, name: str, loader: Callable[[], object]) -> None:
        self.name = name
        self.value = f"test:{name}"
        self._loader = loader

    def load(self) -> object:
        return self._loader()


def _patch_services(monkeypatch: pytest.MonkeyPatch, entry_points: dict[str, _FakeEntryPoint]) -> None:
    monkeypatch.setattr("nemo_helix_plugin.discovery.discover_entry_points", lambda group: entry_points)


def test_deny_field_round_trips_through_wire_format() -> None:
    contrib = AuthzContribution(endpoints={"/x": {"get": AuthzEndpointMethod(permissions=[], deny=True)}})
    serialized = contrib.to_dict()["endpoints"]["/x"]["get"]
    assert serialized["deny"] is True
    assert _method_from_dict(serialized).deny is True
    # Absent deny defaults to False (and is omitted from the wire form).
    assert (
        "deny"
        not in AuthzContribution(endpoints={"/x": {"get": AuthzEndpointMethod(permissions=["a"])}}).to_dict()[
            "endpoints"
        ]["/x"]["get"]
    )
    assert _method_from_dict({"permissions": []}).deny is False


def test_permissions_outside_service_namespace_fail_closed() -> None:
    """A permission whose first segment isn't the service's own name is squatting: every route
    is denied (fail-closed) and no permissions are contributed."""
    router = APIRouter()

    @router.get("/v2/x")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "x", "read", "Read x")])
    async def x() -> None: ...

    @router.get("/v2/y")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("other", "y", "read", "Read y")])
    async def y() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, errors, _warnings = _derive_service_contribution(_Svc())
    assert contrib.endpoints["/apis/svc/v2/x"]["get"].deny is True
    assert contrib.endpoints["/apis/svc/v2/y"]["get"].deny is True
    assert contrib.permissions == {}
    assert any("outside the service namespace" in e and "other.y.read" in e for e in errors)


def test_malformed_permission_id_fails_closed() -> None:
    """A permission id that isn't dot-separated lowercase segments would 500 the bundle's
    validate_static_authz_data if it reached the wire — so it fails the plugin closed here."""
    router = APIRouter()

    @router.get("/v2/x")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "", "bad_segment", "Read x")])
    async def x() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, errors, _warnings = _derive_service_contribution(_Svc())
    assert contrib.endpoints["/apis/svc/v2/x"]["get"].deny is True
    assert contrib.permissions == {}
    assert any("malformed permission id" in e for e in errors)


def test_duplicate_path_method_binding_fails_closed() -> None:
    """Two handlers on the same (path, method): Starlette serves the first, but the derived
    policy could describe the second. Refuse to guess — deny the pair and flag it."""
    router = APIRouter()

    @router.get("/v2/dup")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "", "read", "Read")])
    async def first() -> None: ...

    @router.get("/v2/dup")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "", "read", "Read")])
    async def second() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, errors, _warnings = _derive_service_contribution(_Svc())
    assert contrib.endpoints["/apis/svc/v2/dup"]["get"].deny is True
    assert any("duplicate route binding" in e for e in errors)


def test_websocket_route_is_warned_not_denied() -> None:
    """A WebSocket/ASGI route never reaches the BaseHTTPMiddleware PDP, so a derived deny would
    be inert: it surfaces as a (non-deny) warning rather than an error, and the plugin's HTTP
    routes are unaffected."""
    from fastapi import WebSocket

    router = APIRouter()

    @router.get("/v2/x")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "", "read", "Read")])
    async def x() -> None: ...

    @router.websocket("/v2/stream")
    async def stream(ws: WebSocket) -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, errors, warnings = _derive_service_contribution(_Svc())
    assert errors == []
    assert any("not an APIRoute" in w for w in warnings)
    assert contrib.endpoints["/apis/svc/v2/x"]["get"].deny is False


def test_missing_permission_description_is_warning_not_deny() -> None:
    router = APIRouter()

    @router.get("/v2/x")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "x", "read", "")])
    async def x() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, errors, warnings = _derive_service_contribution(_Svc())
    # A missing description is metadata-only: it's a warning, never an error, and the route
    # still requires the right permission (it is not denied).
    assert errors == []
    assert any("missing a description" in w for w in warnings)
    assert contrib.endpoints["/apis/svc/v2/x"]["get"].deny is False


def test_conflicting_descriptions_for_same_id_is_warning() -> None:
    router = APIRouter()

    @router.get("/v2/a")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "", "read", "Read")])
    async def a() -> None: ...

    @router.get("/v2/b")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "", "read", "Totally different")])
    async def b() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, errors, warnings = _derive_service_contribution(_Svc())
    # A description conflict is cosmetic — it must not deny a route or escalate the fail-mode.
    assert errors == []
    assert any("conflicting descriptions" in w for w in warnings)
    assert contrib.endpoints["/apis/svc/v2/a"]["get"].deny is False
    assert contrib.endpoints["/apis/svc/v2/b"]["get"].deny is False


def test_extra_permissions_adds_non_route_permission_to_catalog() -> None:
    """The escape hatch contributes a permission with no 1:1 route (e.g. middleware-checked)."""
    router = APIRouter()

    @router.get("/v2/x")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "", "read", "Read")])
    async def x() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

        def extra_permissions(self) -> list[Permission]:
            return [Permission("svc", "", "admin", "Administer svc")]

    contrib, errors, _warnings = _derive_service_contribution(_Svc())
    assert errors == []
    assert contrib.permissions == {"svc.read": "Read", "svc.admin": "Administer svc"}
    # The extra permission has no endpoint binding.
    assert all("svc.admin" not in m.permissions for methods in contrib.endpoints.values() for m in methods.values())


def test_extra_permissions_failure_is_reported_routes_survive() -> None:
    router = APIRouter()

    @router.get("/v2/x")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "", "read", "Read")])
    async def x() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

        def extra_permissions(self) -> list[Permission]:
            raise RuntimeError("boom")

    contrib, errors, _warnings = _derive_service_contribution(_Svc())
    # A broken hatch loses its extras but never invalidates the route-derived authz.
    assert any("extra_permissions() raised" in e for e in errors)
    assert contrib.endpoints["/apis/svc/v2/x"]["get"].deny is False
    assert contrib.endpoints["/apis/svc/v2/x"]["get"].permissions == ["svc.read"]


def test_discover_plugin_authz_reports_unruled_route(monkeypatch: pytest.MonkeyPatch) -> None:
    router = APIRouter()

    @router.get("/v2/unruled")
    async def unruled() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    _patch_services(monkeypatch, {"svc": _FakeEntryPoint("svc", lambda: _Svc)})
    clear_plugin_authz_cache()
    try:
        results = discover_plugin_authz()
    finally:
        clear_plugin_authz_cache()

    assert len(results) == 1
    assert results[0].key == "svc"
    assert results[0].problems
    assert results[0].contribution.endpoints["/apis/svc/v2/unruled"]["get"].deny is True


def test_discover_plugin_authz_records_import_load_failure_as_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry point whose ``load()`` raises (broken import) becomes a degraded result keyed
    by the entry-point name, never silently dropped — per-entry-point isolation."""

    def _boom() -> object:
        raise ImportError("module not found")

    _patch_services(monkeypatch, {"broken": _FakeEntryPoint("broken", _boom)})
    clear_plugin_authz_cache()
    try:
        results = discover_plugin_authz()
    finally:
        clear_plugin_authz_cache()

    assert len(results) == 1
    assert results[0].key == "broken"
    assert any("failed to load" in p for p in results[0].problems)
    assert results[0].contribution.endpoints == {}
    assert results[0].mount_name == "broken"


def test_discover_plugin_authz_records_derivation_failure_as_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A class that loads but whose route walk raises is a degraded (derivation) failure,
    distinct from a load failure — and the degraded fence still covers /apis/<name>."""

    class _BadSvc(NemoService):
        name = "bad"

        def get_routers(self) -> list[RouterSpec]:
            raise RuntimeError("boom")

    _patch_services(monkeypatch, {"bad": _FakeEntryPoint("bad", lambda: _BadSvc)})
    clear_plugin_authz_cache()
    try:
        results = discover_plugin_authz()
    finally:
        clear_plugin_authz_cache()

    assert len(results) == 1
    assert results[0].key == "bad"
    assert any("failed to derive" in p for p in results[0].problems)
    assert results[0].contribution.endpoints == {}
    assert results[0].mount_name == "bad"


def test_degraded_result_is_not_cached_but_clean_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """A degraded derivation is never pinned for the process lifetime (that would 403 the
    namespace until restart): the next call re-derives. An all-clean derivation is cached."""
    calls = {"n": 0}

    class _FlakySvc(NemoService):
        name = "flaky"
        fail = True

        def get_routers(self) -> list[RouterSpec]:
            calls["n"] += 1
            if type(self).fail:
                raise RuntimeError("transient boom")
            router = APIRouter()

            @router.get("/v2/x")
            @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("flaky", "", "read", "Read")])
            async def x() -> None: ...

            return [RouterSpec(router)]

    _patch_services(monkeypatch, {"flaky": _FakeEntryPoint("flaky", lambda: _FlakySvc)})
    clear_plugin_authz_cache()
    try:
        first = discover_plugin_authz()
        second = discover_plugin_authz()
        # Degraded: both calls re-derive (the failure is not cached).
        assert first[0].problems and second[0].problems
        assert calls["n"] == 2

        # Failure clears; the next derivation is clean and gets cached.
        _FlakySvc.fail = False
        third = discover_plugin_authz()
        fourth = discover_plugin_authz()
        assert third[0].problems == [] and fourth[0].problems == []
        assert calls["n"] == 3  # third derived; fourth served from cache
    finally:
        clear_plugin_authz_cache()


def test_clean_plugin_has_no_problems(monkeypatch: pytest.MonkeyPatch) -> None:
    router = APIRouter()

    @router.get("/v2/items/{name}")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[Permission("svc", "items", "read", "Read items")])
    async def get_item(name: str) -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    _patch_services(monkeypatch, {"svc": _FakeEntryPoint("svc", lambda: _Svc)})
    clear_plugin_authz_cache()
    try:
        results = discover_plugin_authz()
    finally:
        clear_plugin_authz_cache()

    assert results[0].problems == []
    assert results[0].warnings == []
    assert results[0].contribution.endpoints["/apis/svc/v2/items/{name}"]["get"].deny is False


def test_malformed_route_denies_only_itself_not_the_plugin() -> None:
    """A route whose rules can't collapse denies only itself — the plugin's other routes survive."""
    router = APIRouter()
    svc_read = Permission("svc", "", "read", "Read")

    @router.get("/v2/bad")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[svc_read])
    @path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[Permission("svc", "", "internal", "Internal")])
    async def bad() -> None: ...

    @router.get("/v2/good")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[svc_read])
    async def good() -> None: ...

    class _Svc(NemoService):
        name = "svc"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]

    contrib, errors, _warnings = _derive_service_contribution(_Svc())
    assert contrib.endpoints["/apis/svc/v2/bad"]["get"].deny is True
    assert contrib.endpoints["/apis/svc/v2/good"]["get"].deny is False
    assert contrib.endpoints["/apis/svc/v2/good"]["get"].permissions == ["svc.read"]
    assert any("distinct permission sets" in e for e in errors)
