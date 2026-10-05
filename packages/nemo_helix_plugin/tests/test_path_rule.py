# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoring API: CallerKind, Permission, PermissionSet, PathRule, @path_rule, callers plumbing."""

from __future__ import annotations

import inspect

import pytest
from fastapi import APIRouter
from fastapi.routing import APIRoute
from nemo_helix_plugin.authz import (
    AuthzContribution,
    AuthzEndpointMethod,
    AuthzScope,
    CallerKind,
    PathRule,
    Permission,
    PermissionSet,
    get_path_rules,
    get_path_scope,
    path_rule,
    perm,
    validate_caller_strings,
)
from nemo_helix_plugin.authz_discovery import _iter_composed_routes, _method_from_dict
from nemo_helix_plugin.service import NemoService, RouterSpec

_READ = Permission("x", "", "read", "Read x")


def test_caller_kind_values_and_no_anon() -> None:
    assert CallerKind.PRINCIPAL == "principal"
    assert CallerKind.SERVICE_PRINCIPAL == "service_principal"
    assert {c.value for c in CallerKind} == {"principal", "service_principal"}
    assert not hasattr(CallerKind, "ANON")


def test_permission_id_is_joined_from_structured_parts() -> None:
    permission = Permission(service="agents", resource="deployments", action="read", description="Read deployments")
    # id (and str) are the dotted wire value joined from the parts — never split back out.
    assert (permission.service, permission.resource, permission.action) == ("agents", "deployments", "read")
    assert permission.id == "agents.deployments.read"
    assert permission.namespace == "agents.deployments"
    assert str(permission) == "agents.deployments.read"
    with pytest.raises(Exception):  # frozen dataclass: FrozenInstanceError
        permission.id = "other"  # type: ignore[misc]


def test_service_level_permission_has_no_resource_segment() -> None:
    # An empty resource drops out of the id: a service-level action is just ``service.action``.
    permission = Permission(service="evaluator", resource="", action="create", description="Create")
    assert permission.id == "evaluator.create"
    assert permission.namespace == "evaluator"


def test_permission_set_derives_ids_from_namespace_and_member_name() -> None:
    class WidgetPerms(PermissionSet, namespace="widget"):
        CREATE = perm("Create a widget")
        BULK = perm("Bulk export", suffix="bulk.export")

    assert WidgetPerms.CREATE == Permission("widget", "", "create", "Create a widget")
    # A dotted suffix contributes resource segments; the final segment is the action.
    assert WidgetPerms.BULK == Permission("widget", "bulk", "export", "Bulk export")
    assert (WidgetPerms.BULK.service, WidgetPerms.BULK.resource, WidgetPerms.BULK.action) == (
        "widget",
        "bulk",
        "export",
    )
    assert set(WidgetPerms.all()) == {WidgetPerms.CREATE, WidgetPerms.BULK}
    # A typo'd member doesn't exist — caught at access time, not at the policy layer.
    assert not hasattr(WidgetPerms, "CRAETE")


def test_path_rule_returns_identical_function_and_signature() -> None:
    async def handler(name: str, count: int = 0) -> dict[str, str]:
        return {"name": name}

    before = inspect.signature(handler)
    decorated = path_rule(callers=[CallerKind.PRINCIPAL], permissions=[_READ])(handler)

    # D5: same object, unchanged signature — never wrapped.
    assert decorated is handler
    assert inspect.signature(handler) == before


def test_path_rule_attaches_rule() -> None:
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[_READ])
    async def handler() -> None: ...

    rules = get_path_rules(handler)
    assert rules == [PathRule(callers=[CallerKind.PRINCIPAL], permissions=[_READ])]


def test_scope_decorator_attaches_scope_independent_of_rule() -> None:
    """@AuthzScope.read stamps the OAuth scope on its own, separate from @path_rule."""
    scope = AuthzScope("x")

    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[_READ])
    async def handler() -> None: ...

    # The scope is read back independently of the permission rule.
    assert get_path_scope(handler) == ["x:read", "platform:read"]
    assert get_path_rules(handler) == [PathRule(callers=[CallerKind.PRINCIPAL], permissions=[_READ])]


def test_scope_decorator_returns_same_function() -> None:
    scope = AuthzScope("x")

    async def handler() -> None: ...

    # The decorator never wraps — same object back, scope stamped on it.
    assert scope.write(handler) is handler
    assert get_path_scope(handler) == ["x:write", "platform:write"]


def test_scope_decorator_order_relative_to_path_rule_is_irrelevant() -> None:
    """Both decorators only stamp attributes, so stacking order doesn't change the result."""
    scope = AuthzScope("x")

    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[_READ])
    @scope.read
    async def handler() -> None: ...

    assert get_path_scope(handler) == ["x:read", "platform:read"]
    assert get_path_rules(handler)[0].permissions == [_READ]


def test_get_path_scope_none_when_unscoped() -> None:
    async def handler() -> None: ...

    assert get_path_scope(handler) is None


def test_same_scope_reattached_is_idempotent() -> None:
    scope = AuthzScope("x")

    async def handler() -> None: ...

    scope.read(handler)
    scope.read(handler)  # idempotent — same scope, no error
    assert get_path_scope(handler) == ["x:read", "platform:read"]


def test_conflicting_scope_on_one_handler_raises() -> None:
    """A handler declares a single scope; a second, different one is caught at decoration."""

    async def handler() -> None: ...

    AuthzScope("x").read(handler)
    with pytest.raises(ValueError, match="conflicting scope"):
        AuthzScope("y").read(handler)


def test_read_and_write_scope_on_one_endpoint_is_rejected() -> None:
    """An endpoint has a single scope: stacking @scope.read and @scope.write is rejected.

    Read and write of the same area are different scope lists, so attaching both to one handler
    trips the conflict guard at decoration (import) time — never silently last-writer-wins. Both
    stacking orders are checked, since decorators apply bottom-up.
    """
    scope = AuthzScope("x")

    with pytest.raises(ValueError, match="conflicting scope"):

        @scope.write
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[_READ])
        async def read_then_write() -> None: ...

    with pytest.raises(ValueError, match="conflicting scope"):

        @scope.read
        @scope.write
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[_READ])
        async def write_then_read() -> None: ...


def test_path_rule_stacks_as_or() -> None:
    # Decorators apply bottom-up; both rules end up attached as OR alternatives.
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[_READ])
    @path_rule(callers=[CallerKind.SERVICE_PRINCIPAL])
    async def handler() -> None: ...

    rules = get_path_rules(handler)
    assert len(rules) == 2
    assert {tuple(r.callers) for r in rules} == {
        (CallerKind.SERVICE_PRINCIPAL,),
        (CallerKind.PRINCIPAL,),
    }


def test_path_rule_empty_callers_rejected() -> None:
    with pytest.raises(ValueError, match="at least one caller"):
        path_rule(callers=[])


def test_path_rule_coerces_and_validates_caller_strings() -> None:
    # Strings are coerced to CallerKind; unknown values raise.
    @path_rule(callers=["principal"])  # type: ignore[list-item]
    async def ok() -> None: ...

    assert get_path_rules(ok)[0].callers == [CallerKind.PRINCIPAL]

    with pytest.raises(ValueError):
        path_rule(callers=["anon"])  # type: ignore[list-item]


def test_path_rule_rejects_bare_string_permission() -> None:
    """A permission must be a Permission object. A bare string is rejected at decoration so a
    typo (or a forgotten PermissionSet member) can't silently reach the policy layer."""
    with pytest.raises(TypeError, match="must be Permission objects"):
        path_rule(callers=[CallerKind.PRINCIPAL], permissions=["x.read"])  # type: ignore[list-item]


def test_get_path_rules_empty_for_undecorated() -> None:
    async def handler() -> None: ...

    assert get_path_rules(handler) == []


def test_path_rule_survives_router_prefix_rebasing() -> None:
    """D5: function-attached metadata must survive include_router(prefix=...) rebasing.

    fastapi 0.138.0 makes ``include_router(prefix=...)`` lazy (rebased routes live behind a
    ``_IncludedRouter`` proxy, not in ``.routes``), so discoverability is asserted via the
    derivation's composed-route enumeration rather than by scanning raw ``.routes``.
    """
    router = APIRouter()
    items_read = Permission("items", "", "read", "Read items")

    @router.get("/items/{name}")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[items_read])
    async def get_item(name: str) -> dict[str, str]:
        return {"name": name}

    # Two prefix hops, as a real plugin mount does (/apis/<plugin> then workspace prefix).
    # _iter_composed_routes re-creates the /apis/<name> mount, so the spec supplies only the
    # inner workspace prefix; the helper prepends /apis/example.
    inner = APIRouter()
    inner.include_router(router, prefix="/v2/workspaces/{workspace}")

    class _Svc(NemoService):
        name = "example"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(inner)]

    matching = [
        r for r in _iter_composed_routes(_Svc()) if isinstance(r, APIRoute) and r.path.endswith("/items/{name}")
    ]
    assert len(matching) == 1
    final_route = matching[0]
    assert final_route.path == "/apis/example/v2/workspaces/{workspace}/items/{name}"

    # Metadata survived the rebase: the rule is still readable off the (identity-preserved) endpoint.
    rules = get_path_rules(final_route.endpoint)
    assert len(rules) == 1
    assert rules[0].callers == [CallerKind.PRINCIPAL]
    assert rules[0].permissions == [items_read]


def test_extra_permissions_default_empty() -> None:
    class _Svc(NemoService):
        name = "example-svc"

        def get_routers(self) -> list[RouterSpec]:
            return []

    assert _Svc().extra_permissions() == []


def test_authz_endpoint_method_callers_roundtrip() -> None:
    contrib = AuthzContribution(
        endpoints={
            "/apis/x/v2/thing": {
                "post": AuthzEndpointMethod(
                    permissions=["x.create"],
                    scopes=["x:write"],
                    callers=["service_principal"],
                ),
                "get": AuthzEndpointMethod(permissions=["x.read"]),
            }
        }
    )
    serialized = contrib.to_dict()
    post = serialized["endpoints"]["/apis/x/v2/thing"]["post"]
    get = serialized["endpoints"]["/apis/x/v2/thing"]["get"]

    # Present callers serialize; absent callers are omitted (default ⇒ PRINCIPAL).
    assert post["callers"] == ["service_principal"]
    assert "callers" not in get

    # Round-trip through the parse chokepoint.
    assert _method_from_dict(post).callers == ["service_principal"]
    assert _method_from_dict(get).callers is None


def test_validate_caller_strings() -> None:
    validate_caller_strings(None, context="t")  # absence allowed
    validate_caller_strings(["principal", "service_principal"], context="t")
    with pytest.raises(ValueError, match="Invalid caller kind"):
        validate_caller_strings(["anon"], context="t")
