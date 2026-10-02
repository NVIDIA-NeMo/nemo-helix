# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization policy for NeMo Helix plugins.

A plugin attaches :func:`path_rule` rules to its route handlers, referencing
:class:`Permission` objects from a typed :class:`PermissionSet`, and declares the route's
OAuth scope with ``@AuthzScope.read`` / ``@AuthzScope.write``. The platform derives
the normalized policy — the permission catalog (id + description), the per-endpoint
bindings, and the namespace — *entirely from the routes* (see
:mod:`nemo_helix_plugin.authz_discovery`) when the OPA bundle is built; it can also be
materialized into ``static-authz.yaml`` via ``auth-tools sync-plugins``.

The permission rule (callers + permissions) and the OAuth scope gate are independent
concerns, so they ride on the handler as two separate decorators: :func:`path_rule` and
``@AuthzScope.read``/``.write``. That keeps each declaration single-purpose and lets the
scope be read back and verified on its own (:func:`get_path_scope`).

There is no separate permission declaration to keep in sync: the permission *is* the
object referenced on the route, and it carries its own description. The only thing a
service declares apart from its routes is the optional escape hatch
:meth:`NemoService.extra_permissions` — for permissions that are not 1:1 with a route
(e.g. checked in middleware).

Example::

    from fastapi import APIRouter
    from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
    from nemo_helix_plugin.service import NemoService, RouterSpec

    scope = AuthzScope("example")

    class ExamplePerms(PermissionSet, namespace="example"):
        READ = perm("Read example items")  # -> Permission("example.read", ...)

    router = APIRouter()

    @router.get("/v2/workspaces/{workspace}/items/{name}")
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ExamplePerms.READ])
    async def get_item(workspace: str, name: str) -> dict: ...

    class ExampleService(NemoService):
        name = "example"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router)]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

# ---------------------------------------------------------------------------
# Plugin authoring API: a typed permission vocabulary + path rules.
#
# Plugins declare permissions as ``Permission`` constants (typically grouped in a
# ``PermissionSet``) and attach ``PathRule``s to route handlers with ``@path_rule``,
# referencing those constants. The platform derives the wire-format
# ``AuthzContribution`` (below) from the routes at startup — there is no separate
# permission list.
# ---------------------------------------------------------------------------


class CallerKind(StrEnum):
    """Who a route is intended for — a PDP *subject attribute*, not a permission.

    Plugin routes are ``PRINCIPAL`` (a normal authenticated user) or
    ``SERVICE_PRINCIPAL`` (a caller whose id is prefixed ``service:``). There is
    intentionally no ``ANON``: the only genuinely public routes are core
    infrastructure, hardcoded as a bypass in the PEP.
    """

    PRINCIPAL = "principal"
    SERVICE_PRINCIPAL = "service_principal"


# Caller kinds stamped on every factory-generated route (see ``jobs.api_factory`` and
# ``functions.routes``).
#
# Generated routes admit both kinds. Plugin controllers and other Helix services call
# these routes as ``service:<name>``, and a principal-only route is an unconditional PDP
# *deny* for a service principal — it overrides even the ServiceSystem wildcard, so no role
# grant recovers it. Caller kind is the wrong lever for narrowing service-to-service reach:
# it distinguishes humans from services, not one service from another, so a per-route knob
# here could only encode guesses about which service calls what. Scoping that reach properly
# needs well-scoped service tokens, which the platform does not have yet.
#
# Hand-written routes are unaffected and still choose their own callers via ``@path_rule`` —
# including the service-only routes that a controller writes to.
GENERATED_ROUTE_CALLERS: list[CallerKind] = [CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL]


@dataclass(frozen=True)
class Permission:
    """A service-owned permission, structured as ``service.resource.action``.

    The permission is authored as its parts, and the canonical wire id (:attr:`id`) is
    *derived* from them by joining — never the other way around. So reading a part is a
    field access, not a string split, and the id is computed exactly once (the object is
    frozen). Every id follows the same grammar:

    - ``service`` — the owning plugin (the first id segment; equals ``NemoService.name`` —
      the fail-closed namespace check enforces it).
    - ``resource`` — the collection / sub-resource acted on. ``""`` for a service-level
      action; may itself be dotted (e.g. ``deployment-configs.status``).
    - ``action`` — the operation (the final segment, e.g. ``create`` / ``read`` / ``update``).

    ``description`` is the one piece of authz data that cannot be derived from anything else,
    so it rides on the permission itself rather than in a parallel list. ``str(permission)``
    and :attr:`id` are the dotted wire value, so a ``Permission`` can be used wherever the id
    string is expected.
    """

    service: str
    resource: str
    action: str
    description: str
    id: str = field(init=False, compare=False, default="")
    """Canonical dotted wire id, joined from ``service.resource.action`` at construction.
    ``compare=False``: equality is over the structured parts, and the id is just their cached
    view (an empty ``resource`` drops out, so a service-level action is ``service.action``)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", ".".join(seg for seg in (self.service, self.resource, self.action) if seg))

    @property
    def namespace(self) -> str:
        """The dotted prefix (``service.resource``) — the id with the action removed."""
        return ".".join(seg for seg in (self.service, self.resource) if seg)

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True)
class AuthzScope:
    """The OAuth scope a plugin owns, plus the permission namespace minted beneath it.

    A plugin declares one ``AuthzScope`` (``AuthzScope("data-designer")``) and the route
    adapters derive every :class:`Permission` and scope list from it — the single front
    door for plugin route authz, so the ``<namespace>.<action>`` id format lives in one
    place rather than being hand-built per adapter.

    ``scope`` is the coarse OAuth grouping fed to :func:`scopes_for` (→ ``"<scope>:read"`` /
    ``"<scope>:write"``). ``namespace`` is the dotted prefix for permission ids and defaults
    to ``scope``; use :meth:`child` when the permission namespace nests deeper than the scope
    (an ``agents`` scope with per-collection ``agents.<collection>`` permissions, say)::

        AuthzScope("agents").child("deployments").permission("create", description="...")
        # -> Permission("agents.deployments.create", ...); scope stays "agents"
    """

    scope: str
    # Empty string is the "default to ``scope``" sentinel, resolved in ``__post_init__``;
    # a real permission namespace is never empty. Keeping the field ``str`` (not ``str | None``)
    # means ``namespace`` is always the effective dotted prefix for readers and the type checker.
    namespace: str = ""

    def __post_init__(self) -> None:
        if not self.namespace:
            object.__setattr__(self, "namespace", self.scope)

    def child(self, *segments: str) -> AuthzScope:
        """Return a scope whose permission namespace is deepened by *segments*; scope unchanged."""
        return AuthzScope(self.scope, ".".join((self.namespace, *segments)))

    @property
    def resource(self) -> str:
        """The permission ``resource`` path this scope mints under — its namespace beyond the service.

        ``""`` for a bare service-level scope; deepened by :meth:`child` (an ``agents`` scope
        with a ``deployments`` child has resource ``"deployments"``).
        """
        return self.namespace[len(self.scope) + 1 :] if self.namespace != self.scope else ""

    def permission(self, *segments: str, description: str) -> Permission:
        """Build the :class:`Permission` for an action under this scope.

        The final segment is the ``action``; any leading segments extend the ``resource`` path.

        ``AuthzScope("agents").child("deployments").permission("create", description="...")`` →
        ``Permission(service="agents", resource="deployments", action="create", ...)`` (id
        ``agents.deployments.create``).
        """
        if not segments:
            raise ValueError("AuthzScope.permission() requires at least one segment (the action)")
        *lead, action = segments
        resource = ".".join(seg for seg in (self.resource, *lead) if seg)
        return Permission(self.scope, resource, action, description)

    def read_scopes(self) -> list[str]:
        """The read scope strings for this area, e.g. ``["agents:read", "platform:read"]``.

        Built from :attr:`scope`, not :attr:`namespace`, so a :meth:`child` scope keeps the
        parent area. This is the raw list; :attr:`read` is the route decorator built from it.
        """
        return scopes_for(self.scope, write=False)

    def write_scopes(self) -> list[str]:
        """The write scope strings for this area, e.g. ``["agents:write", "platform:write"]``."""
        return scopes_for(self.scope, write=True)

    @property
    def read(self) -> Callable[[_F], _F]:
        """Route decorator stamping this area's **read** scope requirement (``@scope.read``).

        The scope gate is declared separately from :func:`path_rule`: the permission/caller
        rule and the OAuth scope are orthogonal, so each is attached, read back, and verified
        on its own (the scope via :func:`get_path_scope`). Order relative to ``@path_rule`` is
        irrelevant — both only stamp attributes on the handler and return it unchanged.
        """
        return _attach_scope(self.read_scopes())

    @property
    def write(self) -> Callable[[_F], _F]:
        """Route decorator stamping this area's **write** scope requirement (``@scope.write``)."""
        return _attach_scope(self.write_scopes())


@dataclass(frozen=True)
class _PendingPermission:
    """A permission declared inside a :class:`PermissionSet` body before its namespace
    is known. :meth:`PermissionSet.__init_subclass__` resolves it into a
    :class:`Permission` once the namespace is bound."""

    description: str
    suffix: str | None = None


def perm(description: str, *, suffix: str | None = None) -> Any:
    """Declare a permission inside a :class:`PermissionSet` body.

    The id is built as ``<namespace>.<member-name-lowercased>`` unless *suffix* is given
    (use *suffix* for compound ids, e.g. ``perm("...", suffix="configs.create")``). The
    return type is ``Any`` so the class attribute type-checks as a :class:`Permission`
    after ``__init_subclass__`` rewrites it.
    """
    return _PendingPermission(description, suffix)


class PermissionSet:
    """A closed, typed group of permissions under one namespace.

    Subclass with ``namespace=`` and assign ``perm(...)`` members; each becomes a
    :class:`Permission` whose id is ``<namespace>.<member-name-lowercased>`` (or the
    explicit ``suffix``). Referencing a member that doesn't exist is an ``AttributeError``
    at import — a permission typo can't reach the policy layer.

        class WidgetPerms(PermissionSet, namespace="widget"):
            CREATE = perm("Create a widget")   # -> Permission("widget.create", ...)
    """

    namespace: str
    _members: dict[str, Permission]

    def __init_subclass__(cls, *, namespace: str, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.namespace = namespace
        # The declared namespace is ``service`` + an optional resource prefix; the member's
        # suffix contributes any further resource segments plus the trailing action. Split
        # here, once, so each Permission is stored as structured parts (never re-split).
        service, _, namespace_resource = namespace.partition(".")
        cls._members = {}
        for name, value in list(vars(cls).items()):
            if isinstance(value, _PendingPermission):
                suffix = value.suffix or name.lower()
                *lead, action = suffix.split(".")
                resource = ".".join(seg for seg in (namespace_resource, *lead) if seg)
                resolved = Permission(service, resource, action, value.description)
                setattr(cls, name, resolved)
                cls._members[name] = resolved

    @classmethod
    def all(cls) -> list[Permission]:
        """Every permission declared on this set (handy for ``extra_permissions``)."""
        return list(cls._members.values())


@dataclass(frozen=True, kw_only=True)
class PathRule:
    """One alternative authorization rule for a route handler.

    Within a rule, ``callers`` are OR'd and ``permissions`` are AND'd. Multiple rules on
    one endpoint are OR'd (any satisfied rule allows access).

    ``method`` and ``path`` are unknown at decoration time and are filled in during
    derivation once the route is mounted (see ``authz_discovery``).
    """

    callers: list[CallerKind]
    permissions: list[Permission] = field(default_factory=list)
    method: str | None = None
    path: str | None = None


# Attribute used to stash the (OR-combined) ``PathRule``s on a route handler.
# Mutated in place — the function is never wrapped — so ``route.endpoint``
# identity survives FastAPI ``include_router(prefix=...)`` rebasing, which
# rebuilds ``APIRoute`` objects but passes the endpoint through by identity.
PATH_RULES_ATTR = "__nemo_path_rules__"

_F = TypeVar("_F", bound=Callable[..., Any])


def path_rule(
    *,
    callers: list[CallerKind],
    permissions: list[Permission] | None = None,
) -> Callable[[_F], _F]:
    """Attach an authorization rule (callers + permissions) to a route handler.

    Stacking ``@path_rule`` on the same handler adds alternative (OR) rules. The
    handler is returned **unchanged** (same object, same signature): the rule is
    stored on the function itself so it survives router rebasing.

    The route's OAuth **scope** is declared separately, with ``@AuthzScope.read`` /
    ``@AuthzScope.write`` (see :meth:`AuthzScope.read`) — the scope gate is orthogonal to
    the permission rule, so it lives in its own decorator and is read back independently
    via :func:`get_path_scope`.

    Args:
        callers: Non-empty list of caller kinds this rule applies to (OR'd).
        permissions: :class:`Permission` objects the caller must hold (AND'd). May be
            empty for authenticated-but-permissionless routes.

    Raises:
        ValueError: if *callers* is empty or contains an unknown caller kind.
        TypeError: if any *permissions* entry is not a :class:`Permission` (e.g. a bare
            string) — caught at decoration so a typo can't silently reach the policy layer.
    """
    resolved_callers = [CallerKind(c) for c in callers]
    if not resolved_callers:
        raise ValueError("@path_rule requires at least one caller kind")
    resolved_permissions = list(permissions or [])
    for p in resolved_permissions:
        if not isinstance(p, Permission):
            raise TypeError(
                f"@path_rule permissions must be Permission objects, got {type(p).__name__} ({p!r}). "
                f"Reference a PermissionSet member (e.g. MyPerms.READ) rather than a bare string."
            )
    rule = PathRule(callers=resolved_callers, permissions=resolved_permissions)

    def decorate(func: _F) -> _F:
        rules = func.__dict__.get(PATH_RULES_ATTR)
        if rules is None:
            rules = []
            setattr(func, PATH_RULES_ATTR, rules)
        rules.append(rule)
        return func

    return decorate


def get_path_rules(func: Callable[..., Any]) -> list[PathRule]:
    """Return the ``PathRule``s attached to *func* by :func:`path_rule` (empty if none)."""
    return list(getattr(func, PATH_RULES_ATTR, []))


# Attribute used to stash a route's required OAuth scope (an ``area:verb`` list) on its
# handler. Like ``PATH_RULES_ATTR`` it is set in place so the endpoint identity survives
# FastAPI ``include_router(prefix=...)`` rebasing.
PATH_SCOPE_ATTR = "__nemo_path_scope__"


def _attach_scope(scopes: list[str]) -> Callable[[_F], _F]:
    """Stamp a route's required OAuth scopes (``area:verb`` list) onto its handler.

    Backs ``@AuthzScope.read`` / ``@AuthzScope.write``. A route has a single scope
    requirement (the wire format and the Rego scope check hold one scope list per
    ``(path, method)``), so re-stamping the same scopes is idempotent but a *different*
    scope on the same handler is a :class:`ValueError`, surfaced at decoration rather than
    silently letting one win.
    """
    resolved = list(scopes)

    def decorate(func: _F) -> _F:
        existing = func.__dict__.get(PATH_SCOPE_ATTR)
        if existing is not None and existing != resolved:
            raise ValueError(
                f"conflicting scope on {getattr(func, '__name__', func)!r}: {existing} vs {resolved}; "
                f"a route declares a single scope (one @AuthzScope.read/.write)."
            )
        setattr(func, PATH_SCOPE_ATTR, resolved)
        return func

    return decorate


def get_path_scope(func: Callable[..., Any]) -> list[str] | None:
    """Return the OAuth scopes attached by ``@AuthzScope.read`` / ``.write`` (``None`` if unscoped)."""
    scopes = getattr(func, PATH_SCOPE_ATTR, None)
    return list(scopes) if scopes is not None else None


def validate_caller_strings(callers: list[str] | None, *, context: str) -> None:
    """Validate wire-format caller kinds. Absence (``None``) is allowed (⇒ PRINCIPAL).

    The valid set is derived from :class:`CallerKind` rather than hardcoded, so it
    cannot drift from the enum.

    Raises:
        ValueError: if any value is not a known :class:`CallerKind`.
    """
    if callers is None:
        return
    valid = {c.value for c in CallerKind}
    for c in callers:
        if c not in valid:
            raise ValueError(f"Invalid caller kind {c!r} in {context}: expected one of {sorted(valid)}.")


@dataclass(frozen=True)
class AuthzEndpointMethod:
    """One HTTP method binding for an API route."""

    permissions: list[str]
    scopes: list[str] | None = None
    callers: list[str] | None = None
    """Allowed caller kinds (:class:`CallerKind` values). ``None`` ⇒ PRINCIPAL (default)."""

    deny: bool = False
    """When True the route is unconditionally denied — the fail-closed marker for an
    unruled or invalid plugin route. The PDP denies it outright, overriding every allow
    rule (including the service ``*`` wildcard and the HelixAdmin bypass), so an
    un-annotated route can never fall through to the ``service:`` no-match bypass."""


@dataclass
class AuthzContribution:
    """Authorization data contributed by a plugin."""

    permissions: dict[str, str] = field(default_factory=dict)
    """Flat registry entries: ``permission_id`` → human-readable description."""

    endpoints: dict[str, dict[str, AuthzEndpointMethod]] = field(default_factory=dict)
    """Full API paths (``/apis/...``) → lower-case HTTP method → spec."""

    role_permissions: dict[str, list[str]] = field(default_factory=dict)
    """Optional explicit role → permission grants (merged with defaults)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for :func:`nhx.common.auth.authz_merge.merge_authz_contributions`."""
        return {
            "permissions": dict(self.permissions),
            "endpoints": {
                path: {
                    method: {
                        "permissions": spec.permissions,
                        **({"scopes": spec.scopes} if spec.scopes is not None else {}),
                        **({"callers": spec.callers} if spec.callers is not None else {}),
                        **({"deny": True} if spec.deny else {}),
                    }
                    for method, spec in methods.items()
                }
                for path, methods in self.endpoints.items()
            },
            "role_permissions": {role: list(perms) for role, perms in self.role_permissions.items()},
        }


def scopes_for(api_area: str, write: bool) -> list[str]:
    """Normalized NeMo scopes for a route: the api-area scope plus the platform scope."""
    verb = "write" if write else "read"
    return [f"{api_area}:{verb}", f"platform:{verb}"]
