# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive plugin authorization contributions from the ``NemoService`` route surface.

Plugins attach :func:`~nemo_helix_plugin.authz.path_rule` rules to route handlers,
referencing :class:`~nemo_helix_plugin.authz.Permission` constants. This module
instantiates each discovered ``NemoService``, walks its mounted routes — computing the
same ``/apis/<name>/<prefix>`` paths the platform mounts at runtime — reads the
function-attached :class:`~nemo_helix_plugin.authz.PathRule`\\ s, and builds the
wire-format :class:`~nemo_helix_plugin.authz.AuthzContribution` consumed by the OPA
bundle builder and ``auth-tools sync-plugins``.

The permission catalog (ids + descriptions) and the service namespace are derived
*entirely from the routes* (plus the optional :meth:`NemoService.extra_permissions`
hatch). There is no separately-declared permission list: the permission is the object
referenced on the route, and it carries its own description.

Path composition mirrors production: the platform runner mounts each service app at
``/apis/<service.name>`` and the service app includes each ``RouterSpec`` router at its
``prefix`` (see ``nhx.platform_runner.server`` and ``nhx.common.service.base``). We
re-create that composition with a throwaway router so FastAPI computes the final paths —
including prefix joins and ``{param:path}`` wildcards — exactly as it does at runtime.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter
from fastapi.routing import APIRoute
from nemo_helix_plugin.authz import (
    AuthzContribution,
    AuthzEndpointMethod,
    CallerKind,
    PathRule,
    Permission,
    get_path_rules,
    get_path_scope,
)
from nemo_helix_plugin.authz_format import is_valid_permission_id
from nemo_helix_plugin.service import NemoService
from starlette.routing import BaseRoute

logger = logging.getLogger(__name__)


def _method_from_dict(spec: dict[str, Any]) -> AuthzEndpointMethod:
    """Parse a serialized endpoint-method dict back into :class:`AuthzEndpointMethod`.

    This is the single chokepoint that decides which wire fields are preserved; unknown
    keys are dropped. Kept as the canonical inverse of ``AuthzContribution.to_dict`` for
    round-trip validation and bundle-side parsing.
    """
    return AuthzEndpointMethod(
        permissions=list(spec.get("permissions") or []),
        scopes=list(spec["scopes"]) if spec.get("scopes") is not None else None,
        callers=list(spec["callers"]) if spec.get("callers") is not None else None,
        deny=bool(spec.get("deny", False)),
    )


def _wire_callers(rules: list[PathRule]) -> list[str] | None:
    """Union the caller kinds across an endpoint's (OR'd) rules into the wire list.

    Returns ``None`` when no rule declares callers (the route falls back to the PRINCIPAL
    default and the Rego layer adds no caller-kind restriction).
    """
    kinds = {c.value if isinstance(c, CallerKind) else str(c) for rule in rules for c in rule.callers}
    return sorted(kinds) if kinds else None


def _collapse_rules(
    rules: list[PathRule], *, path: str, method: str, service: str
) -> tuple[list[Permission], list[str] | None]:
    """Collapse the (OR'd) ``PathRule``\\ s on one ``(path, method)`` into one binding.

    v1 supports OR across rules only in the **caller** dimension: caller kinds are unioned,
    but ``permissions`` must agree across rules. The single-slot wire format (one AND'd
    ``permissions`` list per method) and the Rego permission check cannot represent an OR of
    *distinct* permission sets, so that case is rejected loudly rather than silently
    mis-authorized.

    Scope is not a rule dimension — it is declared once per route with ``@AuthzScope.read`` /
    ``.write`` and read separately via :func:`~nemo_helix_plugin.authz.get_path_scope`, so
    there is nothing to reconcile across rules here.

    Returns ``(permissions, callers)`` for the representative rule.
    """
    # Only callers are OR'd across rules (unioned below); their permissions must match, since
    # the wire format holds one permissions list per (path, method). Compare by Permission.id —
    # the ``service.resource.action`` wire key joined from each Permission's structured parts —
    # not by object, so a description-only difference isn't read as a distinct set. Reject a
    # mismatch rather than silently picking one rule's.
    distinct_permission_sets = {frozenset(p.id for p in rule.permissions) for rule in rules}
    if len(distinct_permission_sets) > 1:
        raise ValueError(
            f"{service}: {method.upper()} {path} has @path_rule rules with differing "
            f"permissions ({[sorted(p) for p in distinct_permission_sets]}). v1 cannot represent an OR of "
            f"distinct permission sets — use one rule with shared permissions, or a single "
            f"rule listing multiple callers."
        )

    representative = rules[0]
    return list(representative.permissions), _wire_callers(rules)


@dataclass
class PluginAuthzResult:
    """One plugin's derived authz, before the bundle applies its fail-mode policy.

    ``problems`` are deny-worthy **errors**: unruled routes, an OR of distinct permission
    sets, a duplicate ``(path, method)``, a malformed permission id, a permission outside
    the service's own namespace, or a load/derivation failure. The affected routes are
    already emitted as explicit DENY bindings in ``contribution`` (fail-closed), and the
    bundle decides — via ``authz.on_invalid_plugin`` — whether to keep just those denies
    (``deny_route``), deny the whole plugin (``quarantine``), or refuse to build the bundle
    (``hard_fail``).

    ``warnings`` are non-deny-worthy: a missing or conflicting permission *description*.
    These are metadata-only — the route still requires the right permission, so they are
    surfaced (logged / status endpoint) but never escalate ``on_invalid_plugin`` and never
    deny a route. Keeping them out of ``problems`` is what stops a cosmetic description
    typo from quarantining a whole plugin (or hard-failing the bundle).
    """

    key: str
    contribution: AuthzContribution
    problems: list[str]
    warnings: list[str] = field(default_factory=list)
    mount_name: str = ""
    """The ``/apis/<mount_name>`` segment the runner mounts this service at (its
    ``NemoService.name``). Captured so the degraded-plugin namespace fence can cover the real
    mount path even when it diverges from the entry-point ``key`` — the ``name == key``
    invariant is only warned, not enforced (see ``discover_services``)."""


def _deny_binding() -> AuthzEndpointMethod:
    """A wire binding that the PDP denies unconditionally (fail-closed marker)."""
    return AuthzEndpointMethod(permissions=[], deny=True)


def _register_permission(catalog: dict[str, Permission], perm: Permission, warnings: list[str]) -> None:
    """Record *perm* in *catalog*, flagging a missing or conflicting description as a warning.

    Description problems are metadata-only (the route still requires the right permission),
    so they are surfaced but never deny a route. Id-format validity and namespace ownership
    are checked over the whole catalog in :func:`_derive_service_contribution`.
    """
    if not perm.description:
        warnings.append(f"permission {perm.id!r} is missing a description")
    previous = catalog.get(perm.id)
    if previous is not None and previous.description != perm.description:
        warnings.append(
            f"permission {perm.id!r} defined with conflicting descriptions: "
            f"{previous.description!r} != {perm.description!r}"
        )
    catalog.setdefault(perm.id, perm)


def _iter_composed_routes(service: NemoService) -> Iterator[BaseRoute]:
    """Yield the fully-composed leaf routes of *service*, one ``BaseRoute`` per mounted route.

    This re-creates the runtime mount (``/apis/<name>`` + ``RouterSpec.prefix`` + route path)
    and flattens it to leaves so the derivation can read each route's final ``.path``,
    ``.methods``, and (for ``APIRoute``) original ``.endpoint``.

    Lazy-include workaround (fastapi 0.138.0 / starlette 1.3.1): ``include_router(prefix=...)``
    no longer materializes rebased ``APIRoute`` objects into ``.routes`` — it stores a
    ``fastapi.routing._IncludedRouter`` proxy, so walking ``.routes`` for ``APIRoute`` finds
    nothing. We descend each proxy via ``effective_route_contexts()`` (which also recurses
    through nested includes) and reconstruct the composed leaves:

    - For an ``APIRoute`` we shallow-copy the original and overwrite ``.path``/``.methods`` with
      the context's composed values. The copy keeps ``isinstance(route, APIRoute)`` true and
      preserves the original ``.endpoint`` object so ``get_path_rules(route.endpoint)`` still
      finds the function-attached rules; copying (rather than mutating) avoids corrupting the
      shared original route.
    - For non-``APIRoute`` leaves (WebSocketRoute / Mount / plain Route) we yield the context's
      ``starlette_route``, which already carries the composed path — so the existing
      fail-closed / warning branches still fire and no route is silently dropped.

    A concrete route appearing directly in ``.routes`` (e.g. a future eager-include path) is
    yielded as-is, so this keeps working if the proxy behavior changes again.
    """
    composed = APIRouter()
    for spec in service.get_routers():
        composed.include_router(spec.router, prefix=f"/apis/{service.name}{spec.prefix}")

    for route in composed.routes:
        contexts = getattr(route, "effective_route_contexts", None)
        if contexts is None:
            # Already a concrete leaf (no lazy-include proxy) — pass it through unchanged.
            yield route
            continue
        for ctx in contexts():
            original = ctx.original_route
            if isinstance(original, APIRoute):
                # Rebased APIRoute: copy + composed path/methods, original endpoint preserved.
                rebased = copy.copy(original)
                rebased.path = ctx.path
                rebased.methods = ctx.methods
                yield rebased
            else:
                # WS / Mount / plain Route: the composed-path route is on the context.
                yield ctx.starlette_route or original


def _derive_service_contribution(service: NemoService) -> tuple[AuthzContribution, list[str], list[str]]:
    """Derive one plugin's wire contribution, split into deny-worthy errors and warnings.

    Every mounted route must carry a valid ``@path_rule``. A route that doesn't — unruled,
    an unrepresentable OR of distinct permission sets, or a duplicate ``(path, method)`` — is
    emitted as an explicit DENY binding (never omitted), so it can never fall through to the
    ``service:`` no-match bypass. The permission catalog is derived from the permissions the
    routes reference plus ``extra_permissions()``; if any permission id is malformed, or sits
    outside the service's own ``/apis/<name>`` namespace, the whole plugin fails closed.

    Returns ``(contribution, errors, warnings)``. ``errors`` are deny-worthy and drive the
    bundle fail-mode; ``warnings`` (missing/conflicting descriptions) are metadata-only and
    never deny a route.
    """
    errors: list[str] = []
    warnings: list[str] = []
    catalog: dict[str, Permission] = {}

    # Pass 1: walk the fully-composed leaf routes (/apis/<name> + RouterSpec.prefix + route
    # path), collapse OR'd rules, and collect referenced permissions. ``bindings`` holds the
    # tentative allow binding per (path, method); unruled / invalid routes are recorded as None
    # and become DENY regardless of namespace validity.
    bindings: dict[str, dict[str, AuthzEndpointMethod | None]] = {}
    for route in _iter_composed_routes(service):
        if not isinstance(route, APIRoute):
            # Mount / plain Starlette Route / WebSocket route — not an HTTP API route the PDP
            # binds by (path, method). Never silently skip it (that lets it fall through the
            # service: no-match bypass).
            other_path = getattr(route, "path", None) or repr(route)
            other_methods = sorted(getattr(route, "methods", None) or set())
            if other_methods:
                # Has HTTP methods (Mount / plain Route): the PDP could enforce it but we can't
                # derive a rule, so deny those methods (fail-closed) and flag it as an error.
                errors.append(f"{other_path} is a {type(route).__name__}, not an APIRoute — denied (fail-closed)")
                for http_method in other_methods:
                    bindings.setdefault(other_path, {})[http_method.lower()] = None
            else:
                # Method-less (WebSocket / ASGI): AuthorizationMiddleware is BaseHTTPMiddleware,
                # which only sees the http scope, so a WS handshake never reaches the PDP — a
                # derived deny would be inert. Surface it as a (non-deny) warning;
                # actually closing the WS gap needs pure-ASGI middleware.
                warnings.append(
                    f"{other_path} is a {type(route).__name__}, not an APIRoute — HTTP authz cannot "
                    f"cover it (WebSocket/ASGI routes bypass the BaseHTTPMiddleware PDP)"
                )
            continue
        methods = sorted(route.methods or set())
        rules = get_path_rules(route.endpoint)

        binding: AuthzEndpointMethod | None
        if not rules:
            binding = None
            errors.append(f"{route.path} ({', '.join(methods) or 'no methods'}) has no @path_rule")
        else:
            try:
                permissions, callers = _collapse_rules(
                    rules, path=route.path, method=methods[0] if methods else "", service=service.name
                )
            except (ValueError, AttributeError, TypeError) as exc:
                # A single malformed rule denies only its own route — it never crashes the
                # plugin (which would empty the whole contribution and fall open). The broad
                # catch also covers a bare-string permission that slipped past @path_rule
                # (``p.id`` raises AttributeError), not just the unrepresentable-OR ValueError.
                binding = None
                errors.append(str(exc))
            else:
                for perm in permissions:
                    _register_permission(catalog, perm, warnings)
                # Scope is declared on the route by @AuthzScope.read/.write, independent of the
                # permission rules, and read straight off the (identity-preserved) endpoint.
                scopes = get_path_scope(route.endpoint)
                binding = AuthzEndpointMethod(
                    permissions=[perm.id for perm in permissions], scopes=scopes, callers=callers
                )

        for http_method in methods:
            method_key = http_method.lower()
            route_methods = bindings.setdefault(route.path, {})
            if method_key in route_methods:
                # Two handlers claim the same (path, method): Starlette serves the first
                # registered, but the derived policy could describe the second. Rather than
                # let the last writer silently win, fail the pair closed and flag it.
                errors.append(
                    f"duplicate route binding for {http_method.upper()} {route.path} — a second "
                    f"handler would shadow the first; refusing to guess which policy applies"
                )
                route_methods[method_key] = None
            else:
                route_methods[method_key] = binding

    # Permissions with no 1:1 route (middleware-checked, declared-before-wired). A broken
    # extra_permissions() must NOT abort derivation — that would omit the route bindings and
    # let them fall through the service: bypass. Record it and keep the route-derived authz.
    try:
        extra = service.extra_permissions()
    except Exception as exc:
        extra = []
        errors.append(f"extra_permissions() raised {exc!r}")
    for perm in extra:
        _register_permission(catalog, perm, warnings)

    # Extra role grants: permissions the suffix heuristic (.list/.read → Viewer+Editor, else
    # Editor) wouldn't assign to a role — e.g. the gateway .invoke permission a Viewer needs.
    # Register each in the catalog (so it carries a registry entry + description) and thread
    # role -> [id] into the contribution; merge_authz_contributions unions these with the
    # suffix-derived defaults. A broken hook must NOT abort derivation (that would drop the
    # route bindings and let them fall through the service: bypass), so it's recorded, not raised.
    role_grants: dict[str, list[str]] = {}
    try:
        extra_roles = service.extra_role_permissions()
    except Exception as exc:
        extra_roles = {}
        errors.append(f"extra_role_permissions() raised {exc!r}")
    for role_name, perms in extra_roles.items():
        for perm in perms:
            _register_permission(catalog, perm, warnings)
        granted = sorted({perm.id for perm in perms})
        if granted:
            role_grants[role_name] = granted

    # Pass 2: validate the catalog. A malformed permission id would 500 the bundle's
    # ``validate_static_authz_data`` if it reached the wire; a permission whose first segment
    # isn't the service's own name is namespace squatting (it would silently widen the
    # Viewer/Editor role grants for another service's namespace). Either is a fail-closed
    # error: deny every route and contribute no permissions, so nothing malformed or
    # cross-namespace can reach the merged policy.
    # Role-granted permissions are registered in the catalog above, so the ownership fence below
    # covers them too: a plugin cannot grant a role a permission outside its own namespace.
    owner = service.name
    malformed = sorted(pid for pid in catalog if not is_valid_permission_id(pid))
    out_of_namespace = sorted(p.id for p in catalog.values() if p.service != owner)
    if malformed:
        errors.append(f"malformed permission id(s) (fail-closed): {malformed}")
    if out_of_namespace:
        errors.append(f"permission id(s) outside the service namespace {owner!r} (fail-closed): {out_of_namespace}")
    if malformed or out_of_namespace:
        denied = {path: {method: _deny_binding() for method in methods} for path, methods in bindings.items()}
        return AuthzContribution(permissions={}, endpoints=denied), errors, warnings

    endpoints: dict[str, dict[str, AuthzEndpointMethod]] = {
        path: {method: (binding if binding is not None else _deny_binding()) for method, binding in methods.items()}
        for path, methods in bindings.items()
    }
    permissions = {perm.id: perm.description for perm in catalog.values()}
    return (
        AuthzContribution(permissions=permissions, endpoints=endpoints, role_permissions=role_grants),
        errors,
        warnings,
    )


_plugin_authz_cache: list[PluginAuthzResult] | None = None


def _cached_plugin_authz() -> list[PluginAuthzResult] | None:
    """The cached derivation, or ``None`` if nothing is cached."""
    return _plugin_authz_cache


def _cache_plugin_authz(results: list[PluginAuthzResult]) -> None:
    """Cache *results*, but only a fully clean derivation.

    Pinning a degraded result (say, a transient import error on first build) would 403 the
    plugin's namespace until the process restarts.
    """
    global _plugin_authz_cache
    if all(not result.problems for result in results):
        _plugin_authz_cache = results


def discover_plugin_authz() -> list[PluginAuthzResult]:
    """Derive per-plugin authz results from every installed ``nemo.services`` entry point.

    Each entry point is loaded and its service class instantiated in its own ``try/except``,
    so a single broken plugin can never take down derivation for the others. Both a *load*
    failure (the module won't import) and a *derivation* failure (instantiation / route walk
    raises) are recorded as a fully-degraded result — a problem, no usable contribution —
    rather than silently dropped. Silent drop would omit the plugin's routes and let them
    fall through the ``service:`` no-match bypass once enforcement is on; the bundle instead
    fences ``/apis/<name>`` for a degraded plugin.

    Entry points are enumerated directly (not via ``discover_services``) because
    ``discover()`` swallows load failures and excludes the plugin entirely — exactly the
    silent drop this fail-closed path must avoid.

    Only an **all-clean** derivation is cached. A degraded result (e.g. a transient first-build
    import error) is never pinned for the process lifetime — that would 403 the plugin's
    namespace until restart — so the next call re-derives until the failure clears. Call
    ``clear_plugin_authz_cache()`` (and ``discover_entry_points.cache_clear()``) in tests after
    changing the installed plugin set.
    """
    cached = _cached_plugin_authz()
    if cached is not None:
        return cached

    from nemo_helix_plugin.discovery import discover_entry_points

    results: list[PluginAuthzResult] = []
    for ep_name, ep in discover_entry_points("nemo.services").items():
        try:
            service_cls = ep.load()
        except Exception as exc:
            logger.warning("Failed to load nemo.services %r — recording as degraded", ep_name, exc_info=True)
            results.append(
                PluginAuthzResult(
                    key=ep_name,
                    contribution=AuthzContribution(),
                    problems=[f"failed to load plugin: {exc!r}"],
                    mount_name=ep_name,
                )
            )
            continue
        # Read the mount name off the class (a ClassVar, available even if instantiation
        # below fails) so the degraded fence can cover /apis/<name>, not just /apis/<key>.
        mount_name = getattr(service_cls, "name", ep_name) or ep_name
        try:
            contribution, errors, warnings = _derive_service_contribution(service_cls())
        except Exception as exc:
            logger.warning(
                "Failed to derive authz from nemo.services %r — recording as degraded", ep_name, exc_info=True
            )
            results.append(
                PluginAuthzResult(
                    key=ep_name,
                    contribution=AuthzContribution(),
                    problems=[f"failed to derive plugin authz: {exc!r}"],
                    mount_name=mount_name,
                )
            )
            continue
        results.append(
            PluginAuthzResult(
                key=ep_name, contribution=contribution, problems=errors, warnings=warnings, mount_name=mount_name
            )
        )

    _cache_plugin_authz(results)
    return results


def clear_plugin_authz_cache() -> None:
    """Reset the cached all-clean plugin-authz derivation.

    Call in tests after changing the installed plugin set (alongside
    ``discover_entry_points.cache_clear()``).
    """
    global _plugin_authz_cache
    _plugin_authz_cache = None


def discover_authz_contributions() -> list[AuthzContribution]:
    """Plugin contributions with content (compat shim over :func:`discover_plugin_authz`)."""
    return [r.contribution for r in discover_plugin_authz() if r.contribution.permissions or r.contribution.endpoints]


def discover_authz_contribution_dicts() -> list[dict[str, Any]]:
    """Return contributions as dicts for :func:`nhx.common.auth.authz_merge.merge_authz_contributions`."""
    return [c.to_dict() for c in discover_authz_contributions()]
