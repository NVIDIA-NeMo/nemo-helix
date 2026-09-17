# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization metadata stamped onto factory-generated job and function routes."""

from __future__ import annotations

import pytest
from fastapi import APIRouter
from fastapi.routing import APIRoute
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.authz import (
    AuthzScope,
    CallerKind,
    get_path_rules,
    get_path_scope,
)
from nemo_platform_plugin.authz_discovery import _derive_service_contribution
from nemo_platform_plugin.entities import EntityClient
from nemo_platform_plugin.function import NemoFunction
from nemo_platform_plugin.functions.routes import add_function_routes
from nemo_platform_plugin.jobs.api_factory import (
    FileResultSerializer,
    JobRouteOption,
    PlatformJobResultRoute,
    job_route_factory,
)
from nemo_platform_plugin.jobs.routes import _rebase_job_collection_routes
from nemo_platform_plugin.jobs.spec import PlatformJobSpec
from nemo_platform_plugin.service import NemoService, RouterSpec
from pydantic import BaseModel

_READ = ["customization:read", "platform:read"]
_WRITE = ["customization:write", "platform:write"]


class _Spec(BaseModel):
    value: str = "x"


async def _compiler(
    workspace: str,
    input_spec: _Spec,
    output_spec: _Spec,
    entity_client: EntityClient,
    job_name: str | None,
    sdk: AsyncNeMoPlatform,
    /,
) -> PlatformJobSpec:
    del workspace, input_spec, output_spec, entity_client, job_name, sdk
    raise NotImplementedError


def _rules_by_path_method(router: APIRouter) -> dict[tuple[str, str], tuple[list, list[str] | None]]:
    """Map (composed-path, lower-method) -> (attached PathRules, attached scope) per APIRoute.

    fastapi 0.138.0 makes ``include_router(prefix=...)`` lazy: rebased ``APIRoute``\\ s live
    behind a ``_IncludedRouter`` proxy rather than in ``.routes``. Descend the proxy via
    ``effective_route_contexts()`` to read each route's composed path/methods and original
    endpoint (which still carries the ``@path_rule`` metadata).
    """
    out: dict[tuple[str, str], tuple[list, list[str] | None]] = {}
    for route in router.routes:
        contexts = getattr(route, "effective_route_contexts", None)
        if contexts is None:
            if isinstance(route, APIRoute):
                for method in route.methods or set():
                    out[(route.path, method.lower())] = (
                        get_path_rules(route.endpoint),
                        get_path_scope(route.endpoint),
                    )
            continue
        for ctx in contexts():
            if isinstance(ctx.original_route, APIRoute):
                for method in ctx.methods or set():
                    out[(ctx.path, method.lower())] = (
                        get_path_rules(ctx.original_route.endpoint),
                        get_path_scope(ctx.original_route.endpoint),
                    )
    return out


def _mounted_customization_jobs(**factory_kwargs) -> APIRouter:
    """Build the customization job router the way production does: factory -> rebase -> mount."""
    factory_router = job_route_factory(
        service_name="customization",
        job_type="Widget",
        job_input=_Spec,
        platform_job_config_compiler=_compiler,
        authz=AuthzScope("customization").child("jobs"),
        **factory_kwargs,
    )
    rebased = _rebase_job_collection_routes(factory_router, "/widget-jobs")
    mounted = APIRouter()
    mounted.include_router(rebased, prefix="/apis/customization/v2/workspaces/{workspace}")
    return mounted


def _assert_single_rule(entry: tuple[list, list[str] | None], perm: str, scopes: list[str]) -> None:
    """Assert *entry* (rules, scope) has exactly one PRINCIPAL rule for *perm* and scope *scopes*."""
    rules, scope = entry
    assert len(rules) == 1
    rule = rules[0]
    assert rule.callers == [CallerKind.PRINCIPAL]
    assert [p.id for p in rule.permissions] == [perm]
    assert scope == scopes


def test_job_factory_stamps_every_core_route_exactly() -> None:
    """Exact-set assertion: every CORE route carries the right rule, and nothing else is stamped."""
    rules = _rules_by_path_method(_mounted_customization_jobs())
    base = "/apis/customization/v2/workspaces/{workspace}/widget-jobs"

    # (path, method) -> (permission, scopes). The full CORE surface, including both
    # /results read routes and the generic download closure.
    expected = {
        (base, "post"): ("customization.jobs.create", _WRITE),
        (base, "get"): ("customization.jobs.list", _READ),
        (f"{base}/{{name}}", "get"): ("customization.jobs.read", _READ),
        (f"{base}/{{name}}", "delete"): ("customization.jobs.delete", _WRITE),
        (f"{base}/{{name}}/status", "get"): ("customization.jobs.read", _READ),
        (f"{base}/{{name}}/cancel", "post"): ("customization.jobs.cancel", _WRITE),
        (f"{base}/{{name}}/logs", "get"): ("customization.jobs.read", _READ),
        (f"{base}/{{name}}/results", "get"): ("customization.jobs.read", _READ),
        (f"{base}/{{job}}/results/{{name}}", "get"): ("customization.jobs.read", _READ),
        (f"{base}/{{job}}/results/{{name}}/download", "get"): ("customization.jobs.read", _READ),
    }

    # Exact set — catches both a dropped stamp and an unexpected/extra stamped route.
    assert set(rules) == set(expected)
    for key, (perm, scopes) in expected.items():
        _assert_single_rule(rules[key], perm, scopes)


def test_job_factory_stamps_pause_resume_when_enabled() -> None:
    rules = _rules_by_path_method(
        _mounted_customization_jobs(route_options=[JobRouteOption.CORE, JobRouteOption.PAUSE_RESUME])
    )
    base = "/apis/customization/v2/workspaces/{workspace}/widget-jobs"
    _assert_single_rule(rules[(f"{base}/{{name}}/pause", "post")], "customization.jobs.pause", _WRITE)
    _assert_single_rule(rules[(f"{base}/{{name}}/resume", "post")], "customization.jobs.resume", _WRITE)


def test_job_factory_core_only_omits_pause_resume() -> None:
    rules = _rules_by_path_method(_mounted_customization_jobs())
    assert not any(path.endswith(("/pause", "/resume")) for path, _ in rules)


def test_job_factory_stamps_explicit_result_download_closure() -> None:
    rules = _rules_by_path_method(
        _mounted_customization_jobs(
            job_result_routes=[PlatformJobResultRoute(name="metrics", serializer=FileResultSerializer())]
        )
    )
    base = "/apis/customization/v2/workspaces/{workspace}/widget-jobs"
    _assert_single_rule(
        rules[(f"{base}/{{job}}/results/metrics/download", "get")],
        "customization.jobs.read",
        _READ,
    )


def test_job_factory_no_namespace_is_inert() -> None:
    factory_router = job_route_factory(
        service_name="customization",
        job_type="Bare",
        job_input=_Spec,
        platform_job_config_compiler=_compiler,
    )
    for route in factory_router.routes:
        if isinstance(route, APIRoute):
            assert get_path_rules(route.endpoint) == []


def test_job_factory_derivation_end_to_end() -> None:
    class _Svc(NemoService):
        name = "customization"

        def get_routers(self) -> list[RouterSpec]:
            router = job_route_factory(
                service_name="customization",
                job_type="E2E",
                job_input=_Spec,
                platform_job_config_compiler=_compiler,
                authz=AuthzScope("customization").child("jobs"),
            )
            return [RouterSpec(router, prefix="/v2/workspaces/{workspace}")]

    contrib, problems, _warnings = _derive_service_contribution(_Svc())
    assert problems == []

    # The catalog, derived from the routes, carries every id the stamped rules reference.
    assert {
        "customization.jobs.create",
        "customization.jobs.list",
        "customization.jobs.read",
        "customization.jobs.delete",
        "customization.jobs.cancel",
    } <= set(contrib.permissions)

    collection = "/apis/customization/v2/workspaces/{workspace}/jobs"
    assert contrib.endpoints[collection]["post"].permissions == ["customization.jobs.create"]
    assert contrib.endpoints[collection]["post"].callers == ["principal"]
    assert contrib.endpoints[f"{collection}/{{name}}"]["delete"].permissions == ["customization.jobs.delete"]


def test_add_function_routes_stamps_invoke_permission() -> None:
    class _GreetFn(NemoFunction):
        name = "greet"
        spec_schema = _Spec

        async def run(self, spec: _Spec) -> dict[str, bool]:
            return {"ok": True}

    router = add_function_routes(_GreetFn, authz=AuthzScope("example"))

    routes = [r for r in router.routes if isinstance(r, APIRoute)]
    assert len(routes) == 1
    _assert_single_rule(
        (get_path_rules(routes[0].endpoint), get_path_scope(routes[0].endpoint)),
        "example.greet",
        ["example:write", "platform:write"],
    )


def test_add_function_routes_no_namespace_is_inert() -> None:
    class _BareFn(NemoFunction):
        name = "bare"
        spec_schema = _Spec

        async def run(self, spec: _Spec) -> dict[str, bool]:
            return {"ok": True}

    router = add_function_routes(_BareFn)
    routes = [r for r in router.routes if isinstance(r, APIRoute)]
    assert len(routes) == 1
    assert get_path_rules(routes[0].endpoint) == []


def test_add_function_routes_description_without_authz_raises() -> None:
    class _DescFn(NemoFunction):
        name = "desc"
        spec_schema = _Spec

        async def run(self, spec: _Spec) -> dict[str, bool]:
            return {"ok": True}

    with pytest.raises(ValueError, match="permission_description requires authz"):
        add_function_routes(_DescFn, permission_description="Invoke desc")  # authz omitted
