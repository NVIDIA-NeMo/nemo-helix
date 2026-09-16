# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

import pytest
import typer
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from nemo_customizer.router import (
    CustomizationRouterError,
    CustomizationRouterService,
    merge_router_dependencies,
)
from nemo_platform_plugin.service import RouterSpec


class _FakeContributor:
    name: ClassVar[str] = "fake"
    dependencies: ClassVar[list[str]] = ["studio"]

    def get_routers(self) -> list[RouterSpec]:
        router = APIRouter()

        @router.get("/ping")
        async def ping() -> dict[str, str]:
            return {"backend": "fake"}

        return [
            RouterSpec(
                router=router,
                prefix="/v2/workspaces/{workspace}/fake",
                tag="Fake",
            ),
        ]

    def get_cli(self) -> typer.Typer:
        app = typer.Typer()

        @app.command("info")
        def info() -> None:
            typer.echo("fake")

        return app


def test_merge_router_dependencies_unions_contributor_deps() -> None:
    deps = merge_router_dependencies({"fake": _FakeContributor()})
    assert "studio" in deps
    assert "jobs" in deps


def test_router_sets_merged_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_customizer.router.discover_customization_contributors",
        lambda: {"fake": _FakeContributor()},
    )
    CustomizationRouterService()
    assert "studio" in CustomizationRouterService.dependencies


def test_router_raises_without_contributors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_customizer.router.discover_customization_contributors",
        lambda: {},
    )
    with pytest.raises(CustomizationRouterError, match="no contributors"):
        CustomizationRouterService()


def test_router_merges_contributor_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_customizer.router.discover_customization_contributors",
        lambda: {"fake": _FakeContributor()},
    )
    service = CustomizationRouterService()
    app = FastAPI()
    for spec in service.get_routers():
        if spec.prefix:
            app.include_router(spec.router, prefix=spec.prefix)
        else:
            app.include_router(spec.router)

    client = TestClient(app)
    assert client.get("/v2/healthz").json()["contributors"] == ["fake"]
    assert client.get("/v2/workspaces/ws-a/fake/ping").json() == {"backend": "fake"}


def test_prefix_collision_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DupA(_FakeContributor):
        name = "a"

    class _DupB(_FakeContributor):
        name = "b"

    monkeypatch.setattr(
        "nemo_customizer.router.discover_customization_contributors",
        lambda: {"a": _DupA(), "b": _DupB()},
    )
    with pytest.raises(CustomizationRouterError, match="collision"):
        CustomizationRouterService()


def test_template_route_collision_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TemplateDup(_FakeContributor):
        name = "template-dup"

        def get_routers(self) -> list[RouterSpec]:
            router = APIRouter()

            @router.get("/job-templates")
            async def list_templates() -> dict[str, str]:
                return {"backend": "template-dup"}

            return [
                RouterSpec(
                    router=router,
                    prefix="/v2/workspaces/{workspace}",
                    tag="Template Dup",
                ),
            ]

    monkeypatch.setattr(
        "nemo_customizer.router.discover_customization_contributors",
        lambda: {"template-dup": _TemplateDup()},
    )
    with pytest.raises(CustomizationRouterError, match="collision"):
        CustomizationRouterService()


def test_shared_parent_prefix_with_disjoint_routes_is_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two contributors mounting at the same parent prefix is fine as long as their
    actual routes underneath don't collide. This is the automodel + unsloth case:
    both register their jobs router at ``/v2/workspaces/{workspace}`` and add a
    backend-scoped collection path via ``job_collection_path_for``.
    """

    def _make_contributor(backend_name: str) -> object:
        class _Contributor:
            name: ClassVar[str] = backend_name
            dependencies: ClassVar[list[str]] = []

            def get_routers(self) -> list[RouterSpec]:
                router = APIRouter()

                @router.post(f"/{backend_name}/jobs")
                async def submit() -> dict[str, str]:
                    return {"backend": backend_name}

                return [
                    RouterSpec(
                        router=router,
                        prefix="/v2/workspaces/{workspace}",
                        tag=backend_name.title(),
                    ),
                ]

            def get_cli(self) -> typer.Typer:
                return typer.Typer()

        return _Contributor()

    monkeypatch.setattr(
        "nemo_customizer.router.discover_customization_contributors",
        lambda: {"automodel": _make_contributor("automodel"), "unsloth": _make_contributor("unsloth")},
    )
    # Should not raise.
    service = CustomizationRouterService()
    assert sorted(service._contributors.keys()) == ["automodel", "unsloth"]


def test_authz_derives_from_contributor_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hub's authz is derived from the ``@path_rule``-decorated routes its
    contributors mount — there is no separate ``get_authz_contribution`` declaration.

    Doubles as the Phase-0 derivation gate: the customization hub plus backends
    must derive with no problems and no fail-closed DENY bindings.
    """
    from nemo_platform_plugin.authz import CallerKind, Permission, path_rule
    from nemo_platform_plugin.authz_discovery import _derive_service_contribution

    def _make_contributor(backend: str) -> object:
        class _Contributor:
            name: ClassVar[str] = backend
            dependencies: ClassVar[list[str]] = []

            def get_routers(self) -> list[RouterSpec]:
                router = APIRouter()
                create_perm = Permission("customization", f"{backend}.jobs", "create", f"Create {backend} jobs")

                @router.post(f"/{backend}/jobs")
                @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[create_perm])
                async def submit() -> dict[str, str]:
                    return {"backend": backend}

                @router.get(f"/{backend}/healthz")
                @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
                async def healthz() -> dict[str, str]:
                    return {"backend": backend}

                return [RouterSpec(router=router, prefix="/v2/workspaces/{workspace}", tag=backend.title())]

            def get_cli(self) -> typer.Typer:
                return typer.Typer()

        return _Contributor()

    monkeypatch.setattr(
        "nemo_customizer.router.discover_customization_contributors",
        lambda: {"automodel": _make_contributor("automodel"), "unsloth": _make_contributor("unsloth")},
    )

    service = CustomizationRouterService()
    contribution, problems, _warnings = _derive_service_contribution(service)

    assert problems == []
    assert not any(spec.deny for methods in contribution.endpoints.values() for spec in methods.values())
    assert "customization.automodel.jobs.create" in contribution.permissions
    assert "customization.unsloth.jobs.create" in contribution.permissions
    # The hub's own /v2/healthz is authenticated-but-permissionless (ruled, not denied).
    hub_healthz = contribution.endpoints["/apis/customization/v2/healthz"]["get"]
    assert hub_healthz.permissions == [] and not hub_healthz.deny
