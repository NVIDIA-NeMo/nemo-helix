# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard auth-context response policy coverage for installed services."""

from typing import Annotated, Any, get_args, get_origin, get_type_hints

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute, iter_route_contexts
from nemo_helix_plugin.api.response_policy import PolicyRoute, PolicyRoutes
from nemo_helix_plugin.auth.access import AuthContextResponsePolicy
from nemo_helix_plugin.discovery import discover_entry_points
from nemo_helix_plugin.schema import NemoListResponse, Page
from nhx.platform_runner import server
from nhx.platform_runner.loader import load_service
from nhx.platform_runner.registry import get_available_services
from pydantic import BaseModel, PrivateAttr, computed_field


class Item(BaseModel):
    auth_context: dict[str, str] | None = None
    other: str | None = None


class ComputedItem(BaseModel):
    _auth_context: dict[str, str] | None = PrivateAttr(default=None)

    @computed_field
    @property
    def auth_context(self) -> dict[str, str] | None:
        return self._auth_context


class OtherItem(BaseModel):
    auth_context: dict[str, str] | None = None


def _auth_context_models(response_type: Any) -> set[type[BaseModel]]:
    seen: set[type[BaseModel]] = set()

    def visit(field_type: Any) -> set[type[BaseModel]]:
        origin = get_origin(field_type)
        if origin is Annotated:
            return visit(get_args(field_type)[0])
        if origin is not None:
            return set().union(*(visit(argument) for argument in get_args(field_type)))
        if not isinstance(field_type, type) or not issubclass(field_type, BaseModel) or field_type in seen:
            return set()

        seen.add(field_type)
        models: set[type[BaseModel]] = set()
        if "auth_context" in field_type.model_fields or "auth_context" in field_type.model_computed_fields:
            models.add(field_type)
        for field in field_type.model_fields.values():
            models.update(visit(field.annotation))
        return models

    return visit(response_type)


def _audit_auth_context_routes(app: FastAPI) -> tuple[int, list[str]]:
    checked = 0
    failures: list[str] = []
    for context in iter_route_contexts(app.router.routes):
        route = context.original_route
        if not isinstance(route, APIRoute):
            continue
        response_type = route.response_model
        if response_type is None:
            response_type = get_type_hints(route.endpoint).get("return")
        models = _auth_context_models(response_type)
        if not models:
            continue

        policy = type(route).response_policy if isinstance(route, PolicyRoute) else None
        for model in models:
            checked += 1
            if (
                not isinstance(policy, AuthContextResponsePolicy)
                or policy.dto_type is not model
                or policy.field != "auth_context"
            ):
                methods = ",".join(sorted(route.methods or []))
                failures.append(f"{methods} {context.path}: {model.__name__}.auth_context lacks its response policy")
    return checked, failures


def test_installed_auth_context_response_routes_have_matching_policy() -> None:
    available_services = get_available_services()
    missing_plugins = set(discover_entry_points("nemo.services")) - available_services.keys()
    assert not missing_plugins, f"Plugin services were not loaded for the policy audit: {sorted(missing_plugins)}"
    assert {"models", "jobs", "agents", "deployments"} <= available_services.keys()

    services = []
    for name, service_value in sorted(available_services.items()):
        if name == "entities":
            continue
        services.append(load_service(name, service_value) if isinstance(service_value, str) else service_value)

    app = server.create_app(services=services)
    checked, failures = _audit_auth_context_routes(app)
    assert checked > 0
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("response_model", [Item, Page[Item], NemoListResponse[Item], list[Item]])
def test_detects_unprotected_item_page_and_list_routes(response_model: Any) -> None:
    router = APIRouter()
    router.add_api_route("/unprotected", lambda: None, methods=["GET"], response_model=response_model)
    app = FastAPI()
    app.include_router(router)

    assert _audit_auth_context_routes(app) == (
        1,
        ["GET /unprotected: Item.auth_context lacks its response policy"],
    )


def test_detects_unprotected_computed_auth_context() -> None:
    router = APIRouter()
    router.add_api_route("/computed", lambda: None, methods=["GET"], response_model=ComputedItem)
    app = FastAPI()
    app.include_router(router)

    assert _audit_auth_context_routes(app) == (
        1,
        ["GET /computed: ComputedItem.auth_context lacks its response policy"],
    )


def test_detects_unprotected_return_annotation_without_response_model() -> None:
    router = APIRouter()

    @router.get("/untyped", response_model=None)
    def untyped() -> Item:
        return Item()

    app = FastAPI()
    app.include_router(router)

    assert _audit_auth_context_routes(app) == (
        1,
        ["GET /untyped: Item.auth_context lacks its response policy"],
    )


def test_detects_policy_bound_to_wrong_field() -> None:
    router = APIRouter()
    routes = PolicyRoutes(router, AuthContextResponsePolicy(dto_type=Item, field="other"))

    @routes.get("/wrong-field")
    def wrong_field() -> Item:
        return Item()

    app = FastAPI()
    app.include_router(router)

    assert _audit_auth_context_routes(app) == (
        1,
        ["GET /wrong-field: Item.auth_context lacks its response policy"],
    )


def test_detects_policy_bound_to_wrong_dto() -> None:
    class WrongDTORoute(PolicyRoute):
        response_policy = AuthContextResponsePolicy(dto_type=OtherItem, field="auth_context")

    router = APIRouter()
    router.add_api_route(
        "/wrong-dto", lambda: None, methods=["GET"], response_model=Item, route_class_override=WrongDTORoute
    )
    app = FastAPI()
    app.include_router(router)

    assert _audit_auth_context_routes(app) == (
        1,
        ["GET /wrong-dto: Item.auth_context lacks its response policy"],
    )


def test_accepts_matching_policy() -> None:
    router = APIRouter()
    routes = PolicyRoutes(router, AuthContextResponsePolicy(dto_type=Item, field="auth_context"))

    @routes.get("/protected")
    def protected() -> Item:
        return Item()

    app = FastAPI()
    app.include_router(router)

    assert _audit_auth_context_routes(app) == (1, [])
