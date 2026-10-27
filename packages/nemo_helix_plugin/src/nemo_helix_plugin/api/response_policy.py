# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, request-bound field projection for HTTP response models."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import iscoroutinefunction
from typing import Annotated, Any, ClassVar, Coroutine, get_args, get_type_hints, overload

from fastapi import APIRouter, Request, Response
from fastapi.routing import APIRoute
from nemo_helix_plugin.auth import RuntimeAuthClient, current_auth_client
from nemo_helix_plugin.schema import NemoListResponse, Page
from pydantic import BaseModel, BeforeValidator, SerializerFunctionWrapHandler, WrapSerializer
from starlette.responses import JSONResponse


@dataclass(frozen=True)
class Box[ResultT]:
    value: ResultT


def unbox[ResultT](boxed: Box[ResultT]) -> ResultT:
    return boxed.value


read_allowed: ContextVar[bool] = ContextVar("response_policy_read_allowed", default=False)


@dataclass(frozen=True)
class ResponsePolicy[DTO: BaseModel](ABC):
    dto_type: type[DTO]
    field: str

    @abstractmethod
    def can_read(self, auth_client: RuntimeAuthClient) -> bool: ...


def wire_field_for[DTO: BaseModel](policy: ResponsePolicy[DTO]) -> str:
    field_info = policy.dto_type.model_fields.get(policy.field)
    if field_info is not None:
        if field_info.exclude or field_info.is_required():
            raise ValueError(f"Protected response field must be optional and serializable: {policy.field}")
        wire_field = field_info.serialization_alias or field_info.alias or policy.field
    else:
        computed_info = policy.dto_type.model_computed_fields.get(policy.field)
        if (
            computed_info is None
            or computed_info.return_type is None
            or type(None) not in get_args(computed_info.return_type)
        ):
            raise ValueError(f"Protected response field must be optional and serializable: {policy.field}")
        wire_field = computed_info.alias or policy.field
    if any(
        name != policy.field and (info.serialization_alias or info.alias or name) == wire_field
        for name, info in policy.dto_type.model_fields.items()
    ):
        raise ValueError(f"Protected response alias collides with another field: {wire_field}")
    return wire_field


def item_response_model[DTO: BaseModel](policy: ResponsePolicy[DTO], wire_field: str) -> Any:
    def serialize_item(value: DTO, handler: SerializerFunctionWrapHandler):
        content: dict[str, Any] = handler(value)
        if not read_allowed.get():
            content.pop(wire_field, None)
        return content

    return Annotated.__getitem__((policy.dto_type, BeforeValidator(unbox), WrapSerializer(serialize_item)))


def page_response_model[PageT: BaseModel](page_type: type[PageT], wire_field: str) -> Any:
    def serialize_page(value: PageT, handler: SerializerFunctionWrapHandler):
        content: dict[str, Any] = handler(value)
        if not read_allowed.get():
            for item in content["data"]:
                item.pop(wire_field, None)
        return content

    return Annotated.__getitem__((page_type, BeforeValidator(unbox), WrapSerializer(serialize_page)))


def list_response_model[DTO: BaseModel](policy: ResponsePolicy[DTO], wire_field: str) -> Any:
    def serialize_list(value: list[DTO], handler: SerializerFunctionWrapHandler):
        content: list[dict[str, Any]] = handler(value)
        if not read_allowed.get():
            for item in content:
                item.pop(wire_field, None)
        return content

    list_type = list.__class_getitem__(policy.dto_type)
    return Annotated.__getitem__((list_type, BeforeValidator(unbox), WrapSerializer(serialize_list)))


class PolicyRoute(APIRoute):
    response_policy: ClassVar[ResponsePolicy[Any]]

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        ordinary_handler = super().get_route_handler()
        policy = type(self).response_policy

        async def handler(request: Request) -> Response:
            token = read_allowed.set(policy.can_read(current_auth_client(request)))
            try:
                return await ordinary_handler(request)
            finally:
                read_allowed.reset(token)

        return handler


def bound_route_class[DTO: BaseModel](policy: ResponsePolicy[DTO]) -> type[PolicyRoute]:
    class BoundPolicyRoute(PolicyRoute):
        response_policy = policy

    return BoundPolicyRoute


def boxed_endpoint(endpoint: Callable[..., Any]) -> Callable[..., Any]:
    if iscoroutinefunction(endpoint):

        @wraps(endpoint)
        async def boxed(**values: Any) -> Box[Any]:
            return Box(await endpoint(**values))

    else:

        @wraps(endpoint)
        def boxed(**values: Any) -> Box[Any]:
            return Box(endpoint(**values))

    return boxed


class ProtectedRouteDecorator[ResultT]:
    def __init__(
        self,
        router: APIRouter,
        path: str,
        method: str,
        expected_type: object,
        response_model: Any,
        route_class: type[PolicyRoute],
        options: dict[str, Any],
    ) -> None:
        self.router = router
        self.path = path
        self.method = method
        self.expected_type = expected_type
        self.response_model = response_model
        self.route_class = route_class
        self.options = options

    @overload
    def __call__[**Params](self, endpoint: Callable[Params, ResultT]) -> Callable[Params, ResultT]: ...

    @overload
    def __call__[**Params](
        self, endpoint: Callable[Params, Awaitable[ResultT]]
    ) -> Callable[Params, Awaitable[ResultT]]: ...

    def __call__(self, endpoint: Callable[..., Any]) -> Callable[..., Any]:
        if get_type_hints(endpoint).get("return") != self.expected_type:
            raise ValueError("Protected route must declare its registered DTO return type")
        status_code = self.options.get("status_code")
        response_class = self.options.get("response_class", self.router.default_response_class)
        if (
            status_code is not None
            and (status_code < 200 or status_code in (204, 205, 304))
            or isinstance(response_class, type)
            and not issubclass(response_class, JSONResponse)
            or "response_class" in self.options
            and not isinstance(response_class, type)
            or self.options.get("response_model_by_alias") is False
            or self.options.get("response_model_include") is not None
            or self.options.get("response_model_exclude") is not None
        ):
            raise ValueError("Protected route requires a JSON response with aliased, complete model fields")
        self.router.add_api_route(
            self.path,
            boxed_endpoint(endpoint),
            methods=[self.method],
            response_model=self.response_model,
            route_class_override=self.route_class,
            **self.options,
        )
        return endpoint


class ItemPolicyRoutes[DTO: BaseModel]:
    def __init__(self, router: APIRouter, policy: ResponsePolicy[DTO]) -> None:
        self.router = router
        self.dto_type = policy.dto_type
        wire_field = wire_field_for(policy)
        self.response_model = item_response_model(policy, wire_field)
        self.route_class = bound_route_class(policy)

    def get(self, path: str, **options: Any) -> ProtectedRouteDecorator[DTO]:
        return ProtectedRouteDecorator(
            self.router, path, "GET", self.dto_type, self.response_model, self.route_class, options
        )

    def post(self, path: str, **options: Any) -> ProtectedRouteDecorator[DTO]:
        return ProtectedRouteDecorator(
            self.router, path, "POST", self.dto_type, self.response_model, self.route_class, options
        )

    def put(self, path: str, **options: Any) -> ProtectedRouteDecorator[DTO]:
        return ProtectedRouteDecorator(
            self.router, path, "PUT", self.dto_type, self.response_model, self.route_class, options
        )


class _PagePolicyRoutes[PageT: BaseModel]:
    def __init__(self, router: APIRouter, policy: ResponsePolicy[Any], page_type: type[PageT]) -> None:
        self.router = router
        self.page_type = page_type
        wire_field = wire_field_for(policy)
        self.response_model = page_response_model(page_type, wire_field)
        self.route_class = bound_route_class(policy)

    def get(self, path: str, **options: Any) -> ProtectedRouteDecorator[PageT]:
        return ProtectedRouteDecorator(
            self.router, path, "GET", self.page_type, self.response_model, self.route_class, options
        )


def page_policy_routes[DTO: BaseModel](
    router: APIRouter, policy: ResponsePolicy[DTO], page_type: type[Page[DTO]]
) -> _PagePolicyRoutes[Page[DTO]]:
    return _PagePolicyRoutes(router, policy, page_type)


def nemo_page_policy_routes[DTO: BaseModel](
    router: APIRouter, policy: ResponsePolicy[DTO], page_type: type[NemoListResponse[DTO]]
) -> _PagePolicyRoutes[NemoListResponse[DTO]]:
    return _PagePolicyRoutes(router, policy, page_type)


class ListPolicyRoutes[DTO: BaseModel]:
    def __init__(self, router: APIRouter, policy: ResponsePolicy[DTO]) -> None:
        self.router = router
        self.dto_type = policy.dto_type
        wire_field = wire_field_for(policy)
        self.response_model = list_response_model(policy, wire_field)
        self.route_class = bound_route_class(policy)

    def get(self, path: str, **options: Any) -> ProtectedRouteDecorator[list[DTO]]:
        list_type = list.__class_getitem__(self.dto_type)
        return ProtectedRouteDecorator(
            self.router, path, "GET", list_type, self.response_model, self.route_class, options
        )
