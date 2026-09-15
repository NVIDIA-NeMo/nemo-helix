# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections.abc import MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from typing import Any, Optional, Self, Type, TypeVar, cast

from fastapi import Request
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span, Tracer

AUTH_HEADER_USERID = "x-nmp-principal-id"
AUTH_HEADER_ACCOUNT_ID = "x-nmp-actor-account-id"
AUTH_HEADER_EMAIL = "x-nmp-principal-email"
AUTH_HEADER_FILTERS = "x-nmp-principal-filters"
AUTH_HEADER_GROUPS = "x-nmp-principal-groups"
AUTH_HEADER_ALIASES = "x-nmp-actor-aliases"
AUTH_HEADER_ON_BEHALF_OF = "x-nmp-principal-on-behalf-of"
AUTH_HEADER_ON_BEHALF_OF_EMAIL = "x-nmp-principal-on-behalf-of-email"
AUTH_HEADER_ON_BEHALF_OF_GROUPS = "x-nmp-principal-on-behalf-of-groups"
AUTH_HEADER_ON_BEHALF_OF_ACCOUNT_ID = "x-nmp-subject-account-id"
AUTH_HEADER_ON_BEHALF_OF_ALIASES = "x-nmp-subject-aliases"


def _get_trace_id():
    return trace.get_current_span().get_span_context().trace_id


@dataclass
class BaseContext(ABC):
    otel_prefix: str = ""

    @cached_property
    def _fields(self) -> dict[str, Any]:
        ret = {}
        for f in fields(self):
            if f.name == "otel_prefix":
                continue
            value = getattr(self, f.name)
            field_key = f.name
            if self.otel_prefix:
                field_key = f"{self.otel_prefix}.{field_key}"
            ret[field_key] = value
        return ret

    def add_to_span(self, span: Span):
        for key, value in self._fields.items():
            if value is None:
                continue
            if isinstance(value, dict):
                value = str(value)
            span.set_attribute(key, value)

    def add_to_log(self, event_dict: MutableMapping[str, Any]) -> None:
        for key, value in self._fields.items():
            if value is None:
                continue
            event_dict[key] = value


@dataclass
class AuthContext(BaseContext):
    """
    Stores auth information for the current context.
    """

    otel_prefix: str = "auth"

    principal_id: str | None = None
    account_id: str | None = None
    email: str | None = None
    filters: str | None = None
    groups: str | None = None
    authz_aliases: str | None = None
    on_behalf_of: str | None = None
    on_behalf_of_email: str | None = None
    on_behalf_of_groups: str | None = None
    on_behalf_of_account_id: str | None = None
    on_behalf_of_authz_aliases: str | None = None

    def to_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.principal_id is not None:
            headers[AUTH_HEADER_USERID] = self.principal_id
        if self.account_id is not None:
            headers[AUTH_HEADER_ACCOUNT_ID] = self.account_id
        if self.email is not None:
            headers[AUTH_HEADER_EMAIL] = self.email
        if self.filters is not None:
            headers[AUTH_HEADER_FILTERS] = self.filters
        if self.groups is not None:
            headers[AUTH_HEADER_GROUPS] = self.groups
        if self.authz_aliases is not None:
            headers[AUTH_HEADER_ALIASES] = self.authz_aliases
        if self.on_behalf_of is not None:
            headers[AUTH_HEADER_ON_BEHALF_OF] = self.on_behalf_of
        if self.on_behalf_of_email is not None:
            headers[AUTH_HEADER_ON_BEHALF_OF_EMAIL] = self.on_behalf_of_email
        if self.on_behalf_of_groups is not None:
            headers[AUTH_HEADER_ON_BEHALF_OF_GROUPS] = self.on_behalf_of_groups
        if self.on_behalf_of_account_id is not None:
            headers[AUTH_HEADER_ON_BEHALF_OF_ACCOUNT_ID] = self.on_behalf_of_account_id
        if self.on_behalf_of_authz_aliases is not None:
            headers[AUTH_HEADER_ON_BEHALF_OF_ALIASES] = self.on_behalf_of_authz_aliases
        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> Self:
        ctx = cls()
        if userid := headers.get(AUTH_HEADER_USERID):
            ctx.principal_id = userid
        if account_id := headers.get(AUTH_HEADER_ACCOUNT_ID):
            ctx.account_id = account_id
        if email := headers.get(AUTH_HEADER_EMAIL):
            ctx.email = email
        if filters := headers.get(AUTH_HEADER_FILTERS):
            ctx.filters = filters
        if groups := headers.get(AUTH_HEADER_GROUPS):
            ctx.groups = groups
        if aliases := headers.get(AUTH_HEADER_ALIASES):
            ctx.authz_aliases = aliases
        if on_behalf_of := headers.get(AUTH_HEADER_ON_BEHALF_OF):
            ctx.on_behalf_of = on_behalf_of
        if on_behalf_of_email := headers.get(AUTH_HEADER_ON_BEHALF_OF_EMAIL):
            ctx.on_behalf_of_email = on_behalf_of_email
        if on_behalf_of_groups := headers.get(AUTH_HEADER_ON_BEHALF_OF_GROUPS):
            ctx.on_behalf_of_groups = on_behalf_of_groups
        if on_behalf_of_account_id := headers.get(AUTH_HEADER_ON_BEHALF_OF_ACCOUNT_ID):
            ctx.on_behalf_of_account_id = on_behalf_of_account_id
        if on_behalf_of_aliases := headers.get(AUTH_HEADER_ON_BEHALF_OF_ALIASES):
            ctx.on_behalf_of_authz_aliases = on_behalf_of_aliases
        return ctx


CtxT = TypeVar("CtxT", bound=BaseContext)


@dataclass
class AppContext:
    """
    Primary dataclass used for collecting all context-specific
    variables. This AppContext is retrievable via `get_app_ctx`.
    """

    trace_id: int = field(default_factory=_get_trace_id)
    auth_ctx: AuthContext | None = None
    service_name: str | None = None
    workspace: str | None = None

    _custom_ctx: dict[Type[BaseContext], BaseContext] = field(default_factory=dict)

    def with_custom_ctx(self, custom_ctx: BaseContext) -> Self:
        """Return a new AppContext with the custom context added."""
        new_custom_ctx_dict = self._custom_ctx.copy()
        new_custom_ctx_dict[type(custom_ctx)] = custom_ctx
        return replace(self, _custom_ctx=new_custom_ctx_dict)

    def get_custom_ctx(self, ctx_cls: Type[CtxT]) -> CtxT | None:
        ret = self._custom_ctx.get(ctx_cls)
        if ret is None:
            return None
        return cast(CtxT, ret)

    def add_to_span(self, span: Span):
        if self.service_name:
            span.set_attribute("nmp.service", self.service_name)
        if self.workspace:
            span.set_attribute("nmp.workspace", self.workspace)
        if self.auth_ctx:
            self.auth_ctx.add_to_span(span)
        for ctx in self._custom_ctx.values():
            ctx.add_to_span(span)

    @cached_property
    def _log_fields(self) -> dict[str, Any]:
        """Pre-computed log fields for fast logging. Cached after first access."""
        result: dict[str, Any] = {}
        if self.service_name:
            result["service"] = self.service_name
        if self.workspace:
            result["workspace"] = self.workspace
        if self.auth_ctx:
            for key, value in self.auth_ctx._fields.items():
                if value is not None:
                    result[key] = value
        for ctx in self._custom_ctx.values():
            for key, value in ctx._fields.items():
                if value is not None:
                    result[key] = value
        return result

    def add_to_log(self, event_dict: MutableMapping[str, Any]) -> None:
        event_dict.update(self._log_fields)

    @classmethod
    def from_request(cls, request: Request, include_auth_context: bool = True) -> Self:
        auth_ctx = None
        if include_auth_context:
            auth_ctx = AuthContext.from_headers(dict(request.headers))

        trace_id = trace.get_current_span().get_span_context().trace_id
        return cls(trace_id=trace_id, auth_ctx=auth_ctx)


_app_context: ContextVar[AppContext | None] = ContextVar("app_context", default=None)


def get_app_ctx() -> AppContext | None:
    """
    Retrieve the current AppContext.

    Returns:
        The current AppContext if one exists, None otherwise

    Example:
        initialize_app_ctx(AppContext())
        ctx = get_app_ctx()
    """
    return _app_context.get()


def initialize_app_ctx(app_ctx: AppContext) -> None:
    """
    Initialize the AppContext for the current request.

    This should only be called by middleware or router dependencies at the
    start of a request. For adding custom contexts during request handling,
    use `update_app_ctx()` or `scoped_app_ctx()` instead.

    Args:
        app_ctx: The AppContext to set
    """
    _app_context.set(app_ctx)


def update_app_ctx(custom_ctx: BaseContext) -> None:
    """
    Add a custom context to the current AppContext.

    If no AppContext exists, one is created. The custom context is added
    to the AppContext and the updated context is stored.

    Note: This does NOT restore the previous context when done. For scoped
    context that restores on exit, use `scoped_app_ctx()` context manager.

    Args:
        custom_ctx: The custom context to add (e.g., JobContext)

    Example:
        update_app_ctx(JobContext(id="job-123"))
    """
    current = get_app_ctx()
    if current is None:
        current = AppContext()
    new_ctx = current.with_custom_ctx(custom_ctx)
    _app_context.set(new_ctx)


@contextmanager
def scoped_app_ctx(custom_ctx: BaseContext):
    """
    Context manager that adds custom context and restores previous state on exit.

    Use this in controller loops or other cases where you need isolation between
    iterations. For HTTP request handlers where cleanup isn't needed, use
    `update_app_ctx()` instead.

    Args:
        custom_ctx: The custom context to add

    Example:
        for item in items:
            with scoped_app_ctx(JobContext(id=item.job_id)):
                # Process item with its own context
                pass
            # Context restored to what it was before
    """
    current = get_app_ctx()
    if current is None:
        current = AppContext()
    new_ctx = current.with_custom_ctx(custom_ctx)
    token = _app_context.set(new_ctx)
    try:
        yield
    finally:
        _app_context.reset(token)


def start_span_with_ctx(tracer: Tracer, name: str, custom_ctx: BaseContext, **kwargs):
    """
    Start a span and add a custom context to the AppContext.

    This is a convenience function that adds the custom context and starts a span.
    The custom context is added to spans and logs via the AppContext.

    Args:
        tracer: The OpenTelemetry tracer to use
        name: The name of the span
        custom_ctx: The custom context to add
        **kwargs: Additional arguments passed to start_as_current_span

    Returns:
        A context manager that yields the span
    """
    update_app_ctx(custom_ctx)
    return tracer.start_as_current_span(name, **kwargs)


def create_app_context_dependency(service_name: str, include_auth_context: bool = True):
    """Create a FastAPI dependency that initializes AppContext for a router.

    Used with include_router(dependencies=[...]) in api.py to automatically
    create AppContext for all routes in a router. This captures auth headers,
    trace ID, service name, and workspace in one place.

    Args:
        service_name: The service name to set on the context
        include_auth_context: Whether to extract auth headers from the request

    Example:
        app.include_router(
            router=service_app.router,
            prefix=f"/apis/{service.name}",
            dependencies=[Depends(create_app_context_dependency(service.name))]
        )

    Note:
        The inner dependency function MUST be async. Starlette's BaseHTTPMiddleware
        uses task groups internally which can break ContextVar propagation across
        task boundaries. An async dependency runs in the same coroutine chain as
        the endpoint handler, ensuring the ContextVar set here is visible to the
        endpoint. A sync dependency may run in a different task context.
    """

    async def dependency(request: Request) -> None:
        app_ctx = AppContext.from_request(request, include_auth_context=include_auth_context)
        app_ctx.service_name = service_name
        app_ctx.workspace = request.path_params.get("workspace")
        initialize_app_ctx(app_ctx)
        app_ctx.add_to_span(trace.get_current_span())

        # Also store on request.state to enable middleware access.
        # ContextVars don't propagate back through BaseHTTPMiddleware's call_next,
        # but request.state does since it's on the request object itself.
        request.state.service = service_name
        request.state.workspace = app_ctx.workspace

    return dependency


# --------- Processors --------- #


def AppContextLogProcessor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    if (ctx := get_app_ctx()) is None:
        return event_dict
    ctx.add_to_log(event_dict)
    return event_dict


class AppContextSpanProcessor(SpanProcessor):
    """
    This processor adds application context to spans.
    """

    def on_start(
        self,
        span: Span,
        parent_context: Optional[Context] = None,
    ) -> None:
        if app_ctx := get_app_ctx():
            app_ctx.add_to_span(span)
