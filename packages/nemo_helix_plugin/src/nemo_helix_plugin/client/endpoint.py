# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions using ParamSpec-based decorators.

Endpoints are declared as decorated methods on a class. The decorator
replaces each method with a callable that builds a :class:`PreparedRequest`,
preserving the original call signature for autocomplete and type checking::

    class ExampleEndpoints:
        @post("/apis/example/v2/workspaces/{workspace}/items")
        def create_item(self, body: CreateItemRequest, *, workspace: str) -> Item:
            raise NotImplementedError

        @get("/apis/example/hello/{name}")
        def hello(self, *, name: str) -> HelloResponse:
            raise NotImplementedError

    endpoints = ExampleEndpoints()
    req = endpoints.create_item(workspace="default", body=CreateItemRequest(name="x"))
    resp = client.send(req)  # NemoResponse[Item]

Parameter conventions:
- ``body`` — JSON request body (Pydantic model, serialized automatically)
- ``content`` — binary request body (raw bytes)
- ``query_params`` — query parameters (dict or TypedDict)
- Blessed client options (e.g. ``exist_ok``) — client-side behavior (stripped before sending)
- All other keyword parameters — path parameters (matched to ``{placeholders}`` in the path template)
"""

from __future__ import annotations

import functools
import inspect
import string
from collections.abc import AsyncIterable, Callable, Iterable
from typing import get_type_hints

from nemo_helix_plugin.client.types import (
    BLESSED_CLIENT_PARAMS,
    RESERVED_PARAM_NAMES,
    ConflictResolver,
    P,
    PreparedRequest,
    ResponseT,
)
from pydantic import BaseModel


def _identify_client_option_params(fn: Callable) -> set[str]:
    """Return parameter names that are blessed client-side options.

    A parameter is a client option if its name appears in
    :data:`BLESSED_CLIENT_PARAMS`.
    """
    sig = inspect.signature(fn)
    return set(sig.parameters.keys()) & BLESSED_CLIENT_PARAMS.keys()


def _validate_params(fn: Callable, path_param_names: set[str], client_option_names: set[str]) -> None:
    """Raise ``TypeError`` at decoration time if any parameter is unrecognised.

    Every parameter must be one of:
    - ``self``
    - A path placeholder (``{name}`` in the URL template)
    - ``body``, ``content``, or ``query_params``
    - A blessed client option (e.g. ``exist_ok``) with the correct type
    """
    sig = inspect.signature(fn)
    known = RESERVED_PARAM_NAMES | path_param_names | client_option_names
    unknown = set(sig.parameters.keys()) - known
    fn_name = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
    if unknown:
        blessed = ", ".join(sorted(BLESSED_CLIENT_PARAMS.keys()))
        raise TypeError(
            f"Endpoint {fn_name} has unrecognised parameters: {unknown}. "
            f"Parameters must be path params {path_param_names}, "
            f"'body', 'content', 'query_params', or a client option ({blessed})."
        )

    # Validate that blessed client option params have the expected type annotation.
    hints = get_type_hints(fn)
    for param_name in client_option_names:
        expected_type = BLESSED_CLIENT_PARAMS[param_name]
        actual_type = hints.get(param_name)
        if actual_type is not None and actual_type is not expected_type:
            raise TypeError(
                f"Endpoint {fn_name}: client option '{param_name}' must be "
                f"annotated as '{expected_type.__name__}', got '{actual_type}'."
            )


def _build_prepared_request(
    method: str,
    path: str,
    sig: inspect.Signature,
    path_param_names: set[str],
    client_option_names: set[str],
    response_type: type | None,
    get_on_conflict: ConflictResolver | None,
    additional_success_status_codes: tuple[int, ...],
    args: tuple,
    kwargs: dict,
) -> PreparedRequest:
    """Build a PreparedRequest by binding call arguments to the endpoint signature.

    Uses ``bind_partial`` so that path parameters with client-level defaults
    (e.g. ``workspace``) can be omitted by the caller.

    Client option parameters (blessed names like ``exist_ok``) are stripped
    from the HTTP request and stashed in ``PreparedRequest.client_options``.

    If ``get_on_conflict`` is set, the resolver is called here with the live
    ``body`` model and resolved ``workspace`` to build the retrieve request that
    ``send()`` replays on a 409. It runs before ``body`` is serialised because the
    resolver needs the model, not the JSON bytes.
    """
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    path_params: dict[str, str] = {}
    query_params: dict[str, str | int | bool | None] | None = None
    content: bytes | Iterable[bytes] | AsyncIterable[bytes] | None = None
    content_type: str | None = None
    client_options: dict[str, object] | None = None
    body_model: BaseModel | None = None

    for name, value in bound.arguments.items():
        if name == "self":
            continue
        if name in client_option_names:
            if client_options is None:
                client_options = {}
            client_options[name] = value
        elif name in path_param_names:
            if value is not None:
                path_params[name] = str(value)
        elif name == "body":
            if not isinstance(value, BaseModel):
                raise TypeError(f"body must be a BaseModel instance, got {type(value).__name__}")
            body_model = value
            content = value.model_dump_json(exclude_unset=True).encode()
            content_type = "application/json"
        elif name == "content":
            content = value
            content_type = "application/octet-stream"
        elif name == "query_params":
            if value is not None:
                query_params = dict(value)

    on_conflict_get: PreparedRequest | None = None
    if get_on_conflict is not None and body_model is not None:
        # ``workspace`` may be omitted by the caller (filled from the client
        # default at send() time). Pass through whatever was bound, if any.
        workspace = bound.arguments.get("workspace")
        on_conflict_get = get_on_conflict(body_model, workspace)

    return PreparedRequest(
        path_template=path,
        path_params=path_params,
        method=method,
        content=content,
        content_type=content_type,
        response_type=response_type,
        query_params=query_params,
        client_options=client_options,
        additional_success_status_codes=additional_success_status_codes,
        on_conflict_get=on_conflict_get,
    )


def _make_endpoint(
    http_method: str,
    path: str,
    fn: Callable[P, ResponseT],
    get_on_conflict: ConflictResolver | None = None,
    additional_success_status_codes: tuple[int, ...] = (),
) -> Callable[P, PreparedRequest[ResponseT]]:
    """Create a callable that builds PreparedRequests from the function's signature."""
    sig = inspect.signature(fn)
    path_param_names = {field_name for _, field_name, _, _ in string.Formatter().parse(path) if field_name}
    client_option_names = _identify_client_option_params(fn)
    _validate_params(fn, path_param_names, client_option_names)

    # Fail fast: exist_ok relies on a get_on_conflict resolver to fetch the
    # existing entity on 409. Without one, the option is inert and a real
    # conflict would raise at send() time — catch the misconfiguration here.
    if "exist_ok" in client_option_names and get_on_conflict is None:
        fn_name = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
        raise TypeError(
            f"Endpoint {fn_name} declares the 'exist_ok' option but no "
            "get_on_conflict resolver. Pass get_on_conflict=<resolver> to the "
            "@post decorator so the existing entity can be retrieved on 409."
        )

    hints = get_type_hints(fn)
    ret = hints.get("return")
    response_type = ret if ret is not None and ret is not type(None) else None

    @functools.wraps(fn)
    def prepare(*args: P.args, **kwargs: P.kwargs) -> PreparedRequest[ResponseT]:
        return _build_prepared_request(
            http_method,
            path,
            sig,
            path_param_names,
            client_option_names,
            response_type,
            get_on_conflict,
            additional_success_status_codes,
            args,
            kwargs,
        )

    return prepare


# ---------------------------------------------------------------------------
# Decorator factories
# ---------------------------------------------------------------------------


def get(
    path: str, *, additional_success_status_codes: tuple[int, ...] = ()
) -> Callable[[Callable[P, ResponseT]], Callable[P, PreparedRequest[ResponseT]]]:
    """Define a GET endpoint (no request body)."""

    def decorator(fn: Callable[P, ResponseT]) -> Callable[P, PreparedRequest[ResponseT]]:
        return _make_endpoint("GET", path, fn, additional_success_status_codes=additional_success_status_codes)

    return decorator


def post(
    path: str, *, get_on_conflict: ConflictResolver | None = None
) -> Callable[[Callable[P, ResponseT]], Callable[P, PreparedRequest[ResponseT]]]:
    """Define a POST endpoint.

    Args:
        path: URL path template with ``{placeholder}`` path params.
        get_on_conflict: Optional resolver enabling ``exist_ok`` on a create
            endpoint. When the request is sent with ``exist_ok=True`` and the
            server responds 409, the client replays the retrieve request this
            resolver builds and returns its entity instead of raising
            :class:`ConflictError`. See :class:`ConflictResolver`.
    """

    def decorator(fn: Callable[P, ResponseT]) -> Callable[P, PreparedRequest[ResponseT]]:
        return _make_endpoint("POST", path, fn, get_on_conflict=get_on_conflict)

    return decorator


def put(path: str) -> Callable[[Callable[P, ResponseT]], Callable[P, PreparedRequest[ResponseT]]]:
    """Define a PUT endpoint."""

    def decorator(fn: Callable[P, ResponseT]) -> Callable[P, PreparedRequest[ResponseT]]:
        return _make_endpoint("PUT", path, fn)

    return decorator


def patch(path: str) -> Callable[[Callable[P, ResponseT]], Callable[P, PreparedRequest[ResponseT]]]:
    """Define a PATCH endpoint."""

    def decorator(fn: Callable[P, ResponseT]) -> Callable[P, PreparedRequest[ResponseT]]:
        return _make_endpoint("PATCH", path, fn)

    return decorator


def delete(path: str) -> Callable[[Callable[P, ResponseT]], Callable[P, PreparedRequest[ResponseT]]]:
    """Define a DELETE endpoint (no request body, optional response body)."""

    def decorator(fn: Callable[P, ResponseT]) -> Callable[P, PreparedRequest[ResponseT]]:
        return _make_endpoint("DELETE", path, fn)

    return decorator
