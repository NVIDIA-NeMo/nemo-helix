# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-derived FastAPI route for a :class:`NemoFunction` subclass.

``add_function_routes(MyFunction)`` is the one-liner that mirrors
:func:`~nemo_platform_plugin.jobs.routes.add_job_routes` for the function tier.
It returns an :class:`APIRouter` carrying a single ``POST`` route that:

- Validates the request body against
  :attr:`~nemo_platform_plugin.function.NemoFunction.spec_schema`.
- Builds a :class:`~nemo_platform_plugin.function_context.FunctionContext` from
  the workspace path parameter and the optional ``X-Request-ID``
  header.
- Resolves keyword-only DI parameters on
  :meth:`~nemo_platform_plugin.function.NemoFunction.run` by parameter name —
  ``ctx`` (FunctionContext), ``sdk`` (``NeMoPlatform``), ``async_sdk``
  (``AsyncNeMoPlatform``, resolved from
  :func:`~nemo_platform_plugin.dependencies.get_sdk_client`), and ``is_local=False``.
  Functions that need sync-only libraries may request ``sdk`` and run that
  work inside :func:`anyio.to_thread.run_sync` / :func:`asyncio.to_thread`.
- Awaits ``run(spec, **resolved)``.
- If the result is an async iterator/generator, wraps it in a
  :class:`StreamingResponse` with media type
  :data:`NDJSON_MEDIA_TYPE`. Each yielded
  :class:`pydantic.BaseModel` becomes ``model_dump_json() + "\\n"``;
  any other value is JSON-encoded the same way.
- Otherwise returns the value as a normal JSON response (FastAPI
  serialises ``BaseModel`` instances).

The streaming branch wraps the user's iterator with a heartbeat
injector that emits a :class:`~nemo_platform_plugin.functions.frames.Heartbeat`
frame every :data:`HEARTBEAT_INTERVAL_SECONDS` of idle time. Heartbeat
emission is unconditional on streaming routes — proxies between the
plugin service and the caller need it to keep the connection open.

Usage::

    from nemo_platform_plugin.functions.routes import add_function_routes

    router = add_function_routes(GreetFunction)
    app.include_router(
        router,
        prefix="/apis/example/v2/workspaces/{workspace}",
        tags=["Example"],
    )

The mounted route resolves to
``POST /apis/example/v2/workspaces/{workspace}/{function-name}``.
Override the trailing path segment per-class via
:attr:`NemoFunction.endpoint` when a legacy URL needs to be preserved.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.authz import AuthzScope, CallerKind, path_rule
from nemo_platform_plugin.dependencies import get_sdk_client, get_sync_sdk_client
from nemo_platform_plugin.function import NemoFunction, returns_async_iterator
from nemo_platform_plugin.function_context import FunctionContext
from nemo_platform_plugin.functions.frames import DEFAULT_FUNCTION_PATH, NDJSON_MEDIA_TYPE, Heartbeat
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class NdjsonFrameResponse(JSONResponse):
    """Shapes the OpenAPI 200 block; never instantiated.

    Subclasses ``JSONResponse`` because ``fastapi/openapi/utils.py`` hardcodes the
    schema to ``{"type": "string"}`` for anything else, discarding the declared model.
    """

    media_type = NDJSON_MEDIA_TYPE


# Default idle interval after which the route adapter emits a
# :class:`Heartbeat` frame. Matches DD `preview`'s existing
# ``HEARTBEAT_SECONDS = 5`` and `plan-functions.md`'s recommendation.
# Override per-call via the ``add_function_routes`` keyword argument
# when wiring a router for a function with very different latency
# expectations.
HEARTBEAT_INTERVAL_SECONDS: float = 5.0


def add_function_routes(
    function_cls: type[NemoFunction],
    *,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
    authz: AuthzScope | None = None,
    permission_description: str | None = None,
) -> APIRouter:
    """Mount a single ``POST`` route for *function_cls* on a fresh router.

    Args:
        function_cls: The :class:`NemoFunction` subclass to expose. Must
            declare :attr:`~NemoFunction.spec_schema` — that's how the
            adapter knows what to validate the request body against.
        heartbeat_interval_seconds: Idle gap before a
            :class:`Heartbeat` frame is injected on streaming
            responses. Defaults to :data:`HEARTBEAT_INTERVAL_SECONDS`.
            Lower values are useful in tests; production callers
            usually leave the default.
        authz: The plugin's :class:`~nemo_platform_plugin.authz.AuthzScope`.
            When set, a PRINCIPAL ``@path_rule`` is stamped on the route with
            an invoke permission minted from it (``<namespace>.<function-name>``,
            a write action). When omitted the route is left unruled — denied
            fail-closed at bundle time.
        permission_description: Optional human description for the invoke
            permission. Defaults to ``function_cls.description`` or
            ``"Invoke the <name> function"``. Requires ``authz``.

    Returns:
        An :class:`APIRouter` with one route. The caller mounts it
        with the URL prefix of their choice (typically
        ``/apis/<plugin>/v2/workspaces/{workspace}``).

    Raises:
        TypeError: If ``function_cls`` doesn't declare a
            ``spec_schema``.
        ValueError: If ``permission_description`` is given without ``authz``
            (the description rides on the permission stamped from ``authz``,
            so alone it would be silently discarded).
    """
    spec_schema = getattr(function_cls, "spec_schema", None)
    if spec_schema is None:
        raise TypeError(
            f"{function_cls.__name__}.spec_schema is None; add_function_routes "
            f"requires a declared spec_schema. Set it to a Pydantic BaseModel "
            f"subclass on the NemoFunction class."
        )

    if permission_description is not None and authz is None:
        raise ValueError(
            "permission_description requires authz to be set (the description rides on the "
            "permission stamped from authz); supplying it alone would be silently discarded."
        )

    router = APIRouter()
    instance = function_cls()
    run_params = function_cls.run_signature().parameters
    wants_ctx = "ctx" in run_params
    wants_sdk = "sdk" in run_params
    wants_async_sdk = "async_sdk" in run_params
    wants_is_local = "is_local" in run_params

    path = _resolve_route_path(function_cls)

    # The handler signature is built dynamically so we only declare
    # ``Depends(get_sdk_client)`` for functions that actually opt in.
    # Otherwise FastAPI would resolve it on every request and the
    # placeholder raises ``RuntimeError`` when no ``dependency_overrides``
    # entry has been registered. We also only declare the request-id
    # header when the function wants ``ctx`` — pure-spec functions stay
    # purely positional.
    handler = _build_route_handler(
        instance=instance,
        spec_schema=spec_schema,
        wants_ctx=wants_ctx,
        wants_sdk=wants_sdk,
        wants_async_sdk=wants_async_sdk,
        wants_is_local=wants_is_local,
        send_headers_before_first_frame=function_cls.send_headers_before_first_frame,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    handler.__name__ = f"{function_cls.__name__}__route"
    handler.__doc__ = function_cls.description or f"Invoke the {function_cls.name} function."

    if authz is not None:
        # Invoking a function is a write action; the permission id defaults to <namespace>.<function>.
        permission = authz.permission(
            function_cls.name,
            description=permission_description
            or function_cls.description
            or f"Invoke the {function_cls.name} function",
        )
        path_rule(callers=[CallerKind.PRINCIPAL], permissions=[permission])(handler)
        # Invoking a function is a write action; the scope rides on the route via @AuthzScope.write.
        authz.write(handler)

    summary = function_cls.description or f"Invoke {function_cls.name}"

    if function_cls.frame_schema is None:
        router.post(path, summary=summary, response_model=None)(handler)
    else:
        # Not ``responses[200]["model"]`` — that merges onto FastAPI's string default,
        # leaving a schema that is both a string and a union.
        router.post(
            path,
            summary=summary,
            response_model=function_cls.frame_schema,
            response_class=NdjsonFrameResponse,
            responses={
                200: {
                    "description": (
                        "Newline-delimited JSON stream. Each line is one frame; read "
                        "them incrementally rather than buffering the whole body."
                    )
                }
            },
        )(handler)

    return router


def _build_route_handler(
    *,
    instance: NemoFunction,
    spec_schema: type[BaseModel],
    wants_ctx: bool,
    wants_sdk: bool,
    wants_async_sdk: bool,
    wants_is_local: bool,
    send_headers_before_first_frame: bool,
    heartbeat_interval_seconds: float,
):
    """Construct an async route handler whose signature matches *only* the DI the function asks for.

    FastAPI inspects the handler's signature at registration time;
    every parameter with a ``Depends`` is resolved on every request.
    Declaring SDK dependencies on a function that doesn't want them
    would force a runtime error on every request until the platform
    installs ``app.dependency_overrides`` for the SDKs. Building the
    closure body's call-shape from booleans keeps each function's
    surface as narrow as the function declared.
    """

    async def _invoke(
        spec_obj: BaseModel,
        ctx: FunctionContext | None,
        sdk: NeMoPlatform | None,
        async_sdk: AsyncNeMoPlatform | None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if ctx is not None:
            kwargs["ctx"] = ctx
        if wants_sdk:
            kwargs["sdk"] = sdk
        if wants_async_sdk:
            kwargs["async_sdk"] = async_sdk
        if wants_is_local:
            kwargs["is_local"] = False

        result = instance.run(spec_obj, **kwargs)
        if returns_async_iterator(result):
            if not send_headers_before_first_frame:
                result = await _prime_async_iterator(result)
            stream = _with_heartbeats(result, interval_seconds=heartbeat_interval_seconds)
            return StreamingResponse(_to_ndjson(stream), media_type=NDJSON_MEDIA_TYPE)
        return await result

    if wants_ctx and wants_sdk and wants_async_sdk:

        async def handler(
            workspace: str,
            request_body,
            x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
            sdk: NeMoPlatform = Depends(get_sync_sdk_client),
            async_sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> Any:
            ctx = FunctionContext(workspace=workspace, request_id=x_request_id)
            return await _invoke(request_body, ctx, sdk, async_sdk)

    elif wants_ctx and wants_sdk:

        async def handler(
            workspace: str,
            request_body,
            x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
            sdk: NeMoPlatform = Depends(get_sync_sdk_client),
        ) -> Any:
            ctx = FunctionContext(workspace=workspace, request_id=x_request_id)
            return await _invoke(request_body, ctx, sdk, None)

    elif wants_ctx and wants_async_sdk:

        async def handler(
            workspace: str,
            request_body,
            x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
            async_sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> Any:
            ctx = FunctionContext(workspace=workspace, request_id=x_request_id)
            return await _invoke(request_body, ctx, None, async_sdk)

    elif wants_ctx:

        async def handler(
            workspace: str,
            request_body,
            x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
        ) -> Any:
            ctx = FunctionContext(workspace=workspace, request_id=x_request_id)
            return await _invoke(request_body, ctx, None, None)

    elif wants_sdk and wants_async_sdk:

        async def handler(
            workspace: str,
            request_body,
            sdk: NeMoPlatform = Depends(get_sync_sdk_client),
            async_sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> Any:
            return await _invoke(request_body, None, sdk, async_sdk)

    elif wants_sdk:

        async def handler(
            workspace: str,
            request_body,
            sdk: NeMoPlatform = Depends(get_sync_sdk_client),
        ) -> Any:
            return await _invoke(request_body, None, sdk, None)

    elif wants_async_sdk:

        async def handler(
            workspace: str,
            request_body,
            async_sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
        ) -> Any:
            return await _invoke(request_body, None, None, async_sdk)

    else:

        async def handler(
            workspace: str,
            request_body,
        ) -> Any:
            return await _invoke(request_body, None, None, None)

    # FastAPI inspects ``__annotations__`` (via ``get_type_hints``)
    # rather than the literal source annotation, and a closure-captured
    # ``spec_schema`` doesn't resolve from the handler's local scope.
    # Assigning the Pydantic class directly to the parameter's
    # annotation ensures FastAPI treats ``request_body`` as a Body
    # rather than defaulting to a Query parameter.
    handler.__annotations__["request_body"] = spec_schema
    return handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_route_path(function_cls: type[NemoFunction]) -> str:
    """Return the route path the adapter mounts the handler under.

    When :attr:`NemoFunction.endpoint` is set, it's a full template
    that the operator wants to keep verbatim — the only substitution
    we apply is ``{name}`` (so the same template can be reused across
    functions in a plugin without writing the suffix twice). The
    ``{api}`` and ``{workspace}`` placeholders stay as literal route
    parameters; FastAPI will refuse to mount the router otherwise.

    The default :data:`DEFAULT_FUNCTION_PATH` template is the trailing
    ``/{name}`` segment, intended to be mounted under the standard
    ``/apis/<plugin>/v2/workspaces/{workspace}`` prefix.
    """
    template = function_cls.endpoint or DEFAULT_FUNCTION_PATH
    return template.replace("{name}", function_cls.name)


async def _to_ndjson(source: AsyncIterator[Any]) -> AsyncGenerator[str, None]:
    """Encode each frame from *source* as a single NDJSON line.

    Pydantic models flow through ``model_dump_json()``; every other
    value goes through ``json.dumps`` so plugin authors can yield
    plain dicts during prototyping without constructing wrapper
    models.
    """
    async for frame in source:
        if isinstance(frame, BaseModel):
            yield frame.model_dump_json() + "\n"
        else:
            yield json.dumps(frame, default=str) + "\n"


async def _prime_async_iterator(source: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Start *source* and wait for its first frame before returning a wrapper.

    This lets route handlers surface validation failures raised before a
    streaming function's first frame as normal HTTP errors, before
    ``StreamingResponse`` sends response headers.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    sentinel = object()
    first_ready = asyncio.Event()
    producer_error: list[BaseException] = []
    frames_produced = 0

    async def _drain() -> None:
        nonlocal frames_produced
        try:
            async for frame in source:
                frames_produced += 1
                await queue.put(frame)
                if frames_produced == 1:
                    first_ready.set()
        except BaseException as exc:  # noqa: BLE001 — re-raised by the consumer
            producer_error.append(exc)
        finally:
            await queue.put(sentinel)
            first_ready.set()

    drain_task = asyncio.create_task(_drain())
    try:
        await first_ready.wait()
    except asyncio.CancelledError:
        if not drain_task.done():
            drain_task.cancel()
            with suppress(asyncio.CancelledError):
                await drain_task
        raise
    if frames_produced == 0 and producer_error:
        if not drain_task.done():
            drain_task.cancel()
            with suppress(asyncio.CancelledError):
                await drain_task
        else:
            await drain_task
        raise producer_error[0]

    return _consume_primed_queue(queue, sentinel, drain_task, producer_error)


async def _consume_primed_queue(
    queue: asyncio.Queue[Any],
    sentinel: object,
    drain_task: asyncio.Task[None],
    producer_error: list[BaseException],
) -> AsyncIterator[Any]:
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                await drain_task
                if producer_error:
                    raise producer_error[0]
                return
            yield item
    finally:
        if not drain_task.done():
            drain_task.cancel()
            with suppress(asyncio.CancelledError):
                await drain_task


async def _with_heartbeats(
    source: AsyncIterator[Any],
    *,
    interval_seconds: float,
) -> AsyncGenerator[Any, None]:
    """Wrap *source* with a heartbeat injector.

    Yields whatever *source* yields, with a :class:`Heartbeat` frame
    inserted whenever the source has been quiet for at least
    *interval_seconds*. The timer resets on every user-yielded frame
    so back-to-back work doesn't accumulate spurious heartbeats.

    The implementation drains *source* into an unbounded
    :class:`asyncio.Queue` from a background task so the heartbeat
    timer can run on the caller side. When *source* is exhausted (or
    raises) the background task pushes a sentinel to terminate the
    output stream cleanly; producer exceptions ride alongside the
    sentinel and are re-raised on the consumer side so the caller
    sees the failure rather than a clean EOF on a truncated stream.
    Cancellation propagates: cancelling the consumer cancels the
    drain task too.

    *interval_seconds* of ``0`` or less disables heartbeats — useful
    for tests that want pure pass-through and for plugins that yield
    fast enough to never hit the idle threshold.
    """
    if interval_seconds <= 0:
        async for frame in source:
            yield frame
        return

    queue: asyncio.Queue[Any] = asyncio.Queue()
    sentinel = object()
    # Captured outside the queue so the consumer can re-raise it once the
    # sentinel arrives. Putting the exception on the queue alongside frames
    # would be ambiguous — a plugin yielding a bare ``Exception`` instance
    # as a frame would otherwise look indistinguishable from a producer crash.
    producer_error: list[BaseException] = []

    async def _drain() -> None:
        try:
            async for frame in source:
                await queue.put(frame)
        except BaseException as exc:  # noqa: BLE001 — re-raised on consumer side
            producer_error.append(exc)
        finally:
            await queue.put(sentinel)

    drain_task = asyncio.create_task(_drain())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                yield Heartbeat()
                continue
            if item is sentinel:
                # Drain finished — surface a producer crash if there was one.
                # Awaiting the task lets ``CancelledError`` propagate naturally
                # while making sure any synchronous bookkeeping in ``_drain``
                # has run to completion before we re-raise.
                await drain_task
                if producer_error:
                    raise producer_error[0]
                return
            yield item
    finally:
        if not drain_task.done():
            drain_task.cancel()
            # Suppress the cancellation exception so we don't mask the
            # original error if the consumer is also being torn down
            # because of one.
            try:
                await drain_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — see comment above
                pass


__all__ = [
    "DEFAULT_FUNCTION_PATH",
    "HEARTBEAT_INTERVAL_SECONDS",
    "NDJSON_MEDIA_TYPE",
    "add_function_routes",
]
