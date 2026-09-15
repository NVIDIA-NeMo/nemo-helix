# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route-level tests for :func:`nemo_platform_plugin.functions.routes.add_function_routes`.

Covers:

- Non-streaming: 200 + JSON body matching the validated return shape.
- Streaming: ``application/x-ndjson`` content type, ``\\n``-delimited
  frames, terminator-frame semantics.
- Heartbeat: a Heartbeat frame is injected when the user stream goes
  quiet (uses a tiny ``heartbeat_interval_seconds`` in lieu of a fake
  clock — keeps the test deterministic on slow CI).
- 422 on invalid spec.
- DI: ``ctx`` annotation gets a :class:`FunctionContext` with the
  workspace from the path and the request id from the
  ``X-Request-ID`` header.
- DI: ``sdk`` and ``async_sdk`` annotations get the request-scoped platform
  handle (overridden via ``app.dependency_overrides``).

The tests mount the auto-derived router under
``/apis/example/v2/workspaces/{workspace}`` so the full route matches
the documented default — same prefix the production runner uses.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from nemo_platform_plugin.dependencies import get_sdk_client, get_sync_sdk_client
from nemo_platform_plugin.function import NemoFunction
from nemo_platform_plugin.function_context import FunctionContext
from nemo_platform_plugin.functions.frames import Done, FrameModel, Heartbeat
from nemo_platform_plugin.functions.routes import (
    NDJSON_MEDIA_TYPE,
    _prime_async_iterator,
    _with_heartbeats,
    add_function_routes,
)
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared fixtures — schemas, function classes
# ---------------------------------------------------------------------------


class GreetSpec(BaseModel):
    name: str


class GreetResponse(BaseModel):
    message: str


@dataclass(frozen=True)
class _MarkerSdk:
    marker: str


class _NonStreamingGreet(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "greet"
    description: ClassVar[str] = "Say hello."
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec) -> GreetResponse:
        return GreetResponse(message=f"Hello, {spec.name}!")


class _StreamingGreet(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "stream-greet"
    description: ClassVar[str] = "Stream greetings."
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec) -> AsyncIterator[BaseModel]:
        yield GreetResponse(message=f"Hello, {spec.name}!")
        yield GreetResponse(message="Goodbye!")
        yield Done()


class _PrimedStreamingGreet(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "primed-stream-greet"
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec) -> AsyncIterator[BaseModel]:
        yield GreetResponse(message=f"Hello, {spec.name}!")
        await asyncio.sleep(0)
        yield GreetResponse(message="Goodbye!")
        yield Done()


class _PrimedEarlyFailure(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "primed-early-failure"
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec) -> AsyncIterator[BaseModel]:
        raise HTTPException(status_code=418, detail=f"Cannot greet {spec.name}")
        yield GreetResponse(message="unreachable")  # pragma: no cover


class _ContextSensitiveGreet(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "ctx-greet"
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec, *, ctx: FunctionContext) -> dict:
        return {
            "name": spec.name,
            "workspace": ctx.workspace,
            "request_id": ctx.request_id,
        }


class _SdkGreet(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "sdk-greet"
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec, *, async_sdk: _MarkerSdk) -> dict:
        # Echo a marker attribute so the test can assert the override
        # actually flowed through.
        return {"name": spec.name, "sdk_marker": async_sdk.marker}


class _SyncSdkGreet(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "sync-sdk-greet"
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec, *, sdk: _MarkerSdk) -> dict:
        return {"name": spec.name, "sdk_marker": sdk.marker}


class _BothSdkGreet(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "both-sdk-greet"
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec, *, sdk: _MarkerSdk, async_sdk: _MarkerSdk) -> dict:
        return {
            "name": spec.name,
            "sdk_marker": sdk.marker,
            "async_sdk_marker": async_sdk.marker,
        }


class _LocalityGreet(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "locality-greet"
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec, *, is_local: bool) -> dict:
        return {"name": spec.name, "is_local": is_local}


def _build_app(function_cls: type[NemoFunction]) -> FastAPI:
    """Mount *function_cls* under the canonical workspace prefix.

    Mirrors the production layout: routers from
    :func:`add_function_routes` are included with prefix
    ``/apis/<plugin>/v2/workspaces/{workspace}`` so the trailing
    function name resolves to the canonical URL.
    """
    app = FastAPI()
    router = add_function_routes(function_cls)
    app.include_router(router, prefix="/apis/example/v2/workspaces/{workspace}")
    return app


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


class TestNonStreamingResponse:
    def test_returns_json_body(self) -> None:
        client = TestClient(_build_app(_NonStreamingGreet))
        resp = client.post(
            "/apis/example/v2/workspaces/default/greet",
            json={"name": "world"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json() == {"message": "Hello, world!"}

    def test_invalid_spec_returns_422(self) -> None:
        client = TestClient(_build_app(_NonStreamingGreet))
        resp = client.post(
            "/apis/example/v2/workspaces/default/greet",
            json={"wrong_field": "world"},
        )
        assert resp.status_code == 422

    def test_route_injects_is_local_false_when_requested(self) -> None:
        client = TestClient(_build_app(_LocalityGreet))
        resp = client.post(
            "/apis/example/v2/workspaces/default/locality-greet",
            json={"name": "world"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"name": "world", "is_local": False}


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreamingResponse:
    def test_emits_ndjson_lines(self) -> None:
        client = TestClient(_build_app(_StreamingGreet))
        with client.stream(
            "POST",
            "/apis/example/v2/workspaces/default/stream-greet",
            json={"name": "world"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith(NDJSON_MEDIA_TYPE)
            lines = [ln for ln in resp.iter_lines() if ln]

        decoded = [json.loads(ln) for ln in lines]
        # User-yielded frames first; the route does not synthesise
        # anything when the user provides an explicit terminator.
        assert decoded == [
            {"message": "Hello, world!"},
            {"message": "Goodbye!"},
            {"kind": "done"},
        ]

    def test_default_primed_stream_still_emits_ndjson_lines(self) -> None:
        client = TestClient(_build_app(_PrimedStreamingGreet))
        with client.stream(
            "POST",
            "/apis/example/v2/workspaces/default/primed-stream-greet",
            json={"name": "world"},
        ) as resp:
            assert resp.status_code == 200
            lines = [ln for ln in resp.iter_lines() if ln]

        decoded = [json.loads(ln) for ln in lines]
        assert decoded == [
            {"message": "Hello, world!"},
            {"message": "Goodbye!"},
            {"kind": "done"},
        ]

    def test_default_priming_surfaces_pre_frame_http_exception(self) -> None:
        client = TestClient(_build_app(_PrimedEarlyFailure), raise_server_exceptions=False)
        resp = client.post(
            "/apis/example/v2/workspaces/default/primed-early-failure",
            json={"name": "world"},
        )
        assert resp.status_code == 418
        assert resp.json() == {"detail": "Cannot greet world"}

    @pytest.mark.asyncio
    async def test_priming_cancellation_closes_source_before_first_frame(self) -> None:
        source_closed = asyncio.Event()

        async def never_yields() -> AsyncIterator[BaseModel]:
            try:
                await asyncio.Event().wait()
                yield GreetResponse(message="unreachable")  # pragma: no cover
            finally:
                source_closed.set()

        prime_task = asyncio.create_task(_prime_async_iterator(never_yields()))
        await asyncio.sleep(0)

        prime_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await prime_task

        await asyncio.wait_for(source_closed.wait(), timeout=1)


# ---------------------------------------------------------------------------
# Heartbeat injector — tested at the helper level to avoid wall-clock
# coupling in the HTTP test path.
# ---------------------------------------------------------------------------


class TestHeartbeatInjection:
    @pytest.mark.asyncio
    async def test_inserts_heartbeat_on_idle(self) -> None:
        async def slow_source() -> AsyncIterator[BaseModel]:
            yield GreetResponse(message="hi")
            await asyncio.sleep(0.05)  # > heartbeat interval below
            yield GreetResponse(message="bye")

        # Tiny interval so the test runs fast; production default is 5 s.
        wrapped = _with_heartbeats(slow_source(), interval_seconds=0.01)
        kinds: list[str] = []
        messages: list[str] = []
        async for frame in wrapped:
            if isinstance(frame, Heartbeat):
                kinds.append("heartbeat")
            elif isinstance(frame, GreetResponse):
                messages.append(frame.message)

        assert messages == ["hi", "bye"]
        assert "heartbeat" in kinds  # at least one heartbeat slipped in during the sleep

    @pytest.mark.asyncio
    async def test_disabled_when_interval_non_positive(self) -> None:
        async def fast_source() -> AsyncIterator[BaseModel]:
            yield GreetResponse(message="a")
            yield GreetResponse(message="b")

        # interval_seconds=0 disables heartbeats — pass-through only.
        wrapped = _with_heartbeats(fast_source(), interval_seconds=0)
        out = [f.message async for f in wrapped]
        assert out == ["a", "b"]

    @pytest.mark.asyncio
    async def test_propagates_producer_exception_after_partial_yield(self) -> None:
        # Earlier the drainer swallowed producer exceptions and the
        # consumer saw a clean EOF, so an HTTP client got a truncated
        # ``200`` instead of a failure. Pin the propagation contract so
        # the route surfaces the error path.
        class _Boom(RuntimeError):
            pass

        async def flaky_source() -> AsyncIterator[BaseModel]:
            yield GreetResponse(message="hi")
            raise _Boom("source exploded")

        wrapped = _with_heartbeats(flaky_source(), interval_seconds=0.5)
        seen: list[str] = []
        with pytest.raises(_Boom, match="source exploded"):
            async for frame in wrapped:
                if isinstance(frame, GreetResponse):
                    seen.append(frame.message)
        assert seen == ["hi"]

    @pytest.mark.asyncio
    async def test_propagates_producer_exception_before_first_yield(self) -> None:
        # Same propagation guarantee when the producer fails before
        # emitting anything — no frames, but still an exception, not
        # a clean EOF.
        class _BoomEarly(RuntimeError):
            pass

        async def early_failure() -> AsyncIterator[BaseModel]:
            raise _BoomEarly("never started")
            yield GreetResponse(message="unreachable")  # pragma: no cover

        wrapped = _with_heartbeats(early_failure(), interval_seconds=0.5)
        with pytest.raises(_BoomEarly, match="never started"):
            async for _ in wrapped:
                pass


# ---------------------------------------------------------------------------
# DI: ctx and async_sdk via run signature
# ---------------------------------------------------------------------------


class TestSignatureDi:
    def test_ctx_carries_workspace_and_request_id(self) -> None:
        client = TestClient(_build_app(_ContextSensitiveGreet))
        resp = client.post(
            "/apis/example/v2/workspaces/prod/ctx-greet",
            json={"name": "world"},
            headers={"X-Request-ID": "req-42"},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "name": "world",
            "workspace": "prod",
            "request_id": "req-42",
        }

    def test_ctx_request_id_optional(self) -> None:
        client = TestClient(_build_app(_ContextSensitiveGreet))
        resp = client.post(
            "/apis/example/v2/workspaces/prod/ctx-greet",
            json={"name": "world"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["workspace"] == "prod"
        assert body["request_id"] is None

    def test_async_sdk_resolved_from_dependency_override(self) -> None:
        app = _build_app(_SdkGreet)
        app.dependency_overrides[get_sdk_client] = lambda: _MarkerSdk("fake-sdk")
        client = TestClient(app)
        resp = client.post(
            "/apis/example/v2/workspaces/default/sdk-greet",
            json={"name": "world"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"name": "world", "sdk_marker": "fake-sdk"}

    def test_async_sdk_unconfigured_surfaces_runtime_error(self) -> None:
        # Without ``app.dependency_overrides`` the placeholder
        # raises at request time, not at import — which is the
        # point: missing wiring fails per-request, loud and clear.
        app = _build_app(_SdkGreet)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/apis/example/v2/workspaces/default/sdk-greet",
            json={"name": "world"},
        )
        assert resp.status_code == 500

    def test_sync_sdk_resolved_from_dependency_override(self) -> None:
        app = _build_app(_SyncSdkGreet)
        app.dependency_overrides[get_sync_sdk_client] = lambda: _MarkerSdk("fake-sync-sdk")
        client = TestClient(app)
        resp = client.post(
            "/apis/example/v2/workspaces/default/sync-sdk-greet",
            json={"name": "world"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"name": "world", "sdk_marker": "fake-sync-sdk"}

    def test_sync_and_async_sdks_resolved_independently(self) -> None:
        app = _build_app(_BothSdkGreet)
        app.dependency_overrides[get_sync_sdk_client] = lambda: _MarkerSdk("fake-sync-sdk")
        app.dependency_overrides[get_sdk_client] = lambda: _MarkerSdk("fake-async-sdk")
        client = TestClient(app)
        resp = client.post(
            "/apis/example/v2/workspaces/default/both-sdk-greet",
            json={"name": "world"},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "name": "world",
            "sdk_marker": "fake-sync-sdk",
            "async_sdk_marker": "fake-async-sdk",
        }


# ---------------------------------------------------------------------------
# Endpoint override — when the function carries a custom path template
# ---------------------------------------------------------------------------


class _LegacyEndpointGreet(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "legacy"
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec
    endpoint: ClassVar[str | None] = "/legacy-{name}"

    async def run(self, spec: GreetSpec) -> dict:
        return {"name": spec.name}


class TestEndpointOverride:
    def test_endpoint_template_replaces_name_placeholder(self) -> None:
        client = TestClient(_build_app(_LegacyEndpointGreet))
        resp = client.post(
            "/apis/example/v2/workspaces/default/legacy-legacy",
            json={"name": "world"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"name": "world"}


# ---------------------------------------------------------------------------
# Mount-time validation
# ---------------------------------------------------------------------------


class TestMountValidation:
    def test_missing_spec_schema_raises_typeerror(self) -> None:
        # Concrete subclass with no ``spec_schema`` — class definition
        # would normally fail at the ``ClassVar`` requirement, but we
        # subvert it by deleting the attribute on a copy so the
        # mounting check has something to react to.
        cls = type(
            "_NoSpec",
            (_NonStreamingGreet,),
            {"name": "no-spec", "spec_schema": None},
        )
        with pytest.raises(TypeError, match="spec_schema is None"):
            add_function_routes(cls)

    def test_permission_description_without_authz_is_rejected(self) -> None:
        # permission_description only takes effect when authz is set (it rides on the stamped
        # permission); supplying it alone would be silently discarded and leave the route
        # unruled (→ DENY at bundle time), so it must raise rather than fail open.
        with pytest.raises(ValueError, match="permission_description requires authz"):
            add_function_routes(_NonStreamingGreet, permission_description="Greet someone")


# ---------------------------------------------------------------------------
# frame_schema -> OpenAPI
# ---------------------------------------------------------------------------


class GreetFrame(FrameModel):
    kind: Literal["greeting"] = "greeting"
    message: str


class _FramedGreet(NemoFunction[GreetSpec]):
    name: ClassVar[str] = "framed-greet"
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec
    frame_schema: ClassVar[Any] = Annotated[GreetFrame | Done, Field(discriminator="kind")]

    async def run(self, spec: GreetSpec) -> AsyncIterator[BaseModel]:
        yield GreetFrame(message=f"Hello, {spec.name}!")
        yield Done()


def _preview_200(function_cls: type[NemoFunction]) -> dict:
    spec = _build_app(function_cls).openapi()
    path = f"/apis/example/v2/workspaces/{{workspace}}/{function_cls.name}"
    return spec["paths"][path]["post"]["responses"]["200"]


class TestFrameSchemaDocumentation:
    def test_declared_frames_are_documented_under_the_ndjson_media_type(self) -> None:
        content = _preview_200(_FramedGreet)["content"]

        assert list(content) == [NDJSON_MEDIA_TYPE]

    def test_the_frame_union_is_emitted_with_its_discriminator(self) -> None:
        schema = _preview_200(_FramedGreet)["content"][NDJSON_MEDIA_TYPE]["schema"]

        assert schema["discriminator"]["propertyName"] == "kind"
        assert set(schema["discriminator"]["mapping"]) == {"greeting", "done"}
        # A bare ``{"type": "string"}`` would mean the union was dropped.
        assert "type" not in schema

    def test_frame_models_land_in_components_so_clients_can_reference_them(self) -> None:
        schemas = _build_app(_FramedGreet).openapi()["components"]["schemas"]

        assert {"GreetFrame", "Done"} <= set(schemas)

    def test_functions_without_a_frame_schema_are_left_alone(self) -> None:
        content = _preview_200(_StreamingGreet).get("content", {})

        assert NDJSON_MEDIA_TYPE not in content

    def test_declaring_frames_does_not_change_what_is_streamed(self) -> None:
        client = TestClient(_build_app(_FramedGreet))
        with client.stream(
            "POST",
            "/apis/example/v2/workspaces/default/framed-greet",
            json={"name": "world"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith(NDJSON_MEDIA_TYPE)
            frames = [json.loads(ln) for ln in resp.iter_lines() if ln]

        assert frames[0]["message"] == "Hello, world!"
        assert frames[-1]["kind"] == "done"
