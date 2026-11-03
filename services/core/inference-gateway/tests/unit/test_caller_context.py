# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A router calls models back as its caller without ever holding the caller's credential."""

from __future__ import annotations

import base64
from collections.abc import Iterator

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from nemo_helix_plugin.inference_middleware_models import VirtualModel
from nhx.core.inference_gateway.api import caller_context
from nhx.core.inference_gateway.api.caller_context import (
    CALLER_CONTEXT_HEADER,
    CONTEXT_LIFETIME_SEC,
    CallerContextSeal,
    InvalidCallerContextError,
    caller_headers,
    refuse_router_keys,
)
from nhx.core.inference_gateway.api.caller_context_middleware import GATEWAY_PATH_PREFIX, CallerContextMiddleware
from nhx.core.inference_gateway.api.dependencies import global_virtual_model_cache
from nhx.core.inference_gateway.api.model_cache import ModelCache
from nhx.core.inference_gateway.api.virtual_model_cache import VirtualModelCache
from nhx.core.inference_gateway.config import InferenceGatewayConfig
from pydantic import ValidationError
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse

KEY = bytes(range(32))
CALLER = "trial-token"
ROUTER_KEY = "router-key"
GATEWAY_PATH = f"{GATEWAY_PATH_PREFIX}v2/workspaces/default/openai/-/v1/models"


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def seal() -> Iterator[CallerContextSeal]:
    seal = CallerContextSeal(KEY)
    previous, caller_context._seal = caller_context._seal, seal
    try:
        yield seal
    finally:
        caller_context._seal = previous


@pytest.fixture
def router(model_cache: ModelCache) -> str:
    """Provider ``default/tot`` as a router holding its own key."""
    info = model_cache.get_from_provider("default", "tot")
    assert info is not None
    info.model_provider.calls_back_as_caller = True
    info.secret_value = ROUTER_KEY
    return "default/tot"


def test_a_context_opens_to_the_credential_it_sealed() -> None:
    seal = CallerContextSeal(KEY)

    assert seal.open("ws/router", seal.seal("ws/router", CALLER)) == CALLER
    assert seal.open("ws/router", seal.seal("ws/router", None)) is None


def test_a_context_never_shows_the_credential_it_carries() -> None:
    assert CALLER not in CallerContextSeal(KEY).seal("ws/router", CALLER)


@pytest.mark.parametrize(
    "spoil",
    [
        pytest.param(lambda value, seal: CallerContextSeal(bytes(32)).seal("ws/router", CALLER), id="another key"),
        pytest.param(lambda value, seal: seal.seal("ws/other", CALLER), id="another provider"),
        pytest.param(lambda value, seal: value[:-4] + ("AAAA" if not value.endswith("AAAA") else "BBBB"), id="altered"),
        pytest.param(lambda value, seal: "v1.not-a-context", id="malformed"),
    ],
)
def test_a_context_sealed_elsewhere_or_altered_doesnt_open(spoil) -> None:  # noqa: ANN001
    seal = CallerContextSeal(KEY)
    value = seal.seal("ws/router", CALLER)

    with pytest.raises(InvalidCallerContextError):
        seal.open("ws/router", spoil(value, seal))


def test_a_context_expires() -> None:
    clock = Clock()
    seal = CallerContextSeal(KEY, clock=clock)
    value = seal.seal("ws/router", CALLER)

    clock.now += CONTEXT_LIFETIME_SEC + 1

    with pytest.raises(InvalidCallerContextError, match="expired"):
        seal.open("ws/router", value)


def test_a_call_back_with_the_routers_key_and_its_context_acts_as_the_caller(
    model_cache: ModelCache, router: str, seal: CallerContextSeal
) -> None:
    headers = {"Authorization": f"Bearer {ROUTER_KEY}", CALLER_CONTEXT_HEADER: seal.seal(router, CALLER)}

    acting = caller_headers(headers, model_cache.router_keys(), seal)

    assert acting["authorization"] == f"Bearer {CALLER}"
    assert CALLER_CONTEXT_HEADER not in acting


def test_a_caller_with_no_credential_stays_without_one_on_its_routers_calls_back(
    model_cache: ModelCache, router: str, seal: CallerContextSeal
) -> None:
    headers = {"x-api-key": ROUTER_KEY, CALLER_CONTEXT_HEADER: seal.seal(router, None)}

    acting = caller_headers(headers, model_cache.router_keys(), seal)

    assert "authorization" not in acting
    assert "x-api-key" not in acting


@pytest.mark.parametrize("context", [None, "v1.bm90.aGVsbG8"], ids=["no context", "invalid context"])
def test_a_call_with_a_routers_key_and_no_valid_context_is_refused(
    model_cache: ModelCache, router: str, seal: CallerContextSeal, context: str | None
) -> None:
    """Otherwise a router that drops the context would call models with its own key, outside its caller's policy."""
    headers = {"authorization": f"Bearer {ROUTER_KEY}"}
    if context is not None:
        headers[CALLER_CONTEXT_HEADER] = context

    with pytest.raises(HTTPException) as exc_info:
        caller_headers(headers, model_cache.router_keys(), seal)

    assert exc_info.value.status_code == 403
    assert ROUTER_KEY not in str(exc_info.value.detail)


def test_an_expired_context_on_a_call_back_is_refused(model_cache: ModelCache, router: str) -> None:
    clock = Clock()
    seal = CallerContextSeal(KEY, clock=clock)
    headers = {"authorization": f"Bearer {ROUTER_KEY}", CALLER_CONTEXT_HEADER: seal.seal(router, CALLER)}
    clock.now += CONTEXT_LIFETIME_SEC + 1

    with pytest.raises(HTTPException) as exc_info:
        caller_headers(headers, model_cache.router_keys(), seal)

    assert exc_info.value.status_code == 403


def test_a_context_sent_without_its_routers_key_is_dropped(
    model_cache: ModelCache, router: str, seal: CallerContextSeal
) -> None:
    """A caller can't borrow another credential by sending a context along with its own."""
    headers = {"authorization": "Bearer someone-else", CALLER_CONTEXT_HEADER: seal.seal(router, CALLER)}

    acting = caller_headers(headers, model_cache.router_keys(), seal)

    assert acting == {"authorization": "Bearer someone-else"}


def test_calling_a_router_sends_its_own_key_and_the_callers_context(
    client: TestClient, mock_proxy_client, router: str, seal: CallerContextSeal
) -> None:
    response = client.post(
        "/v2/workspaces/default/provider/tot/-/v1/chat/completions", headers={"Authorization": f"Bearer {CALLER}"}
    )

    assert response.status_code == 200
    sent = CIMultiDict(mock_proxy_client.request.call_args.kwargs["headers"])
    assert sent["authorization"] == f"Bearer {ROUTER_KEY}"
    assert CALLER not in str(dict(sent))
    assert seal.open(router, sent[CALLER_CONTEXT_HEADER]) == CALLER


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("/v2/workspaces/default/provider/ollama/-/v1/chat/completions", id="provider route"),
        pytest.param("/v2/workspaces/e2e-test/openai/-/v1/chat/completions", id="virtual model route"),
    ],
)
def test_a_routers_key_that_reaches_a_handler_is_refused_even_with_its_context(
    app: FastAPI, client: TestClient, mock_proxy_client, router: str, seal: CallerContextSeal, path: str
) -> None:
    """The service app alone has no caller-context middleware: a call back reaching it must not run as the router."""
    vm_cache = VirtualModelCache()
    vm_cache.rebuild(
        [VirtualModel(name="alias", workspace="e2e-test", default_model_entity="e2e-test/meta_llama-3.2-1b-instruct")]
    )
    app.dependency_overrides[global_virtual_model_cache] = lambda: vm_cache

    response = client.post(
        path,
        headers={"Authorization": f"Bearer {ROUTER_KEY}", CALLER_CONTEXT_HEADER: seal.seal(router, CALLER)},
        json={"model": "e2e-test/alias", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 403
    mock_proxy_client.request.assert_not_called()


async def _seen_through_middleware(headers: list[tuple[bytes, bytes]], path: str = GATEWAY_PATH) -> dict | int:
    """The headers the app behind the middleware receives, or the status the middleware answers with itself."""
    seen: dict[str, str] = {}

    async def app(scope, receive, send) -> None:  # noqa: ANN001
        seen.update({name.decode(): value.decode() for name, value in scope["headers"]})
        await PlainTextResponse("ok")(scope, receive, send)

    middleware = CallerContextMiddleware(
        app, router_keys=lambda: {"default/tot": ROUTER_KEY}, seal=lambda: CallerContextSeal(KEY)
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://testserver") as client:
        response = await client.get(path, headers=headers)
    return seen if response.status_code == 200 and seen else response.status_code


async def test_the_middleware_puts_the_caller_in_place_of_its_router() -> None:
    context = CallerContextSeal(KEY).seal("default/tot", CALLER)

    seen = await _seen_through_middleware(
        [(b"authorization", f"Bearer {ROUTER_KEY}".encode()), (CALLER_CONTEXT_HEADER.encode(), context.encode())]
    )

    assert isinstance(seen, dict)
    assert seen["authorization"] == f"Bearer {CALLER}"
    assert CALLER_CONTEXT_HEADER not in seen


async def test_the_middleware_drops_a_context_sent_without_its_routers_key() -> None:
    context = CallerContextSeal(KEY).seal("default/tot", CALLER)

    seen = await _seen_through_middleware(
        [(b"x-api-key", b"someone-else"), (CALLER_CONTEXT_HEADER.encode(), context.encode())]
    )

    assert isinstance(seen, dict)
    assert seen["x-api-key"] == "someone-else"
    assert "authorization" not in seen
    assert CALLER_CONTEXT_HEADER not in seen


async def test_the_middleware_refuses_a_routers_key_without_a_context() -> None:
    assert await _seen_through_middleware([(b"authorization", f"Bearer {ROUTER_KEY}".encode())]) == 403


@pytest.mark.parametrize("name", [b"authorization", b"x-api-key"])
async def test_the_middleware_refuses_a_second_credential_header(name: bytes) -> None:
    """Platform auth reads the first credential; a router's key sent first, with another after it, must not slip by."""
    first = f"Bearer {ROUTER_KEY}".encode() if name == b"authorization" else ROUTER_KEY.encode()

    assert await _seen_through_middleware([(name, first), (name, b"someone-else")]) == 400


def test_the_handler_refusal_reads_the_credential_platform_auth_reads() -> None:
    headers = Headers(raw=[(b"authorization", f"Bearer {ROUTER_KEY}".encode()), (b"authorization", b"Bearer other")])

    with pytest.raises(HTTPException) as exc_info:
        refuse_router_keys(headers, {"default/tot": ROUTER_KEY})

    assert exc_info.value.status_code == 403


async def test_the_middleware_leaves_other_platform_routes_alone() -> None:
    seen = await _seen_through_middleware(
        [(b"authorization", f"Bearer {ROUTER_KEY}".encode())], path="/apis/models/v2/x"
    )

    assert isinstance(seen, dict)
    assert seen["authorization"] == f"Bearer {ROUTER_KEY}"


@pytest.mark.parametrize(
    "key",
    [pytest.param("not*base64!!", id="not base64"), pytest.param(base64.b64encode(bytes(16)).decode(), id="16 bytes")],
)
def test_a_bad_caller_context_key_fails_at_startup_without_echoing_it(key: str) -> None:
    """Otherwise every inference request would fail when the gateway first builds its seal."""
    with pytest.raises(ValidationError) as exc_info:
        InferenceGatewayConfig(caller_context_key=key)

    assert key not in str(exc_info.value)


def test_a_32_byte_caller_context_key_is_accepted() -> None:
    config = InferenceGatewayConfig(caller_context_key=base64.b64encode(KEY).decode())

    assert config.caller_context_key is not None
