# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_helix_plugin.inference_middleware import (
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceRequest,
    ModelProviderInferenceTarget,
)
from nemo_helix_plugin.inference_middleware_models import MiddlewareCall, VirtualModel
from nemo_switchyard import _state
from nemo_switchyard._native_config import (
    map_random_routing_config,
    models_map_from_config,
    validate_llm_classifier_config,
    validate_stage_router_config,
)
from nemo_switchyard._native_host import IgwJudgeTransport, NativeBinding, run_native_stream
from nemo_switchyard.middleware import SwitchyardMiddleware


@pytest.fixture(autouse=True)
def clear_native_state() -> Iterator[None]:
    yield
    _state.clear_all()


class FakeCall:
    def __init__(self, models: list[str], request: dict[str, Any]) -> None:
        self.models = models
        self.request = request
        self.responses: list[Any] = []
        self.failures: list[BaseException] = []

    def respond(self, response: Any) -> None:
        self.responses.append(response)

    def fail(self, error: BaseException) -> None:
        self.failures.append(error)


class FakeCallModel:
    def __init__(self, call: FakeCall) -> None:
        self.call = call


class FakeOutcome:
    def __init__(
        self,
        selected: str,
        request: dict[str, Any],
        response: Any = None,
    ) -> None:
        self.selected_model_ids = [selected]
        self.request = request
        self.response = response


class FakeDone:
    def __init__(self, outcome: FakeOutcome) -> None:
        self.outcome = outcome


class FakeAlgorithm:
    def __init__(self, selected: str = "ws/strong", *, judge: bool = False) -> None:
        self.selected = selected
        self.judge = judge
        self.active = 0
        self.max_active = 0

    async def run_stream(
        self,
        request: dict[str, Any],
        models: dict[str, list[str]],
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[Any]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        if self.judge:
            yield FakeCallModel(FakeCall(["ws/judge"], request))
        yield FakeDone(FakeOutcome(self.selected, request))
        self.active -= 1


class RecordingTransport:
    def __init__(self, error: InferenceMiddlewareError | None = None) -> None:
        self.error = error
        self.models: list[str] = []
        self.bodies: list[dict[str, Any]] = []

    async def complete(
        self,
        model_entity_id: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        del headers
        self.models.append(model_entity_id)
        self.bodies.append(deepcopy(body))
        if self.error is not None:
            raise self.error
        return {"id": "judge", "choices": []}


def _request(path: str = "v1/chat/completions") -> InferenceRequest:
    body = {"model": "ws/router", "messages": [{"role": "user", "content": "hello"}]}
    return InferenceRequest(
        body=body,
        headers={"authorization": "Bearer caller", "x-request-id": "r1"},
        path=path,
        typed_body=body,
    )


def _ctx(request: InferenceRequest | None = None) -> InferenceMiddlewareContext:
    original = request or _request()
    return InferenceMiddlewareContext(
        request_id="r1",
        workspace="ws",
        virtual_model_name="router",
        original_request=original,
    )


def _random_call() -> MiddlewareCall:
    return MiddlewareCall(
        name="nemo-switchyard",
        config_type="random_routing",
        config={
            "strong": {"model": "ws/strong"},
            "weak": {"model": "ws/weak"},
            "strong_probability": 1.0,
            "rng_seed": 7,
        },
    )


def _vm(
    *, request_calls: list[MiddlewareCall] | None = None, response_calls: list[MiddlewareCall] | None = None
) -> VirtualModel:
    return VirtualModel.model_validate(
        {
            "id": "vm-1",
            "workspace": "ws",
            "name": "router",
            "models": [],
            "request_middleware": request_calls or [],
            "response_middleware": response_calls or [],
        }
    )


@pytest.mark.asyncio
async def test_run_stream_serves_judge_then_routes_user_request() -> None:
    transport = RecordingTransport()
    request = _request()
    request.body["stream"] = True

    result = await run_native_stream(
        algorithm=FakeAlgorithm(judge=True),
        request=request,
        models={"judge": ["ws/judge"], "any": ["ws/strong"]},
        headers=dict(request.headers),
        transport=transport,
    )

    assert result is request
    assert request.body["model"] == "ws/strong"
    assert request.body["stream"] is True
    assert transport.models == ["ws/judge"]
    assert transport.bodies[0]["stream"] is False


@pytest.mark.asyncio
async def test_noop_outcome_preserves_original_openai_body() -> None:
    body = {
        "model": "ws/router",
        "messages": [
            {"role": "user", "name": "customer", "content": "hello"},
            {"role": "system", "content": "late instruction"},
            {"role": "assistant", "content": None, "refusal": "cannot comply"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "search", "description": "find", "parameters": {"type": "object"}},
            }
        ],
        "parallel_tool_calls": False,
    }
    original = deepcopy(body)
    request = InferenceRequest(body=body, headers={}, path="v1/chat/completions", typed_body=body)

    await run_native_stream(
        algorithm=FakeAlgorithm(),
        request=request,
        models={"any": ["ws/strong"]},
        headers={},
        transport=RecordingTransport(),
    )

    assert request.body == {**original, "model": "ws/strong"}


@pytest.mark.asyncio
async def test_failed_judge_call_allows_algorithm_fallback_done() -> None:
    algorithm = FakeAlgorithm(judge=True)
    request = _request()

    await run_native_stream(
        algorithm=algorithm,
        request=request,
        models={"judge": ["ws/judge"], "any": ["ws/strong"]},
        headers={},
        transport=RecordingTransport(InferenceMiddlewareError("judge unavailable", status_code=502)),
    )

    assert request.body["model"] == "ws/strong"


@pytest.mark.asyncio
async def test_run_stream_rejects_non_openai_chat_path() -> None:
    with pytest.raises(InferenceMiddlewareError) as exc:
        await run_native_stream(
            algorithm=FakeAlgorithm(),
            request=_request("v1/messages"),
            models={"any": ["ws/strong"]},
            headers={},
            transport=RecordingTransport(),
        )

    assert exc.value.status_code == 400
    assert "translation is not available" in str(exc.value)


@pytest.mark.asyncio
async def test_run_stream_fails_closed_on_inlined_response() -> None:
    class InlinedResponse:
        async def run_stream(
            self,
            request: dict[str, Any],
            models: dict[str, list[str]],
            headers: dict[str, str] | None = None,
        ) -> AsyncIterator[Any]:
            yield FakeDone(FakeOutcome("ws/strong", request, {"id": "response"}))

    with pytest.raises(InferenceMiddlewareError) as exc:
        await run_native_stream(
            algorithm=InlinedResponse(),
            request=_request(),
            models={"any": ["ws/strong"]},
            headers={},
            transport=RecordingTransport(),
        )

    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_run_stream_times_out() -> None:
    class Stuck:
        async def run_stream(
            self,
            request: dict[str, Any],
            models: dict[str, list[str]],
            headers: dict[str, str] | None = None,
        ) -> AsyncIterator[Any]:
            await asyncio.sleep(10)
            yield FakeDone(FakeOutcome("ws/strong", request))

    with pytest.raises(InferenceMiddlewareError) as exc:
        await run_native_stream(
            algorithm=Stuck(),
            request=_request(),
            models={"any": ["ws/strong"]},
            headers={},
            transport=RecordingTransport(),
            timeout=0.01,
        )

    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_run_stream_timeout_includes_lock_wait() -> None:
    lock = asyncio.Lock()
    await lock.acquire()
    try:
        with pytest.raises(InferenceMiddlewareError) as exc:
            await run_native_stream(
                algorithm=FakeAlgorithm(),
                request=_request(),
                models={"any": ["ws/strong"]},
                headers={},
                transport=RecordingTransport(),
                timeout=0.01,
                lock=lock,
            )
    finally:
        lock.release()

    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_upsert_replaces_and_destroy_removes_per_vm_binding() -> None:
    middleware = SwitchyardMiddleware()
    first = FakeAlgorithm("ws/strong")
    second = FakeAlgorithm("ws/weak")
    with patch(
        "nemo_switchyard.middleware.build_native_algorithm",
        side_effect=[first, second],
    ):
        await middleware.on_virtual_model_upserted(_vm(request_calls=[_random_call()]))
        assert _state.BINDINGS[("ws/router", "random_routing")].algorithm is first
        await middleware.on_virtual_model_upserted(_vm(request_calls=[_random_call()]))

    assert _state.BINDINGS[("ws/router", "random_routing")].algorithm is second
    await middleware.on_virtual_model_destroyed(_vm())
    assert not _state.BINDINGS
    assert not _state.VM_BINDING_KEYS


@pytest.mark.asyncio
async def test_response_registration_is_rejected() -> None:
    middleware = SwitchyardMiddleware()
    with pytest.raises(InferenceMiddlewareError) as exc:
        await middleware.on_virtual_model_upserted(_vm(response_calls=[_random_call()]))

    assert exc.value.status_code == 400
    assert "request-only" in str(exc.value)


@pytest.mark.asyncio
async def test_duplicate_config_type_is_rejected() -> None:
    middleware = SwitchyardMiddleware()
    call = _random_call()
    with (
        patch("nemo_switchyard.middleware.build_native_algorithm", return_value=FakeAlgorithm()),
        pytest.raises(InferenceMiddlewareError) as exc,
    ):
        await middleware.on_virtual_model_upserted(_vm(request_calls=[call, call]))

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_binding_lock_serializes_algorithm_state() -> None:
    middleware = SwitchyardMiddleware()
    algorithm = FakeAlgorithm()
    binding = NativeBinding(
        algorithm=algorithm,
        models={"any": ["ws/strong"]},
        config_type="random_routing",
    )
    _state.BINDINGS[("ws/router", "random_routing")] = binding

    await asyncio.gather(
        *[
            middleware.process_request(
                _ctx(request := _request()),
                request,
                {"config_type": "random_routing"},
            )
            for _ in range(3)
        ]
    )

    assert algorithm.max_active == 1


def test_random_mapper_preserves_existing_config_shape() -> None:
    config = _random_call().config
    assert config is not None
    weights, seed, models = map_random_routing_config(config)

    assert weights == [1.0, 0.0]
    assert seed == 7
    assert models == {"any": ["ws/strong", "ws/weak"]}


def test_random_mapper_defaults_to_even_probability() -> None:
    weights, _seed, _models = map_random_routing_config(
        {
            "strong": {"model": "ws/strong"},
            "weak": {"model": "ws/weak"},
        }
    )

    assert weights == [0.5, 0.5]


def test_models_any_excludes_judge() -> None:
    mapping = models_map_from_config(
        {"models": {"judge": ["ws/j"], "capable": ["ws/s"], "efficient": ["ws/w"]}},
        required=("judge", "capable", "efficient"),
    )

    assert mapping["any"] == ["ws/s", "ws/w"]


@pytest.mark.parametrize(
    "validator,config,error_text",
    [
        (
            validate_llm_classifier_config,
            {
                "base_threshold": 0.5,
                "message_hash_fallback": True,
                "models": {"judge": ["ws/j"], "capable": ["ws/s"], "efficient": ["ws/w"]},
            },
            "session_affinity",
        ),
        (
            validate_stage_router_config,
            {
                "confidence_threshold": 0.5,
                "handoff_notes": {"deescalation_note": "down"},
                "models": {"capable": ["ws/s"], "efficient": ["ws/w"]},
            },
            "escalation_note",
        ),
        (
            validate_stage_router_config,
            {
                "confidence_threshold": 0.5,
                "handoff_notes": "not-an-object",
                "models": {"capable": ["ws/s"], "efficient": ["ws/w"]},
            },
            "handoff_notes must be an object",
        ),
        (
            validate_stage_router_config,
            {
                "confidence_threshold": 0.5,
                "classifier": {"base_threshold": 2},
                "models": {"capable": ["ws/s"], "efficient": ["ws/w"]},
            },
            "classifier.base_threshold",
        ),
        (
            validate_stage_router_config,
            {
                "confidence_threshold": 0.5,
                "classifier": {"base_threshold": 0.5},
                "models": {"capable": ["ws/s"], "efficient": ["ws/w"]},
            },
            "models.judge is required",
        ),
        (
            validate_llm_classifier_config,
            {
                "base_threshold": 0.5,
                "session_affinity": "false",
                "models": {"judge": ["ws/j"], "capable": ["ws/s"], "efficient": ["ws/w"]},
            },
            "session_affinity must be a boolean",
        ),
    ],
)
def test_invalid_native_config_is_rejected(
    validator: Any,
    config: dict[str, Any],
    error_text: str,
) -> None:
    with pytest.raises(InferenceMiddlewareError) as exc:
        validator(config)

    assert error_text in str(exc.value)


@pytest.mark.asyncio
async def test_judge_transport_does_not_forward_caller_credentials() -> None:
    middleware = SwitchyardMiddleware()
    middleware.get_inference_url_and_model = MagicMock(
        return_value=ModelProviderInferenceTarget(
            model_provider_gateway_url="https://provider.example/v1",
            served_model_name="served-judge",
            default_extra_body={"temperature": 0.2, "provider_default": True},
            required_extra_body={"temperature": 0.0, "provider_required": True},
            outbound_headers={
                "X-Api-Key": "provider-secret",
                "X-Default": "default",
                "X-Required": "required",
            },
        )
    )
    response = MagicMock(status_code=200)
    response.json.return_value = {"id": "ok"}
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post.return_value = response

    with patch("nemo_switchyard._native_host.httpx.AsyncClient", return_value=client):
        await IgwJudgeTransport(middleware).complete(
            "ws/judge",
            {"messages": [], "temperature": 0.8},
            {"authorization": "Bearer caller"},
        )

    posted = client.post.await_args
    assert posted.args[0] == "https://provider.example/v1/chat/completions"
    assert posted.kwargs["json"] == {
        "messages": [],
        "model": "served-judge",
        "provider_default": True,
        "provider_required": True,
        "temperature": 0.0,
    }
    assert posted.kwargs["headers"] == {
        "X-Api-Key": "provider-secret",
        "X-Default": "default",
        "X-Required": "required",
    }


@pytest.mark.asyncio
async def test_judge_transport_maps_timeout() -> None:
    middleware = SwitchyardMiddleware()
    middleware.get_inference_url_and_model = MagicMock(
        return_value=ModelProviderInferenceTarget(
            model_provider_gateway_url="https://provider.example/v1",
            served_model_name="served-judge",
        )
    )
    middleware.get_model_entity = MagicMock(return_value=None)
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post.side_effect = httpx.TimeoutException("slow")

    with (
        patch("nemo_switchyard._native_host.httpx.AsyncClient", return_value=client),
        pytest.raises(InferenceMiddlewareError) as exc,
    ):
        await IgwJudgeTransport(middleware, timeout=0.01).complete("ws/judge", {}, {})

    assert exc.value.status_code == 504
