# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fake Algorithm + host-loop tests (no switchyard_rust)."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_platform_plugin.inference_middleware import (
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceRequest,
    InferenceResponse,
    ModelProviderInferenceTarget,
)
from nemo_platform_plugin.inference_middleware_models import MiddlewareCall, VirtualModel
from nemo_switchyard import _state
from nemo_switchyard._native_config import map_random_routing_config, models_map_from_config
from nemo_switchyard._native_host import (
    IgwJudgeTransport,
    NativeBinding,
    native_request_dict,
    run_native_stream,
)
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
    def __init__(self, selected: str, request: dict[str, Any], response: Any = None) -> None:
        self.selected_model_ids = [selected]
        self.request = request
        self.response = response


class FakeDone:
    def __init__(self, outcome: FakeOutcome) -> None:
        self.outcome = outcome


class CountingAlgorithm:
    def __init__(self, selected: str = "workspace/llama-3-70b", *, calls: int = 1) -> None:
        self.selected = selected
        self.judge_calls = calls
        self.user_http = 0
        self.run_count = 0

    async def run_stream(
        self,
        request: dict[str, Any],
        models: dict[str, list[str]],
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[Any]:
        self.run_count += 1
        for _ in range(self.judge_calls):
            yield FakeCallModel(FakeCall(["workspace/judge"], request))
        yield FakeDone(FakeOutcome(self.selected, request, None))


class InlinedResponseAlgorithm:
    async def run_stream(self, request: dict[str, Any], models: Any, headers: Any = None) -> AsyncIterator[Any]:
        yield FakeDone(FakeOutcome("workspace/llama-3-70b", request, {"id": "inlined"}))


class RecordingTransport:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
        *,
        fail_first: int | None = None,
    ) -> None:
        self.payload = payload or {"id": "judge", "choices": []}
        self.error = error
        self.fail_first = fail_first
        self.models: list[str] = []
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    async def complete(self, model_entity_id: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        self.models.append(model_entity_id)
        self.bodies.append(body)
        self.headers.append(headers)
        if self.error is not None:
            if self.fail_first is None:
                raise self.error
            if self.fail_first > 0:
                self.fail_first -= 1
                raise self.error
        return self.payload


def _provider_target() -> ModelProviderInferenceTarget:
    return ModelProviderInferenceTarget(
        model_provider_gateway_url="https://provider.example/v1",
        served_model_name="meta/llama-3.1-70b-instruct",
    )


def _async_http_client(*, post: Any) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post = post
    return client


def _openai_request(*, content: Any = "hello") -> InferenceRequest:
    body: dict[str, Any] = {
        "model": "ws/router",
        "messages": [{"role": "user", "content": content}],
    }
    return InferenceRequest(
        body=body, headers={"authorization": "Bearer t"}, path="v1/chat/completions", typed_body=body
    )


def _ctx(request: InferenceRequest | None = None) -> InferenceMiddlewareContext:
    req = request or _openai_request()
    return InferenceMiddlewareContext(
        request_id="r1",
        workspace="ws",
        virtual_model_name="router",
        original_request=req,
    )


@pytest.mark.asyncio
async def test_counting_callmodel_uses_judge_transport_not_user_model() -> None:
    algorithm = CountingAlgorithm(calls=2)
    request = _openai_request()
    transport = RecordingTransport()
    out = await run_native_stream(
        algorithm=algorithm,
        request=request,
        models={"any": ["workspace/llama-3-70b"]},
        headers=dict(request.headers),
        transport=transport,
    )
    assert out is request
    assert request.body["model"] == "workspace/llama-3-70b"
    assert transport.models == ["workspace/judge", "workspace/judge"]
    assert algorithm.user_http == 0


@pytest.mark.asyncio
async def test_done_none_response_does_not_call_user_model() -> None:
    algorithm = CountingAlgorithm(calls=0)
    request = _openai_request()
    transport = RecordingTransport()
    await run_native_stream(
        algorithm=algorithm,
        request=request,
        models={"any": ["workspace/llama-3-70b"]},
        headers={},
        transport=transport,
    )
    assert transport.models == []
    assert request.body["model"] == "workspace/llama-3-70b"


@pytest.mark.asyncio
async def test_inlined_done_response_fails_closed() -> None:
    with pytest.raises(InferenceMiddlewareError) as exc:
        await run_native_stream(
            algorithm=InlinedResponseAlgorithm(),
            request=_openai_request(),
            models={"any": ["workspace/llama-3-70b"]},
            headers={},
            transport=RecordingTransport(),
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_judge_fail_closed_4xx() -> None:
    with pytest.raises(InferenceMiddlewareError) as exc:
        await run_native_stream(
            algorithm=CountingAlgorithm(),
            request=_openai_request(),
            models={"any": ["workspace/llama-3-70b"]},
            headers={},
            transport=RecordingTransport(error=InferenceMiddlewareError("judge 502", status_code=502)),
        )
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_call_fail_is_invoked_on_judge_error() -> None:
    class CaptureFail:
        def __init__(self) -> None:
            self.call: FakeCall | None = None

        async def run_stream(self, request: dict[str, Any], models: Any, headers: Any = None) -> AsyncIterator[Any]:
            self.call = FakeCall(["workspace/judge"], request)
            yield FakeCallModel(self.call)
            yield FakeDone(FakeOutcome("workspace/llama-3-70b", request, None))

    algo = CaptureFail()
    with pytest.raises(InferenceMiddlewareError):
        await run_native_stream(
            algorithm=algo,
            request=_openai_request(),
            models={"any": ["workspace/llama-3-70b"]},
            headers={},
            transport=RecordingTransport(error=InferenceMiddlewareError("nope", status_code=502)),
        )
    assert algo.call is not None
    assert algo.call.failures


def test_string_content_is_coerced_to_blocks() -> None:
    request = _openai_request(content="plain")
    converted = native_request_dict(request.body)
    assert converted["messages"][0]["content"] == [{"type": "text", "text": "plain"}]
    assert request.body["messages"][0]["content"] == "plain"


def test_content_blocks_pass_through() -> None:
    blocks = [{"type": "text", "text": "already"}]
    converted = native_request_dict(_openai_request(content=blocks).body)
    assert converted["messages"][0]["content"] == blocks


def test_random_mapper_weight_zero_never_selected() -> None:
    weights, seed, models = map_random_routing_config(
        {
            "strong": {"model": "ws/strong"},
            "weak": {"model": "ws/weak"},
            "strong_probability": 1.0,
            "rng_seed": 7,
        }
    )
    assert weights == [1.0, 0.0]
    assert seed == 7
    assert models == {"any": ["ws/strong", "ws/weak"]}


def test_random_mapper_requires_strong_weak() -> None:
    with pytest.raises(InferenceMiddlewareError) as exc:
        map_random_routing_config({"strong_probability": 0.5})
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_igw_judge_uses_cache_accessor_not_localhost() -> None:
    mw = SwitchyardMiddleware()
    mw.get_inference_url_and_model = MagicMock(return_value=_provider_target())
    mw.get_model_entity = MagicMock(return_value=None)
    transport = IgwJudgeTransport(mw, timeout=5.0)
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"id": "ok"}
    client = _async_http_client(post=AsyncMock(return_value=response))
    with patch("nemo_switchyard._native_host.httpx.AsyncClient", return_value=client) as ctor:
        payload = await transport.complete(
            "ws/judge",
            {"messages": []},
            {"authorization": "Bearer leaked", "x-switchyard-session-id": "s1"},
        )
    assert payload == {"id": "ok"}
    timeout = ctor.call_args.kwargs["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    posted = client.post.await_args
    assert posted.args[0] == "https://provider.example/v1/chat/completions"
    assert "localhost" not in posted.args[0]
    assert posted.kwargs["json"]["model"] == "meta/llama-3.1-70b-instruct"
    assert "authorization" not in {k.lower() for k in posted.kwargs["headers"]}
    assert posted.kwargs["headers"] == {}


@pytest.mark.asyncio
async def test_igw_judge_timeout() -> None:
    mw = SwitchyardMiddleware()
    mw.get_inference_url_and_model = MagicMock(return_value=_provider_target())
    mw.get_model_entity = MagicMock(return_value=None)
    client = _async_http_client(post=AsyncMock(side_effect=httpx.TimeoutException("slow")))
    with patch("nemo_switchyard._native_host.httpx.AsyncClient", return_value=client):
        with pytest.raises(InferenceMiddlewareError) as exc:
            await IgwJudgeTransport(mw, timeout=0.01).complete("ws/judge", {}, {})
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_igw_judge_http_error_fail_closed() -> None:
    mw = SwitchyardMiddleware()
    mw.get_inference_url_and_model = MagicMock(return_value=_provider_target())
    mw.get_model_entity = MagicMock(return_value=None)
    response = MagicMock()
    response.status_code = 400
    response.text = "model does not exist"
    client = _async_http_client(post=AsyncMock(return_value=response))
    with patch("nemo_switchyard._native_host.httpx.AsyncClient", return_value=client):
        with pytest.raises(InferenceMiddlewareError) as exc:
            await IgwJudgeTransport(mw).complete("ws/judge", {}, {})
    assert exc.value.status_code == 502
    assert "400" in str(exc.value)
    assert "model does not exist" in str(exc.value)


@pytest.mark.asyncio
async def test_process_request_native_binding_and_streaming_identity() -> None:
    mw = SwitchyardMiddleware()
    await mw.on_startup()
    cfg_hash = "native-test-hash"
    _state.NATIVE_BY_CONFIG_HASH[cfg_hash] = NativeBinding(
        algorithm=CountingAlgorithm(calls=0),
        models={"any": ["workspace/llama-3-70b"]},
        config_type="stage_router",
    )
    _state.VM_NAME_TO_CONFIG_HASH[("ws/router", "stage_router", "request")] = cfg_hash
    request = _openai_request()
    out = await mw.process_request(_ctx(request), request, {"config_type": "stage_router"})
    assert out is request
    assert request.body["model"] == "workspace/llama-3-70b"

    async def stream() -> AsyncIterator[dict[str, Any]]:
        yield {"id": "chunk"}

    body = stream()
    response = InferenceResponse(result=body, headers={})
    routed = await mw.process_response(_ctx(), response, {"config_type": "stage_router"})
    assert routed.result is body
    await mw.on_shutdown()


def _stage_router_vm() -> VirtualModel:
    return VirtualModel(
        id="vm-lazy",
        workspace="ws",
        name="router",
        models=[],
        request_middleware=[
            MiddlewareCall(
                name="nemo-switchyard",
                config_type="stage_router",
                config={
                    "confidence_threshold": 0.5,
                    "models": {"capable": "workspace/strong", "efficient": "workspace/weak"},
                },
            )
        ],
    )


@pytest.mark.asyncio
async def test_process_request_rebuilds_native_binding_from_virtual_model() -> None:
    mw = SwitchyardMiddleware()
    await mw.on_startup()
    algorithm = CountingAlgorithm(selected="workspace/weak", calls=0)
    mw.get_virtual_model = MagicMock(return_value=_stage_router_vm())
    request = _openai_request()
    with (
        patch("nemo_switchyard._native_config.native_rust_available", return_value=True),
        patch("nemo_switchyard.middleware.build_native_algorithm", return_value=algorithm),
    ):
        out = await mw.process_request(_ctx(request), request, {"config_type": "stage_router"})
    assert out is request
    assert request.body["model"] == "workspace/weak"
    mw.get_virtual_model.assert_called_once_with("ws/router")
    await mw.on_shutdown()


@pytest.mark.asyncio
async def test_process_request_native_miss_without_virtual_model_stays_400() -> None:
    mw = SwitchyardMiddleware()
    await mw.on_startup()
    mw.get_virtual_model = MagicMock(return_value=None)
    request = _openai_request()
    with pytest.raises(InferenceMiddlewareError) as exc:
        await mw.process_request(_ctx(request), request, {"config_type": "stage_router"})
    assert exc.value.status_code == 400
    assert "No native Algorithm registered" in str(exc.value)
    await mw.on_shutdown()


@pytest.mark.asyncio
async def test_vm_destroy_unregisters_native_algorithm() -> None:
    mw = SwitchyardMiddleware()
    await mw.on_startup()
    cfg_hash = "native-destroy"
    _state.NATIVE_BY_CONFIG_HASH[cfg_hash] = NativeBinding(
        algorithm=CountingAlgorithm(),
        models={"any": ["workspace/llama-3-70b"]},
        config_type="stage_router",
    )
    _state.VM_NAME_TO_CONFIG_HASH[("ws/router", "stage_router", "request")] = cfg_hash
    _state.VM_CONFIG_MAPPING["vm-native"] = [cfg_hash]
    vm = VirtualModel(id="vm-native", workspace="ws", name="router", models=[])
    await mw.on_virtual_model_destroyed(vm)
    assert cfg_hash not in _state.NATIVE_BY_CONFIG_HASH
    await mw.on_shutdown()


@pytest.mark.asyncio
async def test_upsert_without_switchyard_drops_stale_native_binding() -> None:
    mw = SwitchyardMiddleware()
    await mw.on_startup()
    cfg_hash = "native-stale"
    _state.NATIVE_BY_CONFIG_HASH[cfg_hash] = NativeBinding(
        algorithm=CountingAlgorithm(),
        models={"any": ["workspace/llama-3-70b"]},
        config_type="stage_router",
    )
    _state.VM_CONFIG_MAPPING["vm-stale"] = [cfg_hash]
    vm = VirtualModel(id="vm-stale", workspace="ws", name="router", models=[], request_middleware=[])
    await mw.on_virtual_model_upserted(vm)
    assert cfg_hash not in _state.NATIVE_BY_CONFIG_HASH
    await mw.on_shutdown()


@pytest.mark.asyncio
async def test_concurrent_native_requests_share_binding() -> None:
    mw = SwitchyardMiddleware()
    await mw.on_startup()
    algorithm = CountingAlgorithm(calls=0)
    cfg_hash = "native-conc"
    _state.NATIVE_BY_CONFIG_HASH[cfg_hash] = NativeBinding(
        algorithm=algorithm,
        models={"any": ["workspace/llama-3-70b"]},
        config_type="llm_classifier",
    )
    _state.VM_NAME_TO_CONFIG_HASH[("ws/router", "llm_classifier", "request")] = cfg_hash

    async def one() -> None:
        req = _openai_request()
        await mw.process_request(_ctx(req), req, {"config_type": "llm_classifier"})

    await asyncio.gather(one(), one(), one())
    assert algorithm.run_count == 3
    await mw.on_shutdown()
    import nemo_switchyard.middleware as mod

    source = inspect.getsource(mod)
    assert "import switchyard_rust" not in source
    assert "switchyard_rust" not in __import__("sys").modules


@pytest.mark.asyncio
async def test_native_upsert_400_without_rust() -> None:
    mw = SwitchyardMiddleware()
    await mw.on_startup()
    with pytest.raises(InferenceMiddlewareError) as exc:
        await mw.validate_middleware_config(
            "stage_router",
            {
                "confidence_threshold": 0.5,
                "models": {"capable": ["ws/strong"], "efficient": ["ws/weak"]},
            },
        )
    assert exc.value.status_code == 400
    assert "switchyard_rust" in str(exc.value)
    with pytest.raises(InferenceMiddlewareError) as exc2:
        await mw.validate_middleware_config(
            "llm_classifier",
            {
                "base_threshold": 0.5,
                "models": {"judge": ["ws/j"], "capable": ["ws/s"], "efficient": ["ws/w"]},
            },
        )
    assert exc2.value.status_code == 400
    await mw.on_shutdown()


@pytest.mark.asyncio
async def test_native_missing_required_models() -> None:
    from nemo_switchyard._native_config import validate_stage_router_config

    with pytest.raises(InferenceMiddlewareError) as exc:
        validate_stage_router_config({"confidence_threshold": 0.5, "models": {}})
    assert "capable" in str(exc.value)


@pytest.mark.asyncio
async def test_callmodel_falls_back_to_next_model() -> None:
    class TwoModelCall(CountingAlgorithm):
        async def run_stream(self, request: dict[str, Any], models: Any, headers: Any = None) -> AsyncIterator[Any]:
            self.run_count += 1
            yield FakeCallModel(FakeCall(["workspace/judge-a", "workspace/judge-b"], request))
            yield FakeDone(FakeOutcome(self.selected, request, None))

    transport = RecordingTransport(
        error=InferenceMiddlewareError("first down", status_code=502),
        fail_first=1,
    )
    request = _openai_request()
    await run_native_stream(
        algorithm=TwoModelCall(),
        request=request,
        models={"any": ["workspace/llama-3-70b"]},
        headers={},
        transport=transport,
    )
    assert transport.models == ["workspace/judge-a", "workspace/judge-b"]
    assert "instructions" not in transport.bodies[0]


@pytest.mark.asyncio
async def test_outcome_request_rewrites_user_messages() -> None:
    class RewriteAlgorithm:
        async def run_stream(self, request: dict[str, Any], models: Any, headers: Any = None) -> AsyncIterator[Any]:
            yield FakeDone(
                FakeOutcome(
                    "workspace/llama-3-70b",
                    {
                        "instructions": [{"role": "system", "content": [{"type": "text", "text": "note"}]}],
                        "messages": request["messages"],
                    },
                    None,
                )
            )

    request = _openai_request()
    await run_native_stream(
        algorithm=RewriteAlgorithm(),
        request=request,
        models={"any": ["workspace/llama-3-70b"]},
        headers={},
        transport=RecordingTransport(),
    )
    assert request.body["messages"][0]["role"] == "system"
    assert request.body["messages"][0]["content"] == "note"


@pytest.mark.asyncio
async def test_igw_judge_injects_cached_provider_secret() -> None:
    mw = SwitchyardMiddleware()
    mw.get_inference_url_and_model = MagicMock(return_value=_provider_target())
    provider = MagicMock()
    provider.api_key_secret_name = "nvidia-key"
    provider.auth_header_format = None
    provider.default_extra_headers = {}
    provider.required_extra_headers = {}
    info = MagicMock()
    info.secret_value = "sk-test"
    info.model_provider = provider
    entity = MagicMock()
    entity.model_providers = [("meta/llama", info)]
    mw.get_model_entity = MagicMock(return_value=entity)
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"id": "ok"}
    client = _async_http_client(post=AsyncMock(return_value=response))
    with patch("nemo_switchyard._native_host.httpx.AsyncClient", return_value=client):
        await IgwJudgeTransport(mw).complete("ws/judge", {"messages": []}, {"authorization": "Bearer caller"})
    posted = client.post.await_args
    assert posted.kwargs["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_run_stream_overall_timeout() -> None:
    class Stuck:
        async def run_stream(self, request: dict[str, Any], models: Any, headers: Any = None) -> AsyncIterator[Any]:
            await asyncio.sleep(10)
            yield FakeDone(FakeOutcome("workspace/llama-3-70b", request, None))

    with pytest.raises(InferenceMiddlewareError) as exc:
        await run_native_stream(
            algorithm=Stuck(),
            request=_openai_request(),
            models={"any": ["workspace/llama-3-70b"]},
            headers={},
            transport=RecordingTransport(),
            timeout=0.01,
        )
    assert exc.value.status_code == 504


def test_models_any_excludes_judge() -> None:
    mapping = models_map_from_config(
        {"models": {"judge": ["ws/j"], "capable": ["ws/s"], "efficient": ["ws/w"]}},
        required=("judge", "capable", "efficient"),
    )
    assert mapping["any"] == ["ws/s", "ws/w"]


def test_bool_probability_rejected() -> None:
    with pytest.raises(InferenceMiddlewareError):
        map_random_routing_config(
            {
                "strong": {"model": "ws/s"},
                "weak": {"model": "ws/w"},
                "strong_probability": True,
            }
        )


def test_message_hash_fallback_requires_session_affinity() -> None:
    from nemo_switchyard._native_config import validate_llm_classifier_config

    with pytest.raises(InferenceMiddlewareError) as exc:
        validate_llm_classifier_config(
            {
                "base_threshold": 0.5,
                "message_hash_fallback": True,
                "models": {"judge": ["ws/j"], "capable": ["ws/s"], "efficient": ["ws/w"]},
            }
        )
    assert "session_affinity" in str(exc.value)


def test_deescalation_note_requires_escalation_note() -> None:
    from nemo_switchyard._native_config import validate_stage_router_config

    with pytest.raises(InferenceMiddlewareError) as exc:
        validate_stage_router_config(
            {
                "confidence_threshold": 0.5,
                "handoff_notes": {"deescalation_note": "down"},
                "models": {"capable": ["ws/s"], "efficient": ["ws/w"]},
            }
        )
    assert "escalation_note" in str(exc.value)
