# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for translate factory with path update.

Tests that the Switchyard middleware correctly:
1. Registers translate factory with _VMFactoryInstance
2. Adds _PathUpdateProcessor to request pipeline
3. Updates request.path based on target format from context
"""

from __future__ import annotations

import pytest
from nemo_helix_plugin.inference_middleware import (
    BackendFormat,
    InferenceMiddlewareContext,
    InferenceRequest,
    InferenceResponse,
)
from nemo_helix_plugin.inference_middleware_models import (
    MiddlewareCall,
    VirtualModel,
    VirtualModelInferenceConfig,
)
from nemo_switchyard import _state as middleware_state
from nemo_switchyard.middleware import SwitchyardMiddleware


class TestE2ETranslate:
    """End-to-end translate routing scenarios."""

    @pytest.fixture
    async def middleware(self) -> SwitchyardMiddleware:
        mw = SwitchyardMiddleware()
        await mw.on_startup()
        yield mw
        await mw.on_shutdown()

    @pytest.fixture
    def virtual_model_translate(self) -> VirtualModel:
        """VirtualModel configured to use translate factory.

        Routes from OpenAI format to Anthropic format.
        """
        return VirtualModel(
            id="vm-translate-123",
            workspace="test-workspace",
            name="translate-model",
            models=[
                VirtualModelInferenceConfig(
                    model="workspace/claude-opus",
                    backend_format=BackendFormat.ANTHROPIC_MESSAGES,
                ),
                VirtualModelInferenceConfig(
                    model="workspace/gpt-4",
                    backend_format=BackendFormat.OPENAI_CHAT,
                ),
            ],
            request_middleware=[
                MiddlewareCall(
                    name="nemo-switchyard",
                    config_type="translate",
                    config={
                        "target_format": "anthropic",
                        "enable_stats": False,
                    },
                ),
            ],
            response_middleware=[
                MiddlewareCall(
                    name="nemo-switchyard",
                    config_type="translate",
                    config={
                        "target_format": "anthropic",
                        "enable_stats": False,
                    },
                ),
            ],
            post_response_middleware=[],
        )

    @pytest.fixture
    def middleware_ctx(self) -> InferenceMiddlewareContext:
        """Create a test InferenceMiddlewareContext."""
        return InferenceMiddlewareContext(
            request_id="translate-123",
            workspace="test-workspace",
            virtual_model_name="translate-model",
            original_request=InferenceRequest(
                body={
                    "model": "test-workspace/translate-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers={},
                path="v1/chat/completions",
            ),
        )

    @pytest.mark.asyncio
    async def test_translate_registers_factory_with_vm_instance(
        self,
        middleware: SwitchyardMiddleware,
        virtual_model_translate: VirtualModel,
    ) -> None:
        """Test that translate config creates _VMFactoryInstance and registers it."""
        await middleware.on_virtual_model_upserted(virtual_model_translate)

        vm_key = f"{virtual_model_translate.workspace}/{virtual_model_translate.name}"
        assert (vm_key, "translate", "request") in middleware_state.VM_NAME_TO_CONFIG_HASH
        cfg_hash = middleware_state.VM_NAME_TO_CONFIG_HASH[(vm_key, "translate", "request")]
        assert cfg_hash in middleware_state.FACTORIES_BY_CONFIG_HASH

    @pytest.mark.asyncio
    async def test_translate_factory_processes_request(
        self,
        middleware: SwitchyardMiddleware,
        virtual_model_translate: VirtualModel,
        middleware_ctx: InferenceMiddlewareContext,
    ) -> None:
        """Test that translate factory processes request and updates path."""
        await middleware.on_virtual_model_upserted(virtual_model_translate)

        # Use a model that's in the translate config's models list (anthropic-backed)
        # so ModelFormatLookupProcessor stamps CTX_TARGET_FORMAT and PathUpdateProcessor
        # rewrites request.path. With target_format=anthropic and the model resolving to
        # ANTHROPIC_MESSAGES, the rewritten path should be the Anthropic Messages route.
        request = InferenceRequest(
            body={
                "model": "workspace/claude-opus",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={},
            path="v1/chat/completions",
        )
        request.typed_body = request.body

        result = await middleware.process_request(middleware_ctx, request, {"config_type": "translate"})

        assert result is request
        assert "messages" in result.body
        assert result.path == "v1/messages"

    @pytest.mark.asyncio
    async def test_translate_response_processing(
        self,
        middleware: SwitchyardMiddleware,
        virtual_model_translate: VirtualModel,
        middleware_ctx: InferenceMiddlewareContext,
    ) -> None:
        """Full round-trip: OpenAI request → translate → Anthropic backend →
        Anthropic response → translate response pipeline → OpenAI response back.

        process_request stamps CTX_ORIGINAL_FORMAT=OPENAI_CHAT via
        StampOriginalFormatProcessor. process_response reads it and translates
        the Anthropic Message back to OpenAI format via the Switchyard pipeline."""
        import anthropic.types as anthropic_types
        import openai.types.chat as openai_chat_types

        await middleware.on_virtual_model_upserted(virtual_model_translate)

        # 1. Run the request pipeline — stamps CTX_ORIGINAL_FORMAT into bridged metadata.
        request = InferenceRequest(
            body={"model": "workspace/claude-opus", "messages": [{"role": "user", "content": "hi"}]},
            headers={},
            path="v1/chat/completions",
        )
        request.typed_body = request.body
        await middleware.process_request(middleware_ctx, request, {"config_type": "translate"})

        # 2. Simulate an Anthropic-format response from the backend.
        anthropic_msg = anthropic_types.Message.model_validate(
            {
                "id": "msg_test",
                "content": [{"type": "text", "text": "hello back"}],
                "model": "claude-opus",
                "role": "assistant",
                "stop_reason": "end_turn",
                "type": "message",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            }
        )
        response = InferenceResponse(result={}, headers={}, typed_body=anthropic_msg)

        # 3. Run the response pipeline — FormatTranslateResponseProcessor reads
        # CTX_ORIGINAL_FORMAT=OPENAI_CHAT and translates Anthropic → OpenAI.
        result = await middleware.process_response(middleware_ctx, response, {"config_type": "translate"})

        assert result is response
        # Translated to OpenAI: typed_body updated to ChatCompletion, result updated.
        assert isinstance(result.typed_body, openai_chat_types.ChatCompletion)
        assert result.typed_body.choices[0].message.content == "hello back"

    @pytest.mark.asyncio
    async def test_unified_factory_wrapper_for_all_types(
        self,
        middleware: SwitchyardMiddleware,
    ) -> None:
        """Test that all factory types use _VMFactoryInstance wrapper.

        Verify the unified approach works for different config types.
        """
        # Create VMs with different config types
        vm_random = VirtualModel(
            id="vm-random-unified",
            workspace="test-workspace",
            name="random-unified",
            models=[
                VirtualModelInferenceConfig(
                    model="workspace/model-a",
                    backend_format=BackendFormat.OPENAI_CHAT,
                ),
                VirtualModelInferenceConfig(
                    model="workspace/model-b",
                    backend_format=BackendFormat.OPENAI_CHAT,
                ),
            ],
            request_middleware=[
                MiddlewareCall(
                    name="nemo-switchyard",
                    config_type="random_routing",
                    config={
                        "strong": {"model": "workspace/model-a"},
                        "weak": {"model": "workspace/model-b"},
                        "strong_probability": 0.5,
                        "enable_stats": False,
                    },
                ),
            ],
            response_middleware=[],
            post_response_middleware=[],
        )

        vm_translate = VirtualModel(
            id="vm-translate-unified",
            workspace="test-workspace",
            name="translate-unified",
            models=[
                VirtualModelInferenceConfig(
                    model="workspace/anthropic-model",
                    backend_format=BackendFormat.ANTHROPIC_MESSAGES,
                ),
            ],
            request_middleware=[
                MiddlewareCall(
                    name="nemo-switchyard",
                    config_type="translate",
                    config={
                        "target_format": "anthropic",
                        "enable_stats": False,
                    },
                ),
            ],
            response_middleware=[],
            post_response_middleware=[],
        )

        await middleware.on_virtual_model_upserted(vm_random)
        await middleware.on_virtual_model_upserted(vm_translate)

        random_key = (f"{vm_random.workspace}/{vm_random.name}", "random_routing", "request")
        translate_key = (f"{vm_translate.workspace}/{vm_translate.name}", "translate", "request")
        assert random_key in middleware_state.VM_NAME_TO_CONFIG_HASH
        assert translate_key in middleware_state.VM_NAME_TO_CONFIG_HASH
        assert middleware_state.VM_NAME_TO_CONFIG_HASH[random_key] in middleware_state.FACTORIES_BY_CONFIG_HASH
        assert middleware_state.VM_NAME_TO_CONFIG_HASH[translate_key] in middleware_state.FACTORIES_BY_CONFIG_HASH
