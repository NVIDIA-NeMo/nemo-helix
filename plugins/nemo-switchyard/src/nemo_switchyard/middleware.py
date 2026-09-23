# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway middleware for native Switchyard routing algorithms."""

from __future__ import annotations

import logging
from typing import Any

from nemo_helix_plugin.inference_middleware import (
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceRequest,
    InferenceResponse,
    NemoInferenceMiddleware,
    VirtualModel,
)
from nemo_switchyard import _state
from nemo_switchyard._native_config import (
    NATIVE_CONFIG_TYPES,
    build_native_algorithm,
    models_for_native_config,
    validate_native_config,
)
from nemo_switchyard._native_host import IgwJudgeTransport, NativeBinding, run_native_stream

logger = logging.getLogger(__name__)
_MIDDLEWARE_NAME = "nemo-switchyard"


class SwitchyardMiddleware(NemoInferenceMiddleware):
    """Route OpenAI Chat requests with native Switchyard algorithms."""

    async def on_startup(self) -> None:
        logger.info("Switchyard middleware loaded with config types: %s", sorted(NATIVE_CONFIG_TYPES))

    async def on_shutdown(self) -> None:
        binding_count = len(_state.BINDINGS)
        _state.clear_all()
        logger.info("Switchyard middleware cleared %d native bindings", binding_count)

    async def validate_middleware_config(
        self,
        config_type: str,
        config: Any,
    ) -> dict[str, Any]:
        validated = validate_native_config(config_type, config)
        return {"config_type": config_type, "config": validated}

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: dict[str, Any],
    ) -> InferenceRequest:
        config_type = _config_type(middleware_config)
        binding = self._lookup_binding(ctx, config_type)
        try:
            async with binding.lock:
                return await run_native_stream(
                    algorithm=binding.algorithm,
                    request=request,
                    models=binding.models,
                    headers=dict(request.headers),
                    transport=IgwJudgeTransport(self),
                )
        except InferenceMiddlewareError:
            raise
        except Exception as exc:
            logger.error("Native Switchyard run_stream failed", exc_info=True)
            raise InferenceMiddlewareError(str(exc), status_code=500) from exc

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: dict[str, Any],
    ) -> InferenceResponse:
        del ctx, middleware_config
        return response

    async def on_virtual_model_upserted(self, virtual_model: VirtualModel) -> None:
        vm_key = _vm_key(virtual_model)
        request_entries = [call for call in (virtual_model.request_middleware or []) if call.name == _MIDDLEWARE_NAME]
        response_entries = [call for call in (virtual_model.response_middleware or []) if call.name == _MIDDLEWARE_NAME]
        if response_entries:
            raise InferenceMiddlewareError(
                "Native Switchyard algorithms are request-only; remove nemo-switchyard from response_middleware.",
                status_code=400,
            )

        bindings: dict[_state.BindingKey, NativeBinding] = {}
        for call in request_entries:
            config_type = call.config_type
            key = (vm_key, config_type)
            if key in bindings:
                raise InferenceMiddlewareError(
                    f"VirtualModel {vm_key!r} lists {config_type!r} more than once",
                    status_code=400,
                )
            bindings[key] = self._build_binding(vm_key, config_type, call.config or {})

        _state.replace_vm_bindings(virtual_model.id, bindings)
        logger.info("Registered %d native Switchyard bindings for VM %s", len(bindings), vm_key)

    async def on_virtual_model_destroyed(self, virtual_model: VirtualModel) -> None:
        _state.remove_vm_bindings(virtual_model.id)

    def _lookup_binding(
        self,
        ctx: InferenceMiddlewareContext,
        config_type: str,
    ) -> NativeBinding:
        vm_key = f"{ctx.workspace}/{ctx.virtual_model_name}"
        key = (vm_key, config_type)
        binding = _state.BINDINGS.get(key)
        if binding is not None:
            return binding

        virtual_model = self.get_virtual_model(vm_key)
        if virtual_model is not None:
            for call in virtual_model.request_middleware or []:
                if call.name == _MIDDLEWARE_NAME and call.config_type == config_type:
                    binding = self._build_binding(vm_key, config_type, call.config or {})
                    _state.BINDINGS[key] = binding
                    if key not in _state.VM_BINDING_KEYS.setdefault(virtual_model.id, []):
                        _state.VM_BINDING_KEYS[virtual_model.id].append(key)
                    return binding

        raise InferenceMiddlewareError(
            f"No native Switchyard binding registered for VM {vm_key!r} with config_type {config_type!r}",
            status_code=400,
        )

    @staticmethod
    def _build_binding(
        vm_key: str,
        config_type: str,
        config: dict[str, Any],
    ) -> NativeBinding:
        validated = validate_native_config(config_type, config)
        try:
            algorithm = build_native_algorithm(config_type, validated)
        except InferenceMiddlewareError:
            raise
        except Exception as exc:
            raise InferenceMiddlewareError(
                f"Failed to build native {config_type!r} for VM {vm_key!r}: {exc}",
                status_code=400,
            ) from exc
        return NativeBinding(
            algorithm=algorithm,
            models=models_for_native_config(config_type, validated),
            config_type=config_type,
        )


def _config_type(middleware_config: dict[str, Any]) -> str:
    config_type = middleware_config.get("config_type")
    if not isinstance(config_type, str) or not config_type:
        raise InferenceMiddlewareError("middleware_config is missing config_type", status_code=500)
    if config_type not in NATIVE_CONFIG_TYPES:
        raise InferenceMiddlewareError(
            f"Unknown config_type {config_type!r}. Supported: {sorted(NATIVE_CONFIG_TYPES)}",
            status_code=400,
        )
    return config_type


def _vm_key(virtual_model: VirtualModel) -> str:
    return f"{virtual_model.workspace}/{virtual_model.name}"
