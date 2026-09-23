# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependencies for inference-gateway API routes."""

from aiohttp import ClientSession
from nhx.core.inference_gateway.api.middleware_registry import MiddlewareRegistry
from nhx.core.inference_gateway.api.model_cache import ModelCache
from nhx.core.inference_gateway.api.virtual_model_cache import VirtualModelCache

_HTTP_CLIENT: ClientSession | None = None
_MODEL_CACHE: ModelCache | None = None
_VIRTUAL_MODEL_CACHE: VirtualModelCache | None = None
_MIDDLEWARE_REGISTRY: MiddlewareRegistry | None = None


def set_global_http_client(http_client: ClientSession) -> ClientSession:
    global _HTTP_CLIENT
    _HTTP_CLIENT = http_client
    return _HTTP_CLIENT


def global_http_client() -> ClientSession:
    if _HTTP_CLIENT is None:
        raise RuntimeError(
            "The global http client has not been initialized. Call set_global_http_client() during application startup."
        )

    return _HTTP_CLIENT


def set_global_model_cache(model_cache: ModelCache) -> ModelCache:
    global _MODEL_CACHE
    _MODEL_CACHE = model_cache
    return _MODEL_CACHE


def global_model_cache() -> ModelCache:
    if _MODEL_CACHE is None:
        raise RuntimeError(
            "The global model cache has not been initialized. Call set_global_model_cache() during application startup."
        )

    return _MODEL_CACHE


def reset_global_model_cache() -> None:
    """Reset the global model cache.

    Used by tests to ensure clean state between test functions.
    """
    global _MODEL_CACHE
    _MODEL_CACHE = None


def set_global_virtual_model_cache(cache: VirtualModelCache) -> VirtualModelCache:
    global _VIRTUAL_MODEL_CACHE
    _VIRTUAL_MODEL_CACHE = cache
    return _VIRTUAL_MODEL_CACHE


def global_virtual_model_cache() -> VirtualModelCache:
    if _VIRTUAL_MODEL_CACHE is None:
        raise RuntimeError(
            "The global virtual model cache has not been initialized. "
            "Call set_global_virtual_model_cache() during application startup."
        )
    return _VIRTUAL_MODEL_CACHE


def reset_global_virtual_model_cache() -> None:
    """Reset the global virtual model cache.

    Used by tests to ensure clean state between test functions.
    """
    global _VIRTUAL_MODEL_CACHE
    _VIRTUAL_MODEL_CACHE = None


def set_global_middleware_registry(registry: MiddlewareRegistry) -> MiddlewareRegistry:
    global _MIDDLEWARE_REGISTRY
    _MIDDLEWARE_REGISTRY = registry
    return _MIDDLEWARE_REGISTRY


def global_middleware_registry() -> MiddlewareRegistry:
    if _MIDDLEWARE_REGISTRY is None:
        raise RuntimeError(
            "The global middleware registry has not been initialized. "
            "Call set_global_middleware_registry() during application startup."
        )
    return _MIDDLEWARE_REGISTRY


def reset_global_middleware_registry() -> None:
    """Reset the global middleware registry.

    Used by tests to ensure clean state between test functions.
    """
    global _MIDDLEWARE_REGISTRY
    _MIDDLEWARE_REGISTRY = None
