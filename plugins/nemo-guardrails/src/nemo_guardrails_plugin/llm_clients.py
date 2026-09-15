# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This module provides helpers to store and inject request-scoped headers
into cached ``LLMRails`` instances and their LangChain clients.

Each IGW request leases an ``LLMRails`` instance from the cache, runs the
configured rails, then returns the instance to the pool. The leased instance
already owns LangChain clients used for calls to the guardrail models.

Since the LangChain client for an ``LLMRails`` instance is reused by later requests,
request-specific service-principal and tracing headers cannot be stored on the client
when constructing the ``LLMRails`` instance: the next request may need different
headers while using the same client object. This module stores the current request's
headers in a ``ContextVar`` and merges them into each outbound ``ChatNVIDIA`` call
to a guardrail model.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client_provider import get_forwarding_headers
from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter
from nemoguardrails.llm.providers import register_provider
from nemoguardrails.types import LLMModel

NIM_PROVIDER_NAME = "nim"
"""NeMo Guardrails provider name for NVIDIA NIM chat models."""

_HEADER_AWARE_INITIALIZER_INSTALLED = False
"""Process-local guard so repeated middleware startups do not re-register the
NeMo Guardrails NIM provider."""

RequestHeaders = Mapping[str, str]
_request_headers_ctx: ContextVar[RequestHeaders] = ContextVar(
    "guardrails_request_headers", default=MappingProxyType({})
)


def get_request_headers() -> RequestHeaders:
    """Return the platform headers in scope for the current rail execution."""
    return _request_headers_ctx.get()


@contextmanager
def platform_headers_context(client: AsyncNemoClient) -> Iterator[None]:
    """Make platform headers visible to rail model calls in this context.

    Model calls happen deep inside nemoguardrails/LangChain library code that
    does not receive IGW's middleware context object. ``ContextVar`` gives those
    calls access to request-scoped headers without adding mutable request data
    on the cached LangChain client itself.

    Args:
        client: The client whose forwarding headers should be propagated.
            For per-request auth, pass a request-scoped client built via
            ``client.with_options(headers=...)``.

    The data flow is:

    1. ``get_forwarding_headers(client)`` extracts the per-request
       service-principal, on-behalf-of, and tracing headers.
    2. This context manager stores them in ``_request_headers_ctx`` for the
       current execution context.
    3. Non-streaming rails run via ``asyncio.to_thread``; Python copies the
       caller task's context into the worker thread, and ``run_until_complete``
       schedules ``LLMRails.generate_async`` as a task with that copied context.
    4. Streaming rails enter this context inside the returned async generator
       before iterating ``LLMRails.stream_async``, so the active task context
       stays in scope while chunks are produced.
    5. When LangChain prepares an outbound model request, our
       ``_prepare_inputs_and_payload`` override reads ``_request_headers_ctx``
       and merges those headers into the request.
    """
    headers = _request_headers_ctx.set(get_forwarding_headers(client))
    try:
        yield
    finally:
        _request_headers_ctx.reset(headers)


def _load_chat_nvidia_class():
    """Import ``ChatNVIDIA`` only when a NIM rail model is actually built."""
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA


def build_header_aware_chat_nvidia(*, model_name: str, kwargs: dict[str, Any]) -> BaseChatModel:
    """Build a ``ChatNVIDIA`` client that merges request-scoped headers per call.

    ``ChatNVIDIA.default_headers`` is a Pydantic field that should continue to
    contain only static config headers. Request-scoped headers are merged into
    the per-call ``extra_headers`` returned by ``_prepare_inputs_and_payload``
    so cached clients can serve sequential requests with different headers.
    """
    chat_nvidia_cls = _load_chat_nvidia_class()

    # Patch the internal ``_prepare_inputs_and_payload`` method to merge request-scoped headers.
    # The return value is a tuple of (inputs, payload, extra_headers) used by the internal
    # HTTP client.
    def _prepare_inputs_and_payload(self: Any, *args: Any, **prepare_kwargs: Any):
        inputs, payload, extra_headers = chat_nvidia_cls._prepare_inputs_and_payload(self, *args, **prepare_kwargs)
        return inputs, payload, {**(extra_headers or {}), **dict(get_request_headers())}

    # The subclass is created dynamically because langchain-nvidia-ai-endpoints
    # is optional for tests and for deployments that never configure NIM rails.
    header_aware_cls = type(
        "HeaderAwareChatNVIDIA",
        (chat_nvidia_cls,),
        {"_prepare_inputs_and_payload": _prepare_inputs_and_payload},
    )
    return cast(BaseChatModel, header_aware_cls(model=model_name, **kwargs))


def _init_header_aware_nim_model(*, model: str, **kwargs: Any) -> LLMModel:
    """Adapter matching NeMo Guardrails' provider factory shape."""
    model_name = model
    return LangChainLLMAdapter(build_header_aware_chat_nvidia(model_name=model_name, kwargs=kwargs))


def register_header_aware_nim_provider() -> None:
    """Install the header-aware client for NeMo Guardrails ``engine: nim``.

    Registering the provider preserves the existing ``engine: nim`` config
    surface while ensuring cached NIM clients read request-scoped platform
    headers at call time.

    The process-local guard makes repeated middleware startup calls idempotent
    and avoids rewriting the private table once the replacement is installed.
    """
    global _HEADER_AWARE_INITIALIZER_INSTALLED
    if _HEADER_AWARE_INITIALIZER_INSTALLED:
        return

    register_provider(NIM_PROVIDER_NAME, _init_header_aware_nim_model)
    _HEADER_AWARE_INITIALIZER_INSTALLED = True
