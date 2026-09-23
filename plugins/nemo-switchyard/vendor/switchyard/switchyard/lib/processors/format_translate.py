# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router-agnostic format-translation processors for the IGW path.

When Switchyard owns the LLM call (standalone server), inbound→target
format translation happens *inside* the LLM backend (e.g.
:class:`OpenAILLMBackend.call` runs
:meth:`ChatRequestTranslationEngine.to_openai_chat` at the top), and
outbound translation happens in :class:`DefaultResponseTranslator`.
The host's chain has dedicated slots for both.

When the host (NeMo Helix IGW) owns the LLM call, neither slot is available
— IGW supplies its own backend and there's no
:class:`ResponseTranslator` slot. Format translation has to live in
processors instead. The three processors here are the IGW equivalent
of the standalone path, and reusable by *any* router (random routing
today, RouteLLM later) that needs cross-format dispatch:

* :class:`StampOriginalFormatProcessor` — runs at the head of the
  chain and captures the inbound :class:`ChatRequestType` into
  ``ctx.metadata[CTX_ORIGINAL_FORMAT]`` so the response processor can
  translate back later.
* :class:`FormatTranslateRequestProcessor` — runs after a router
  picks a tier and reads ``ctx.metadata[CTX_TARGET_FORMAT]``.
  No-ops when the target is unset or matches
  ``request.request_type``; otherwise delegates to
  :meth:`ChatRequestTranslationEngine.to_any_of`.
* :class:`FormatTranslateResponseProcessor` — runs after the host
  backend returns. Reads ``ctx.metadata[CTX_ORIGINAL_FORMAT]`` and
  converts the response back to the client's inbound format via
  :class:`ChatResponseTranslationEngine`.

Composability: any router that wants cross-format support stamps a
:class:`ChatRequestType` into ``CTX_TARGET_FORMAT``; these processors
handle the actual conversion. Router and converter are fully
decoupled — neither imports the other.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, cast

from switchyard.lib.backends.backend_tier import BackendFormat
from switchyard.lib.chat_request import ChatRequest, ChatRequestType
from switchyard.lib.chat_response.anthropic import (
    AnthropicChatResponse,
    AnthropicResponseStream,
    AnthropicStreamingChatResponse,
)
from switchyard.lib.chat_response.openai_chat import (
    CompletionChatResponse,
    ResponseStream,
    StreamingChatResponse,
)
from switchyard.lib.chat_response.openai_responses import (
    ResponsesApiChatResponse,
    ResponsesApiStream,
    ResponsesApiStreamingChatResponse,
)
from switchyard.lib.proxy_context import (
    CTX_ORIGINAL_FORMAT,
    CTX_ORIGINAL_REQUEST,
    CTX_PROXY_ACTUAL_MODEL,
    CTX_TARGET_FORMAT,
)
from switchyard.lib.roles import RequestProcessor, ResponseProcessor
from switchyard.lib.translation.request_engine import (
    ChatRequestTranslationEngine,
)
from switchyard.lib.translation.response_engine import (
    ChatResponseTranslationEngine,
)

if TYPE_CHECKING:
    from anthropic.types import RawMessageStreamEvent
    from anthropic.types.message_create_params import MessageCreateParamsBase
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.completion_create_params import CompletionCreateParamsBase
    from openai.types.responses import ResponseStreamEvent
    from openai.types.responses.response_create_params import ResponseCreateParamsBase

    from switchyard.lib.chat_response.base import ChatResponse
    from switchyard.lib.factories.translate.factory import TranslateConfig
    from switchyard.lib.proxy_context import ProxyContext

log = logging.getLogger(__name__)


class ModelFormatLookupProcessor(RequestProcessor):
    """Look up the selected model's target format from the translate config.

    After a router picks a tier and stamps its model name into
    ``ctx.metadata[CTX_PROXY_ACTUAL_MODEL]``, this processor looks up
    the model in the translate config's models list and stamps the
    corresponding backend format into ``ctx.metadata[CTX_TARGET_FORMAT]``.

    ``BackendFormat.AUTO`` is resolved locally because the IGW path has
    no Switchyard-owned backend slot where backend construction can probe
    endpoint capabilities. The resolver keeps Anthropic-native inbound
    traffic native, and otherwise normalizes to OpenAI Chat as the broad
    fallback wire format.

    No-op when the model is not found in the config (passthrough).
    """

    def __init__(self, config: TranslateConfig) -> None:
        self._config = config
        # Build a lookup dict: model name → backend_format
        self._model_to_format: dict[str, BackendFormat] = {}
        for entry in config.models:
            model_name = entry.get("model")
            format_str = entry.get("backend_format")
            if model_name and format_str:
                try:
                    self._model_to_format[model_name] = BackendFormat(format_str)
                except ValueError:
                    log.warning(
                        "Unknown backend_format in translate config: %s", format_str,
                    )

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        selected_model = ctx.metadata.get(CTX_PROXY_ACTUAL_MODEL)
        if selected_model is None:
            return request
        target_format = self._model_to_format.get(selected_model)
        if target_format is not None:
            target_request_type = _target_request_type_for_backend_format(
                target_format, request,
            )
            ctx.metadata[CTX_TARGET_FORMAT] = target_request_type
            log.info(
                "ModelFormatLookupProcessor: model=%s backend_format=%s "
                "target_request_type=%s",
                selected_model,
                target_format.value,
                target_request_type.value,
            )
        return request


class StampOriginalFormatProcessor(RequestProcessor):
    """Capture the inbound request type into ``ctx.metadata``.

    Place at the head of any IGW chain that may rewrite the request
    format before the host backend sees it. The matching response
    processor reads ``CTX_ORIGINAL_FORMAT`` to know what format to
    translate the response back to.
    """

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        ctx.metadata[CTX_ORIGINAL_FORMAT] = request.request_type
        if CTX_ORIGINAL_REQUEST not in ctx.metadata:
            body = getattr(request, "body", None)
            if isinstance(body, Mapping):
                ctx.metadata[CTX_ORIGINAL_REQUEST] = deepcopy(dict(body))
        return request


class FormatTranslateRequestProcessor(RequestProcessor):
    """Translate the request to the target wire format chosen by a router.

    Reads ``ctx.metadata[CTX_TARGET_FORMAT]`` (a
    :class:`ChatRequestType`) and routes through
    :meth:`ChatRequestTranslationEngine.to_any_of` when the target
    differs from ``request.request_type``. Pure passthrough when:

    * No router stamped a target (``CTX_TARGET_FORMAT`` absent).
    * Target matches the inbound type (already in the right format).

    Router-agnostic — no notion of *why* a target was chosen, only
    what to do with it. Reusable by every routing factory that needs
    cross-format dispatch.
    """

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        target = ctx.metadata.get(CTX_TARGET_FORMAT)
        if target is None or target == request.request_type:
            return request
        if not isinstance(target, ChatRequestType):
            raise TypeError(
                f"ctx.metadata[{CTX_TARGET_FORMAT!r}] must be a ChatRequestType, "
                f"got {type(target).__name__}",
            )
        return ChatRequestTranslationEngine.to_any_of(request, [target])


class FormatTranslateResponseProcessor(ResponseProcessor):
    """Translate the response back to the inbound format the client expects.

    Reads ``ctx.metadata[CTX_ORIGINAL_FORMAT]`` (stamped by
    :class:`StampOriginalFormatProcessor` at the head of the chain)
    and converts the response via
    :class:`ChatResponseTranslationEngine`. No-ops when the response
    is already in the original format or when ``CTX_ORIGINAL_FORMAT``
    is absent.

    Streaming responses use the original request body captured in
    context to preserve target-format stream metadata such as model
    names while translating lazily.
    """

    async def process(self, ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        original = ctx.metadata.get(CTX_ORIGINAL_FORMAT)
        if original is None:
            return response
        if not isinstance(original, ChatRequestType):
            raise TypeError(
                f"ctx.metadata[{CTX_ORIGINAL_FORMAT!r}] must be a ChatRequestType, "
                f"got {type(original).__name__}",
            )
        if isinstance(response, (
            StreamingChatResponse,
            AnthropicStreamingChatResponse,
            ResponsesApiStreamingChatResponse,
        )):
            return _translate_streaming_response(ctx, response, original)
        if _matches_format(response, original):
            return response
        if original is ChatRequestType.OPENAI_CHAT:
            return ChatResponseTranslationEngine.to_openai_chat(response)
        if original is ChatRequestType.ANTHROPIC:
            return ChatResponseTranslationEngine.to_anthropic(response)
        if original is ChatRequestType.OPENAI_RESPONSES:
            return ChatResponseTranslationEngine.to_responses(
                response, original_body=_original_body(ctx),
            )
        raise NotImplementedError(
            f"Unsupported original format for response translation: {original!r}",
        )


def _translate_streaming_response(
    ctx: ProxyContext,
    response: ChatResponse,
    original: ChatRequestType,
) -> ChatResponse:
    if _matches_streaming_format(response, original):
        return response

    translated = ChatResponseTranslationEngine.translate_stream(
        _request_for_original_format(original, _original_body(ctx)),
        response,
    )
    if original is ChatRequestType.OPENAI_CHAT:
        return StreamingChatResponse(
            ResponseStream(cast("AsyncIterator[ChatCompletionChunk]", translated)),
        )
    if original is ChatRequestType.ANTHROPIC:
        return AnthropicStreamingChatResponse(
            AnthropicResponseStream(
                cast("AsyncIterator[RawMessageStreamEvent]", translated),
            ),
        )
    if original is ChatRequestType.OPENAI_RESPONSES:
        return ResponsesApiStreamingChatResponse(
            ResponsesApiStream(cast("AsyncIterator[ResponseStreamEvent]", translated)),
        )
    raise NotImplementedError(
        f"Unsupported original format for streaming response translation: {original!r}",
    )


def _request_for_original_format(
    original: ChatRequestType,
    body: dict[str, object],
) -> ChatRequest:
    from switchyard.lib.chat_request.anthropic import AnthropicChatRequest
    from switchyard.lib.chat_request.openai_chat import OpenAIChatRequest
    from switchyard.lib.chat_request.openai_responses import ResponsesChatRequest

    if original is ChatRequestType.OPENAI_CHAT:
        return OpenAIChatRequest(cast("CompletionCreateParamsBase", body))
    if original is ChatRequestType.ANTHROPIC:
        return AnthropicChatRequest(cast("MessageCreateParamsBase", body))
    if original is ChatRequestType.OPENAI_RESPONSES:
        return ResponsesChatRequest(cast("ResponseCreateParamsBase", body))
    raise NotImplementedError(f"Unsupported original format: {original!r}")


def _original_body(ctx: ProxyContext) -> dict[str, object]:
    body = ctx.metadata.get(CTX_ORIGINAL_REQUEST)
    if isinstance(body, Mapping):
        return dict(body)
    return {}


def _matches_format(response: ChatResponse, target: ChatRequestType) -> bool:
    """Return ``True`` when *response* is already in *target*'s wire format."""
    if target is ChatRequestType.OPENAI_CHAT:
        return isinstance(response, CompletionChatResponse)
    if target is ChatRequestType.ANTHROPIC:
        return isinstance(response, AnthropicChatResponse)
    if target is ChatRequestType.OPENAI_RESPONSES:
        return isinstance(response, ResponsesApiChatResponse)
    raise AssertionError(f"Exhausted all ChatRequestType variants: {target}")


def _matches_streaming_format(response: ChatResponse, target: ChatRequestType) -> bool:
    if target is ChatRequestType.OPENAI_CHAT:
        return isinstance(response, StreamingChatResponse)
    if target is ChatRequestType.ANTHROPIC:
        return isinstance(response, AnthropicStreamingChatResponse)
    if target is ChatRequestType.OPENAI_RESPONSES:
        return isinstance(response, ResponsesApiStreamingChatResponse)
    raise AssertionError(f"Exhausted all ChatRequestType variants: {target}")


def _target_request_type_for_backend_format(
    backend_format: BackendFormat,
    request: ChatRequest,
) -> ChatRequestType:
    """Resolve a configured backend wire format to a request format.

    ``BackendFormat.AUTO`` on the processor-only IGW path cannot do the
    backend-owned capability probe used by backend factories. It instead
    chooses the least surprising concrete request shape:

    * Anthropic inbound remains Anthropic, preserving native Claude
      fields instead of translating them away.
    * OpenAI Chat inbound remains OpenAI Chat.
    * Responses inbound normalizes to OpenAI Chat because there is no
      Responses-native backend format in :class:`BackendFormat`.
    """
    if backend_format is BackendFormat.OPENAI:
        return ChatRequestType.OPENAI_CHAT
    if backend_format is BackendFormat.ANTHROPIC:
        return ChatRequestType.ANTHROPIC
    if backend_format is BackendFormat.AUTO:
        if request.request_type is ChatRequestType.ANTHROPIC:
            return ChatRequestType.ANTHROPIC
        return ChatRequestType.OPENAI_CHAT
    raise ValueError(f"Unsupported backend_format: {backend_format!r}")
