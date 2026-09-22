# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example inference middleware — keyword content filter.

Demonstrates the complete :class:`~nemo_helix_plugin.inference_middleware.NemoInferenceMiddleware`
interface, including both config patterns:

**Inline config** (``MiddlewareCall.config``)
    Embed the config directly in the VirtualModel definition.  No extra API or
    entity store needed.  Good for simple, per-VirtualModel configs.

**Config-by-reference** (``MiddlewareCall.config_id``)
    Store the config as a :class:`~nemo_example_plugin.middleware_config.ExampleMiddlewareConfig`
    entity (via the CRUD API in :mod:`nemo_example_plugin.middleware_service`).
    Reference it with ``config_id: "workspace/my-filter"`` in the VirtualModel.
    Updates to the stored config propagate automatically — IGW re-resolves on
    every polling cycle without a VirtualModel edit.

Registration in ``pyproject.toml``::

    [project.entry-points."nemo.inference_middleware"]
    "nemo-example-middleware" = "nemo_example_plugin.middleware:ExampleInferenceMiddleware"

Example VirtualModel (inline config)::

    POST /v2/workspaces/default/virtual-models
    {
      "name": "safe-llama",
      "default_model_entity": "default/llama-3b",
      "request_middleware": [
        {
          "name": "nemo-example-middleware",
          "config_type": "example_middleware_config",
          "config": {
            "blocked_keywords": ["violence", "hate"],
            "block_message": "That topic is off-limits."
          }
        }
      ],
      "response_middleware": [
        {
          "name": "nemo-example-middleware",
          "config_type": "example_middleware_config",
          "config": {
            "blocked_keywords": ["violence", "hate"],
            "block_message": "Response contained restricted content."
          }
        }
      ]
    }

Example VirtualModel (config by reference)::

    # First create the config entity via the plugin's CRUD API:
    POST /apis/example/v2/workspaces/default/middleware-configs
    {
      "name": "global-safety-filter",
      "blocked_keywords": ["violence", "hate"],
      "block_message": "That topic is off-limits."
    }

    # Then reference it by ID in the VirtualModel:
    POST /v2/workspaces/default/virtual-models
    {
      "name": "safe-llama",
      "default_model_entity": "default/llama-3b",
      "request_middleware": [
        {
          "name": "nemo-example-middleware",
          "config_type": "example_middleware_config",
          "config_id": "default/global-safety-filter"
        }
      ]
    }
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from typing import Any, cast

from nemo_example_plugin.middleware_config import ExampleMiddlewareConfig
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from nemo_helix_plugin.inference_middleware import (
    ImmediateResponse,
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceRequest,
    InferenceResponse,
    NemoInferenceMiddleware,
    ResponseResult,
    VirtualModel,
)
from pydantic import BaseModel


class ExampleMiddlewareConfigData(BaseModel):
    """Validated config passed to :meth:`process_request` and :meth:`process_response`.

    This is the *working* form of the config — always a simple Pydantic model
    regardless of whether the source was an inline dict or a stored entity.

    :class:`~nemo_example_plugin.middleware_config.ExampleMiddlewareConfig`
    (a :class:`~nemo_helix_plugin.entity.NemoEntity`) is only used for entity store
    persistence.  :meth:`~ExampleInferenceMiddleware.validate_middleware_config`
    always converts to this type before returning.
    """

    blocked_keywords: list[str] = []
    block_message: str = "Your request contains content that is not permitted."


logger = logging.getLogger(__name__)


class ExampleInferenceMiddleware(NemoInferenceMiddleware):
    """Content-filter middleware that blocks and redacts prohibited keywords.

    **Request phase** (:meth:`process_request`): if any ``blocked_keyword``
    appears in the user's message content, return an
    :class:`~nemo_helix_plugin.inference_middleware.ImmediateResponse` with a refusal
    message — the backend is never called.

    **Response phase** (:meth:`process_response`): replace any ``blocked_keyword``
    found in the assistant's reply with ``"[REDACTED]"``.

    Register as ``"nemo-example-middleware"`` under the
    ``nemo.inference_middleware`` entry-point group.
    """

    def __init__(self) -> None:
        super().__init__()
        self._entity_client: NemoEntitiesClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_startup(self) -> None:
        """Initialise resources and validate platform state at startup.

        Constructs an entity client from the platform SDK so that
        :meth:`get_middleware_config` can fetch stored config entities.
        Logs a warning if no model entities are visible — this does not
        prevent the plugin from loading.
        """
        from nemo_helix_plugin.sdk_provider import get_async_platform_sdk

        sdk = get_async_platform_sdk(as_service="nemo-example-middleware", internal=True)
        self._entity_client = NemoEntitiesClient(client_from_platform(sdk, AsyncEntitiesClient))

        entities = self.list_model_entities_for_workspace()
        if not entities:
            logger.warning(
                "%s: no model entities found in cache at startup — "
                "inference routing will fail until models are registered.",
                type(self).__name__,
            )
        else:
            logger.info(
                "%s: ready. %d model entity/-ies visible.",
                type(self).__name__,
                len(entities),
            )

    async def on_virtual_model_upserted(self, virtual_model: VirtualModel) -> None:
        """Log when a VirtualModel referencing this plugin is created or updated."""
        logger.debug(
            "%s: VirtualModel '%s/%s' upserted.",
            type(self).__name__,
            virtual_model.workspace,
            virtual_model.name,
        )

    async def on_virtual_model_destroyed(self, virtual_model: VirtualModel) -> None:
        """Log when a VirtualModel referencing this plugin is removed."""
        logger.debug(
            "%s: VirtualModel '%s/%s' destroyed.",
            type(self).__name__,
            virtual_model.workspace,
            virtual_model.name,
        )

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    async def get_middleware_config(self, config_type: str, config_id: str) -> ExampleMiddlewareConfig:
        """Fetch a stored :class:`~nemo_example_plugin.middleware_config.ExampleMiddlewareConfig`.

        IGW calls this when a VirtualModel ``MiddlewareCall`` uses ``config_id``
        instead of inline ``config``.  The call happens at VirtualModel create/
        update time and on every polling cycle — never per-request.

        This implementation uses the plugin's own entity client to fetch from the
        entity store.  The entity must have been created via the CRUD API in
        :mod:`nemo_example_plugin.middleware_service`.

        Args:
            config_type: Must be ``"example_middleware_config"``.
            config_id: ``"workspace/name"`` of the stored config entity.

        Returns:
            The resolved :class:`~nemo_example_plugin.middleware_config.ExampleMiddlewareConfig`.

        Raises:
            ValueError: If ``config_type`` is not ``"example_middleware_config"``.
            InferenceMiddlewareError: If the entity is not found or the fetch fails.
        """
        if config_type != ExampleMiddlewareConfig.__entity_type__:
            raise InferenceMiddlewareError(
                f"ExampleInferenceMiddleware does not support config_type={config_type!r}. "
                f"Expected 'example_middleware_config'.",
                status_code=400,
            )

        ws, name = config_id.split("/", 1)

        # Import the entity client lazily so the module can be imported
        # without a running platform (e.g. in unit tests that mock this method).
        if self._entity_client is None:
            raise InferenceMiddlewareError(
                "Entity client not initialised — was on_startup() called?",
                status_code=503,
            )
        try:
            config = await self._entity_client.get(ExampleMiddlewareConfig, name=name, workspace=ws)
        except NemoEntityNotFoundError as exc:
            raise InferenceMiddlewareError(
                f"ExampleMiddlewareConfig '{config_id}' not found.",
                status_code=404,
            ) from exc
        except Exception as exc:
            raise InferenceMiddlewareError(
                f"Could not fetch ExampleMiddlewareConfig '{config_id}': {exc}",
                status_code=503,
            ) from exc

        return config

    async def validate_middleware_config(self, config_type: str, config: Any) -> ExampleMiddlewareConfigData:
        """Validate and normalise *config* into an :class:`ExampleMiddlewareConfigData`.

        Always returns a plain :class:`ExampleMiddlewareConfigData` regardless
        of whether the source was an inline ``config`` dict or a stored
        :class:`~nemo_example_plugin.middleware_config.ExampleMiddlewareConfig`
        entity (returned by :meth:`get_middleware_config`).  This separation
        keeps the entity type (used for persistence) distinct from the working
        config type (used at request time).

        Args:
            config_type: Must be ``"example_middleware_config"``.
            config: Either a plain ``dict`` (inline) or an
                :class:`~nemo_example_plugin.middleware_config.ExampleMiddlewareConfig`
                entity instance.

        Returns:
            A validated :class:`ExampleMiddlewareConfigData`.

        Raises:
            ValueError: If ``config_type`` is unknown or the config is malformed.
        """
        if config_type != ExampleMiddlewareConfig.__entity_type__:
            raise InferenceMiddlewareError(
                f"Unknown config_type={config_type!r}. "
                f"ExampleInferenceMiddleware only handles 'example_middleware_config'.",
                status_code=400,
            )

        if isinstance(config, ExampleMiddlewareConfig):
            # Fetched from entity store — extract the domain fields.
            return ExampleMiddlewareConfigData(
                blocked_keywords=config.blocked_keywords,
                block_message=config.block_message,
            )

        # Inline config arrives as a plain dict — validate against the data model.
        return ExampleMiddlewareConfigData.model_validate(config)

    # ------------------------------------------------------------------
    # Request processing
    # ------------------------------------------------------------------

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: ExampleMiddlewareConfigData,
    ) -> InferenceRequest | ImmediateResponse:
        """Block requests whose message content contains a prohibited keyword.

        Extracts text from every ``messages[*].content`` field (Chat Completions
        format).  Returns an :class:`~nemo_helix_plugin.inference_middleware.ImmediateResponse`
        with a refusal payload if any keyword matches — the backend is never
        called.  Otherwise returns the request unchanged.

        Returning :class:`~nemo_helix_plugin.inference_middleware.ImmediateResponse`
        is the idiomatic way to implement a *blocker* — prefer it over raising
        :class:`~nemo_helix_plugin.inference_middleware.InferenceMiddlewareError` when
        you want the caller to receive a well-formed (non-error) refusal rather
        than an HTTP error status.
        """
        if not middleware_config.blocked_keywords:
            return request

        user_text = _extract_message_text(request.body)
        matched = _find_keyword(user_text, middleware_config.blocked_keywords)

        if matched:
            logger.info(
                "ExampleInferenceMiddleware: blocking request — matched keyword %r",
                matched,
            )
            return ImmediateResponse(
                data={
                    "id": "blocked",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": middleware_config.block_message,
                            },
                            "finish_reason": "content_filter",
                        }
                    ],
                }
            )

        return request

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: ExampleMiddlewareConfigData,
    ) -> InferenceResponse:
        """Redact prohibited keywords from the assistant's reply.

        Handles both non-streaming (``dict``) and streaming
        (``AsyncIterator[dict]``) responses.

        **Streaming** responses are wrapped in :func:`_redact_stream`, which
        uses a sliding-window buffer to catch keywords that are split across
        chunk boundaries (e.g. ``"drug"`` in one chunk, ``"s"`` in the next).
        Each buffered character is held back until enough subsequent content
        has arrived to rule out a keyword match starting at that position,
        then flushed.
        """
        if not middleware_config.blocked_keywords:
            return response

        if isinstance(response.result, dict):
            redacted_result = _redact_keywords(response.result, middleware_config.blocked_keywords)
            if redacted_result is not response.result:
                logger.debug("ExampleInferenceMiddleware: redacted keywords from non-streaming response.")
        else:
            logger.debug("ExampleInferenceMiddleware: wrapping stream with keyword redactor.")
            redacted_result = _redact_stream(response.result, middleware_config.blocked_keywords)

        return InferenceResponse(result=redacted_result, headers=response.headers)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_message_text(body: dict[str, Any]) -> str:
    """Concatenate all message content strings from a Chat Completions body."""
    parts: list[str] = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Multi-modal content: [{type: "text", text: "..."}, ...]
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
    return " ".join(parts)


def _find_keyword(text: str, keywords: list[str]) -> str | None:
    """Return the first keyword found (case-insensitive) in *text*, or ``None``."""
    lower = text.lower()
    for kw in keywords:
        if kw.lower() in lower:
            return kw
    return None


async def _redact_stream(
    stream: AsyncIterator[dict[str, Any]],
    keywords: list[str],
) -> AsyncIterator[dict[str, Any]]:
    """Wrap *stream* and redact *keywords* from each chunk's ``delta.content``.

    **Cross-chunk keyword detection** — a keyword like ``"drugs"`` can arrive
    split across two consecutive chunks (e.g. ``"drug"`` then ``"s"``).  A
    naive per-chunk regex would miss the match.  This generator maintains a
    *lookahead buffer*: it holds back up to ``max_kw_len - 1`` characters from
    the previous chunk's tail and prepends them to the next chunk's delta before
    running the regex.  Once the combined string has been scanned, the portion
    that cannot be the start of any remaining keyword is yielded immediately;
    the rest stays in the buffer until the next chunk arrives.

    When the stream ends the buffer is flushed using the **last real chunk's
    shape** so the synthesised chunk is structurally identical to every other
    chunk in the stream — same ``id``, ``model``, ``created``, and ``usage``
    fields — rather than a stripped-down object that would confuse strict
    parsers.

    Only the ``choices[*].delta.content`` field is modified; all other chunk
    fields are passed through unchanged.
    """
    max_kw_len = max((len(k) for k in keywords), default=0)
    # Buffer holds tail characters that might be the start of a split keyword.
    buf: str = ""
    last_chunk: dict[str, Any] | None = None  # used as the template for buffer flush

    async for chunk in stream:
        delta_content = _get_delta_content(chunk)
        if not delta_content:
            # Preserve content ordering: flush buffered text before any
            # non-content chunk (role announcement, finish_reason, etc.).
            if buf:
                template = last_chunk or chunk
                yield _with_delta_content(template, buf)
                buf = ""
            yield chunk
            continue

        last_chunk = chunk

        # Prepend any buffered tail from the previous chunk.
        combined = buf + delta_content

        # Redact matches across the full combined string.
        redacted = combined
        for kw in keywords:
            redacted = re.sub(re.escape(kw), "[REDACTED]", redacted, flags=re.IGNORECASE)

        # Hold back the last (max_kw_len - 1) characters — they could still be
        # the start of a keyword continued in the next chunk.
        safe_len = max(0, len(redacted) - (max_kw_len - 1))
        to_yield = redacted[:safe_len]
        buf = redacted[safe_len:]

        if to_yield:
            yield _with_delta_content(chunk, to_yield)

    # Flush the remaining buffer.  Re-use the last real chunk's shape so the
    # flushed chunk looks like a normal delta chunk to downstream consumers.
    if buf:
        template = last_chunk or {}
        yield _with_delta_content(template, buf)


def _get_delta_content(chunk: dict[str, Any]) -> str:
    """Extract ``choices[0].delta.content`` from a streaming chunk, or ``""``."""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


def _with_delta_content(chunk: dict[str, Any], content: str) -> dict[str, Any]:
    """Return a shallow copy of *chunk* with ``choices[0].delta.content`` replaced."""
    choices = chunk.get("choices") or []
    if not choices:
        return chunk
    first = choices[0]
    delta = first.get("delta") or {}
    new_delta = {**delta, "content": content}
    new_first = {**first, "delta": new_delta}
    return {**chunk, "choices": [new_first, *choices[1:]]}


def _redact_keywords(response: ResponseResult, keywords: list[str]) -> ResponseResult:
    """Return a copy of *response* with each keyword replaced by ``[REDACTED]``
    in ``choices[*].message.content``.

    Targets the same field as the streaming path (``choices[*].delta.content``),
    keeping both paths at parity.  Non-dict (streaming) responses are returned
    unchanged.
    """
    if not isinstance(response, dict):
        return response

    response_dict = cast(dict[str, Any], response)
    choices = response_dict.get("choices") or []
    if not choices:
        return response_dict

    new_choices: list[Any] = []
    changed = False
    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            new_choices.append(choice)
            continue
        redacted = content
        for kw in keywords:
            redacted = re.sub(re.escape(kw), "[REDACTED]", redacted, flags=re.IGNORECASE)
        if redacted != content:
            changed = True
            new_choices.append({**choice, "message": {**message, "content": redacted}})
        else:
            new_choices.append(choice)

    if not changed:
        return response_dict

    return {**response_dict, "choices": new_choices}
