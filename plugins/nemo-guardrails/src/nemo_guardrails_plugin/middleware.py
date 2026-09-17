# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import logging
from collections.abc import AsyncIterator
from typing import Any

from nemo_guardrails_plugin.constants import (
    GUARDRAILS_PLUGIN_CONFIG_TYPE,
    PROCESS_REQUEST_RAIL_TYPES,
    PROCESS_RESPONSE_RAIL_TYPES,
)
from nemo_guardrails_plugin.llm_clients import platform_headers_context, register_header_aware_nim_provider
from nemo_guardrails_plugin.llmrails_cache import (
    DefaultLLMRailsBuilder,
    EntityGuardrailConfigSource,
    GuardrailConfigSource,
    InlineGuardrailConfigSource,
    LLMRailsCache,
    Provenance,
    StabilizedRailsConfigCache,
    StableRailsConfig,
    extract_output_rails_streaming_config,
    provenance_of,
    source_has_input_flows,
    source_has_output_flows,
    wire_config_id,
)
from nemo_guardrails_plugin.rails import (
    build_generate_async_options,
    build_guardrails_data,
    build_main_llm,
    run_generate_in_new_loop,
)
from nemo_guardrails_plugin.requests import (
    extract_log_options_from_request,
    extract_return_choice_from_request,
    parse_guardrails_request,
    sanitize_request_body_for_proxy,
)
from nemo_guardrails_plugin.responses import (
    GUARDRAILS_DATA_FIELD,
    apply_input_rail_modifications,
    build_assistant_message_from_response_result,
    build_blocked_immediate_response_body,
    build_blocked_output_response_body,
    build_immediate_response,
    build_inference_response,
    build_output_response_body,
    extract_upstream_error,
    is_blocked_generation_response,
)
from nemo_guardrails_plugin.schemas import GuardrailsRequest
from nemo_guardrails_plugin.streaming import (
    ChatCompletionChunkMetadata,
    build_streaming_error_response,
    chunks_to_strings,
    close_async_iterator,
    strings_to_chunks,
)
from nemo_guardrails_plugin.transforms import GenerationResponseMapper
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.config import get_common_service_config
from nemo_platform_plugin.guardrail.client import AsyncGuardrailClient
from nemo_platform_plugin.guardrail.types import GenerationLogOptionsParam
from nemo_platform_plugin.guardrail.types import RailsConfig as PlatformRailsConfig
from nemo_platform_plugin.inference_middleware import (
    ImmediateResponse,
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceMiddlewareUnavailableError,
    InferenceRequest,
    InferenceResponse,
    MiddlewareCall,
    MiddlewareConfigNotFoundError,
    NemoInferenceMiddleware,
    ResponseResult,
    VirtualModel,
)
from nemo_platform_plugin.refs import parse_entity_ref
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.types import LLMModel
from typing_extensions import TypeIs

logger = logging.getLogger(__name__)

PLUGIN_NAME = "nemo-guardrails"
"""Entry-point key under ``nemo.inference_middleware``. Must match pyproject.toml."""

GUARDRAILS_LIBRARY_LOGGER_NAME = "nemoguardrails"
"""Logger namespace owned by the nemo_guardrails library."""

# Keys used inside ``ctx.state(PLUGIN_NAME)`` for cross-hook state. Both are
# written in ``process_request`` and read in ``process_response`` of the same
# request; ``ctx.state`` already namespaces by ``PLUGIN_NAME`` so the bare keys
# below are safe.
STATE_KEY_GUARDRAILS_REQUEST_BODY = "guardrails"
STATE_KEY_INPUT_GENERATION_RESPONSE = "input_generation_response"


def set_guardrails_library_log_level() -> None:
    """Configure the logging level for the nemo_guardrails library.

    By default, keeps the library at WARNING or higher to suppress noisy INFO logs.
    """
    library_logger = logging.getLogger(GUARDRAILS_LIBRARY_LOGGER_NAME)
    platform_log_level = get_common_service_config().log_level.upper()

    if platform_log_level == "INFO":
        library_logger.setLevel(logging.WARNING)
    else:
        library_logger.setLevel(platform_log_level)


def is_streaming_response_result(result: ResponseResult) -> TypeIs[AsyncIterator[dict[str, Any]]]:
    """Discriminate ``ResponseResult`` (``dict[str, Any] | AsyncIterator[dict[str, Any]]``).

    Prefer this over ``isinstance(result, AsyncIterator)`` directly:
    ``AsyncIterator`` is a runtime-checkable *structural* protocol, so a
    type checker can't statically rule out ``dict`` also satisfying it;
    that isinstance check alone leaves an imprecise ``dict & AsyncIterator``
    intersection type in one of the two narrowed branches.
    """
    return not isinstance(result, dict)


def _latest_user_message(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the latest user message as the context for output rails."""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return [message]
    return []


def handle_streaming_output_check(
    llm_rails: LLMRails,
    response_result: AsyncIterator[dict[str, Any]],
    request_body: dict[str, Any],
    messages: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """Run streaming output rails; yields OpenAI chat-completion chunks.

    Blocks emit a blocked final chunk; streaming errors emit an error chunk.
    """
    metadata = ChatCompletionChunkMetadata()
    token_stream = chunks_to_strings(response_result, metadata)
    output_rails_stream = llm_rails.stream_async(
        generator=token_stream,
        messages=list(messages),
    )

    async def _stream_output_chunks() -> AsyncIterator[dict[str, Any]]:
        try:
            async for chunk in strings_to_chunks(
                output_rails_stream,
                metadata=metadata,
                model=str(request_body.get("model", "")),
            ):
                yield chunk
        except Exception as exc:
            logger.error("Exception while streaming output rails: %s", exc, exc_info=True)
            yield build_streaming_error_response(exc)
        finally:
            await close_async_iterator(output_rails_stream)
            await close_async_iterator(response_result)

    return _stream_output_chunks()


class GuardrailsMiddleware(NemoInferenceMiddleware):
    # Class-level ``None`` defaults let the per-request checks raise a clean
    # RuntimeError if a request arrives before on_startup ran.
    _client: AsyncNemoClient | None = None
    # Cache of ``LLMRails`` instances keyed by stabilized content hash.
    _rails_cache: LLMRailsCache | None = None
    # Memoization of ``PlatformRailsConfig`` to ``StableRailsConfig``
    # transform by entity identity ``(workspace, name, updated_at)``.
    _stable_cache: StabilizedRailsConfigCache | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def on_startup(self) -> None:
        # Set log level for the nemo_guardrails library before executing our plugin code.
        set_guardrails_library_log_level()
        # Use our custom header-aware NIM provider adapter for library-initiated model calls.
        register_header_aware_nim_provider()

        self._client = self._get_platform_client("on_startup")
        self._rails_cache = LLMRailsCache(builder=DefaultLLMRailsBuilder())
        self._stable_cache = StabilizedRailsConfigCache()

    async def on_shutdown(self) -> None:
        # Detach attributes before awaiting close so a partial close can't
        # leave a half-torn-down cache visible to a later request.
        cache, self._rails_cache = self._rails_cache, None
        self._client = None
        self._stable_cache = None
        if cache is not None:
            await cache.close()

    async def on_virtual_model_upserted(self, virtual_model: VirtualModel) -> None:
        """Warm the cache for each guardrails config the VM references.

        Both entity-backed and inline configs warm — inline payloads live
        in ``MiddlewareCall.config`` and are fully resolvable at upsert
        time. Dedup is by :attr:`StableRailsConfig.content_hash` (the
        actual outer-cache key), so an inline config and an entity config
        with structurally identical rails warm exactly one pool slot.

        Fire-and-forget per content hash so KB builds don't block IGW
        polling; stale revisions age out under LRU.
        """
        cache = self._rails_cache
        stable_cache = self._stable_cache
        if cache is None or stable_cache is None:
            logger.debug(
                "Cache not initialized; skipping upsert for VM %s/%s",
                virtual_model.workspace,
                virtual_model.name,
            )
            return

        seen_hashes: set[str] = set()
        for call in self._iter_guardrail_calls(virtual_model):
            # ``_resolve_call`` is best-effort: it logs and returns ``None`` for
            # any failure mode. No outer try/except here.
            source = await self._resolve_call(call)
            if source is None:
                continue

            try:
                stable = stable_cache.get_or_compute(source, self.get_openai_compatible_inference_url_and_model)
            except Exception:
                logger.exception(
                    "Failed to stabilize guardrail source %s for VM %s/%s",
                    provenance_of(source).label,
                    virtual_model.workspace,
                    virtual_model.name,
                )
                continue

            if stable.content_hash in seen_hashes:
                continue
            seen_hashes.add(stable.content_hash)

            cache.warm(stable, provenance=provenance_of(source))

    # ------------------------------------------------------------------
    # IGW config interface (called at VM upsert / polling, not per-request)
    # ------------------------------------------------------------------

    async def get_middleware_config(self, config_type: str, config_id: str) -> GuardrailConfigSource:
        """Resolve an entity-stored guardrails config; sets the discriminator exactly once.

        Called by IGW when ``MiddlewareCall.config_id`` is set.

        Raises :class:`MiddlewareConfigNotFoundError` on a definitive 404 from
        the Guardrails service (entity deleted or never created) so IGW can
        evict any previously-resolved middleware referencing it. Any other SDK
        error propagates so IGW treats it as transient and preserves the prior
        resolved config.
        """
        self._require_supported_config_type(config_type)
        client = self._ensure_client()
        guardrail_client = AsyncGuardrailClient.from_client(client)

        ref = parse_entity_ref(config_id)
        try:
            entity = (await guardrail_client.get_guardrail_config(name=ref.name, workspace=ref.workspace)).data()
        except NotFoundError as exc:
            raise MiddlewareConfigNotFoundError(config_id) from exc
        # (workspace, name, updated_at) form the ``StabilizedRailsConfigCache`` key,
        # so an empty value would collide entries across unrelated entities. Fail fast here.
        rails_data = entity.data
        if rails_data is None:
            raise ValueError(f"GuardrailConfig {config_id!r} has no data.")
        if not entity.workspace:
            raise ValueError(f"GuardrailConfig {config_id!r} has empty workspace.")
        if not entity.name:
            raise ValueError(f"GuardrailConfig {config_id!r} has no name.")
        if not entity.updated_at:
            raise ValueError(f"GuardrailConfig {config_id!r} has empty updated_at.")

        return EntityGuardrailConfigSource(
            workspace=entity.workspace,
            name=entity.name,
            updated_at=entity.updated_at.isoformat(),
            rails=rails_data,
        )

    async def validate_middleware_config(self, config_type: str, config: Any) -> GuardrailConfigSource:
        """Validate an inline guardrails config from ``MiddlewareCall.config``.

        Accepts a ``PlatformRailsConfig``-shaped dict; idempotent on a
        :class:`GuardrailConfigSource`.
        """
        self._require_supported_config_type(config_type)
        if isinstance(config, (EntityGuardrailConfigSource, InlineGuardrailConfigSource)):
            return config
        if isinstance(config, PlatformRailsConfig):
            return InlineGuardrailConfigSource(rails=config, label=None)
        if isinstance(config, dict):
            payload = dict(config)
            # ``PlatformRailsConfig`` has no ``name`` field but is configured
            # with ``extra="allow"`` (Stainless default), so leaving ``name``
            # in the payload wouldn't fail validation — it would silently
            # land in ``model_extra`` where the inline source can't see it.
            # Pop it explicitly so envelope-style inline payloads surface
            # ``name`` as the diagnostic label (``<inline:my-test>`` in
            # logs) instead of burying it.
            raw_label = payload.pop("name", None)
            label = raw_label if isinstance(raw_label, str) else None
            if raw_label is not None and label is None:
                logger.debug(
                    "Inline guardrails config 'name' is not a string (got %s); discarding as label",
                    type(raw_label).__name__,
                )
            try:
                rails = PlatformRailsConfig.model_validate(payload)
            except Exception as exc:
                raise ValueError(f"Inline guardrails config failed validation: {exc}") from exc
            return InlineGuardrailConfigSource(rails=rails, label=label)
        raise ValueError(f"Inline guardrails config must be a dict or PlatformRailsConfig, got {type(config).__name__}")

    # ------------------------------------------------------------------
    # Request / Response hooks
    # ------------------------------------------------------------------

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: GuardrailConfigSource,
    ) -> InferenceRequest | ImmediateResponse:
        source = middleware_config
        provenance = provenance_of(source)
        # ``provenance.label`` (with ``updated_at``) is what we log;
        # ``wire_config_id(source)`` is the stable, externally-visible identifier
        # that goes onto the wire in ``guardrails_data.config_ids``.
        config_id = wire_config_id(source)

        logger.info("Processing request with guardrail %s", provenance.label)

        # Stash the user's ``guardrails`` field on ``ctx.state`` so
        # ``process_response`` can read it back without re-parsing the
        # original request body. Cleared implicitly when ``ctx`` is dropped
        # at the end of the request.
        plugin_state = ctx.state(PLUGIN_NAME)
        guardrails = parse_guardrails_request(request.body.get("guardrails", None))
        plugin_state.set(STATE_KEY_GUARDRAILS_REQUEST_BODY, guardrails)

        # Remove Guardrails-specific fields from the request body proxied to the upstream model.
        # Otherwise, the upstream model may reject the request.
        # Assign a new top-level body dict (sanitize returns a shallow copy) so we do not
        # mutate ``ctx.original_request.body``. Nested ``messages`` may still be aliased
        # until replaced below — we only replace that list, never mutate it in place.
        request.body = sanitize_request_body_for_proxy(request.body)

        if not source_has_input_flows(source):
            return request

        messages = request.body.get("messages")
        if not isinstance(messages, list):
            logger.warning("Request body is missing 'messages' key. Skipping input rails.")
            return request

        if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
            logger.debug("Current request turn is not a user message. Skipping input rails.")
            return request

        user_log_options = extract_log_options_from_request(guardrails)
        # Input rails evaluate the current request turn. Do not replay conversation
        # history through LLMRails: historical tool-call events can be interpreted as
        # pending actions and prevent the current turn from reaching the input rails.
        current_messages = messages[-1:]
        generation_response = await self._run_rails(
            ctx,
            source,
            request.body,
            request.headers,
            messages=current_messages,
            rail_types=PROCESS_REQUEST_RAIL_TYPES,
            user_log_options=user_log_options,
            error_msg="Failed to run input rails",
        )

        # Store the input GenerationResponse before returning. IGW still runs
        # response middleware even when the input is blocked, so this ensures
        # `process_response` can access the input rail's log fields before it
        # builds the final `guardrails_data` included in the response body.
        logger.debug("Storing process_request GenerationResponse for %s", provenance.label)
        plugin_state.set(STATE_KEY_INPUT_GENERATION_RESPONSE, generation_response)

        if generation_response.log is None:
            logger.warning(
                "Guardrail %s returned no activation log after input rails; cannot determine rail result",
                provenance.label,
            )
            raise InferenceMiddlewareError(
                "Unable to determine guardrail result for the request",
                status_code=500,
            )

        if is_blocked_generation_response(generation_response.log):
            return build_immediate_response(
                response_body=build_blocked_immediate_response_body(
                    config_id,
                    request.body,
                    generation_response,
                    user_log_options,
                )
            )

        # If an input rail masked/transformed the user message, write the text content
        # back onto the last user message so the upstream model sees the post-rail content.
        updated_messages = apply_input_rail_modifications(messages, generation_response)
        if updated_messages is not messages:
            request.body["messages"] = updated_messages
            logger.debug("Applied input-rail message modifications for %s", provenance.label)

        # Store the generation_response in plugin state so it can be used by the response middleware
        # to build the `guardrails_data` for the input and output rails.
        logger.debug("Storing process_request GenerationResponse for %s", provenance.label)
        plugin_state.set(STATE_KEY_INPUT_GENERATION_RESPONSE, generation_response)

        # If this request does not contain a response middleware call, we need to inject the input rails'
        # `guardrails_data` into the response body. IGW's proxy handles merging `response_body_annotations`
        # into the final response body before returning to the caller.
        guardrails_data = build_guardrails_data(
            config_id,
            input_generation_response=generation_response,
            user_log_options=user_log_options,
        )
        if guardrails_data is not None:
            logger.debug("Storing guardrails_data in response_body_annotations for %s", provenance.label)
            ctx.response_body_annotations[GUARDRAILS_DATA_FIELD] = guardrails_data.model_dump(exclude_none=True)

        return request

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: GuardrailConfigSource,
    ) -> InferenceResponse:
        # ``ctx.original_request`` is the request as IGW received it, before any
        # request middleware ran. Output rails operate against that snapshot so
        # later middleware can't hide content from us by mutating the body.
        request_body = ctx.original_request.body
        request_headers = ctx.original_request.headers
        response_result = response.result

        messages = request_body.get("messages")
        if not isinstance(messages, list):
            logger.warning("Request body is missing 'messages' key. Skipping output rails.")
            return response

        source = middleware_config
        provenance = provenance_of(source)
        config_id = wire_config_id(source)

        logger.info("Processing response with guardrail %s", provenance.label)

        plugin_state = ctx.state(PLUGIN_NAME)
        input_generation_response: GenerationResponse | None = plugin_state.get(STATE_KEY_INPUT_GENERATION_RESPONSE)
        # Get `guardrails` field from the incoming request body, stored in `ctx.state` by
        # `process_request`. If no request middleware ran, parse the field directly from the
        # request body.
        guardrails_request: GuardrailsRequest | None = (
            plugin_state.get(STATE_KEY_GUARDRAILS_REQUEST_BODY)
            if plugin_state.has(STATE_KEY_GUARDRAILS_REQUEST_BODY)
            else parse_guardrails_request(request_body.get("guardrails", None))
        )

        user_log_options = extract_log_options_from_request(guardrails_request)
        return_guardrails_data_as_choice = extract_return_choice_from_request(guardrails_request)

        if not source_has_output_flows(source):
            logger.debug("No output flows configured for %s. Skipping output rails.", provenance.label)
            if is_streaming_response_result(response_result):
                return response

            return build_inference_response(
                response=response,
                response_body=build_output_response_body(
                    config_id=config_id,
                    original_response=response_result,
                    generation_response=None,
                    input_generation_response=input_generation_response,
                    user_log_options=user_log_options,
                    return_guardrails_data_as_choice=return_guardrails_data_as_choice,
                ),
                return_guardrails_data_as_choice=return_guardrails_data_as_choice,
            )

        n = request_body.get("n", None)
        if isinstance(n, int) and n > 1:
            raise InferenceMiddlewareError(
                f"Output rails do not support multiple completion choices (n={n}). "
                "Set n=1 in the request body to use a GuardrailConfig with output rails.",
                status_code=400,
            )

        # Streaming: the lease must stay held for the life of the iterator.
        if is_streaming_response_result(response_result):
            streaming_config = extract_output_rails_streaming_config(source)
            if streaming_config is not None and streaming_config.enabled is False:
                raise InferenceMiddlewareError(
                    f"Streaming output rails are disabled for config {provenance.label}. "
                    "Set rails.output.streaming.enabled=true to use output rails with streaming responses.",
                    status_code=400,
                )

            # Lease setup is split eager vs lazy.
            #
            # Eager: cache null-check, stabilize, build_main_llm. Failures
            # here still surface as a sync response (no streaming committed
            # yet), so the status-code contract is preserved.
            #
            # Lazy: cache.lease(...) runs inside the async generator body.
            # Per PEP 525 the body doesn't run until __anext__, and aclose()
            # on a never-iterated generator is a no-op — so an eager lease
            # would leak a Pool slot any time IGW drops the returned
            # iterator without iterating.
            cache, stable, lease_provenance, main_llm, client = await self._prepare_lease_with_503(
                ctx, source, request_body, request_headers, "Failed to run streaming output rails"
            )

            # Output rails need the latest user message for ``user_input``. Do not
            # replay earlier conversation history: historical tool-call events can
            # prevent the generated response from reaching the output rails.
            # Deep copy because the library walks nested multimodal content async.
            captured_messages = copy.deepcopy(_latest_user_message(messages))
            captured_request_body = dict(request_body)

            async def _streaming_with_lease() -> AsyncIterator[dict[str, Any]]:
                try:
                    with platform_headers_context(client):
                        async with cache.lease(stable, main_llm=main_llm, provenance=lease_provenance) as llm_rails:
                            inner = handle_streaming_output_check(
                                llm_rails,
                                response_result,
                                captured_request_body,
                                captured_messages,
                            )
                            try:
                                async for chunk in inner:
                                    yield chunk
                            finally:
                                # Close inner before the lease exits so LangChain /
                                # action-LLM cleanup runs while rails is still
                                # leased — otherwise it races the next _reset.
                                # CancelledError / GeneratorExit propagates to
                                # cache.lease.__aexit__ and hits Pool.acquire's
                                # discard branch.
                                # Swallow (but log) cleanup exceptions so they
                                # don't replace a legitimate caller exception.
                                try:
                                    await close_async_iterator(inner)
                                except (asyncio.CancelledError, GeneratorExit):
                                    raise
                                except Exception:
                                    logger.exception("Failed to close streaming output rails iterator")
                except Exception as exc:
                    # A lease __aenter__ failure (KB build, builder raise)
                    # arrives after process_response has returned, so the
                    # only way to surface it is an error chunk. Cheaper
                    # setup failures are already caught eagerly above.
                    logger.exception("Streaming output rails lease failed")
                    yield build_streaming_error_response(exc)

            return InferenceResponse(
                result=_streaming_with_lease(),
                headers=response.headers,
                response_body_annotations=dict(response.response_body_annotations),
            )

        output_messages = _latest_user_message(messages)
        output_messages.append(build_assistant_message_from_response_result(response_result))
        generation_response = await self._run_rails(
            ctx,
            source,
            request_body,
            request_headers,
            messages=output_messages,
            rail_types=PROCESS_RESPONSE_RAIL_TYPES,
            user_log_options=user_log_options,
            error_msg="Failed to run output rails",
        )

        if generation_response.log is None:
            logger.warning(
                "Guardrail %s returned no activation log after output rails; cannot determine rail result",
                provenance.label,
            )
            raise InferenceMiddlewareError(
                "Unable to determine guardrail result for the request",
                status_code=500,
            )

        # Both branches share the same kwargs; only the builder differs.
        response_body_builder = (
            build_blocked_output_response_body
            if is_blocked_generation_response(generation_response.log)
            else build_output_response_body
        )
        return build_inference_response(
            response=response,
            response_body=response_body_builder(
                config_id=config_id,
                original_response=response_result,
                generation_response=generation_response,
                input_generation_response=input_generation_response,
                user_log_options=user_log_options,
                return_guardrails_data_as_choice=return_guardrails_data_as_choice,
            ),
            return_guardrails_data_as_choice=return_guardrails_data_as_choice,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_supported_config_type(config_type: str) -> None:
        """Reject any non-plugin ``config_type``.

        A mismatch indicates an IGW routing bug; we still validate per interface.
        """
        if config_type != GUARDRAILS_PLUGIN_CONFIG_TYPE:
            raise ValueError(f"Unsupported config_type {config_type!r}.")

    def _ensure_client(self) -> AsyncNemoClient:
        """Narrow ``self._client`` to non-``None``, raising a clean error otherwise.

        A request arriving before ``on_startup`` ran (or after ``on_shutdown``
        detached it) is a lifecycle bug, not a per-request failure, so we should
        surface it as a ``RuntimeError``.
        """
        client = self._client
        if client is None:
            raise RuntimeError("NeMo Platform client is not initialized. Was on_startup() called?")
        return client

    async def _resolve_call(self, call: MiddlewareCall) -> GuardrailConfigSource | None:
        """Bridge a ``MiddlewareCall`` to a ``GuardrailConfigSource`` for warming.

        Best-effort: failures return ``None`` so one bad reference can't stall
        the polling loop. ``config_id`` wins over ``config`` when both are set,
        mirroring IGW's request-time dispatch (``middleware_registry.py``
        checks ``config_id is not None`` first).
        """
        if call.config_id is not None and call.config is not None:
            logger.debug(
                "MiddlewareCall has both config_id=%r and config set; preferring config_id",
                call.config_id,
            )
        if call.config_id is not None:
            try:
                return await self.get_middleware_config(call.config_type, call.config_id)
            except (MiddlewareConfigNotFoundError, ValueError) as exc:
                logger.debug(
                    "Skipping warm for config_id %r: %s",
                    call.config_id,
                    exc,
                )
                return None
            except Exception:
                logger.exception("Unexpected error resolving config_id %r during warming", call.config_id)
                return None
        if call.config is not None:
            try:
                return await self.validate_middleware_config(call.config_type, call.config)
            except ValueError as exc:
                logger.debug("Skipping warm for inline config: %s", exc)
                return None
            except Exception:
                logger.exception("Unexpected error validating inline config during warming")
                return None
        return None

    @staticmethod
    def _iter_guardrail_calls(virtual_model: VirtualModel) -> list[MiddlewareCall]:
        # ``post_response_middleware`` is excluded: IGW reserves it for
        # fire-and-forget work that doesn't modify the response, so
        # guardrails have no business there and warming would be wasted.
        return [
            call
            for call in [*virtual_model.request_middleware, *virtual_model.response_middleware]
            if call.name == PLUGIN_NAME
        ]

    async def _run_rails(
        self,
        ctx: InferenceMiddlewareContext,
        source: GuardrailConfigSource,
        request_body: dict[str, Any],
        request_headers: dict[str, str],
        *,
        messages: list[dict[str, Any]],
        rail_types: list[str],
        user_log_options: GenerationLogOptionsParam | None,
        error_msg: str,
    ) -> GenerationResponse:
        """Lease a cached :class:`LLMRails` and run ``rail_types`` against ``messages``.

        Pipeline: stabilize → :func:`build_main_llm` → lease →
        ``generate_async``. ``build_main_llm`` and ``generate_async`` are
        offloaded to worker threads — blocking LangChain imports for the
        former, CPU-heavy Colang walk for the latter.

        Error mapping is shared with the streaming output-rails path:

        - Caller-shape errors during eager setup → 400 via
          :meth:`_prepare_lease_with_503`.
        - :class:`InferenceMiddlewareError` from the pipeline →
          propagated so caller-set ``status_code`` survives.
        - Anything else below the lease → 503 with ``error_msg``.
        """
        cache, stable, provenance, main_llm, client = await self._prepare_lease_with_503(
            ctx, source, request_body, request_headers, error_msg
        )
        try:
            with platform_headers_context(client):
                async with cache.lease(stable, main_llm=main_llm, provenance=provenance) as llm_rails:
                    raw_generation_response = await asyncio.to_thread(
                        run_generate_in_new_loop,
                        llm_rails,
                        messages=messages,
                        options=build_generate_async_options(rail_types, user_log_options),
                    )
        except InferenceMiddlewareError:
            # Preserves the caller's explicit ``status_code``; wrapping to
            # 503 here would clobber it.
            raise
        except Exception as exc:
            # Rail-task LLM calls (e.g. a vision-safety judge) can fail with
            # a real upstream status code (bad request shape, unsupported
            # parameter, provider outage, etc.) that nemoguardrails otherwise
            # buries inside a generic exception. Recover and propagate it
            # verbatim so callers see the actual upstream status instead of
            # our own middleware's fallback.
            if upstream_error := extract_upstream_error(exc):
                raise upstream_error from exc
            # ``ValueError`` from below the lease boundary is a library bug,
            # not caller-shape (caller-shape ValueErrors are caught and
            # converted to 400 in ``_prepare_lease_with_503`` before we
            # ever reach the lease), so it correctly wraps to 503.
            raise InferenceMiddlewareUnavailableError(error_msg) from exc

        return GenerationResponseMapper.parse(raw_generation_response)

    async def _prepare_lease_with_503(
        self,
        ctx: InferenceMiddlewareContext,
        source: GuardrailConfigSource,
        request_body: dict[str, Any],
        request_headers: dict[str, str],
        error_msg: str,
    ) -> tuple[LLMRailsCache, StableRailsConfig, Provenance, LLMModel, AsyncNemoClient]:
        """Run :meth:`_prepare_lease` with the plugin's error-mapping policy.

        Single source of truth for "lease setup → HTTP status" so non-streaming
        and streaming paths can't drift apart, and so the wire-status contract
        stays local to the plugin (IGW only catches
        :class:`InferenceMiddlewareError`).

        Also resolves the client via :meth:`_ensure_client` inside the same boundary
        so a request that races ``on_shutdown`` (client detached) maps to 503
        rather than escaping as a raw ``RuntimeError``.

        - ``ValueError`` → 400 (caller-shape: missing ``model``, malformed inline config).
        - :class:`InferenceMiddlewareError` → propagated unchanged.
        - Anything else → 503 with ``error_msg``, original as ``__cause__``.
        """
        try:
            cache, stable, provenance, main_llm = await self._prepare_lease(source, request_body, request_headers)
            client = ctx.request_nemo_client or self._ensure_client()
            return cache, stable, provenance, main_llm, client
        except InferenceMiddlewareError:
            raise
        except ValueError as exc:
            raise InferenceMiddlewareError(str(exc), status_code=400) from exc
        except Exception as exc:
            raise InferenceMiddlewareUnavailableError(error_msg) from exc

    async def _prepare_lease(
        self,
        source: GuardrailConfigSource,
        request_body: dict[str, Any],
        request_headers: dict[str, str],
    ) -> tuple[LLMRailsCache, StableRailsConfig, Provenance, LLMModel]:
        """Run the eager portion of lease setup; return what ``cache.lease`` needs.

        Eager stabilize + :func:`build_main_llm` (lease itself is deferred to
        the streaming generator) lets streaming fail fast on cheap setup
        errors before the wire commits to streaming. ``build_main_llm`` runs
        on a worker thread — ``init_llm_model`` does a blocking LangChain
        provider import.

        Raises raw exceptions; callers MUST route through
        :meth:`_prepare_lease_with_503` or risk leaking a non-
        :class:`InferenceMiddlewareError` past IGW's exception map (would 500).
        """
        # Snapshot the cache attrs before the null-check so a concurrent
        # ``on_shutdown`` that nulls ``self._rails_cache`` between the check
        # and the lease can't drop us into a NoneType attribute error mid
        # request: we either see both bound (and use the snapshot through
        # to the lease) or raise the clean RuntimeError below. The cache's
        # own ``close`` is responsible for draining in-flight leases.
        cache = self._rails_cache
        stable_cache = self._stable_cache
        if cache is None or stable_cache is None:
            raise RuntimeError("LLMRails cache is not initialized. Was on_startup() called?")

        resolver = self.get_openai_compatible_inference_url_and_model
        stable = stable_cache.get_or_compute(source, resolver)
        # Offloaded to a worker thread because init_llm_model (via
        # build_main_llm) does a blocking LangChain provider import.
        #
        # Per-request: request_body["model"] and the request's x-headers
        # vary per call. The main entry is stripped from the cached
        # LLMRails config, and the freshly-built main_llm is injected
        # here instead.
        main_llm = await asyncio.to_thread(
            build_main_llm,
            request_body,
            request_headers,
            resolver,
            stable.main_model_template,
        )
        return cache, stable, provenance_of(source), main_llm
