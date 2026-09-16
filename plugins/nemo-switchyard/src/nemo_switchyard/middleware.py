# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SwitchyardMiddleware: IGW inference middleware backed by Switchyard factories.

Maps VM configs to Switchyard factory classes, registers per-VM factory instances
in Switchyard's registry, and routes requests/responses through the registered
pipelines.

Each registration is tagged with the phase the user listed it under
(``request`` or ``response``); the lookup key is ``(vm_key, config_type, phase)``.
``process_request`` only finds request-phase entries; ``process_response`` only
response-phase. Listing the same config under both lists registers two phase
entries pointing at the same factory — the user is opting into both pipelines.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_platform_plugin.inference_middleware import (
    ImmediateResponse,
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceRequest,
    InferenceResponse,
    NemoInferenceMiddleware,
    VirtualModel,
)
from nemo_switchyard import _state
from nemo_switchyard._bridge import (
    _wrap_non_streaming,
    _wrap_streaming,
    write_back_request,
    write_back_response,
)
from nemo_switchyard._factory import (
    CONFIG_TYPE_TO_FACTORY_CLASS,
    VMFactoryInstance,
    initialize_factory_map,
)
from nemo_switchyard._format import build_chat_request, vm_models_for_switchyard
from nemo_switchyard._native_config import (
    NATIVE_CONFIG_TYPES,
    build_native_algorithm,
    models_map_from_config,
    native_model_categories,
    require_native_rust,
    validate_native_config,
)
from nemo_switchyard._native_host import IgwJudgeTransport, NativeBinding, run_native_stream
from nmp.core.inference_gateway.api.typed_response import TypedResponseStream
from switchyard.lib.proxy_context import (
    CTX_ORIGINAL_REQUEST,
    CTX_PROXY_ACTUAL_MODEL,
    ProxyContext,
)
from switchyard.lib.registry import MiddlewareFactory, lookup, register, unregister  # noqa: PYI021

logger = logging.getLogger(__name__)

# Plugin-state key for bridging Switchyard ProxyContext.metadata across IGW
# hooks. In standalone Switchyard, the chain runner threads one ProxyContext
# through both pipelines, so anything stamped at request time (e.g.
# CTX_ORIGINAL_FORMAT, CTX_ORIGINAL_REQUEST, routing's CTX_PROXY_ACTUAL_MODEL)
# is visible to response-side processors. IGW invokes process_request and
# process_response separately, so we build a fresh ProxyContext each hook —
# this key carries the request-side metadata across.
_SY_METADATA_STATE_KEY = "sy_metadata"
_PLUGIN_STATE_NAMESPACE = "switchyard"


class SwitchyardMiddleware(NemoInferenceMiddleware):
    """IGW inference middleware using Switchyard factories.

    - Maps VM configs to Switchyard factory classes
    - Registers factory instances in Switchyard's registry with VM-scoped names
    - Looks up and uses registered factories during request/response processing
    """

    async def on_startup(self) -> None:
        initialize_factory_map()
        logger.info(
            "SwitchyardMiddleware loaded — May config_types: %s native config_types: %s",
            list(CONFIG_TYPE_TO_FACTORY_CLASS.keys()),
            sorted(NATIVE_CONFIG_TYPES),
        )

    async def on_shutdown(self) -> None:
        """Unregister all factories and clear state to prevent stale entries on reload."""
        factory_names = list(_state.FACTORIES_BY_CONFIG_HASH.values())
        for factory_name in factory_names:
            try:
                unregister(factory_name)
            except Exception as e:
                logger.warning("Failed to unregister factory %r: %s", factory_name, e, exc_info=True)

        CONFIG_TYPE_TO_FACTORY_CLASS.clear()
        _state.clear_all()

        logger.info("SwitchyardMiddleware shutdown — cleared %d factories", len(factory_names))

    async def validate_middleware_config(
        self,
        config_type: str,
        config: Any,
    ) -> dict[str, Any]:
        """Validate config_type and return middleware_config used at request time.

        Raises InferenceMiddlewareError on unknown config_type so IGW rejects the
        VirtualModel upsert at create time rather than crashing on first request.
        """
        if config_type in NATIVE_CONFIG_TYPES:
            require_native_rust(config_type)
            validate_native_config(config_type, config)
            return {"config_type": config_type}
        if config_type not in CONFIG_TYPE_TO_FACTORY_CLASS:
            supported = [*CONFIG_TYPE_TO_FACTORY_CLASS, *sorted(NATIVE_CONFIG_TYPES)]
            raise InferenceMiddlewareError(
                f"Unknown config_type {config_type!r}. Supported: {supported}",
                status_code=400,
            )
        return {"config_type": config_type}

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: dict[str, Any],
    ) -> InferenceRequest | ImmediateResponse:
        """Run the Switchyard request pipeline for this VM + middleware.

        IGW invokes this once per nemo-switchyard middleware in the VM's
        request_middleware list, handling chaining; we use middleware_config
        ["config_type"] (set by validate_middleware_config) to look up the right
        factory.
        """
        config_type = middleware_config.get("config_type")
        if config_type in NATIVE_CONFIG_TYPES:
            return await self._process_native_request(ctx, request, middleware_config)

        factory = self._lookup_factory(ctx, middleware_config, phase="request")

        # IGW must populate typed_body for all recognised API paths. If it is None,
        # the path was unrecognised or body validation failed — a contract violation
        # that we surface immediately rather than silently proceeding with raw body.
        if request.typed_body is None:
            raise InferenceMiddlewareError(
                f"InferenceRequest.typed_body is required for switchyard middleware "
                f"but was not populated by IGW. "
                f"Path {request.path!r} may not be a recognised API path "
                f"(v1/chat/completions, v1/messages, v1/responses).",
                status_code=500,
            )

        # Build a typed ChatRequest from path + body. Dispatch on path (not body
        # structure) because OpenAI multimodal content also uses list-of-dicts.
        chat_request = build_chat_request(request.path, request.body)

        # Pass request.headers into context metadata by reference. Today no Switchyard
        # processor reads or writes metadata["headers"] (verified against switchyard.lib
        # processors), but exposing it preserves the option for future ones. In-place
        # mutations on the dict propagate to request.headers automatically; a processor
        # that *replaces* metadata["headers"] with a new dict would NOT propagate, which
        # is the contract we expect from processors that touch headers.
        sy_context = ProxyContext(metadata={"headers": request.headers})

        # Seed CTX_PROXY_ACTUAL_MODEL from the request model. Switchyard's routing
        # processors normally set this to the routed-to model, and translate reads it
        # to look up backend_format. When translate runs without prior routing, this
        # seed lets translate find the model in its models list. Routing processors
        # will overwrite this if they run, so it's a safe default.
        model = chat_request.body.get("model")
        if model:
            sy_context.metadata[CTX_PROXY_ACTUAL_MODEL] = model

        # Pre-seed CTX_ORIGINAL_REQUEST from ctx.original_request.body so that
        # StampOriginalFormatProcessor (and any translate pipeline that reads it)
        # sees the truly-original body — before routing or any prior middleware
        # mutated body["model"].  StampOriginalFormatProcessor stamps idempotently
        # (only if the key is absent), so this seed wins for the first call in a
        # chained pipeline and is harmlessly ignored on subsequent calls.
        if ctx.original_request is not None:
            original_body = (
                ctx.original_request.typed_body
                if ctx.original_request.typed_body is not None
                else ctx.original_request.body
            )
            sy_context.metadata[CTX_ORIGINAL_REQUEST] = original_body

        try:
            request_pipeline = factory.build_request_pipeline(factory.config)
            processed_request = await request_pipeline.process(sy_context, chat_request)

            # Apply pipeline output back to the InferenceRequest. write_back_request
            # sets request.body, clears request.typed_body, and applies any path
            # update stamped by PathUpdateProcessor into sy_context.metadata.
            write_back_request(request, processed_request, sy_context)

            # Bridge SY metadata to process_response. In standalone Switchyard, one
            # ProxyContext is threaded through both pipelines so anything stamped here
            # (CTX_ORIGINAL_FORMAT, CTX_ORIGINAL_REQUEST, routing decisions) is
            # visible to response-side processors. IGW calls each hook with its own
            # scope, so we explicitly carry the metadata via plugin state.
            # Strip "headers" — it's a per-hook reference and would shadow the
            # response headers we seed in process_response.
            sy_metadata = {k: v for k, v in sy_context.metadata.items() if k != "headers"}
            ctx.state(_PLUGIN_STATE_NAMESPACE).set(_SY_METADATA_STATE_KEY, sy_metadata)

            return request
        except InferenceMiddlewareError:
            raise
        except Exception as e:
            logger.error("Switchyard request pipeline failed: %s", e, exc_info=True)
            raise InferenceMiddlewareError(str(e), status_code=500) from e

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: dict[str, Any],
    ) -> InferenceResponse:
        """Run the Switchyard response pipeline for this VM + middleware.

        Streaming and non-streaming have different contracts:

        - ``typed_body is None`` — contract violation; IGW must always provide a
          typed view for recognised backends. Raises 500.
        - Non-streaming (``ChatCompletion`` / ``Message``): call
          ``_translate_non_streaming`` directly — bypasses the pipeline entirely.
          The translate pipeline is a single processor calling the same conversion
          functions; we skip the ``ChatResponse`` wrapper round-trip.
        - Streaming (``TypedResponseStream``): wrap via ``_wrap_streaming`` and
          run the pipeline asynchronously. Translation is lazy over the iterator.
        """
        config_type = middleware_config.get("config_type")
        if config_type in NATIVE_CONFIG_TYPES:
            return response

        if response.typed_body is None:
            # A prior streaming switchyard pass clears typed_body and sets result
            # to the translated stream. Pass through rather than raising so chained
            # streaming response middleware entries don't 500.
            if hasattr(response.result, "__aiter__"):
                return response
            raise InferenceMiddlewareError(
                "InferenceResponse.typed_body is required for switchyard middleware "
                "but was not populated by IGW. Ensure no prior response middleware "
                "clears typed_body before switchyard runs.",
                status_code=500,
            )

        prior_metadata = ctx.state(_PLUGIN_STATE_NAMESPACE).get(_SY_METADATA_STATE_KEY, {})

        # Wrap the typed_body into a Switchyard ChatResponse — both streaming and
        # non-streaming go through the Switchyard response pipeline so that all
        # translation is handled by FormatTranslateResponseProcessor, not here.
        if isinstance(response.typed_body, TypedResponseStream):
            backend_format = ctx.backend_format
            if backend_format is None:
                logger.debug("No backend_format on ctx; passing streaming typed_body through")
                return response
            chat_response = _wrap_streaming(response.typed_body, backend_format)
            if chat_response is None:
                logger.debug(
                    "No streaming wrapper for backend_format %s; passing through",
                    backend_format,
                )
                return response
        else:
            chat_response = _wrap_non_streaming(response.typed_body)

        # Phase-tagged lookup: only succeeds if the user listed this config under
        # response_middleware. If they only listed it under request_middleware,
        # we raise 400 with a hint pointing them at the right list.
        factory = self._lookup_factory(ctx, middleware_config, phase="response")

        # Reconstruct SY context with whatever the request hook stamped (e.g.
        # CTX_ORIGINAL_FORMAT, CTX_ORIGINAL_REQUEST, routing decisions). Per-request
        # scope lives on InferenceMiddlewareContext._state — no cross-request leak.
        sy_context = ProxyContext(metadata={**prior_metadata, "headers": response.headers})

        try:
            response_pipeline = factory.build_response_pipeline(factory.config)
            processed = await response_pipeline.process(sy_context, chat_response)
            write_back_response(response, processed)
            return response
        except InferenceMiddlewareError:
            raise
        except Exception as e:
            logger.error("Switchyard response pipeline failed: %s", e, exc_info=True)
            raise InferenceMiddlewareError(str(e), status_code=500) from e

    async def on_virtual_model_upserted(self, virtual_model: VirtualModel) -> None:
        """Register nemo-switchyard middlewares on this VM, tagged by phase.

        A VM may have multiple chained nemo-switchyard middlewares (e.g.,
        random_routing + translate) listed under request_middleware,
        response_middleware, or both. Each entry is registered with the phase
        the user listed it under, so process_request only fires for entries
        from request_middleware and process_response only for entries from
        response_middleware. Translate's response pipeline thus only runs if
        the user explicitly listed it under response_middleware too.

        Raises InferenceMiddlewareError on failure so IGW rejects the VM upsert.
        """
        request_entries = [
            mw for mw in getattr(virtual_model, "request_middleware", []) if mw.name == "nemo-switchyard"
        ]
        response_entries = [
            mw for mw in getattr(virtual_model, "response_middleware", []) if mw.name == "nemo-switchyard"
        ]

        previous_hashes = list(_state.VM_CONFIG_MAPPING.get(virtual_model.id, []))
        if not request_entries and not response_entries:
            logger.debug(
                "SwitchyardMiddleware: VirtualModel %r/%r has no nemo-switchyard middleware",
                virtual_model.workspace,
                virtual_model.name,
            )
            self._release_unused_config_hashes(virtual_model.id, previous_hashes, [])
            return

        vm_key = f"{virtual_model.workspace}/{virtual_model.name}"

        try:
            models_list = vm_models_for_switchyard(virtual_model)
        except ValueError as e:
            raise InferenceMiddlewareError(
                f"Failed to extract backend formats for VM {vm_key}: {e}",
                status_code=400,
            ) from e

        registered_hashes: list[str] = []
        for middleware_call in request_entries:
            registered_hashes.append(
                self._register_entry(vm_key, middleware_call, models_list, phase="request", virtual_model=virtual_model)
            )
        for middleware_call in response_entries:
            registered_hashes.append(
                self._register_entry(
                    vm_key, middleware_call, models_list, phase="response", virtual_model=virtual_model
                )
            )

        _state.VM_CONFIG_MAPPING[virtual_model.id] = registered_hashes
        self._release_unused_config_hashes(virtual_model.id, previous_hashes, registered_hashes)

    def _release_unused_config_hashes(
        self,
        vm_id: str,
        previous: list[str],
        current: list[str],
    ) -> None:
        if current:
            _state.VM_CONFIG_MAPPING[vm_id] = current
        else:
            _state.VM_CONFIG_MAPPING.pop(vm_id, None)
        remaining = {h for hashes in _state.VM_CONFIG_MAPPING.values() for h in hashes}
        released = set(previous) - remaining
        for cfg_hash in released:
            _state.NATIVE_BY_CONFIG_HASH.pop(cfg_hash, None)
        for key, cfg_hash in list(_state.VM_NAME_TO_CONFIG_HASH.items()):
            if cfg_hash in released:
                _state.VM_NAME_TO_CONFIG_HASH.pop(key, None)

    def _register_entry(
        self,
        vm_key: str,
        middleware_call: Any,
        models_list: list[dict[str, Any]],
        phase: _state.Phase,
        virtual_model: VirtualModel,
    ) -> str:
        """Validate, register, and tag-by-phase a single nemo-switchyard middleware entry.

        Returns the config_hash so on_virtual_model_destroyed can clean up.
        """
        config_type = middleware_call.config_type
        config = middleware_call.config or {}

        if config_type in NATIVE_CONFIG_TYPES:
            if phase != "request":
                raise InferenceMiddlewareError(
                    f"{config_type!r} is request-only; list it under request_middleware",
                    status_code=400,
                )
            return self._register_native_entry(vm_key, config_type, config, phase)

        factory_class = CONFIG_TYPE_TO_FACTORY_CLASS.get(config_type)
        if not factory_class:
            raise InferenceMiddlewareError(
                f"Unknown config_type {config_type!r} for VM {vm_key}. "
                f"Supported: {[*CONFIG_TYPE_TO_FACTORY_CLASS, *sorted(NATIVE_CONFIG_TYPES)]}",
                status_code=400,
            )

        enriched_config = {**config, "models": models_list}

        try:
            validated_config = factory_class.validate(enriched_config)
        except Exception as e:
            raise InferenceMiddlewareError(
                f"Failed to validate {config_type!r} config for VM {vm_key}: {e}",
                status_code=400,
            ) from e

        cfg_hash = _state.config_hash(enriched_config, config_type)
        _state.VM_NAME_TO_CONFIG_HASH[(vm_key, config_type, phase)] = cfg_hash

        if cfg_hash in _state.FACTORIES_BY_CONFIG_HASH:
            logger.info(
                "SwitchyardMiddleware: VM %r reuses existing factory for %r/%s (hash=%s)",
                vm_key,
                config_type,
                phase,
                cfg_hash,
            )
            return cfg_hash

        factory_name = f"nemo-switchyard-{config_type}-{cfg_hash}"
        try:
            vm_factory: MiddlewareFactory[Any] = VMFactoryInstance(
                factory_class,
                validated_config,
                factory_name,
                config_type=config_type,
            )
            register(vm_factory)
            _state.FACTORIES_BY_CONFIG_HASH[cfg_hash] = factory_name
            logger.info(
                "SwitchyardMiddleware: Registered factory %r for VM %r (config_type=%r, phase=%s, hash=%s, models=%d)",
                factory_name,
                vm_key,
                config_type,
                phase,
                cfg_hash,
                len(virtual_model.models),
            )
        except Exception as e:
            raise InferenceMiddlewareError(
                f"Failed to register factory for VM {vm_key} ({config_type!r}, {phase}): {e}",
                status_code=500,
            ) from e
        return cfg_hash

    async def on_virtual_model_destroyed(self, virtual_model: VirtualModel) -> None:
        """Unregister this VM's middlewares; factories shared with other VMs stay."""
        vm_id = virtual_model.id
        vm_key = f"{virtual_model.workspace}/{virtual_model.name}"

        config_hashes = _state.VM_CONFIG_MAPPING.pop(vm_id, None)
        if not config_hashes:
            logger.debug("SwitchyardMiddleware: VirtualModel %s has no registered configs", vm_key)
            return

        # Remove all (vm_key, config_type, phase) entries for this VM
        for key in [k for k in _state.VM_NAME_TO_CONFIG_HASH if k[0] == vm_key]:
            _state.VM_NAME_TO_CONFIG_HASH.pop(key, None)

        # Unregister factories no other VM uses. VM_CONFIG_MAPPING was already popped,
        # so .values() reflects remaining VMs only.
        remaining_hashes = {h for hashes in _state.VM_CONFIG_MAPPING.values() for h in hashes}
        for cfg_hash in set(config_hashes):
            if cfg_hash in remaining_hashes:
                logger.debug(
                    "SwitchyardMiddleware: Config hash %s still used by other VMs, keeping factory",
                    cfg_hash,
                )
                continue

            _state.NATIVE_BY_CONFIG_HASH.pop(cfg_hash, None)
            factory_name = _state.FACTORIES_BY_CONFIG_HASH.get(cfg_hash)
            if not factory_name:
                continue
            try:
                unregister(factory_name)
                # Only remove from local mapping after successful unregister, so a failed
                # unregister doesn't orphan the entry in Switchyard's registry.
                _state.FACTORIES_BY_CONFIG_HASH.pop(cfg_hash, None)
                logger.info(
                    "SwitchyardMiddleware: Unregistered factory %r (config_hash=%s)",
                    factory_name,
                    cfg_hash,
                )
            except Exception as e:
                logger.error(
                    "SwitchyardMiddleware: Failed to unregister factory %r: %s",
                    factory_name,
                    e,
                    exc_info=True,
                )

    def _lookup_factory(
        self,
        ctx: InferenceMiddlewareContext,
        middleware_config: dict[str, Any],
        phase: _state.Phase,
    ) -> Any:
        """Look up the registered factory for this VM + (config_type, phase).

        Phase is "request" when called from process_request, "response" from
        process_response. If the user didn't list this config under the matching
        VM list, no factory is registered for that phase and we raise 400 — it's
        a misconfiguration, not a server bug.
        """
        vm_key = f"{ctx.workspace}/{ctx.virtual_model_name}"
        config_type = middleware_config.get("config_type")
        if not config_type:
            raise InferenceMiddlewareError(
                f"middleware_config missing 'config_type' for VM {vm_key}",
                status_code=500,
            )

        cfg_hash = _state.VM_NAME_TO_CONFIG_HASH.get((vm_key, config_type, phase))
        if not cfg_hash:
            other_phase: _state.Phase = "response" if phase == "request" else "request"
            other_registered = (vm_key, config_type, other_phase) in _state.VM_NAME_TO_CONFIG_HASH
            hint = (
                f" — it is registered under {other_phase}_middleware; list it under "
                f"{phase}_middleware too if you want it to run on the {phase} side."
                if other_registered
                else ""
            )
            raise InferenceMiddlewareError(
                f"No factory registered for VM {vm_key} with config_type {config_type!r} on the {phase} side{hint}",
                status_code=400,
            )

        factory_name = _state.FACTORIES_BY_CONFIG_HASH.get(cfg_hash)
        if not factory_name:
            raise InferenceMiddlewareError(
                f"Factory not found for config hash {cfg_hash} (VM {vm_key}, {config_type!r})",
                status_code=500,
            )

        try:
            return lookup(factory_name)
        except Exception as e:
            logger.error(
                "Factory %r was registered but not found in registry: %s",
                factory_name,
                e,
                exc_info=True,
            )
            raise InferenceMiddlewareError(f"Factory {factory_name} missing from registry", status_code=500) from e

    async def _process_native_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: dict[str, Any],
    ) -> InferenceRequest | ImmediateResponse:
        binding = self._lookup_native_binding(ctx, middleware_config)
        transport = IgwJudgeTransport(self)
        try:
            return await run_native_stream(
                algorithm=binding.algorithm,
                request=request,
                models=binding.models,
                headers=dict(request.headers),
                transport=transport,
            )
        except InferenceMiddlewareError:
            raise
        except Exception as exc:
            logger.error("Native Switchyard run_stream failed: %s", exc, exc_info=True)
            raise InferenceMiddlewareError(str(exc), status_code=500) from exc

    def _lookup_native_binding(
        self,
        ctx: InferenceMiddlewareContext,
        middleware_config: dict[str, Any],
    ) -> NativeBinding:
        vm_key = f"{ctx.workspace}/{ctx.virtual_model_name}"
        config_type = middleware_config.get("config_type")
        if not config_type:
            raise InferenceMiddlewareError(
                f"middleware_config missing 'config_type' for VM {vm_key}",
                status_code=500,
            )
        cfg_hash = _state.VM_NAME_TO_CONFIG_HASH.get((vm_key, config_type, "request"))
        binding = _state.NATIVE_BY_CONFIG_HASH.get(cfg_hash) if cfg_hash else None
        if binding is not None:
            return binding
        self._rebuild_native_binding(vm_key, config_type)
        cfg_hash = _state.VM_NAME_TO_CONFIG_HASH.get((vm_key, config_type, "request"))
        binding = _state.NATIVE_BY_CONFIG_HASH.get(cfg_hash) if cfg_hash else None
        if binding is None:
            raise InferenceMiddlewareError(
                f"No native Algorithm registered for VM {vm_key} with config_type {config_type!r}",
                status_code=400,
            )
        return binding

    def _rebuild_native_binding(self, vm_key: str, config_type: str) -> None:
        """Rebuild from the live VM when IGW cache refresh dropped in-memory Algorithms."""
        vm = self.get_virtual_model(vm_key)
        if vm is None:
            return
        for call in getattr(vm, "request_middleware", None) or []:
            if getattr(call, "name", None) != "nemo-switchyard":
                continue
            if getattr(call, "config_type", None) != config_type:
                continue
            self._register_native_entry(vm_key, config_type, call.config or {}, "request")
            return

    def _register_native_entry(
        self,
        vm_key: str,
        config_type: str,
        config: dict[str, Any],
        phase: _state.Phase,
    ) -> str:
        validated = validate_native_config(config_type, config)
        required = native_model_categories(config_type)
        models = models_map_from_config(validated, required=required)
        cfg_hash = _state.config_hash(validated, config_type)
        if cfg_hash in _state.NATIVE_BY_CONFIG_HASH:
            _state.VM_NAME_TO_CONFIG_HASH[(vm_key, config_type, phase)] = cfg_hash
            return cfg_hash
        try:
            algorithm = build_native_algorithm(config_type, validated)
        except InferenceMiddlewareError:
            raise
        except Exception as exc:
            raise InferenceMiddlewareError(
                f"Failed to build native {config_type!r} for VM {vm_key}: {exc}",
                status_code=400,
            ) from exc
        _state.NATIVE_BY_CONFIG_HASH[cfg_hash] = NativeBinding(
            algorithm=algorithm,
            models=models,
            config_type=config_type,
        )
        _state.VM_NAME_TO_CONFIG_HASH[(vm_key, config_type, phase)] = cfg_hash
        logger.info(
            "SwitchyardMiddleware: Registered native %r for VM %r (hash=%s)",
            config_type,
            vm_key,
            cfg_hash,
        )
        return cfg_hash
