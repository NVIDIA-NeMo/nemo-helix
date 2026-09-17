# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference middleware plugin interface — what plugin authors implement.

Plugin authors subclass :class:`NemoInferenceMiddleware` and register a
class under the ``nemo.inference_middleware`` entry-point group in their
plugin's ``pyproject.toml``.  IGW discovers and loads these plugins at
startup.

Example::

    # nemo_switchyard_plugin/middleware.py
    from nemo_platform_plugin.inference_middleware import (
        InferenceMiddlewareContext,
        InferenceRequest,
        InferenceResponse,
        ImmediateResponse,
        NemoInferenceMiddleware,
        VirtualModel,
    )

    class SwitchyardMiddleware(NemoInferenceMiddleware):
        async def on_startup(self) -> None:
            # Validate configured model entities exist at startup
            entities = self.list_model_entities_for_workspace("my-workspace")
            ...

        async def get_middleware_config(
            self, config_type: str, config_id: str
        ) -> Any:
            # Plugin owns config storage — fetch from entity store
            ws, name = config_id.split("/", 1)
            return await self._entity_client.get(RouteLLMConfig, workspace=ws, name=name)

        async def validate_middleware_config(
            self, config_type: str, config: Any
        ) -> Any:
            if config_type == "routellm_config":
                return RouteLLMConfig.model_validate(config)
            raise ValueError(f"Unknown config_type: {config_type!r}")

        async def process_request(
            self,
            ctx: InferenceMiddlewareContext,
            request: InferenceRequest,
            middleware_config: Any,
        ) -> InferenceRequest | ImmediateResponse:
            chosen_model = self._run_routellm(request.body, middleware_config)
            request.body["model"] = chosen_model
            return request

        async def process_response(
            self,
            ctx: InferenceMiddlewareContext,
            response: InferenceResponse,
            middleware_config: Any,
        ) -> InferenceResponse:
            # ctx.original_request holds what the client sent (before any plugin ran)
            # ctx.proxied_request holds what was forwarded to the backend
            # ctx.state("my-plugin").get("key") retrieves state set in process_request
            return response

    # pyproject.toml:
    # [project.entry-points."nemo.inference_middleware"]
    # nemo-switchyard = "nemo_switchyard_plugin.middleware:SwitchyardMiddleware"
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, TypeAlias, Union, runtime_checkable

if TYPE_CHECKING:
    # Keep this public base import-light; these imports pull in heavy SDK modules.
    from nemo_platform import AsyncNeMoPlatform
    from nemo_platform_plugin.client.client import AsyncNemoClient


class BackendFormat(str, Enum):
    """Inference backend API wire formats understood by IGW and middleware plugins."""

    OPENAI_CHAT = "OPENAI_CHAT"
    ANTHROPIC_MESSAGES = "ANTHROPIC_MESSAGES"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

ResponseResult = Union[dict[str, Any], AsyncIterator[dict[str, Any]]]
"""The underlying response data flowing through the response middleware chain.

- ``dict[str, Any]`` — Non-streaming response parsed from the backend.
- ``AsyncIterator[dict[str, Any]]`` — Streaming response (sequence of chunk dicts).
"""

TypedRequest: TypeAlias = Mapping[str, Any]
"""Fast-path structural request body type for :attr:`InferenceRequest.typed_body`.

SDK request parameter types are TypedDicts, so they are dict-like at runtime.
Code that needs the exact SDK TypedDict union can import
``TypedRequest`` from ``nemo_platform_plugin.inference_middleware_types``.
"""


@runtime_checkable
class TypedResponse(Protocol):
    """Fast-path structural response model for :attr:`InferenceResponse.typed_body`.

    The concrete SDK response models live in
    ``nemo_platform_plugin.inference_middleware_types``. The base middleware
    interface only needs the shared serialized-model surface.
    """

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class TypedResponseChunk(Protocol):
    """Fast-path structural streaming chunk model for typed response streams."""

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


TypedResponseResult: TypeAlias = Union[TypedResponse, AsyncIterator[TypedResponseChunk]]
"""Structural typed response body or typed streaming chunk iterator."""


@dataclass
class InferenceRequest:
    """Typed envelope for request pipeline data.

    Passed to :meth:`NemoInferenceMiddleware.process_request` and returned from it.
    Plugins may mutate ``body``, ``headers``, and ``path`` in-place and return
    ``self``, or construct a new instance — either style is valid.

    All three fields are required. When constructing a new instance instead of
    mutating in-place, always copy ``path`` from the incoming request::

        # correct
        return InferenceRequest(
            body={**request.body, "model": chosen},
            headers=request.headers,
            path=request.path,
        )

    Omitting ``path`` is a compile-time error. This prevents the silent 404
    that would result from sending the request to the backend root URL.

    ``typed_body`` does not need to be copied or re-derived manually when
    constructing a new instance. IGW automatically re-derives ``typed_body``
    from ``body`` and ``path`` between every plugin in the request middleware
    chain, so the next plugin always receives a fresh, consistent typed view
    regardless of what the previous plugin returned.

    Attributes:
        body: OpenAI-compatible request body dict (e.g. ``{"model": ..., "messages": [...]}``.
            When called, ``body["model"]`` contains the model entity written by IGW from
            ``VirtualModel.default_model_entity`` (if set). Middleware may freely rewrite it.
        headers: HTTP request headers forwarded to each plugin in the chain.
        path: Backend-relative path (e.g. ``"v1/chat/completions"``). IGW sets the
            initial value from the incoming request URI. Middleware may rewrite this to
            any arbitrary path and IGW will proxy to that path instead.
        typed_body: SDK-typed view of the request body validated against the TypedDict
            schema for this path. ``None`` when the path is unknown or body validation
            fails.

            Mirrors ``InferenceResponse.typed_body``: just as
            ``response.typed_body`` is the typed view of ``result``,
            ``request.typed_body`` is the typed view of ``body``.

            All three SDK request param types are TypedDicts — plain dicts at runtime.
            This field is validated by IGW before the request middleware chain runs.
            Plugins that need cross-format awareness read ``request.path`` to determine
            format and use this as a type-checked alias for ``body``.

            Parsed non-fatally: a body that fails TypedDict validation is not an error
            — the plugin receives ``typed_body=None`` and falls back to ``body`` or
            raises a descriptive error as needed.
    """

    body: dict[str, Any]
    headers: dict[str, str]
    path: str
    typed_body: TypedRequest | None = None


@dataclass
class InferenceResponse:
    """Typed envelope for response pipeline data.

    Passed to :meth:`NemoInferenceMiddleware.process_response` and returned from it.
    Plugins may mutate the canonical response view plus ``headers`` in-place and
    return ``self``, or construct a new instance — either style is valid.
    ``typed_body`` is populated by IGW when it can parse the payload for the
    resolved backend format.

    Canonical response contract: for non-streaming responses, when
    ``typed_body`` is not ``None``, it is the canonical response view and IGW
    serializes it instead of ``result``. Streaming ``typed_body`` is a
    middleware-only typed view; IGW serializes the raw ``result`` stream. When
    ``typed_body`` is ``None``, ``result`` is canonical. Middleware should
    mutate exactly one canonical view. IGW does not keep ``result`` and
    ``typed_body`` synchronized between plugins.

    Plugins can manipulate the response payload or headers in the following ways:

    - Mutate existing payload fields (for example, redact
      ``choices[0].message.content``) by mutating ``typed_body`` when available,
      or ``result`` when no typed view exists.
    - Add new top-level response fields (for example, guardrails metadata)
      by writing to ``response_body_annotations``.
    - Modify HTTP response headers by mutating ``headers``.

    Attributes:
        result: The response data — either a ``dict[str, Any]`` (non-streaming) or an
            ``AsyncIterator[dict[str, Any]]`` (streaming). Passed to the next plugin in
            the chain, or returned to the caller if this is the last plugin. For
            non-streaming responses, mutate this only when ``typed_body`` is ``None``.
            For streaming responses, mutate or replace this when changing the outbound
            client stream.
        headers: HTTP response headers returned to the caller. Mutate to add or modify
            headers (e.g. ``X-Guardrails-Status``). Changes flow through the return value.
        typed_body: SDK-native parsed response data for middleware that wants
            typed access. ``None`` when the backend format is unknown or parsing fails.
            For non-streaming responses, if this is non-``None``, it is canonical.
            For streaming responses, it is a middleware-only view over the raw stream.
            If a plugin needs to return a non-streaming shape that does not fit the
            typed SDK schema, it must set ``response.typed_body = None`` and mutate
            ``response.result`` instead.

            Mirrors ``InferenceRequest.typed_body``: just as ``request.typed_body`` is
            the typed view of ``body``, ``response.typed_body`` is the typed view of
            ``result``.
        response_body_annotations: Top-level fields to be merged into the final
            response body after all middleware have run. Use this — not ``result``
            or ``typed_body`` — when adding fields that do not belong to the
            OpenAI/Anthropic payload schema (e.g. guardrails metadata, custom
            plugin telemetry). For non-streaming responses, IGW merges these into
            the serialized payload just before returning to the caller. For
            streaming responses, annotations are not injected in the initial
            implementation.

            Once an :class:`InferenceResponse` exists, this field is the
            canonical response-annotation owner. IGW may seed it from
            :attr:`InferenceMiddlewareContext.response_body_annotations` when
            building the envelope, but later response middleware can read,
            preserve, modify, or remove annotations here.

            Keys in ``response_body_annotations`` take precedence
            over payload keys on collision. Use plugin-scoped names (e.g.
            ``"guardrails_data"``, not ``"status"``) to avoid cross-plugin collisions.
    """

    result: ResponseResult
    headers: dict[str, str]
    typed_body: TypedResponseResult | None = None
    response_body_annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImmediateResponse:
    """Plugin-provided response that short-circuits the IGW proxy.

    Return this from :meth:`NemoInferenceMiddleware.process_request` to signal
    that the plugin has handled the request itself and IGW should skip the
    backend proxy. IGW passes ``data`` to the response middleware chain.

    ``data`` is a :data:`ResponseResult` — either a ``dict[str, Any]`` for a
    non-streaming response, or an ``AsyncIterator[dict[str, Any]]`` for a
    streaming response.

    ``response_body_annotations`` follows the same contract as
    :attr:`InferenceResponse.response_body_annotations`; use it to add
    top-level response fields on short-circuited responses.

    Example::

        async def process_request(
            self, ctx, request: InferenceRequest, middleware_config
        ) -> InferenceRequest | ImmediateResponse:
            result = await self._call_my_model(request.body)
            return ImmediateResponse(data=result)   # skip proxy
    """

    data: ResponseResult
    response_body_annotations: dict[str, Any] = field(default_factory=dict)


class PluginStateNamespace:
    """Per-plugin scratch space inside :class:`InferenceMiddlewareContext`.

    Returned by ``ctx.state("my-plugin")``. Keys are automatically namespaced
    by the plugin name so plugins cannot accidentally collide with each other::

        state = ctx.state("my-plugin")
        state.set("token", auth_token)      # stored as "my-plugin:token"
        token = state.get("token")
        state.has("token")                  # True
        state.delete("token")

    The underlying store is shared across the entire request so data set by
    one plugin in ``process_request`` is visible to the same plugin (and any
    other plugin that knows the key) in ``process_response``.
    """

    def __init__(self, *, plugin_name: str, _store: dict[str, Any]) -> None:
        self._prefix = f"{plugin_name}:"
        self._store = _store

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* (namespaced to this plugin)."""
        self._store[self._prefix + key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if not set."""
        return self._store.get(self._prefix + key, default)

    def delete(self, key: str) -> None:
        """Remove *key* from state (no-op if not present)."""
        self._store.pop(self._prefix + key, None)

    def has(self, key: str) -> bool:
        """Return ``True`` if *key* is present in state."""
        return (self._prefix + key) in self._store


@dataclass
class InferenceMiddlewareContext:
    """Per-request context constructed once by IGW and passed to every middleware hook.

    Plugins should not construct this themselves — IGW creates exactly one
    instance per VirtualModel request and passes it to every ``process_request``
    and ``process_response`` call that runs for that request.

    The read-only fields (``request_id``, ``workspace``, ``virtual_model_name``,
    ``original_request``, ``proxied_request``) are IGW-owned invariants.
    ``state()`` provides the plugin-owned scratch space for cross-hook communication.

    Attributes:
        request_id: Unique identifier for this request, taken from the incoming
            ``X-Request-ID`` header or generated by IGW.
        virtual_model_name: Name of the :class:`VirtualModel` being served
            (``"workspace/name"`` format not included here — just the ``name``).
        workspace: Workspace owning the VirtualModel.
        original_request: The request as IGW received it, after seeding
            ``body["model"]`` from ``VirtualModel.default_model_entity`` but before
            any request middleware plugin ran. Read-only by convention.
            Available in both ``process_request`` and ``process_response``.
        proxied_request: The request as it was forwarded to the backend — set by IGW
            after the full request middleware chain completes, before the response
            middleware chain runs. ``None`` during ``process_request`` and when an
            :class:`ImmediateResponse` short-circuited the proxy. Read-only by
            convention.
        backend_format: Resolved backend API format used to parse typed responses.
            Defaults to ``None`` until IGW has resolved the request target.
        response_body_annotations: Top-level fields to merge into the final
            response body after all middleware have run. Request middleware can
            use this when it needs to annotate the eventual backend response
            before an :class:`InferenceResponse` exists. This is a staging
            bridge only: IGW copies these values into
            :attr:`InferenceResponse.response_body_annotations` when it builds
            the response envelope, and final serialization uses the
            ``InferenceResponse`` field.
        request_nemo_client: Request-scoped typed client carrying the caller's
            delegated identity and trace headers. Middleware can use it for
            nested platform calls that must preserve request context.
    """

    request_id: str
    virtual_model_name: str
    workspace: str
    original_request: InferenceRequest
    proxied_request: InferenceRequest | None = None
    backend_format: BackendFormat | None = None
    response_body_annotations: dict[str, Any] = field(default_factory=dict)
    request_nemo_client: AsyncNemoClient | None = None
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def state(self, plugin_name: str) -> PluginStateNamespace:
        """Return the :class:`PluginStateNamespace` for *plugin_name*.

        Each call returns a lightweight wrapper around the same shared store —
        creating multiple wrappers for the same name is safe.

        Args:
            plugin_name: The entry-point key of the plugin (e.g. ``"nemo-guardrails"``).
                Use your own plugin name to avoid colliding with other plugins' state.
        """
        return PluginStateNamespace(plugin_name=plugin_name, _store=self._state)


@dataclass
class ModelProviderInferenceTarget:
    """Provider gateway URL and served model name for a direct inference call.

    Returned by :meth:`NemoInferenceMiddleware.get_inference_url_and_model`.
    Both values are resolved from the same provider selection, guaranteeing
    they are mutually consistent.

    Use ``model_provider_gateway_url`` as the OpenAI client base URL and
    ``served_model_name`` as the value for ``body["model"]``::

        target = self.get_inference_url_and_model("ws/llama-70b")
        response = await client.post(
            f"{target.model_provider_gateway_url}/chat/completions",
            json={**body, "model": target.served_model_name},
        )
    """

    model_provider_gateway_url: str
    """IGW provider gateway URL — routes directly to the backend provider,
    bypassing virtual model routing and the middleware chain."""

    served_model_name: str
    """The raw model ID the backend expects in ``body["model"]``
    (e.g. ``"meta/llama-3.1-70b-instruct"``)."""


@dataclass
class OpenAICompatibleInferenceTarget:
    """OpenAI-compatible IGW URL and VirtualModel ID for routed inference calls.

    Returned by :meth:`NemoInferenceMiddleware.get_openai_compatible_inference_url_and_model`.
    Use this when plugin-owned model calls should go through IGW's OpenAI-compatible
    VirtualModel route instead of calling a provider directly.
    """

    openai_base_url: str
    """OpenAI-compatible IGW base URL ending in ``/v1``."""

    model: str
    """Workspace-qualified VirtualModel ID to send in ``body["model"]``."""


RequestResult = Union[InferenceRequest, ImmediateResponse]
"""Return type for :meth:`NemoInferenceMiddleware.process_request`.

- :class:`InferenceRequest` — IGW proxies the (possibly modified) request.
- :class:`ImmediateResponse` — Plugin handled the request itself; IGW skips the
  proxy and passes ``data`` (a :data:`ResponseResult`) to the response middleware chain.
"""

# ---------------------------------------------------------------------------
# Platform entity Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class MiddlewareCall(Protocol):
    """Structural middleware pipeline entry passed through VirtualModel hooks."""

    @property
    def name(self) -> str: ...

    @property
    def config_type(self) -> str: ...

    @property
    def config(self) -> dict[str, Any] | None: ...

    @property
    def config_id(self) -> str | None: ...


@runtime_checkable
class VirtualModelInferenceConfig(Protocol):
    """Structural model entry referenced by a VirtualModel hook payload."""

    @property
    def model(self) -> str: ...

    @property
    def backend_format(self) -> BackendFormat | None: ...


@runtime_checkable
class VirtualModel(Protocol):
    """Structural VirtualModel type passed to middleware lifecycle hooks.

    Middleware plugins should import this type from
    ``nemo_platform_plugin.inference_middleware``. The concrete Pydantic entity
    model used by IGW's API layer lives outside this fast import path.
    """

    @property
    def id(self) -> str: ...

    @property
    def workspace(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def default_model_entity(self) -> str | None: ...

    @property
    def autoprovisioned(self) -> bool: ...

    @property
    def models(self) -> Sequence[VirtualModelInferenceConfig]: ...

    @property
    def request_middleware(self) -> Sequence[MiddlewareCall]: ...

    @property
    def response_middleware(self) -> Sequence[MiddlewareCall]: ...

    @property
    def post_response_middleware(self) -> Sequence[MiddlewareCall]: ...

    @property
    def override_proxy(self) -> str | None: ...

    @property
    def guardrail_config_ids(self) -> Sequence[str]: ...

    def middleware_calls(self) -> Iterator[MiddlewareCall]: ...

    def references_guardrail_config(self, config_id: str) -> bool: ...


@runtime_checkable
class ModelSpec(Protocol):
    """Structural view of a model's specification.

    Concrete type: ``nmp.core.models.schemas.ModelSpec``.
    """

    is_chat: bool | None
    """Whether this model is a chat/instruction-tuned model."""

    head_type: str
    """Task-specific head persisted in the checkpoint."""

    is_embedding_model: bool
    """Deprecated compatibility alias for an embedding head."""

    context_size: int | None
    """Maximum context window in tokens, or ``None`` if unspecified."""

    family: str
    """Model family identifier (e.g. ``"llama"``, ``"mistral"``)."""


@runtime_checkable
class ModelEntity(Protocol):
    """Structural view of a registered model entity.

    Concrete type: ``nmp.core.models.schemas.ModelEntity``.
    Defined here as a Protocol so ``nemo_platform_plugin`` does not depend on the
    models service package.

    When the models service is migrated to define ``ModelEntity`` as a
    ``NemoEntity`` subclass in ``nemo_platform_plugin``, this Protocol is replaced
    in-place — plugin authors' import paths and field access are unchanged.
    """

    workspace: str
    """Workspace this entity belongs to."""

    name: str
    """Entity name within the workspace."""

    spec: ModelSpec | None
    """Model specification — capabilities, architecture family, context size.
    ``None`` for externally-hosted models registered without a spec."""

    finetuning_type: str | None
    """How this model was fine-tuned (e.g. ``"lora_merged"``, ``"all_weights"``),
    or ``None`` for base models."""

    backend_format: BackendFormat | None
    """Inference API wire format expected by the backend, or ``None`` when unset.
    IGW treats ``None`` as ``OPENAI_CHAT`` during routing."""

    providers: list[ModelProvider]
    """All :class:`ModelProvider`\\ s currently serving this model entity.
    Empty if no provider is actively serving it."""


@runtime_checkable
class ModelProvider(Protocol):
    """Structural view of a model provider as seen by inference middleware.

    Concrete type: ``nmp.core.models.entities.ModelProvider``.
    Defined here as a Protocol so ``nemo_platform_plugin`` does not depend on the
    models service package.
    """

    workspace: str
    """Workspace this provider belongs to."""

    name: str
    """Provider name within the workspace."""

    host_url: str
    """Base URL of the backend inference service (e.g. ``"http://nim-svc:8080"``)."""


# ---------------------------------------------------------------------------
# Cache accessor Protocol
# ---------------------------------------------------------------------------


class InferenceMiddlewareCacheAccessor(Protocol):
    """Interface IGW must satisfy to inject read-only platform cache access into plugins.

    IGW constructs a concrete implementation and passes it to each plugin
    via :meth:`NemoInferenceMiddleware._inject_cache` before calling
    ``on_startup()``. Plugin authors reference this type in tests to build
    typed mocks::

        cache = MagicMock(spec=InferenceMiddlewareCacheAccessor)
        cache.get_model_providers_for_model.return_value = [...]
        plugin._inject_cache(cache)
    """

    def get_model_providers_for_model(self, model_entity_id: str) -> list[ModelProvider]: ...

    def get_model_entity(self, model_entity_id: str) -> ModelEntity | None: ...

    def list_model_entities_for_workspace(self, workspace: str | None = None) -> list[str]: ...

    def get_virtual_model(self, virtual_model_id: str) -> VirtualModel | None: ...

    def list_virtual_models_for_workspace(self, workspace: str) -> list[str]: ...

    def get_inference_url_and_model(
        self,
        model_entity_id: str,
        append_v1_suffix: bool = True,
    ) -> ModelProviderInferenceTarget: ...

    def get_backend_format(self, virtual_model_id: str, model_entity_id: str) -> BackendFormat | None: ...

    def get_openai_compatible_inference_url_and_model(
        self, virtual_model_id: str
    ) -> OpenAICompatibleInferenceTarget: ...


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InferenceMiddlewareError(Exception):
    """Typed exception for handled error conditions in inference middleware.

    Raise from :meth:`~NemoInferenceMiddleware.process_request` or
    :meth:`~NemoInferenceMiddleware.process_response` to signal an error
    with a specific HTTP status code. IGW catches this and maps
    ``status_code`` and ``detail`` to an OpenAI-compatible HTTP error
    response. Unhandled exceptions (generic ``Exception``) map to 500.

    Example::

        async def process_request(self, body, headers, config) -> RequestResult:
            raise InferenceMiddlewareError("quota exceeded", status_code=429)
    """

    def __init__(self, detail: str, *, status_code: int = 500) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class InferenceMiddlewareUnavailableError(InferenceMiddlewareError):
    """The plugin's upstream service is temporarily unavailable.

    Use when the plugin depends on an external service (e.g. a guardrails
    service, a routing classifier endpoint) that cannot be reached.
    Defaults to HTTP 503.
    """

    def __init__(
        self,
        detail: str = "Middleware service unavailable",
        *,
        status_code: int = 503,
    ) -> None:
        super().__init__(detail, status_code=status_code)


class MiddlewareConfigNotFoundError(InferenceMiddlewareError):
    """The referenced middleware config entity does not exist.

    Plugins MUST raise this from :meth:`NemoInferenceMiddleware.get_middleware_config`
    when the underlying entity store returns a definitive 404 — either because the
    user deleted the config out from under a referencing VirtualModel, or because
    they referenced a ``config_id`` that was never created.

    IGW uses this exception as the explicit signal to evict any previously-resolved
    middleware for VirtualModels that reference this config and to mark those
    VirtualModels as broken until the situation is resolved (either the config is
    recreated, or the VirtualModel is updated to drop the dangling reference).

    Distinguish this from:

    - :class:`ValueError` — the config exists but is malformed (e.g. missing
      required fields). IGW also evicts on this, treating it as caller-shape input.
    - Any other exception — treated as transient (network blip, plugin's upstream
      service unhealthy). IGW preserves the previously-resolved config so a brief
      outage cannot flap the cache.

    Status code defaults to 404 so that the VirtualModel CRUD endpoints surface
    a clean "config not found" error when a user submits a ``config_id`` that does
    not resolve at create / update time.
    """

    def __init__(self, config_id: str, *, detail: str | None = None) -> None:
        super().__init__(
            detail or f"Middleware config {config_id!r} not found",
            status_code=404,
        )
        self.config_id = config_id


# ---------------------------------------------------------------------------
# NemoInferenceMiddleware ABC
# ---------------------------------------------------------------------------


class NemoInferenceMiddleware(ABC):
    """Plugin interface for in-process inference middleware in IGW.

    Register via the ``nemo.inference_middleware`` entry-point group in your
    plugin's ``pyproject.toml``. The entry-point key is the plugin identity —
    it is what ``MiddlewareCall.name`` references in VirtualModel configs::

        [project.entry-points."nemo.inference_middleware"]
        nemo-switchyard = "nemo_switchyard_plugin.middleware:SwitchyardMiddleware"

    Both :meth:`process_request` and :meth:`process_response` are optional.
    Implement whichever phases your plugin participates in — request-only,
    response-only, or both. Unimplemented hooks pass through unchanged.

    Implement :meth:`get_middleware_config` if your VirtualModel configurations
    reference stored config entities via ``MiddlewareCall.config_id``. IGW calls
    this at VirtualModel create/update time and on each polling cycle — never
    per-request.

    Cache accessor methods (:meth:`get_model_providers_for_model`,
    :meth:`get_model_entity`, :meth:`list_model_entities_for_workspace`,
    :meth:`get_virtual_model`, :meth:`list_virtual_models_for_workspace`,
    :meth:`get_inference_url_and_model`, :meth:`get_backend_format`, and
    :meth:`get_openai_compatible_inference_url_and_model`) are injected by IGW before
    :meth:`on_startup` is called. Calling them before injection raises
    ``RuntimeError``.
    """

    def __init__(self) -> None:
        self._cache: InferenceMiddlewareCacheAccessor | None = None
        self._platform_sdk: AsyncNeMoPlatform | None = None
        self._platform_client: AsyncNemoClient | None = None

    # ------------------------------------------------------------------
    # IGW-called injection point (not part of the plugin author API)
    # ------------------------------------------------------------------

    def _inject_cache(self, cache: InferenceMiddlewareCacheAccessor) -> None:
        """Called by IGW to inject read-only cache access before on_startup().

        Plugin authors must not call this method directly.
        """
        self._cache = cache

    def _inject_platform_sdk(self, sdk: AsyncNeMoPlatform) -> None:
        """Called by IGW to inject a caller-owned SDK before on_startup().

        Plugin authors must not call or close this SDK directly.
        """
        self._platform_sdk = sdk

    def _inject_platform_client(self, client: AsyncNemoClient) -> None:
        """Called by IGW to inject a caller-owned typed client before on_startup().

        Plugin authors must not call or close this client directly.
        """
        self._platform_client = client

    def _get_platform_sdk(self, method_name: str) -> AsyncNeMoPlatform:
        if self._platform_sdk is None:
            raise RuntimeError(
                f"{method_name}() is not available before IGW injects the platform SDK. "
                f"Call {method_name}() from on_startup() or later."
            )
        return self._platform_sdk

    def _get_platform_client(self, method_name: str) -> AsyncNemoClient:
        if self._platform_client is None:
            raise RuntimeError(
                f"{method_name}() is not available before IGW injects the platform client. "
                f"Call {method_name}() from on_startup() or later."
            )
        return self._platform_client

    def _get_cache(self, method_name: str) -> InferenceMiddlewareCacheAccessor:
        if self._cache is None:
            raise RuntimeError(
                f"{method_name}() is not available before IGW injects the cache. "
                f"Call {method_name}() from on_startup() or later."
            )
        return self._cache

    # ------------------------------------------------------------------
    # Lifecycle hooks — override as needed; all default to no-ops
    # ------------------------------------------------------------------

    async def on_startup(self) -> None:
        """One-time initialization — load ML models, build HTTP clients, etc.

        Called once by IGW before the service starts handling requests.
        Cache accessor methods are available at this point.
        Default: no-op.
        """

    async def on_shutdown(self) -> None:
        """Cleanup on graceful shutdown.

        Called once by IGW after requests stop being served.
        Default: no-op.
        """

    async def on_virtual_model_upserted(self, virtual_model: VirtualModel) -> None:
        """Called when a VirtualModel referencing this plugin is created or updated.

        Use to pre-warm per-VirtualModel resources (e.g. load a RouteLLM
        classifier for this VirtualModel's strong/weak model pair, or
        pre-fetch and cache a guardrails config).
        Default: no-op.
        """

    async def on_virtual_model_destroyed(self, virtual_model: VirtualModel) -> None:
        """Called when a VirtualModel referencing this plugin is removed.

        Use to release resources that were warmed for that VirtualModel.
        Default: no-op.
        """

    # ------------------------------------------------------------------
    # IGW-injected cache accessors — available from on_startup() onward
    # ------------------------------------------------------------------

    def get_model_providers_for_model(self, model_entity_id: str) -> list[ModelProvider]:
        """Return all ModelProviders currently serving ``model_entity_id``.

        ``model_entity_id`` format: ``"workspace/name"``. Returns an empty
        list if the model entity is not in the cache.

        Primary use: latency-aware or availability-aware routing — inspect
        each provider's ``host_url`` to measure round-trip times and select
        the best backend, or detect that a backend is unavailable and failover.

        Raises:
            RuntimeError: If called before IGW has injected the cache.
        """
        return self._get_cache("get_model_providers_for_model").get_model_providers_for_model(model_entity_id)

    def get_model_entity(self, model_entity_id: str) -> ModelEntity | None:
        """Return the ModelEntity for ``model_entity_id``, or ``None`` if not found.

        ``model_entity_id`` format: ``"workspace/name"``.

        Primary use: inspecting model capabilities at startup or request time —
        e.g. checking ``entity.spec.is_chat`` to verify the model type is
        appropriate for the plugin's operation, or reading ``finetuning_type``
        to apply fine-tune-specific routing logic.

        Raises:
            RuntimeError: If called before IGW has injected the cache.
        """
        return self._get_cache("get_model_entity").get_model_entity(model_entity_id)

    def list_model_entities_for_workspace(self, workspace: str | None = None) -> list[str]:
        """Return model entity IDs (``"workspace/name"``) known to IGW.

        Pass ``workspace`` to filter to a single workspace; omit for all workspaces.

        Primary use: startup validation — confirm that model entities referenced
        in plugin config (e.g. ``strong_model``, ``weak_model``) actually exist
        before the first request arrives.

        Raises:
            RuntimeError: If called before IGW has injected the cache.
        """
        return self._get_cache("list_model_entities_for_workspace").list_model_entities_for_workspace(workspace)

    def get_virtual_model(self, virtual_model_id: str) -> VirtualModel | None:
        """Return the VirtualModel for ``virtual_model_id``, or ``None`` if not found.

        ``virtual_model_id`` format: ``"workspace/name"``.

        Primary use: a meta-routing plugin that routes between VirtualModels
        reads the target VirtualModel's ``default_model_entity`` and middleware
        chain before writing into ``body["model"]``.

        Raises:
            RuntimeError: If called before IGW has injected the cache.
        """
        return self._get_cache("get_virtual_model").get_virtual_model(virtual_model_id)

    def list_virtual_models_for_workspace(self, workspace: str) -> list[str]:
        """Return VirtualModel IDs (``"workspace/name"``) in ``workspace``.

        Primary use: a meta-routing plugin that discovers all VirtualModels
        in a workspace to build a routing table at startup.

        Raises:
            RuntimeError: If called before IGW has injected the cache.
        """
        return self._get_cache("list_virtual_models_for_workspace").list_virtual_models_for_workspace(workspace)

    def get_inference_url_and_model(
        self,
        model_entity_id: str,
        append_v1_suffix: bool = True,
    ) -> ModelProviderInferenceTarget:
        """Return the IGW provider gateway URL and served model name for ``model_entity_id``.

        Both values are resolved from the same provider (the first provider in
        IGW's model cache for this entity, consistent with how IGW itself selects
        a provider at proxy time), ensuring they are mutually consistent.

        Use ``result.model_provider_gateway_url`` as the base URL and
        ``result.served_model_name`` as the value for ``body["model"]``. Use
        :meth:`get_backend_format` separately when translating request or
        response bodies for a VirtualModel route.

        The provider gateway route bypasses virtual model routing and the middleware
        chain entirely — calling it from within a middleware plugin does not
        cause recursion::

            target = self.get_inference_url_and_model("ws/llama-70b")
            response = await httpx.post(
                f"{target.model_provider_gateway_url}/chat/completions",
                json={**body, "model": target.served_model_name},
            )

        For plugins that perform latency-aware routing across multiple providers,
        use :meth:`get_model_providers_for_model` to inspect all providers and
        construct per-provider URLs via the platform SDK instead.

        Args:
            model_entity_id: ``"workspace/name"`` of the model entity.
            append_v1_suffix: Append ``/v1`` to the provider gateway URL.
                Set ``False`` if the URL will be used with a path that includes
                its own version prefix, or if the provider URL already ends with
                ``/v1``. Defaults to ``True``.

        Raises:
            RuntimeError: If called before IGW has injected the cache.
            KeyError: If ``model_entity_id`` is not in the cache.
        """
        return self._get_cache("get_inference_url_and_model").get_inference_url_and_model(
            model_entity_id, append_v1_suffix
        )

    def get_backend_format(self, virtual_model_id: str, model_entity_id: str) -> BackendFormat | None:
        """Return the backend API format for ``model_entity_id`` within a VirtualModel route.

        Resolves a matching VirtualModel ``models`` entry override first, then
        falls back to the ModelEntity ``backend_format`` value. Returns ``None``
        when neither is set, leaving fallback behavior to the caller. Use this
        when a middleware plugin needs to translate request or response bodies
        for a backend selected from a VirtualModel.

        Args:
            virtual_model_id: ``"workspace/name"`` of the VirtualModel route.
            model_entity_id: ``"workspace/name"`` of the selected ModelEntity.

        Raises:
            RuntimeError: If called before IGW has injected the cache.
            KeyError: If either ID is invalid or not in the cache.
        """
        return self._get_cache("get_backend_format").get_backend_format(virtual_model_id, model_entity_id)

    def get_openai_compatible_inference_url_and_model(self, virtual_model_id: str) -> OpenAICompatibleInferenceTarget:
        """Return the OpenAI-compatible IGW URL and model for ``virtual_model_id``.

        Unlike :meth:`get_inference_url_and_model`, this helper does not resolve
        to a backend provider or served model name. It preserves IGW VirtualModel
        routing by returning an OpenAI-compatible base URL and the original
        workspace-qualified VirtualModel ID.

        Raises:
            RuntimeError: If called before IGW has injected the cache.
            KeyError: If ``virtual_model_id`` is not in the cache.
        """
        return self._get_cache(
            "get_openai_compatible_inference_url_and_model"
        ).get_openai_compatible_inference_url_and_model(virtual_model_id)

    # ------------------------------------------------------------------
    # Plugin-implemented config methods — IGW calls these on the plugin
    # ------------------------------------------------------------------

    async def get_middleware_config(self, config_type: str, config_id: str) -> Any:
        """Fetch a stored config entity of ``config_type`` with id ``config_id``.

        IGW calls this method at VirtualModel create/update time (to resolve
        ``MiddlewareCall.config_id`` references before caching) and on every
        polling cycle (to pick up config entity changes — no stale configs).
        It is never called at per-request time.

        Plugin authors implement this method to fetch from wherever they store
        configs — typically the entity store via ``NemoEntitiesClient``. Only
        needed if your plugin exposes its own ``NemoService`` CRUD API for config
        entities and users reference those entities via ``MiddlewareCall.config_id``.
        If all configs are inline (``MiddlewareCall.config``), this method need
        not be implemented.

        ``config_id`` format: ``"workspace/name"``.

        Args:
            config_type: The ``MiddlewareCall.config_type`` value — the
                ``entity_type`` of the plugin's config ``NemoEntity`` subclass
                (e.g. ``"routellm_config"``). Use this to dispatch to the right
                entity type when a plugin supports multiple config schemas.
            config_id: The ``"workspace/name"`` reference from
                :class:`~nemo_platform_plugin.inference_middleware.MiddlewareCall`.

        Returns:
            The resolved config object, in whatever form the plugin chooses.
            This value is passed to :meth:`validate_middleware_config`, and
            then the validated result is passed to :meth:`process_request` /
            :meth:`process_response` as the ``config`` argument.

        Raises:
            MiddlewareConfigNotFoundError: The referenced entity does not exist
                (deleted, or never created). IGW evicts any previously-resolved
                middleware for VirtualModels that reference this ``config_id``
                and marks them as broken until the config is recreated or the
                reference is removed. Plugins MUST raise this — not :class:`ValueError`
                — on a definitive 404 from their upstream store, otherwise IGW
                cannot distinguish deletion from a transient fetch failure and
                the previously-resolved config will persist indefinitely.
            ValueError: The entity exists but is malformed (missing required
                fields, invalid shape). IGW also evicts on this and treats it
                as caller-shape input.
            NotImplementedError: Default. IGW catches this and returns a 4xx
                to the caller informing them that ``config_id`` is not supported
                by this plugin. Override if your VirtualModel configurations use
                ``config_id``.

        Any other exception is treated as transient (network blip, plugin's
        upstream service unhealthy) — IGW preserves the previously-resolved
        config so a brief outage cannot flap the cache.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_middleware_config(). "
            f"Override this method if your MiddlewareCall entries use config_id, "
            f"or use inline config (MiddlewareCall.config) instead."
        )

    async def validate_middleware_config(self, config_type: str, config: Any) -> Any:
        """Validate and coerce a config object for ``config_type``.

        IGW calls this method at VirtualModel create/update time — both for
        inline ``config`` dicts and for objects returned by
        :meth:`get_middleware_config`. Use to enforce schema constraints and
        return a fully-typed config object.

        The validated return value is cached by IGW and passed verbatim to
        :meth:`process_request` or :meth:`process_response` as the ``config``
        argument. Never called at per-request time.

        Args:
            config_type: The ``MiddlewareCall.config_type`` value. Use this
                to dispatch to the right validation schema when a plugin
                supports multiple config types (e.g. ``"routellm_config"``
                vs. ``"streaming_data_log_config"`` for Switchyard).
            config: The raw config object — either an inline ``dict`` from
                ``MiddlewareCall.config``, or the object returned by
                :meth:`get_middleware_config`.

        Returns:
            The validated config (may be the original dict, a Pydantic model
            instance, or any typed object the plugin chooses).

        Raises:
            ValueError: If the config is invalid for this plugin and
                ``config_type``. IGW will reject the VirtualModel
                create/update request.

        Default: return ``config`` unchanged (pass-through). Override to
        add schema validation.
        """
        return config

    # ------------------------------------------------------------------
    # Request / response hooks — override as needed; both default to pass-through
    # ------------------------------------------------------------------

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: Any,
    ) -> InferenceRequest | ImmediateResponse:
        """Process the inference request before it is proxied to the ModelProvider.

        Receive ``request`` (body + headers + path), optionally modify it, and return
        the updated :class:`InferenceRequest`.  Mutate ``request.body``,
        ``request.headers``, and/or ``request.path`` in-place and return ``request``,
        or construct and return a new :class:`InferenceRequest`.

        Use ``ctx.state("my-plugin").set(key, value)`` to store data that
        ``process_response`` will need.  ``ctx.original_request`` provides a
        read-only snapshot of the request as it entered this plugin chain.

        Args:
            ctx: Per-request context — routing metadata, original request snapshot,
                and plugin scratch space.  ``ctx.proxied_request`` is ``None`` here
                (it is set by IGW after the request chain finishes).
            request: The current :class:`InferenceRequest`.  When called,
                ``request.body["model"]`` contains the model entity name written by
                IGW from ``VirtualModel.default_model_entity`` (if set).  Middleware
                may rewrite ``request.body["model"]`` to any valid model entity name
                and IGW will resolve and proxy it.  Middleware may also rewrite
                ``request.path`` to change the backend URI.
            middleware_config: The validated, cached config — the value returned
                by :meth:`validate_middleware_config` at VirtualModel cache-build
                time. Never resolved per-request.

        Returns:
            - :class:`InferenceRequest`: The modified request. IGW resolves
              ``request.body["model"]`` to a ``ModelProvider`` and proxies.
            - :class:`ImmediateResponse`: Plugin handled the request itself. IGW
              skips the proxy and passes ``data`` (a :data:`ResponseResult`) to
              the response middleware chain.

        Raises:
            InferenceMiddlewareUnavailableError: A service the plugin depends on
                is unavailable. IGW returns 503 by default.
            InferenceMiddlewareError: Any other handled error with a custom
                ``status_code``.

        Default: return ``request`` unchanged — IGW proxies the request as-is.
        """
        return request

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: Any,
    ) -> InferenceResponse:
        """Process the inference response after it is received from the ModelProvider.

        Receive ``response`` (result + headers, and possibly ``typed_result``),
        optionally modify it, and return the updated :class:`InferenceResponse`.

        ``ctx.original_request`` provides the read-only client request (available
        from the request phase). ``ctx.proxied_request`` provides the request that
        was forwarded to the backend (``None`` when an :class:`ImmediateResponse`
        short-circuited the proxy). Use ``ctx.state("my-plugin").get(key)`` to
        retrieve data stored during ``process_request``.

        Canonical response contract: for non-streaming responses, when
        ``response.typed_body`` is not ``None``, mutate or replace
        ``response.typed_body`` and leave ``response.result`` as the raw
        fallback. For streaming responses, ``response.typed_body`` is a
        middleware-only typed view and IGW serializes ``response.result``. When
        ``response.typed_body`` is ``None``, mutate ``response.result``. IGW
        does not synchronize the two views between plugins. If you need to return
        a non-streaming shape that does not fit the typed SDK schema, set
        ``response.typed_body = None`` and mutate ``response.result``.

        Args:
            ctx: Per-request context.  ``ctx.original_request`` is the client
                request (before any plugin ran).  ``ctx.proxied_request`` is what
                IGW forwarded to the backend (``None`` for :class:`ImmediateResponse`
                paths).
            response: The current :class:`InferenceResponse`.  ``response.result``
                is either a ``dict[str, Any]`` (non-streaming) or an
                ``AsyncIterator[dict[str, Any]]`` (streaming). For non-streaming
                responses, a non-``None`` ``response.typed_body`` is the canonical
                parsed SDK-native response object. For streaming responses,
                ``response.typed_body`` is a middleware-only parsed view and
                ``response.result`` is the outbound client stream. Modify
                ``response.headers`` to add or change headers returned to the caller.
            middleware_config: The validated, cached config (same semantics as in
                :meth:`process_request`).

        Returns:
            An :class:`InferenceResponse`, possibly with a modified ``result`` or
            ``headers``. Passed to the next plugin in the response middleware chain,
            or returned to the caller if this is the last plugin.

        Raises:
            InferenceMiddlewareUnavailableError: A service the plugin depends on
                is unavailable. IGW returns 503 by default.
            InferenceMiddlewareError: Any other handled error with a custom
                ``status_code``.

        Default: pass-through (return ``response`` unchanged).
        """
        return response
