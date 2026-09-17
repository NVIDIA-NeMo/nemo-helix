# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-scoped Nooa clients backed by configured Platform Model Entities.

This module is an optional integration: consumers must depend on ``nooa``.
Keeping the integration here lets plugins share Platform config, routing, auth,
and client lifetime behavior without making Nooa a required dependency of the
public plugin contract.
"""

import inspect
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from nemo_platform_plugin.client.auth import AsyncTokenProvider, TokenProvider, resolve_token_async
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.config.config import get_context
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.models.refs import parse_workspace_name_ref
from nemo_platform_plugin.models.types import ModelEntity, ModelProvider
from nooa.unifiedllm import CompletionClient, LLMResponse, Tool, UnifiedLLM
from pydantic import BaseModel

_PLACEHOLDER_API_KEY = "not-needed"
_OPENAI_FORMAT = "OPENAI_CHAT"
_ANTHROPIC_FORMAT = "ANTHROPIC_MESSAGES"
_ACCEPT_ENCODING_HEADER = "accept-encoding"
_IDENTITY_ENCODING = "identity"
_AUTHORIZATION_HEADER = "Authorization"


@dataclass(frozen=True)
class ConfiguredModelRefs:
    """Workspace-qualified Model Entity IDs for default and fast agent work."""

    default: str
    fast: str


@dataclass(frozen=True)
class ConfiguredModelClients:
    """Resolved Nooa clients for default and low-latency agent work."""

    default: UnifiedLLM
    fast: UnifiedLLM
    refs: ConfiguredModelRefs | None = None

    async def aclose(self) -> None:
        """Close each distinct client owned by this pair."""
        close_error: Exception | None = None
        clients = (self.default,) if self.fast is self.default else (self.default, self.fast)
        for client in clients:
            try:
                await client.aclose()
            except Exception as exc:
                if close_error is None:
                    close_error = exc
                else:
                    close_error.add_note(f"Another model client also failed to close: {exc!r}")
        if close_error is not None:
            raise close_error


_ACTIVE_MODEL_CLIENTS: ContextVar[ConfiguredModelClients | None] = ContextVar(
    "nemo_platform_active_model_clients",
    default=None,
)


def configured_model_refs() -> ConfiguredModelRefs:
    """Read the default/fast Model Entity pair from the active CLI context."""
    context = get_context()
    default = context.default_model
    fast = context.fast_model or default
    if not default:
        raise ValueError("No default model is configured. Run `nemo setup` and select agent models.")
    if not fast:
        raise ValueError("No fast model is configured. Run `nemo setup` and select agent models.")
    _validate_configured_ref(default, "NEMO_DEFAULT_MODEL")
    _validate_configured_ref(fast, "NEMO_FAST_MODEL")
    return ConfiguredModelRefs(default=default, fast=fast)


def _validate_configured_ref(model_ref: str, env_var: str) -> None:
    """Reject an unqualified model ref where the operator can still act on it.

    Model Entity IDs discovered by ``nemo setup`` are always ``workspace/name``, but
    the ``NEMO_DEFAULT_MODEL`` / ``NEMO_FAST_MODEL`` overrides are read verbatim. An
    unqualified value used to survive every layer and only fail inside a running job,
    long after the job was created, so the operator saw a stack trace instead of the
    one-line fix.
    """
    try:
        parse_workspace_name_ref(model_ref, label="Model reference")
    except ValueError:
        raise ValueError(
            f"Model reference {model_ref!r} must use workspace/name format "
            f"(for example 'default/{model_ref}'). "
            f"Set {env_var} to a workspace-qualified Model Entity ID, or unset it and "
            f"re-run `nemo setup` to select one."
        ) from None


def _normalize_extra_headers(headers: object, *, label: str) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        raise TypeError(f"{label} must be a mapping of string headers")
    normalized: dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError(f"{label} must contain only string header names and values")
        normalized[key] = value
    return normalized


class _AuthenticatedCompletionClient(CompletionClient):
    """CompletionClient that resolves Platform auth at each Nooa call boundary."""

    def __init__(
        self,
        model: str,
        *,
        platform_auth: TokenProvider | AsyncTokenProvider,
        **config,
    ) -> None:
        self._platform_auth = platform_auth
        super().__init__(model, **config)

    def _extra_headers(self, token: str, call_headers: object) -> dict[str, str]:
        headers = _normalize_extra_headers(self.config.get("extra_headers"), label="CompletionClient extra_headers")
        headers.update(_normalize_extra_headers(call_headers, label="Per-call extra_headers"))
        headers = {key: value for key, value in headers.items() if key.lower() != _AUTHORIZATION_HEADER.lower()}
        headers[_AUTHORIZATION_HEADER] = f"Bearer {token}"
        return headers

    def _sync_access_token(self) -> str:
        get_token = self._platform_auth.get_access_token
        if inspect.iscoroutinefunction(get_token):
            raise TypeError("Async token provider cannot be used from a synchronous Nooa completion call")
        token = get_token()
        if inspect.isawaitable(token):
            raise TypeError("Async token provider cannot be used from a synchronous Nooa completion call")
        return token

    def call(
        self,
        messages: list[dict[str, object]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        cache_control_injection_points: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        kwargs["extra_headers"] = self._extra_headers(self._sync_access_token(), kwargs.get("extra_headers"))
        return super().call(
            messages,
            tools=tools,
            output_model=output_model,
            cache_control_injection_points=cache_control_injection_points,
            **kwargs,
        )

    async def acall(
        self,
        messages: list[dict[str, object]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        cache_control_injection_points: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        token = await resolve_token_async(self._platform_auth)
        kwargs["extra_headers"] = self._extra_headers(token, kwargs.get("extra_headers"))
        return await super().acall(
            messages,
            tools=tools,
            output_model=output_model,
            cache_control_injection_points=cache_control_injection_points,
            **kwargs,
        )


def _platform_completion_client(
    model: str,
    *,
    platform_auth: TokenProvider | AsyncTokenProvider | None,
    **config,
) -> CompletionClient:
    if platform_auth is None:
        return CompletionClient(model, **config)
    return _AuthenticatedCompletionClient(model, platform_auth=platform_auth, **config)


def _completion_client(
    models_client: AsyncModelsClient,
    model_entity: ModelEntity,
    served_model_name: str,
) -> CompletionClient:
    """Build a Nooa client that routes through the Model Entity's Platform URL."""
    api_base = models_client.get_model_entity_route_openai_url(model_entity)
    extra_headers = dict(models_client.default_headers)
    # Inference Gateway's direct passthrough session preserves compressed
    # response bytes. Nooa needs decoded JSON/SSE, so make that requirement
    # explicit at this adapter boundary for every configured agent client.
    extra_headers[_ACCEPT_ENCODING_HEADER] = _IDENTITY_ENCODING
    # Backend format is the Platform-facing wire contract, not the upstream
    # provider identity. The LiteLLM prefix selects the adapter for that shape.
    platform_auth = models_client._auth
    if model_entity.backend_format == _OPENAI_FORMAT:
        litellm_model = f"openai/{served_model_name}"
        return _platform_completion_client(
            litellm_model,
            platform_auth=platform_auth,
            api_base=api_base,
            api_key=_PLACEHOLDER_API_KEY,
            # Platform rewrites the response model to the Model Entity ID. Keep
            # the provider's authoritative served name available to LiteLLM for
            # post-response capability and cost lookup.
            base_model=litellm_model,
            extra_headers=extra_headers,
            drop_params=True,
            # The configured backend contract is explicitly OPENAI_CHAT. Newer
            # LiteLLM versions otherwise bridge GPT-5.4+ tool calls with any
            # reasoning_effort value (including "none") to /responses.
            _skip_responses_api_bridge=True,
        )
    elif model_entity.backend_format == _ANTHROPIC_FORMAT:
        api_base = api_base.removesuffix("/v1")
    else:
        raise ValueError(
            f"Model '{model_entity.workspace}/{model_entity.name}' has unsupported backend format "
            f"{model_entity.backend_format!r}; expected {_OPENAI_FORMAT} or {_ANTHROPIC_FORMAT}"
        )

    litellm_model = f"anthropic/{served_model_name}"
    return _platform_completion_client(
        litellm_model,
        platform_auth=platform_auth,
        api_base=api_base,
        api_key=_PLACEHOLDER_API_KEY,
        base_model=litellm_model,
        extra_headers=extra_headers,
        drop_params=True,
    )


async def _served_model_name(
    client: AsyncModelsClient,
    model_entity: ModelEntity,
    provider_cache: dict[str, ModelProvider],
) -> str:
    """Return the provider-recorded model name, without inferring from the entity ID."""
    entity_ref = f"{model_entity.workspace}/{model_entity.name}"
    for provider_ref in model_entity.model_providers or ():
        provider = provider_cache.get(provider_ref)
        if provider is None:
            workspace, name = parse_workspace_name_ref(provider_ref, label="Provider reference")
            provider = (await client.get_provider(name=name, workspace=workspace)).data()
            provider_cache[provider_ref] = provider
        for mapping in provider.served_models or ():
            if mapping.model_entity_id == entity_ref:
                return mapping.served_model_name

    if model_entity.api_endpoint is not None and model_entity.api_endpoint.model_id:
        return model_entity.api_endpoint.model_id
    return model_entity.name


async def resolve_model_clients(
    client: AsyncNemoClient | AsyncModelsClient,
    refs: ConfiguredModelRefs | None = None,
) -> ConfiguredModelClients:
    """Resolve configured Model Entities and construct each distinct client once."""
    models_client = client if isinstance(client, AsyncModelsClient) else AsyncModelsClient.from_client(client)
    selected = refs or configured_model_refs()
    resolved: dict[str, UnifiedLLM] = {}
    provider_cache: dict[str, ModelProvider] = {}
    try:
        for model_ref in (selected.default, selected.fast):
            if model_ref in resolved:
                continue
            workspace, name = parse_workspace_name_ref(model_ref, label="Model reference")
            entity = (await models_client.get_model(name=name, workspace=workspace)).data()
            served_model_name = await _served_model_name(models_client, entity, provider_cache)
            resolved[model_ref] = _completion_client(models_client, entity, served_model_name)
    except Exception as resolution_error:
        for model_client in resolved.values():
            try:
                await model_client.aclose()
            except Exception as cleanup_error:
                resolution_error.add_note(f"A constructed model client also failed to close: {cleanup_error!r}")
        raise
    return ConfiguredModelClients(
        default=resolved[selected.default],
        fast=resolved[selected.fast],
        refs=selected,
    )


@contextmanager
def activate_model_clients(model_clients: ConfiguredModelClients) -> Iterator[None]:
    """Expose a resolved pair to agents constructed within this run context."""
    token = _ACTIVE_MODEL_CLIENTS.set(model_clients)
    try:
        yield
    finally:
        _ACTIVE_MODEL_CLIENTS.reset(token)


def _active_model_clients() -> ConfiguredModelClients:
    model_clients = _ACTIVE_MODEL_CLIENTS.get()
    if model_clients is None:
        raise RuntimeError("Configured model clients were not activated for this agent run")
    return model_clients


def get_default_model() -> UnifiedLLM:
    """Return the active default model client."""
    return _active_model_clients().default


def get_fast_model() -> UnifiedLLM:
    """Return the active low-latency model client."""
    return _active_model_clients().fast


def get_configured_model_refs() -> ConfiguredModelRefs:
    """Return the model references resolved for the active agent run."""
    refs = _active_model_clients().refs
    if refs is None:
        raise RuntimeError("Configured model references are unavailable for this agent run")
    return refs
