# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway middleware that keeps a VirtualModel closed-book.

Closed-book means the model answers only from the request: no provider-executed
tool (hosted web search, fetch, MCP, code execution) and no model that searches on
its own. Client-executed tools are kept, because the agent runs them inside its
sandbox, where egress policy already applies.
"""

from __future__ import annotations

import copy
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nemo_evals.gateway.policy import ModelAccess
from nemo_evals.gateway.tokens import TrialTokenReader, claims_evals_credential, platform_token_reader
from nemo_helix_plugin.auth import inference_gateway_credential, platform_auth_enabled, verified_access_key_claims
from nemo_helix_plugin.inference_middleware import (
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceMiddlewareUnavailableError,
    InferenceRequest,
    MiddlewareCall,
    ModelProviderInferenceTarget,
    NemoInferenceMiddleware,
)
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

MIDDLEWARE_NAME = "nemo-evals-closed-book"
CONFIG_TYPE = "closed_book"
REMOVED_TOOLS_STATE_KEY = "removed_tools"

_CLIENT_TOOL_TYPE = re.compile(
    r"^(?:function|custom|namespace|local_shell|shell|apply_patch|computer|computer_use_preview"
    r"|(?:bash|text_editor|str_replace_based_edit_tool|memory|computer)_\d{8})$"
)
_TOOL_CHOICE_MODES = frozenset({"auto", "any", "none", "required"})
_PROVIDER_FEATURE_KEYS = (
    "container",
    "mcp_servers",
    "plugins",
    "search_parameters",
    "web_search_options",
)
_PROVIDER_STORED_HISTORY_KEYS = ("conversation", "previous_response_id")
_PLATFORM_KEY_CACHE_SEC = 60.0
_MAX_CACHED_PLATFORM_KEYS = 1024
_SELF_SEARCHING_MODEL = re.compile(
    r"(?:^|/)perplexity/|(?:^|[/_-])(?:sonar|search)(?:-|$)|deep-research|:online$", re.IGNORECASE
)


def ends_closed_book(request_middleware: Sequence[MiddlewareCall]) -> bool:
    """Whether a VirtualModel's request middleware ends with the closed-book middleware."""
    return bool(request_middleware) and request_middleware[-1].name == MIDDLEWARE_NAME


class ClosedBookConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def client_tool_label(tool: object) -> str | None:
    """Return ``None`` for a client-executed tool, else a label naming what was found."""
    if not isinstance(tool, dict):
        return type(tool).__name__
    kind = tool.get("type")
    if kind is None:
        return None if "input_schema" in tool else next(iter(tool), "untyped")
    if not (isinstance(kind, str) and _CLIENT_TOOL_TYPE.match(kind)):
        return str(kind)
    environment = tool.get("environment")
    if kind == "shell" and environment is not None:
        environment_type = environment.get("type") if isinstance(environment, dict) else type(environment).__name__
        if environment_type != "local":
            return f"shell:{environment_type}"
    return None


def is_self_searching_model(model: object) -> bool:
    return isinstance(model, str) and bool(_SELF_SEARCHING_MODEL.search(model))


def strip_hosted_tools(body: MutableMapping[str, Any], *, anthropic: bool = False) -> list[str]:
    """Remove provider-executed tools and features from ``body`` in place; return a label for each removal."""
    removed: list[str] = []
    kept = _KeptTools()
    had_tools = bool(body.get("tools"))
    _strip_list(body, "tools", removed, kept)
    input_items = body.get("input")
    if isinstance(input_items, list):
        for item in input_items:
            if isinstance(item, dict):
                _strip_list(item, "tools", removed, kept)
    for key in _PROVIDER_FEATURE_KEYS:
        if key in body:
            del body[key]
            removed.append(key)
    if isinstance(body.get("prompt"), dict):
        del body["prompt"]
        removed.append("prompt")
    if had_tools and not body["tools"]:
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            body.pop(key, None)
        return removed
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and not _tool_choice_survives(choice, kept, removed):
        body["tool_choice"] = {"type": "auto"} if anthropic else "auto"
    return removed


@dataclass
class _KeptTools:
    names: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)


def _strip_list(container: MutableMapping[str, Any], key: str, removed: list[str], kept: _KeptTools) -> None:
    tools = container.get(key)
    if not isinstance(tools, list):
        return
    kept_tools = []
    for tool in tools:
        label = client_tool_label(tool)
        if label is not None:
            removed.append(label)
            continue
        _strip_list(tool, "tools", removed, kept)
        kept_tools.append(tool)
        name = _tool_name(tool)
        if isinstance(name, str):
            kept.names.add(name)
        if isinstance(tool.get("type"), str):
            kept.types.add(tool["type"])
    container[key] = kept_tools


def _tool_name(tool: MutableMapping[str, Any]) -> object:
    function = tool.get("function")
    return tool.get("name") or (function.get("name") if isinstance(function, dict) else None)


def _tool_choice_survives(choice: MutableMapping[str, Any], kept: _KeptTools, removed: list[str]) -> bool:
    kind = choice.get("type")
    if kind in _TOOL_CHOICE_MODES:
        return True
    if kind == "allowed_tools":
        _strip_list(choice, "tools", removed, _KeptTools())
        return bool(choice.get("tools"))
    name = _tool_name(choice)
    if kind in {"function", "tool", "custom"}:
        survives = name in kept.names
    else:
        survives = isinstance(kind, str) and bool(_CLIENT_TOOL_TYPE.match(kind)) and kind in kept.types
    if not survives:
        removed.append(f"tool_choice:{name or kind}")
    return survives


PlatformKeyVerifier = Callable[[str], Awaitable[Mapping[str, Any] | None]]


class RouterCredentialCheck:
    """Whether a provider that calls back into the gateway is held to a closed model policy.

    A provider such as a router deployment holds a platform credential and calls other models back through the
    gateway on its caller's behalf. Closed-book can't see those calls, so it serves such a provider only when its
    credential is a closed evals router key, which the gateway holds to closed-book models. Any other provider
    credential, such as a model provider's own API key, is not a platform credential and passes.
    """

    def __init__(
        self,
        reader: TrialTokenReader,
        verify_platform_key: PlatformKeyVerifier | None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._reader = reader
        self._verify_platform_key = verify_platform_key
        self._clock = clock
        self._platform_keys: dict[str, tuple[float, bool]] = {}

    async def refusal(self, provider_headers: Mapping[str, str], model: str) -> str | None:
        """Why the provider sending ``provider_headers`` can't serve ``model`` closed-book; ``None`` when it can."""
        credential = inference_gateway_credential(provider_headers)
        if credential is None:
            return None
        if claims_evals_credential(credential):
            claims = await self._reader.claims(credential)
            if claims is not None and claims.model_access is ModelAccess.CLOSED:
                return None
        elif not await self._is_platform_key(credential):
            return None
        return (
            f"The provider serving {model!r} calls models back through the gateway with a credential that isn't a "
            "closed evals router key, so the models it calls might not be closed-book."
        )

    async def _is_platform_key(self, credential: str) -> bool:
        if self._verify_platform_key is None:
            return False
        now = self._clock()
        cached = self._platform_keys.get(credential)
        if cached is not None and now < cached[0]:
            return cached[1]
        is_platform_key = await self._verify_platform_key(credential) is not None
        if len(self._platform_keys) >= _MAX_CACHED_PLATFORM_KEYS:
            self._platform_keys.clear()
        self._platform_keys[credential] = (now + _PLATFORM_KEY_CACHE_SEC, is_platform_key)
        return is_platform_key


def references_stored_items(body: Mapping[str, Any]) -> bool:
    """Whether a Responses request's ``input`` refers to items the provider stored, by ``item_reference``."""
    items = body.get("input")
    return isinstance(items, list) and any(
        isinstance(item, Mapping) and item.get("type") == "item_reference" for item in items
    )


class ClosedBookMiddleware(NemoInferenceMiddleware):
    """Strip provider-executed tools and refuse self-searching models.

    Place it last in ``request_middleware`` so it sees the model a routing
    middleware selected.
    """

    def __init__(self) -> None:
        super().__init__()
        self._router_check: RouterCredentialCheck | None = None

    def _router_credentials(self) -> RouterCredentialCheck:
        if self._router_check is None:
            self._router_check = RouterCredentialCheck(
                platform_token_reader(), verified_access_key_claims if platform_auth_enabled() else None
            )
        return self._router_check

    def supports_middleware_phase(self, phase: str) -> bool:
        return phase == "request"

    async def validate_middleware_config(self, config_type: str, config: Any) -> ClosedBookConfig:
        if config_type != CONFIG_TYPE:
            raise ValueError(f"{MIDDLEWARE_NAME} supports config_type {CONFIG_TYPE!r}, not {config_type!r}")
        return ClosedBookConfig.model_validate(config or {})

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: ClosedBookConfig,
    ) -> InferenceRequest:
        model = request.body.get("model")
        if not isinstance(model, str):
            raise _unresolved_model(ctx, model)
        target = self._provider_target(ctx, model)
        if any(
            is_self_searching_model(name)
            for name in (model.split("/", 1)[-1], target.served_model_name, target.required_extra_body.get("model"))
        ):
            raise _refusal(
                ctx,
                model,
                "self_searching_model",
                f"Model {model!r} searches the web on its own, so it cannot be served closed-book.",
            )
        if any(
            not label.startswith("tool_choice:")
            for policy in (target.default_extra_body, target.required_extra_body)
            for label in strip_hosted_tools(copy.deepcopy(dict(policy)))
        ):
            raise _refusal(
                ctx,
                model,
                "provider_adds_hosted_tools",
                f"The provider serving {model!r} adds provider-executed tools or features to requests, "
                "so it cannot be served closed-book.",
            )
        try:
            router_refusal = await self._router_credentials().refusal(target.outbound_headers, model)
        except Exception as exc:
            logger.warning("Couldn't check the credential of the provider serving %r", model, exc_info=True)
            raise InferenceMiddlewareUnavailableError(
                f"The credential of the provider serving {model!r} couldn't be checked."
            ) from exc
        if router_refusal is not None:
            raise _refusal(ctx, model, "router_credential_not_closed", router_refusal)
        forwarded = {**target.default_extra_body, **request.body, **target.required_extra_body}
        stored = [key for key in _PROVIDER_STORED_HISTORY_KEYS if forwarded.get(key) is not None]
        if references_stored_items(forwarded):
            stored.append("item_reference input items")
        if stored:
            raise _refusal(
                ctx,
                model,
                "provider_stored_history",
                f"The request continues provider-stored history through {', '.join(stored)}, which closed-book "
                "cannot inspect; send the conversation history in the request instead.",
            )
        removed = strip_hosted_tools(request.body, anthropic=request.path.lstrip("/").startswith("v1/messages"))
        ctx.state(MIDDLEWARE_NAME).set(REMOVED_TOOLS_STATE_KEY, removed)
        if removed:
            logger.info(
                "Closed-book middleware removed hosted tools",
                extra={
                    "request_id": ctx.request_id,
                    "workspace": ctx.workspace,
                    "virtual_model": ctx.virtual_model_name,
                    "removed_tools": removed,
                },
            )
        return request

    def _provider_target(self, ctx: InferenceMiddlewareContext, model: str) -> ModelProviderInferenceTarget:
        try:
            return self.get_inference_url_and_model(model)
        except KeyError:
            raise _unresolved_model(ctx, model) from None


def _unresolved_model(ctx: InferenceMiddlewareContext, model: object) -> InferenceMiddlewareError:
    return _refusal(
        ctx,
        model,
        "unresolved_model",
        f"Model {model!r} does not resolve to a model provider, so closed-book cannot check it.",
    )


def _refusal(ctx: InferenceMiddlewareContext, model: object, reason: str, detail: str) -> InferenceMiddlewareError:
    logger.warning(
        "Closed-book middleware refused a request",
        extra={
            "request_id": ctx.request_id,
            "workspace": ctx.workspace,
            "virtual_model": ctx.virtual_model_name,
            "model": model,
            "reason": reason,
        },
    )
    return InferenceMiddlewareError(detail, status_code=403)
