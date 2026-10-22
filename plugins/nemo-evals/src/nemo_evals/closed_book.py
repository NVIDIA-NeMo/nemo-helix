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
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

from nemo_helix_plugin.inference_middleware import (
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceRequest,
    ModelProviderInferenceTarget,
    NemoInferenceMiddleware,
)
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

MIDDLEWARE_NAME = "nemo-evals-closed-book"
CONFIG_TYPE = "closed_book"
REMOVED_TOOLS_STATE_KEY = "removed_tools"

_CLIENT_TOOL_TYPE = re.compile(
    r"^(?:function|custom|namespace|local_shell|shell|apply_patch"
    r"|(?:bash|text_editor|str_replace_based_edit_tool|memory)_\d{8})$"
)
_TOOL_CHOICE_MODES = frozenset({"auto", "any", "none", "required"})
_PROVIDER_FEATURE_KEYS = (
    "container",
    "mcp_servers",
    "plugins",
    "prompt",
    "search_parameters",
    "web_search_options",
)
_SELF_SEARCHING_MODEL = re.compile(r"(?:^|/)perplexity/|sonar|search|:online$", re.IGNORECASE)


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
        return True
    name = _tool_name(choice)
    if kind in {"function", "tool", "custom"}:
        survives = name in kept.names
    else:
        survives = isinstance(kind, str) and bool(_CLIENT_TOOL_TYPE.match(kind)) and kind in kept.types
    if not survives:
        removed.append(f"tool_choice:{name or kind}")
    return survives


class ClosedBookMiddleware(NemoInferenceMiddleware):
    """Strip provider-executed tools and refuse self-searching models.

    Place it last in ``request_middleware`` so it sees the model a routing
    middleware selected.
    """

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
        target = self._provider_target(model)
        if any(
            is_self_searching_model(name)
            for name in (model, target.served_model_name, target.required_extra_body.get("model"))
        ):
            raise InferenceMiddlewareError(
                f"Model {model!r} searches the web on its own, so it cannot be served closed-book.",
                status_code=403,
            )
        if any(
            strip_hosted_tools(copy.deepcopy(dict(policy)))
            for policy in (target.default_extra_body, target.required_extra_body)
        ):
            raise InferenceMiddlewareError(
                f"The provider serving {model!r} adds provider-executed tools or features to requests, "
                "so it cannot be served closed-book.",
                status_code=403,
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

    def _provider_target(self, model: object) -> ModelProviderInferenceTarget:
        try:
            if isinstance(model, str):
                return self.get_inference_url_and_model(model)
        except KeyError:
            pass
        raise InferenceMiddlewareError(
            f"Model {model!r} does not resolve to a model provider, so closed-book cannot check it.",
            status_code=403,
        )
