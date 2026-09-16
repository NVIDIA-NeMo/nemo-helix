# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format mappings between nemo_helix_plugin and Switchyard, and to API paths."""

from __future__ import annotations

from typing import Any

from nemo_helix_plugin.inference_middleware import InferenceMiddlewareError, VirtualModel
from switchyard.lib.chat_request.anthropic import AnthropicChatRequest
from switchyard.lib.chat_request.base import ChatRequest
from switchyard.lib.chat_request.openai_chat import OpenAIChatRequest
from switchyard.lib.chat_request.openai_responses import ResponsesChatRequest

# Map ChatRequestType values (from Switchyard) to API paths
# Values are from ChatRequestType enum (openai_chat, anthropic, openai_responses, etc.)
FORMAT_TO_PATH: dict[str, str] = {
    "openai_chat": "v1/chat/completions",
    "openai_responses": "v1/responses",
    "anthropic": "v1/messages",
}

# Valid nemo_helix_plugin BackendFormat values
_VALID_NEMO_FORMATS = {"OPENAI_CHAT", "ANTHROPIC_MESSAGES", "AUTO"}

# Map nemo_helix_plugin BackendFormat → switchyard BackendFormat
# nemo_helix_plugin uses uppercase, switchyard uses lowercase
_NEMO_TO_SWITCHYARD_FORMAT: dict[str, str] = {
    "OPENAI_CHAT": "openai",
    "ANTHROPIC_MESSAGES": "anthropic",
    "AUTO": "auto",
}


def vm_model_format_map(virtual_model: VirtualModel) -> dict[str, str]:
    """Extract model→backend_format mapping from VirtualModel.

    Returns a dict mapping model names to nemo_helix_plugin backend_format strings.
    Raises ValueError on unknown formats so VM upsert fails fast with a clear error.
    """
    result: dict[str, str] = {}
    for entry in virtual_model.models:
        if entry.backend_format is None:
            continue
        fmt_str = entry.backend_format.value
        if fmt_str not in _VALID_NEMO_FORMATS:
            raise ValueError(
                f"Unknown backend_format {fmt_str!r} on "
                f"VirtualModel {virtual_model.workspace}/{virtual_model.name} "
                f"model {entry.model!r}. Expected one of {sorted(_VALID_NEMO_FORMATS)}."
            )
        result[entry.model] = fmt_str
    return result


def to_switchyard_format(nemo_format: str) -> str:
    """Convert nemo_helix_plugin BackendFormat (uppercase) to switchyard BackendFormat (lowercase)."""
    return _NEMO_TO_SWITCHYARD_FORMAT.get(nemo_format, nemo_format.lower())


def vm_models_for_switchyard(virtual_model: VirtualModel) -> list[dict[str, Any]]:
    """Build the models list expected by Switchyard factories from a VirtualModel."""
    format_map = vm_model_format_map(virtual_model)
    return [{"model": model, "backend_format": to_switchyard_format(fmt)} for model, fmt in format_map.items()]


# Map API path → ChatRequest constructor, normalised (no leading slash).
_PATH_TO_CHAT_REQUEST: dict[str, type[ChatRequest]] = {
    "v1/chat/completions": OpenAIChatRequest,
    "v1/messages": AnthropicChatRequest,
    "v1/responses": ResponsesChatRequest,
}


def build_chat_request(path: str, body: dict[str, Any]) -> ChatRequest:
    """Construct the appropriate Switchyard ``ChatRequest`` for *path*.

    Dispatches on the normalised path (leading slash stripped) and wraps
    *body* in the matching ``ChatRequest`` subclass.  *body* is passed by
    reference — the ``ChatRequest`` stores the same dict object, so any
    mutations the Switchyard pipeline makes to ``chat_request.body`` are
    directly visible via the original dict.

    Raises:
        InferenceMiddlewareError (500): *path* is not a recognised API path.
    """
    cls = _PATH_TO_CHAT_REQUEST.get(path.lstrip("/"))
    if cls is None:
        raise InferenceMiddlewareError(
            f"Cannot build a Switchyard ChatRequest for unrecognised path {path!r}. "
            f"Recognised paths: {sorted(_PATH_TO_CHAT_REQUEST)}.",
            status_code=500,
        )
    return cls(body)
