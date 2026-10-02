# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for resolving inference backend API wire formats."""

from __future__ import annotations

from nemo_helix_plugin.inference_middleware import BackendFormat


def resolve_backend_format(model_entity_info: object, virtual_model: object | None = None) -> BackendFormat | None:
    """Resolve the inference backend API format for a selected model.

    IGW calls this after a request has resolved to a concrete ModelEntity, and
    optionally while it still has the VirtualModel route context. A VirtualModel
    may carry per-model inference config entries whose ``backend_format`` should
    override the ModelEntity value for requests that came through that route.
    If no valid override or entity value is present, this returns ``None`` and
    leaves any defaulting decision to the caller.

    The inputs are intentionally duck-typed because the call sites may hold
    different model classes depending on context: cached ``ModelEntityInfo``,
    generated SDK VirtualModel objects, plugin VirtualModel objects, or plain
    dictionaries in tests. Only ``workspace``, ``name``, ``backend_format``,
    and ``models``/``model`` fields are read.
    """
    qualified_model = f"{_get_field(model_entity_info, 'workspace')}/{_get_field(model_entity_info, 'name')}"
    for model_entry in _get_field(virtual_model, "models") or []:
        if _get_field(model_entry, "model") != qualified_model:
            continue
        backend_format = _backend_format_value(_get_field(model_entry, "backend_format"))
        if backend_format:
            return backend_format

    return _backend_format_value(_get_field(model_entity_info, "backend_format"))


def _get_field(value: object | None, field: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _backend_format_value(value: object | None) -> BackendFormat | None:
    if isinstance(value, BackendFormat):
        return value
    if isinstance(value, str):
        try:
            return BackendFormat(value)
        except ValueError:
            return None
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        try:
            return BackendFormat(enum_value)
        except ValueError:
            return None
    return None
