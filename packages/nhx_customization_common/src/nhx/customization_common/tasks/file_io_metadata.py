# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for stamping output fileset metadata during customization uploads."""

from typing import Any


def extract_tool_calling_metadata(model_entity: Any) -> dict | None:
    """Extract tool_calling fields from a model entity for output fileset metadata."""
    spec = getattr(model_entity, "spec", None)
    if spec is None:
        return None

    tool_calling: dict[str, Any] = {}

    chat_template = getattr(spec, "chat_template", None)
    if chat_template:
        tool_calling["chat_template"] = chat_template

    tcc = getattr(spec, "tool_call_config", None)
    if tcc is not None:
        if getattr(tcc, "tool_call_parser", None):
            tool_calling["tool_call_parser"] = tcc.tool_call_parser
        if getattr(tcc, "tool_call_plugin", None):
            tool_calling["tool_call_plugin"] = tcc.tool_call_plugin
        if getattr(tcc, "auto_tool_choice", None) is not None:
            tool_calling["auto_tool_choice"] = tcc.auto_tool_choice

    return tool_calling or None


def build_model_fileset_metadata(*, tool_calling: dict | None = None) -> dict | None:
    """Build a ``FilesetMetadata`` dict for model-purpose filesets."""
    if not tool_calling:
        return None
    return {"model": {"tool_calling": tool_calling}}


def build_output_fileset_metadata_from_model_entity(model_entity: Any) -> dict | None:
    """Build output fileset metadata by propagating tool_calling from a source model entity."""
    return build_model_fileset_metadata(tool_calling=extract_tool_calling_metadata(model_entity))
