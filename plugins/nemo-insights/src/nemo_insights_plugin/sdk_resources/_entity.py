# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydrate entity-store metadata omitted from Pydantic private attributes."""

from typing import TypeVar

from nemo_helix_plugin.client.types import RESPONSE_VALIDATION_CONTEXT
from nemo_helix_plugin.entity import NemoEntity

_EntityT = TypeVar("_EntityT", bound=NemoEntity)


def object_dict(value: object) -> dict[str, object] | None:
    """Return a string-keyed object dict from raw JSON-like data."""
    if not isinstance(value, dict):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(key, str):
            result[key] = item
    return result


def entity_from_response(entity_type: type[_EntityT], data: dict[str, object]) -> _EntityT:
    """Parse an entity response and restore its store-managed metadata."""
    return entity_type.model_validate(data, context=dict(RESPONSE_VALIDATION_CONTEXT))


def hydrate_page(items: list[_EntityT], raw_items: object) -> None:
    """Restore metadata on validated page items in response order."""
    if not isinstance(raw_items, list):
        return
    for item, raw in zip(items, raw_items, strict=True):
        raw_data = object_dict(raw)
        if raw_data is None:
            continue
        hydrated = entity_from_response(type(item), raw_data)
        item.__pydantic_private__ = hydrated.__pydantic_private__
