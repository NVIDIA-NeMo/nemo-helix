# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydrate entity-store metadata omitted from Pydantic private attributes."""

from datetime import datetime
from typing import TypeVar

from nemo_platform_plugin.entity import NemoEntity

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
    entity = entity_type.model_validate(data)
    entity._id = _optional_str(data.get("id"))
    entity._parent = _optional_str(data.get("parent"))
    entity._created_by = _optional_str(data.get("created_by"))
    entity._updated_by = _optional_str(data.get("updated_by"))
    entity._created_at = _optional_datetime(data.get("created_at"))
    entity._updated_at = _optional_datetime(data.get("updated_at"))
    db_version = data.get("db_version")
    if isinstance(db_version, int):
        entity._db_version = db_version
    return entity


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


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value)
    return None
