# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity relationship registry for cross-entity filter queries.

Relationships allow filtering entities based on attributes of related entities.
For example, filtering models by whether they have adapters with a specific
finetuning type.

Phase 0: Hardcoded registry. Phase 1: Dynamic registration via entity type schemas.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Relationship:
    """A named, typed link between two entity types."""

    kind: Literal["one_to_many"]
    target_entity_type: str
    via: str  # "parent" means child.parent == parent.id


_REGISTRY: dict[str, dict[str, Relationship]] = {
    "model": {
        "adapters": Relationship(
            kind="one_to_many",
            target_entity_type="adapter",
            via="parent",
        ),
    },
}


ENTITY_BASE_FIELDS = frozenset(
    {
        "id",
        "name",
        "entity_type",
        "workspace",
        "parent",
        "project",
        "created_at",
        "updated_at",
        "created_by",
        "updated_by",
        "db_version",
    }
)


def get_relationships(entity_type: str | None) -> dict[str, Relationship]:
    if entity_type is None:
        return {}
    return _REGISTRY.get(entity_type, {})


def resolve_child_field(field: str) -> str:
    """Auto-prefix with 'data.' for fields stored in JSONB.

    Base columns (name, created_at, etc.) are returned as-is.
    Domain fields (finetuning_type, status, etc.) become data.finetuning_type, etc.
    """
    if field in ENTITY_BASE_FIELDS or field.startswith("data."):
        return field
    return f"data.{field}"
