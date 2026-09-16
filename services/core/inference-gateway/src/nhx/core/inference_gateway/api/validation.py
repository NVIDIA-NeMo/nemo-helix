# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request validation helpers for entity names and workspaces.

Uses the same NAME_PATTERN as the entity store so that invalid names
return 422 Unprocessable Entity instead of 404 Not Found.
"""

import re

from fastapi import HTTPException, status
from nhx.common.entities.constants import NAME_PATTERN, NAME_PATTERN_DESCRIPTION

_ENTITY_NAME_PATTERN = re.compile(NAME_PATTERN)


def validate_entity_name(value: str, *, field_name: str = "name") -> None:
    """Raise 422 if value does not match entity store NAME_PATTERN.

    Args:
        value: The workspace or entity name to validate.
        field_name: Label for the field in the error detail.

    Raises:
        HTTPException: 422 if value does not match NAME_PATTERN.
    """
    if not _ENTITY_NAME_PATTERN.match(value):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid {field_name}: {NAME_PATTERN_DESCRIPTION}",
        )


def validate_model_entity_name(value: str, *, field_name: str = "model") -> None:
    """Raise 422 if value is not a valid model entity name.

    Allows simple names (NAME_PATTERN) or LoRA-style compound names
    (base&adapters/adapter-workspace/adapter-name).
    For compound names, each segment is validated with NAME_PATTERN.

    Args:
        value: The model_entity_name (may contain "&adapters/" for LoRA).
        field_name: Label for the field in the error detail.

    Raises:
        HTTPException: 422 if value is invalid.
    """
    if "&adapters/" in value:
        base, _, adapter_part = value.partition("&adapters/")
        if base and adapter_part and "/" in adapter_part:
            adapter_workspace, _, adapter_name = adapter_part.partition("/")
            if adapter_workspace and adapter_name:
                validate_entity_name(base, field_name=f"{field_name} (base)")
                validate_entity_name(adapter_workspace, field_name=f"{field_name} (adapter workspace)")
                validate_entity_name(adapter_name, field_name=f"{field_name} (adapter)")
                return
    validate_entity_name(value, field_name=field_name)


def validate_workspace_and_name(workspace: str, name: str) -> None:
    """Raise 422 if workspace or name does not match entity store NAME_PATTERN.

    Args:
        workspace: The workspace path parameter.
        name: The entity name path parameter (model entity or provider name).

    Raises:
        HTTPException: 422 if either value does not match NAME_PATTERN.
    """
    validate_entity_name(workspace, field_name="workspace")
    validate_entity_name(name, field_name="name")
