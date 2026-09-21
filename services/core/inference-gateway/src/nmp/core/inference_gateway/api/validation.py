# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request validation helpers for entity names and workspaces.

Uses the same NAME_PATTERN as the entity store so that invalid names
return 422 Unprocessable Entity instead of 404 Not Found.
"""

import re

from fastapi import HTTPException, status
from nmp.common.entities.constants import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nmp.common.entities.utils import parse_adapters_suffix

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
    # A LoRA composite name is ``base&adapters/adapter-workspace/adapter-name``.
    # parse_adapters_suffix owns that grammar split (shared with the models-service
    # reconciler + IGW routing); here we only apply NAME_PATTERN to each recovered
    # segment. A non-composite (or malformed composite) name falls through to the
    # plain single-name validation below.
    adapter_parts = parse_adapters_suffix(value)
    if adapter_parts is not None:
        base, adapter_workspace, adapter_name = adapter_parts
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
