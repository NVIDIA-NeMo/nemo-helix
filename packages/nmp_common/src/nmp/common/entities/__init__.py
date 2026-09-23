# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity client modules.

This package provides the EntityClient for database operations.

EntityClient (client.py) - Unified store pattern (recommended)
   SQLAlchemy-style API: one client, type specified per operation.
"""

# Unified client (recommended - primary implementation)
from nmp.common.entities.client import (
    EntityBase,
    EntityClient,
    EntityConflictError,
    EntityNotFoundError,
    EntityStoreError,
    EntityValidationError,
    ListResponse,
    PaginationInfo,
    SyncEntityClient,
)

# Constants for validation
from nmp.common.entities.constants import (
    ALL_WORKSPACES,
    DEFAULT_WORKSPACE,
    MAX_LENGTH_255,
    REGEX_WORD_CHARACTER_DOT_DASH,
    REGEX_WORD_CHARACTER_DOT_DASH_OR_BLANK,
    REGEX_WORD_CHARACTER_DOT_DASH_OR_BLANK_OR_PLUS,
    REGEX_WORD_CHARACTER_DOT_DASH_SLASH,
    SYSTEM_WORKSPACE,
)

# Entity utilities and values
from nmp.common.entities.filters import make_filter_obj_dep
from nmp.common.entities.global_workspace import (
    GLOBAL_WORKSPACE,
    is_global_workspace,
    workspace_lookup_order,
)
from nmp.common.entities.query_utils import coerce_existence_operator
from nmp.common.entities.utils import (
    get_random_bytes,
    get_random_id,
    make_filter_class,
    make_search_class,
    normalize_filter_list,
    normalize_search_list,
)
from nmp.common.entities.values import (
    ENTITY_BASE_FIELDS,
    DatetimeFilter,
    Filter,
    StringFilter,
    Value,
    map_entity_field,
)

__all__ = [
    # Unified client (recommended - primary implementation)
    "EntityBase",
    "EntityClient",
    "SyncEntityClient",
    "EntityStoreError",
    "EntityNotFoundError",
    "EntityConflictError",
    "EntityValidationError",
    "ListResponse",
    "PaginationInfo",
    "ALL_WORKSPACES",
    # Constants for validation
    "MAX_LENGTH_255",
    "REGEX_WORD_CHARACTER_DOT_DASH",
    "REGEX_WORD_CHARACTER_DOT_DASH_OR_BLANK",
    "REGEX_WORD_CHARACTER_DOT_DASH_OR_BLANK_OR_PLUS",
    "REGEX_WORD_CHARACTER_DOT_DASH_SLASH",
    # Entity utilities and values
    "coerce_existence_operator",
    "DEFAULT_WORKSPACE",
    "ENTITY_BASE_FIELDS",
    "SYSTEM_WORKSPACE",
    "GLOBAL_WORKSPACE",
    "is_global_workspace",
    "workspace_lookup_order",
    "DatetimeFilter",
    "Filter",
    "StringFilter",
    "Value",
    "get_random_bytes",
    "get_random_id",
    "make_filter_class",
    "make_filter_obj_dep",
    "make_search_class",
    "map_entity_field",
    "normalize_filter_list",
    "normalize_search_list",
]
