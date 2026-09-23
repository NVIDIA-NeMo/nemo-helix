# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions for entity repository operations."""


class EntityRepositoryError(Exception):
    """Base exception for entity repository errors."""

    pass


class EntityNotFoundError(EntityRepositoryError):
    """Raised when an entity is not found."""

    pass


class EntityVersionConflictError(EntityRepositoryError):
    """Raised when entity version doesn't match (optimistic locking conflict)."""

    pass


class ForeignChildEntitiesError(EntityRepositoryError):
    """Raised when a delete would cascade into workspaces other than the entity's own."""

    def __init__(self, workspaces: set[str]) -> None:
        self.workspaces = sorted(workspaces)
        super().__init__(f"Entity has child entities in other workspaces: {', '.join(self.workspaces)}")
