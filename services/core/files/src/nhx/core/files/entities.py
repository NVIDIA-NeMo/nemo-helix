# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain entities for the Files service."""

from datetime import datetime
from typing import Any, ClassVar, Dict

from nemo_helix_plugin.files.types import FilesetPurpose as FilesetPurpose
from nhx.common.entities import constants
from nhx.common.entities.client import EntityBase
from nhx.common.files.metadata import FilesetMetadata
from nhx.core.files.app.backends.factory import StorageConfig
from pydantic import Field


class Fileset(EntityBase):
    """Fileset domain model - represents a fileset entity."""

    __entity_type__: ClassVar[str] = "fileset"

    description: str | None = Field(
        default=None,
        description="The description of the fileset.",
        max_length=constants.MAX_LENGTH_255,
    )
    storage: StorageConfig = Field(description="The storage configuration for the fileset.")

    purpose: FilesetPurpose = Field(description="The purpose of the fileset.")
    metadata: FilesetMetadata = Field(
        default_factory=FilesetMetadata,
        description="Purpose-specific metadata for the fileset.",
    )

    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom fields for the fileset.")


class FileLock(EntityBase):
    """File lock entity for coordinating file writes across requests.

    Used to prevent multiple requests from caching the same file simultaneously.
    Locks are acquired before writing to cache and released after completion.
    Stale locks can be cleaned up by other requests based on acquired_at + TTL.
    """

    __entity_type__: ClassVar[str] = "file_lock"

    path: str = Field(description="The cache path being locked")
    acquired_at: datetime = Field(description="When the lock was acquired")
