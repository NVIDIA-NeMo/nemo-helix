# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes for storage backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    AsyncIterator,
)

from nmp.common.files.storage_config import BaseStorageConfig as BaseStorageConfig
from nmp.common.files.storage_config import StorageConfigType as StorageConfigType
from nmp.core.files.exceptions import NotFoundError


@dataclass
class ByteRange:
    start: int
    end: int  # inclusive, which matches the ranged GET header


@dataclass
class FileInfo:
    path: str
    size: int


class StorageImpl(ABC):
    config: BaseStorageConfig

    @abstractmethod
    async def list_files(self, path: str | None = None) -> list[FileInfo]: ...

    @abstractmethod
    async def download(self, path: str, byte_range: ByteRange | None) -> AsyncIterator[bytes]: ...

    @abstractmethod
    async def upload(
        self,
        path: str,
        fstream: AsyncIterator[bytes],
        content_length: int | None = None,
    ) -> FileInfo: ...

    @abstractmethod
    async def validate_storage(self): ...

    @abstractmethod
    async def delete(self, path: str) -> FileInfo: ...

    async def delete_all(self) -> None:
        """Delete all files in the storage backend.

        This is a no-op by default. Only storage backends that manage their own
        data (like local storage) should override this to actually delete files.
        External storage backends (like NGC, HuggingFace) should not delete
        data they don't own.
        """
        pass

    async def resolve_config(self) -> BaseStorageConfig:
        """Resolve any mutable references in the storage config.

        External storage backends (HuggingFace, NGC) should override this to resolve
        mutable references (like 'main' or 'latest') to specific immutable identifiers
        (commit SHAs, version IDs). The resolved config should be persisted with the
        fileset entity.

        Returns:
            A new StorageConfig with resolved references. The original user-provided
            value should be stored in `original_revision` or `original_version` fields.
            Returns the unchanged config by default (for backends that don't need resolution).
        """
        return self.config

    @property
    def tracked_revision(self) -> str | None:
        """The mutable ref this fileset was created from, if it tracks one.

        The counterpart to :meth:`resolve_config`: a backend that pins a mutable
        ref to an immutable id at create time reports that original ref here, so
        the fileset can be re-resolved against it later. None when the fileset was
        created from an already-immutable id, which has nothing to move to.
        """
        return self.config.tracked_revision

    def config_at_tracked_revision(self) -> BaseStorageConfig:
        """Return the config pointed back at :attr:`tracked_revision` for re-resolution.

        Only meaningful when :attr:`tracked_revision` is set; callers check first.
        """
        tracked = self.tracked_revision
        if tracked is None:
            raise NotImplementedError(f"{type(self.config).__name__} tracks no revision to re-resolve against")
        return self.config.model_copy(update={"revision": tracked})

    async def get_file(self, path: str) -> FileInfo:
        files = await self.list_files(path)
        if not files:
            raise NotFoundError(f"File not found for path: {path}")
        return files[0]

    async def get_cache_path_key(self, path: str | None = None) -> str | None:
        """
        Return path within cache storage for this file, or None if not cacheable.

        External storage backends (HuggingFace, NGC) should override this to return
        a unique cache key that includes version/revision information.

        Args:
            path: File path within the storage. If None, returns the cache root prefix.

        Returns:
            Path to use within cache storage, or None if this file shouldn't be cached
        """
        return None  # Default: not cacheable

    def get_duckdb_path(self, path: str) -> str:
        """Return a path/URI suitable for DuckDB file access.

        For local storage: returns filesystem path
        For S3 storage: returns s3://bucket/prefix/path URL

        Override in subclasses that support DuckDB access.

        Args:
            path: Relative path within the storage

        Returns:
            Full path/URI for DuckDB to access
        """
        raise NotImplementedError(f"DuckDB path access not supported for {type(self).__name__}")
