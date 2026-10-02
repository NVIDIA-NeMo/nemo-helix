# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for Files service tests."""

from typing import AsyncIterator

import pytest
from nhx.core.files.app.backends.base import ByteRange, FileInfo, StorageImpl
from nhx.testing.blockbuster import blockbuster_fixture

# Enable BlockBuster to detect blocking calls in async code
blockbuster = blockbuster_fixture(autouse=True)


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.

    Auto-marks tests based on their location:

    - Tests in e2e/ directories get the 'e2e' marker
    - Tests in integration/ directories get the 'integration' marker
    - Tests without category markers get the 'unit' marker
    """
    # Category markers that determine test type
    category_markers = {
        "unit",
        "e2e",
        "integration",
        "regression",
        "canary",
        "slow",
        "skip_in_ci",
    }

    for item in items:
        # Get current marker names
        marker_names = {marker.name for marker in item.iter_markers()}

        # Auto-mark tests in e2e directories
        if "/e2e/" in str(item.fspath):
            if "e2e" not in marker_names:
                item.add_marker(pytest.mark.e2e)
                marker_names.add("e2e")

        # Auto-mark tests in integration directories
        elif "/integration/" in str(item.fspath):
            if "integration" not in marker_names:
                item.add_marker(pytest.mark.integration)
                marker_names.add("integration")

        # Auto-mark tests without category markers as unit tests
        if not marker_names.intersection(category_markers):
            item.add_marker(pytest.mark.unit)


# ============================================================================
# Storage Test Helpers
# ============================================================================


class InMemoryStorage(StorageImpl):
    """In-memory storage implementation for testing."""

    def __init__(self):
        self.files: dict[str, bytes] = {}

    async def list_files(self, path: str | None = None) -> list[FileInfo]:
        """List all files with given prefix."""
        prefix = path or ""
        matching_files = []
        for file_path, data in self.files.items():
            if file_path.startswith(prefix):
                matching_files.append(FileInfo(path=file_path, size=len(data)))
        return matching_files

    async def upload(self, path: str, data: AsyncIterator[bytes]) -> FileInfo:
        """Store file data in memory."""
        chunks = []
        async for chunk in data:
            chunks.append(chunk)
        file_data = b"".join(chunks)
        self.files[path] = file_data
        return FileInfo(path=path, size=len(file_data))

    async def download(self, path: str, byte_range: ByteRange | None = None) -> AsyncIterator[bytes]:
        """Retrieve file data from memory."""
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        data = self.files[path]
        if byte_range:
            data = data[byte_range.start : byte_range.end + 1]
        yield data

    async def delete(self, path: str) -> FileInfo:
        """Delete file from memory."""
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        size = len(self.files[path])
        del self.files[path]
        return FileInfo(path=path, size=size)

    async def validate_storage(self) -> None:
        """Validate storage - no-op for in-memory storage."""
        pass

    async def list(self, prefix: str = "") -> list[str]:
        """List all file paths with given prefix (helper method)."""
        return [path for path in self.files.keys() if path.startswith(prefix)]

    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        return path in self.files
