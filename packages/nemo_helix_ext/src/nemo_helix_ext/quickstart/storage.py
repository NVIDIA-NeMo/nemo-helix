# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage and volume management for quickstart."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path


class StorageManager:
    """Manages quickstart storage directory and permissions.

    The storage layout is:
        base_path/
        └── data/           # Shared data directory (mounted in container)
    """

    def __init__(self, base_path: Path):
        """Initialize storage manager.

        Args:
            base_path: Base storage directory path.
        """
        self.base_path = base_path

    @property
    def data_path(self) -> Path:
        """Get the data directory path."""
        return self.base_path / "data"

    def initialize(self) -> None:
        """Create all storage directories with proper permissions.

        Creates the directory structure and sets permissions to allow
        container processes (typically running as uid 1000) to write.
        """
        directories = [
            self.base_path,
            self.data_path,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            # Set permissions: rwxrwxrwx (777) to allow container access
            os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    def cleanup(self) -> None:
        """Remove all storage data."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)

    def exists(self) -> bool:
        """Check if storage directory exists."""
        return self.base_path.exists()

    def get_size(self) -> int:
        """Get total storage size in bytes.

        Returns:
            Total size of all files in the data directory.
        """
        if not self.data_path.exists():
            return 0

        total = 0
        for path in self.data_path.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    # Skip files we can't access
                    pass
        return total

    def get_size_human(self) -> str:
        """Get human-readable storage size.

        Returns:
            Size formatted as a human-readable string (e.g., "1.5 GB").
        """
        size = self.get_size()
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
