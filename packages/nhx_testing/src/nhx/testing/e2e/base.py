# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for E2E test backends.

This module provides the common interface that all E2E test backends must implement.
Backends are responsible for starting/stopping test infrastructure and providing
SDK clients for tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from nemo_helix import NeMoHelix
from nemo_helix_plugin.jobs.image import image_builder

if TYPE_CHECKING:
    from .config import E2EConfig

DEFAULT_REGISTRY = "my-registry"
DEFAULT_TAG = "local"


class E2EBackend(ABC):
    """Abstract base class for e2e test backends.

    Supports context manager protocol for automatic cleanup.
    Produces a NeMoHelix SDK client for tests.

    Args:
        config_path: Path to the NeMo Helix configuration YAML file, or an E2EConfig object.
            When an E2EConfig is provided, its path (the config file) is used as-is.
        registry: Docker registry for NeMo Helix images (overrides platform config).
        tag: Docker image tag (overrides platform config).
    """

    def __init__(
        self,
        config_path: str | Path | E2EConfig,
        registry: str | None = None,
        tag: str | None = None,
        *,
        gpu_requested: bool = False,
    ):
        # Import here to avoid circular dependency at module level
        from .config import E2EConfig as E2EConfigClass

        # Handle E2EConfig object: use its path (config file used as-is)
        if isinstance(config_path, E2EConfigClass):
            self.config_path = config_path.path
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        self.registry = registry
        self.tag = tag
        self.gpu_requested = gpu_requested
        # Create an image builder that uses overrides if provided, otherwise platform config
        self._build_image = image_builder(registry=registry, tag=tag)

    @property
    def image(self) -> str:
        """Get the NeMo Helix API image name."""
        return self._build_image("nhx-api")

    def get_image(self, name: str) -> str:
        """Get a fully qualified image name for any NeMo Helix image.

        Args:
            name: The image name (e.g., 'nhx-api', 'nhx-tasks').

        Returns:
            Fully qualified image name.
        """
        return self._build_image(name)

    @abstractmethod
    def start(self) -> None:
        """Start the test environment."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop and cleanup the test environment."""
        pass

    @abstractmethod
    def get_sdk(self, principal_id: str | None = None) -> NeMoHelix:
        """Create an SDK client for the test environment.

        Args:
            principal_id: Optional principal ID for authentication (X-NHX-Principal-Id header).

        Returns:
            Configured NeMoHelix SDK client.
        """
        pass

    def get_logs(self, tail: int | None = 200) -> tuple[str, str] | None:
        """Get backend logs for debugging.

        Override in subclasses that support log retrieval.

        Args:
            tail: Number of lines to return from the end. None for all logs.

        Returns:
            Tuple of (stdout, stderr) as strings, or None if not supported.
        """
        return None

    def __enter__(self) -> "E2EBackend":
        """Start backend when entering context."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop backend when exiting context."""
        self.stop()
