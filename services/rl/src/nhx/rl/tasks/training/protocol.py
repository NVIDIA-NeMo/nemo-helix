# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from nhx.customization_common.training.progress import JobsServiceProgressReporter
from nhx.rl.app.jobs.training.schemas import (
    CheckpointInfo,
    TrainingMetrics,
    TrainingStepConfig,
)
from nhx.rl.app.jobs.training.schemas import TrainingBackend as TrainingBackendEnum


@dataclass
class LibraryConfig:
    """
    Library-specific configuration ready for training.

    We track both the config dict and the path, and let the consumer decide how to use them.
    """

    config_dict: dict[str, Any]  # Library-specific config dict
    config_path: Path  # Path to the config file (managed by runner)


@runtime_checkable
class SupportsPreprocessing(Protocol):
    """Protocol for backends that need pre-training preprocessing.

    Backends that implement this protocol will have their `run_preprocessing`
    method called before config compilation. Use this for operations like
    model format conversion that must happen before training.
    """

    def run_preprocessing(
        self,
        customizer_config: TrainingStepConfig,
    ) -> None:
        """Run pre-training conversions (e.g., model format conversion).

        Called before config compilation on the coordinator node only.

        Args:
            customizer_config: Standardized training configuration
        """
        ...


@runtime_checkable
class TrainingBackend(Protocol):
    """
    Interface for training backends (Strategy Pattern).

    Each backend (e.g. nemo_rl) implements this interface.
    Backends are responsible for:

    1. Compiling library-specific configuration (pure transformation)
    2. Executing training using library-specific wrappers/recipes
    3. Processing checkpoints to standard output format

    Note: Pre-training conversions are optional. Backends that need them
    should also implement `SupportsPreprocessing`.
    """

    @property
    def backend_type(self) -> TrainingBackendEnum:
        """Backend type identifier."""
        ...

    def compile_config(
        self,
        config: TrainingStepConfig,
        workspace_dir: Path,
    ) -> dict[str, Any]:
        """
        Transform standardized config to library-specific config.

        This is a pure transformation - no file I/O. The runner handles
        writing the config to disk.

        Called by the coordinator node only.

        Args:
            config: Standardized training configuration
            workspace_dir: Directory for training artifacts (for paths in config)

        Returns:
            Library-specific config dict ready to be written as YAML
        """
        ...

    def execute_training(
        self,
        customizer_config: TrainingStepConfig,
        library_config: LibraryConfig,
        progress: JobsServiceProgressReporter,
    ) -> TrainingMetrics:
        """
        Execute training using library-specific wrappers.
        """
        ...

    def find_best_checkpoint(
        self,
        workspace_dir: Path,
        customizer_config: TrainingStepConfig,
        library_config: Optional[LibraryConfig] = None,
    ) -> Path:
        """
        Find the best checkpoint after training.
        """
        ...

    def process_checkpoint(
        self,
        checkpoint_path: Path,
        output_path: Path,
        customizer_config: TrainingStepConfig,
        library_config: LibraryConfig | None = None,
    ) -> CheckpointInfo:
        """
        Process checkpoint to standard output format.

        Args:
            checkpoint_path: Path to the checkpoint directory
            output_path: Where to write the processed checkpoint
            customizer_config: Training configuration
            library_config: Library-specific config (contains resolved chat template, etc.)
        """
        ...
