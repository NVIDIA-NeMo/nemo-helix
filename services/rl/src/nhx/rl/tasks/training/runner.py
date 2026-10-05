# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Training runner with distributed coordination support.

Orchestrates training execution across single-node and multi-node environments,
using file-based barriers for cross-pod synchronization.
"""

import json
import logging
import random
import time
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Self

import yaml
from nhx.customization_common.service.context import NHXJobContext
from nhx.rl.app.constants import DEFAULT_TRAINING_RESULT_FILE_NAME
from nhx.rl.app.jobs.training.schemas import (
    GPUInfo,
    TrainingMetrics,
    TrainingResult,
    TrainingStepConfig,
)
from nhx.rl.app.jobs.training.schemas import TrainingBackend as TrainingBackendEnum
from nhx.rl.tasks.training.progress import JobsServiceProgressReporter

from .distributed import DistributedContext
from .errors.converter import create_error_details
from .protocol import LibraryConfig, SupportsPreprocessing, TrainingBackend
from .utils import get_gpu_info


# Custom YAML representer to serialize Enum values as their string values
def _enum_representer(dumper: yaml.Dumper, data: Enum) -> yaml.Node:
    """Represent Enum as its value (string) rather than a Python object tag."""
    return dumper.represent_str(str(data.value))


yaml.add_representer(Enum, _enum_representer)
# Also add for all Enum subclasses
yaml.add_multi_representer(Enum, _enum_representer)

logger = logging.getLogger(__name__)

# Barrier names for distributed synchronization
BARRIER_CONFIG_READY = "config_ready"
BARRIER_TRAINING_COMPLETE = "training_complete"
BARRIER_PREPROCESSING_COMPLETE = "preprocessing_complete"


class TrainingRunner:
    """
    Orchestrates training execution across single-node and multi-node environments.

    Initializes from environment variables and coordinates training phases:
    - Config compilation: Coordinator only, workers wait
    - Training: All ranks participate (via torchrun)
    - Post-processing: Coordinator only, workers exit after training sync

    Usage:
        with TrainingRunner() as runner:
            result = runner.run()
    """

    def __init__(self, backend: TrainingBackend | None = None) -> None:
        """Initialize the runner from environment variables."""
        self._job_ctx = NHXJobContext.from_env()

        self._config = self._load_config(self._job_ctx.config_path)
        self._progress = JobsServiceProgressReporter(self._job_ctx)
        self._dist_ctx = DistributedContext.from_env(self._get_barrier_dir())
        self._backend = backend or self._load_backend(self._config.backend)
        # workspace_path and output_path are absolute paths from the config
        self._workspace_path = Path(self._config.workspace_path)
        self._output_path = Path(self._config.output_path)

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit - ensures progress reporter is closed."""
        self.close()

    def close(self) -> None:
        """Clean up resources (progress reporter)."""
        self._progress.close()

    # --- Training execution ---

    def run(self) -> TrainingResult:
        """
        Execute training with distributed coordination.

        Phases:
        1. Config compilation (coordinator, workers wait)
        2. Training (all ranks via torchrun)
        3. Sync point (all ranks) — workers return success and exit here
        4. Post-processing (coordinator only)
        5. Result writing (coordinator only)

        Returns:
            TrainingResult with success/failure and metrics
        """
        # Set global seed as first layer of defense for reproducibility
        random.seed(self._config.seed)
        logger.info(f"Global random seed set to {self._config.seed}")

        start_time = time.time()
        gpu_info = get_gpu_info()
        result = TrainingResult(success=False, error_message="No result")

        try:
            # === Phase 0: Pre-training conversions (coordinator only) ===
            self._preprocessing_phase()

            # === Phase 1: Config compilation (coordinator, workers wait) ===
            library_config = self._compile_config_phase()

            # === Phase 2: Training (all ranks) ===
            metrics = self._training_phase(library_config)

            # === Phase 3: Sync after training ===
            self._dist_ctx.sync_point(BARRIER_TRAINING_COMPLETE)

            # === Phase 4: Post-processing (coordinator only, workers exit) ===
            result = self._postprocess_phase(gpu_info, metrics, start_time, library_config)

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            # Convert exception to user-friendly error details using error mapping rules
            error_details = create_error_details(e)
            result = TrainingResult(
                success=False,
                error_message=error_details.get("message", str(e)),
                gpu_info=gpu_info,
                training_duration_seconds=time.time() - start_time,
            )
            if self._dist_ctx.is_coordinator:
                self._progress.report_error(error_details)
                # Publish a failure marker so workers blocked on a coordinator
                # barrier exit promptly instead of waiting out the full timeout.
                self._dist_ctx.signal_failure()
        finally:
            # === Phase 5: Write result (coordinator only) ===
            self._write_result(result)

        # Returning outside `finally` so an uncaught BaseException (e.g.
        # KeyboardInterrupt) propagates instead of being swallowed by the return.
        return result

    # --- Helper methods ---
    def _load_backend(self, backend_type: TrainingBackendEnum) -> TrainingBackend:
        """Load the backend for the given backend type."""
        if backend_type == TrainingBackendEnum.NEMO_RL:
            from .backends.nemo_rl.backend import NemoRLBackend

            return NemoRLBackend(self._job_ctx)

        raise ValueError(f"Unknown backend type: {backend_type}")

    def _get_barrier_dir(self) -> Path:
        """Get the barrier directory for distributed coordination."""
        return self._job_ctx.storage_path / self._job_ctx.attempt_id / "distributed" / "barriers"

    def _load_config(self, config_path: Path) -> TrainingStepConfig:
        """Load the training step config."""
        with open(config_path) as f:
            config = TrainingStepConfig.model_validate(json.load(f))
        return config

    def _get_library_config_path(self) -> Path:
        """
        Get the path for the library-specific config file.

        We define it here and pass it to the backend so that the backend can read it as-is without constructing paths.
        """
        return self._workspace_path / f"{self._backend.backend_type.value}_config.yaml"

    def _preprocessing_phase(self) -> None:
        """
        Run pre-training conversions if the backend supports them.

        Coordinator runs conversions (e.g., model format conversion), workers skip.
        This phase runs before config compilation so that compiled configs can
        reference converted artifacts.

        Only backends implementing SupportsPreprocessing will have conversions run.

        !!! Important !!!
        Only coordinator runs _preprocessing_phase. Workers wait for coordinator to finish.
        Any changes to the configs would affect only coordinator, so avoid any config changes.
        """
        if self._dist_ctx.is_coordinator:
            if isinstance(self._backend, SupportsPreprocessing):
                self._progress.report_running("conversions")
                self._backend.run_preprocessing(self._config)
                logger.info("Pre-training conversions complete")
            # Always release workers, even if no conversions are needed
            self._dist_ctx.signal(BARRIER_PREPROCESSING_COMPLETE)
        else:
            self._dist_ctx.wait_for_coordinator(BARRIER_PREPROCESSING_COMPLETE)

    def _compile_config_phase(self) -> LibraryConfig:
        """
        Compile library-specific config.

        Coordinator compiles config and writes to disk, then signals.
        Workers wait for signal, then load the config file.

        The runner handles all file I/O; backend just compiles.
        """
        config_path = self._get_library_config_path()

        if self._dist_ctx.is_coordinator:
            self._progress.report_running("compiling_config")

            # Backend compiles config (pure transformation, no I/O)
            config_dict = self._backend.compile_config(self._config, self._workspace_path)

            # Runner writes config to disk
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # safe_dump, not dump: the driver reads this back with OmegaConf.load, whose
            # loader is SafeLoader-based. Plain yaml.dump serializes a tuple (or any other
            # Python object) as a !!python/... tag that the driver then cannot construct,
            # so the job fails minutes later with a line number instead of a cause.
            # safe_dump raises here, at compile time, naming the offending value.
            with open(config_path, "w") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)

            logger.info(f"Library config written to: {config_path}")
            self._dist_ctx.signal(BARRIER_CONFIG_READY)

            return LibraryConfig(config_dict=config_dict, config_path=config_path)
        else:
            self._dist_ctx.wait_for_coordinator(BARRIER_CONFIG_READY)
            return self._load_library_config(config_path)

    def _load_library_config(self, config_path: Path) -> LibraryConfig:
        """Load library config from disk (used by workers)."""
        if not config_path.exists():
            raise FileNotFoundError(
                f"Library config not found at {config_path}. Coordinator may not have written it yet."
            )

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        logger.info(f"Loaded library config from: {config_path}")
        return LibraryConfig(config_dict=config_dict, config_path=config_path)

    def _training_phase(self, library_config: LibraryConfig) -> TrainingMetrics:
        """
        Execute training on all ranks.

        Training itself is distributed via Ray, which handles inter-process coordination internally.
        """
        return self._backend.execute_training(
            self._config,
            library_config,
            self._progress,
        )

    def _postprocess_phase(
        self,
        gpu_info: GPUInfo | None,
        metrics: TrainingMetrics,
        start_time: float,
        library_config: LibraryConfig,
    ) -> TrainingResult:
        """
        Process checkpoint and create result.

        Workers return immediately with a minimal success result. They have no
        post-training responsibilities, so letting them exit avoids barrier
        timeouts that would cause Volcano to kill the coordinator mid-copy
        because checkpoint copies for large models can take more than 600s
        which is the default barrier timeout

        The coordinator finds the best checkpoint, copies/processes it to the
        output path, and reports completion.
        """
        if not self._dist_ctx.is_coordinator:
            return TrainingResult(
                success=True,
                gpu_info=gpu_info,
                training_duration_seconds=time.time() - start_time,
            )

        self._progress.report_running("processing_checkpoint")
        checkpoint_path = self._backend.find_best_checkpoint(self._workspace_path, self._config, library_config)
        checkpoint_info = self._backend.process_checkpoint(
            checkpoint_path, self._output_path, self._config, library_config
        )

        result = TrainingResult(
            success=True,
            checkpoint=checkpoint_info,
            gpu_info=gpu_info,
            metrics=metrics,
            training_duration_seconds=time.time() - start_time,
        )

        self._progress.report_completed("Training completed")
        return result

    def _write_result(self, result: TrainingResult) -> None:
        """Write result for downstream tasks."""
        if not self._dist_ctx.is_coordinator:
            return

        result_path = self._workspace_path / DEFAULT_TRAINING_RESULT_FILE_NAME
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            f.write(result.model_dump_json(indent=2))
        logger.info(f"Result written to: {result_path}")
