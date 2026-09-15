# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Training runner with distributed coordination support.

Orchestrates Automodel training in single-node and multi-node environments,
using file-based barriers for cross-pod synchronization.
"""

import json
import logging
import random
import time
from enum import Enum
from pathlib import Path
from types import TracebackType

import yaml
from nmp.automodel.app.constants import DEFAULT_TRAINING_RESULT_FILE_NAME
from nmp.customization_common.service.context import NMPJobContext
from nmp.customization_common.service.path_utils import remap_job_storage_path

from .backends.backend import AUTOMODEL_CONFIG_FILENAME, AutomodelBackend
from .distributed import DistributedContext
from .errors.converter import create_error_details
from .progress import JobsServiceProgressReporter
from .protocol import LibraryConfig
from .schemas import (
    GPUInfo,
    TrainingMetrics,
    TrainingResult,
    TrainingStepConfig,
)
from .utils import get_gpu_info


# Custom YAML representer to serialize Enum values as their string values
def _enum_representer(dumper: yaml.Dumper, data: Enum) -> yaml.Node:
    """Represent Enum as its value (string) rather than a Python object tag."""
    return dumper.represent_str(str(data.value))


yaml.add_representer(Enum, _enum_representer)
yaml.add_multi_representer(Enum, _enum_representer)

logger = logging.getLogger(__name__)

BARRIER_CONFIG_READY = "config_ready"
BARRIER_TRAINING_COMPLETE = "training_complete"


class TrainingRunner:
    """
    Orchestrates Automodel training across single-node and multi-node environments.

    Usage:
        with TrainingRunner() as runner:
            result = runner.run()
    """

    def __init__(self, backend: AutomodelBackend | None = None) -> None:
        self._job_ctx = NMPJobContext.from_env()
        self._config = self._load_config(self._job_ctx.config_path)
        self._progress = JobsServiceProgressReporter(self._job_ctx)
        self._dist_ctx = DistributedContext.from_env(self._get_barrier_dir())
        self._backend = backend or AutomodelBackend(self._job_ctx)
        self._workspace_path = Path(self._config.workspace_path)
        self._output_path = Path(self._config.output_path)

    def __enter__(self) -> "TrainingRunner":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._progress.close()

    def run(self) -> TrainingResult:
        random.seed(self._config.seed)
        logger.info(f"Global random seed set to {self._config.seed}")

        start_time = time.time()
        gpu_info = get_gpu_info()
        result = TrainingResult(success=False, error_message="No result")

        try:
            library_config = self._compile_config_phase()
            metrics = self._training_phase(library_config)
            self._dist_ctx.sync_point(BARRIER_TRAINING_COMPLETE)
            result = self._postprocess_phase(gpu_info, metrics, start_time, library_config)

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            error_details = create_error_details(e)
            result = TrainingResult(
                success=False,
                error_message=error_details.get("message", str(e)),
                gpu_info=gpu_info,
                training_duration_seconds=time.time() - start_time,
            )
            if self._dist_ctx.is_coordinator:
                self._progress.report_error(error_details)
        finally:
            self._write_result(result)

        return result

    def _get_barrier_dir(self) -> Path:
        return self._job_ctx.storage_path / self._job_ctx.attempt_id / "distributed" / "barriers"

    def _load_config(self, config_path: Path) -> TrainingStepConfig:
        with open(config_path) as f:
            config = TrainingStepConfig.model_validate(json.load(f))
        return self._normalize_storage_paths(config)

    def _normalize_storage_paths(self, config: TrainingStepConfig) -> TrainingStepConfig:
        storage_path = self._job_ctx.storage_path

        model = config.model.model_copy(
            update={"path": str(remap_job_storage_path(storage_path, config.model.path))},
        )
        dataset = config.dataset.model_copy(
            update={"path": str(remap_job_storage_path(storage_path, config.dataset.path))},
        )
        training = config.training
        if training.kd is not None:
            teacher_model = training.kd.teacher_model.model_copy(
                update={"path": str(remap_job_storage_path(storage_path, training.kd.teacher_model.path))},
            )
            training = training.model_copy(
                update={"kd": training.kd.model_copy(update={"teacher_model": teacher_model})},
            )

        return config.model_copy(
            update={
                "model": model,
                "dataset": dataset,
                "training": training,
                "workspace_path": str(remap_job_storage_path(storage_path, config.workspace_path)),
                "output_path": str(remap_job_storage_path(storage_path, config.output_path)),
            },
        )

    def _get_library_config_path(self) -> Path:
        return self._workspace_path / AUTOMODEL_CONFIG_FILENAME

    def _compile_config_phase(self) -> LibraryConfig:
        config_path = self._get_library_config_path()

        if self._dist_ctx.is_coordinator:
            self._progress.report_running("compiling_config")
            config_dict = self._backend.compile_config(self._config, self._workspace_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Library config written to: {config_path}")
            self._dist_ctx.signal(BARRIER_CONFIG_READY)
            return LibraryConfig(config_dict=config_dict, config_path=config_path)

        self._dist_ctx.wait_for_coordinator(BARRIER_CONFIG_READY)
        return self._load_library_config(config_path)

    def _load_library_config(self, config_path: Path) -> LibraryConfig:
        if not config_path.exists():
            raise FileNotFoundError(
                f"Library config not found at {config_path}. Coordinator may not have written it yet."
            )
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        logger.info(f"Loaded library config from: {config_path}")
        return LibraryConfig(config_dict=config_dict, config_path=config_path)

    def _training_phase(self, library_config: LibraryConfig) -> TrainingMetrics:
        return self._backend.execute_training(self._config, library_config, self._progress)

    def _postprocess_phase(
        self,
        gpu_info: GPUInfo | None,
        metrics: TrainingMetrics,
        start_time: float,
        library_config: LibraryConfig,
    ) -> TrainingResult:
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
        if not self._dist_ctx.is_coordinator:
            return
        result_path = self._workspace_path / DEFAULT_TRAINING_RESULT_FILE_NAME
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            f.write(result.model_dump_json(indent=2))
        logger.info(f"Result written to: {result_path}")
