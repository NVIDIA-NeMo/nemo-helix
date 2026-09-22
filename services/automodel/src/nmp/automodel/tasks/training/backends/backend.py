# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from nmp.automodel.tasks.training.errors.parser import (
    MAX_OUTPUT_LINES,
    parse_error_from_output,
    read_subprocess_output,
)
from nmp.automodel.tasks.training.progress import JobsServiceProgressReporter
from nmp.automodel.tasks.training.protocol import LibraryConfig
from nmp.automodel.tasks.training.schemas import (
    CheckpointInfo,
    TrainingMetrics,
    TrainingRecipe,
    TrainingStepConfig,
)
from nmp.automodel.tasks.training.utils import generate_torchrun_flags_from_env
from nmp.customization_common.service.context import NMPJobContext
from nmp.customization_common.training.nccl import get_nccl_ib_env

from .checkpoints import (
    ModelType,
    find_selected_checkpoints,
    process_selected_checkpoints,
)
from .config import compile_automodel_config, resolve_compiled_recipe

logger = logging.getLogger(__name__)

AUTOMODEL_CONFIG_FILENAME = "automodel_config.yaml"


def _checkpoint_model_type(config: TrainingStepConfig) -> ModelType:
    recipe = resolve_compiled_recipe(config)
    if recipe == TrainingRecipe.BI_ENCODER:
        return ModelType.EMBEDDING
    if recipe == TrainingRecipe.CROSS_ENCODER:
        return ModelType.CROSS_ENCODER
    return ModelType.LLM


class AutomodelBackend:
    """Compiles and runs nemo-automodel training for customization jobs."""

    def __init__(self, job_ctx: NMPJobContext):
        self.job_ctx = job_ctx

    def compile_config(
        self,
        config: TrainingStepConfig,
        workspace_dir: Path,
    ) -> dict[str, Any]:
        """
        Compile Automodel-specific configuration.

        Pure transformation - no file I/O. The runner handles writing to disk.
        """
        return compile_automodel_config(config, workspace_dir, self.job_ctx)

    def execute_training(
        self,
        customizer_config: TrainingStepConfig,
        library_config: LibraryConfig,
        progress: JobsServiceProgressReporter,
    ) -> TrainingMetrics:
        """Execute training using CustomizerTrainFinetuneRecipe or CustomizerBiencoderRecipe.

        The config file has already been written to disk by the runner.
        Progress reporting happens within the training subprocess via
        TrainingProgressCallback, which reads job context from environment
        variables.
        """
        progress.report_running("training", backend="automodel")

        # Run training with our custom recipe
        # Note: The progress parameter is not passed to run_training_with_customizer_recipe
        # because progress reporting now happens inside the subprocess via
        # TrainingProgressCallback using environment variables.
        command = ["torchrun"]
        command.extend(generate_torchrun_flags_from_env())
        command.extend(
            [
                "-m",
                "nmp.automodel.tasks.training.backends.finetune",
                "--config",
                str(library_config.config_path),
            ]
        )

        logger.info(f"Executing: {' '.join(command)}")

        training_process: subprocess.Popen | None = None

        # Rolling buffer to keep recent output lines for error extraction
        output_lines: deque[str] = deque(maxlen=MAX_OUTPUT_LINES)
        reader_thread: threading.Thread | None = None

        def cleanup(signum, frame):
            logger.warning(f"Signal {signum} received, terminating...")
            if training_process:
                training_process.send_signal(signum)
                try:
                    training_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    training_process.kill()
            raise SystemExit(signum)

        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)

        start_time = time.time()

        training_env = os.environ.copy()
        if customizer_config.parallelism.num_nodes > 1:
            training_env.update(get_nccl_ib_env())
        training_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            env=training_env,
        )

        # Start reader thread to capture output without blocking
        reader_thread = threading.Thread(
            target=read_subprocess_output,
            args=(training_process, output_lines),
            daemon=True,
        )
        reader_thread.start()

        try:
            training_process.wait(timeout=customizer_config.training_timeout)
        except subprocess.TimeoutExpired:
            logger.exception("Training timed out")
            training_process.kill()
            # Reap the killed process to avoid zombies
            try:
                training_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Killed training process did not terminate within 30s - "
                    "process may be stuck in uninterruptible state"
                )
            # Wait for reader thread to capture any remaining output before re-raising
            if reader_thread and reader_thread.is_alive():
                reader_thread.join(timeout=5)
            raise  # Let runner.py convert via create_error_details()

        # Wait for reader thread to finish capturing output
        if reader_thread and reader_thread.is_alive():
            reader_thread.join(timeout=5)

        duration = time.time() - start_time
        logger.info(f"Training finished in {duration:.1f} seconds")

        if training_process.returncode != 0:
            parsed = parse_error_from_output(output_lines, training_process.returncode)
            raise parsed.to_exception()

        # Return empty metrics (actual metrics are reported via callbacks during training)
        # TODO: Consider parsing training logs or checkpoints to extract final metrics.
        return TrainingMetrics(total_steps=0, total_epochs=0)

    def find_checkpoints(
        self,
        workspace_dir: Path,
        customizer_config: TrainingStepConfig,
    ) -> dict[str, Path]:
        """Find the checkpoint(s) selected for publication."""
        model_type = _checkpoint_model_type(customizer_config)
        return find_selected_checkpoints(workspace_dir, customizer_config, model_type=model_type)

    def process_checkpoints(
        self,
        checkpoints: dict[str, Path],
        output_path: Path,
        workspace_dir: Path,
        customizer_config: TrainingStepConfig,
        library_config: LibraryConfig | None = None,
    ) -> CheckpointInfo:
        """Process the selected checkpoint(s) into the output fileset."""
        model_type = _checkpoint_model_type(customizer_config)
        resolved_template = None
        if model_type == ModelType.LLM and library_config and library_config.config_dict:
            resolved_template = library_config.config_dict.get("_resolved_chat_template")

        return process_selected_checkpoints(
            checkpoints,
            output_path,
            workspace_dir,
            customizer_config,
            model_type=model_type,
            resolved_chat_template=resolved_template,
        )
