# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import signal

from nhx.common.jobs.constants import DEFAULT_JOB_STORAGE_PATH
from nhx.customization_common.service.constants import (
    DEFAULT_DATASET_OUTPUT_DIR_NAME,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_OUTPUT_DIR_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_MODEL_DIR_NAME,
    DEFAULT_OUTPUT_MODEL_PATH,
    NHX_FILES_URL_ENVVAR,
    NHX_JOBS_URL_ENVVAR,
)

__all__ = [
    "DEFAULT_DATASET_OUTPUT_DIR_NAME",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_MODEL_OUTPUT_DIR_NAME",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_OUTPUT_MODEL_DIR_NAME",
    "DEFAULT_OUTPUT_MODEL_PATH",
    "DEFAULT_SEED",
    "DEFAULT_TEACHER_MODEL_DIR_NAME",
    "DEFAULT_TEACHER_MODEL_PATH",
    "DEFAULT_TRAINING_OUTPUT_DIR_NAME",
    "DEFAULT_TRAINING_OUTPUT_PATH",
    "DEFAULT_TRAINING_RESULT_FILE_NAME",
    "PAUSE_SENTINEL_FILE_NAME",
    "PAUSE_SIGNAL",
    "NHX_FILES_URL_ENVVAR",
    "NHX_JOBS_URL_ENVVAR",
    "SERVICE_NAME",
]

SERVICE_NAME = "customizer"

# Global default seed for reproducibility
DEFAULT_SEED = 1111

# Relative directory names unique to automodel (teacher model + training outputs).
DEFAULT_TEACHER_MODEL_DIR_NAME = "teacher_model"
DEFAULT_TRAINING_OUTPUT_DIR_NAME = "training"
DEFAULT_TRAINING_RESULT_FILE_NAME = "customizer_training_result.json"
# Global rank 0 writes this into the training workspace once the pause checkpoint is saved,
# and the runner reads it after torchrun exits. The Jobs service only sees the exit code.
PAUSE_SENTINEL_FILE_NAME = "PAUSED"
# Asks Automodel to checkpoint and stop. Not SIGTERM: the runner and torchrun use that for cancel.
PAUSE_SIGNAL = signal.Signals.SIGUSR1

# Absolute paths (used in HelixJobSpec for cross-step file sharing via PVC)
DEFAULT_TEACHER_MODEL_PATH = f"{DEFAULT_JOB_STORAGE_PATH}/{DEFAULT_TEACHER_MODEL_DIR_NAME}"
DEFAULT_TRAINING_OUTPUT_PATH = f"{DEFAULT_JOB_STORAGE_PATH}/{DEFAULT_TRAINING_OUTPUT_DIR_NAME}"
