# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for the unsloth container job pipeline.

Shared container-path/env constants (including the validation-dataset path) come
from :mod:`nhx.customization_common.service.constants`; this module adds the unsloth
``SERVICE_NAME`` and checkpoint path.
"""

from nhx.common.jobs.constants import DEFAULT_JOB_STORAGE_PATH
from nhx.customization_common.service.constants import (
    DEFAULT_DATASET_OUTPUT_DIR_NAME,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_OUTPUT_DIR_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_MODEL_DIR_NAME,
    DEFAULT_OUTPUT_MODEL_PATH,
    DEFAULT_VALIDATION_DATASET_OUTPUT_DIR_NAME,
    NHX_FILES_URL_ENVVAR,
    NHX_JOBS_URL_ENVVAR,
)

__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_DATASET_OUTPUT_DIR_NAME",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_MODEL_OUTPUT_DIR_NAME",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_OUTPUT_MODEL_DIR_NAME",
    "DEFAULT_OUTPUT_MODEL_PATH",
    "DEFAULT_VALIDATION_DATASET_OUTPUT_DIR_NAME",
    "NHX_FILES_URL_ENVVAR",
    "NHX_JOBS_URL_ENVVAR",
    "SERVICE_NAME",
]

SERVICE_NAME = "unsloth"

# Trainer checkpoints, kept out of output_model, which is published as the model.
DEFAULT_CHECKPOINT_PATH = f"{DEFAULT_JOB_STORAGE_PATH}/training/checkpoints"
