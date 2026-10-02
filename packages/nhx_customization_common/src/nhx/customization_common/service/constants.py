# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container-path and env-var constants shared by the customization backends.

The unsloth and automodel services expose the same path layout to the 4-step
container submit pipeline (download -> train -> upload -> model-entity). The
shared subset lives here; each backend's ``app/constants.py`` re-exports these
and adds its own ``SERVICE_NAME`` plus any backend-specific constants.
"""

from nhx.common.jobs.constants import DEFAULT_JOB_STORAGE_PATH

# Subdirectory names under the job's persistent storage root.
DEFAULT_MODEL_OUTPUT_DIR_NAME = "model"
DEFAULT_DATASET_OUTPUT_DIR_NAME = "dataset"
DEFAULT_VALIDATION_DATASET_OUTPUT_DIR_NAME = "validation_dataset"
DEFAULT_ENVIRONMENT_OUTPUT_DIR_NAME = "environment"
DEFAULT_OUTPUT_MODEL_DIR_NAME = "output_model"

# Absolute paths used by the compiler when wiring step-to-step file sharing
# inside the platform Jobs runner's mounted storage layout.
DEFAULT_MODEL_PATH = f"{DEFAULT_JOB_STORAGE_PATH}/{DEFAULT_MODEL_OUTPUT_DIR_NAME}"
DEFAULT_DATASET_PATH = f"{DEFAULT_JOB_STORAGE_PATH}/{DEFAULT_DATASET_OUTPUT_DIR_NAME}"
DEFAULT_VALIDATION_DATASET_PATH = f"{DEFAULT_JOB_STORAGE_PATH}/{DEFAULT_VALIDATION_DATASET_OUTPUT_DIR_NAME}"
DEFAULT_ENVIRONMENT_PATH = f"{DEFAULT_JOB_STORAGE_PATH}/{DEFAULT_ENVIRONMENT_OUTPUT_DIR_NAME}"
DEFAULT_OUTPUT_MODEL_PATH = f"{DEFAULT_JOB_STORAGE_PATH}/{DEFAULT_OUTPUT_MODEL_DIR_NAME}"

# Paths visible inside the job sandbox (OpenSandbox mount layout).
SANDBOX_ENVIRONMENT_PATH = "/job/environment"
SANDBOX_DATASET_PATH = "/job/dataset"
SANDBOX_WORK_PATH = "/job/work"

NHX_JOBS_URL_ENVVAR = "NHX_JOBS_URL"
NHX_FILES_URL_ENVVAR = "NHX_FILES_URL"
