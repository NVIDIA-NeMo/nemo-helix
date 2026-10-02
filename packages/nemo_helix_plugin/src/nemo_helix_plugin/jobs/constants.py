# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

DEFAULT_TASK_STORAGE_PATH = "/var/run/scratch/task"
DEFAULT_JOB_STORAGE_PATH = "/var/run/scratch/job"
DEFAULT_CONFIG_STORAGE_PATH = "/var/run/scratch/config"

EPHEMERAL_TASK_STORAGE_PATH_ENVVAR = "NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH"
PERSISTENT_JOB_STORAGE_PATH_ENVVAR = "NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH"
CONFIG_TASK_STORAGE_PATH_ENVVAR = "NEMO_JOB_STEP_CONFIG_STORAGE_PATH"

TASK_CONFIG_ENVVAR = "NHX_TASK_CONFIG"

NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR = "NEMO_JOB_STEP_CONFIG_FILE_PATH"
NEMO_JOB_STEP_CONFIG_FILE_NAME = "job_step_config.json"
DEFAULT_NEMO_JOB_STEP_CONFIG_FILE_PATH = f"{DEFAULT_CONFIG_STORAGE_PATH}/{NEMO_JOB_STEP_CONFIG_FILE_NAME}"

NEMO_JOB_ID_ENVVAR = "NEMO_JOB_ID"
NEMO_JOB_ATTEMPT_ID_ENVVAR = "NEMO_JOB_ATTEMPT_ID"
NEMO_JOB_STEP_ENVVAR = "NEMO_JOB_STEP"
NEMO_JOB_TASK_ENVVAR = "NEMO_JOB_TASK"
NEMO_JOB_WORKSPACE_ENVVAR = "NEMO_JOB_WORKSPACE"
NEMO_JOB_FILESET_ENVVAR = "NEMO_JOB_FILESET"

NEMO_JOB_SECRETS_ENVVAR = "NEMO_JOB_SECRETS"

TERMINAL_EXIT_CODES = (0, 137)


def job_storage_subpath(workspace: str, job_id: str) -> str:
    """Path of a job's storage directory *relative to the job-storage PVC root*.

    The Kubernetes jobs backend mounts the shared PVC at the job storage path
    with this value as ``subPath``. Anything that needs to hand the same data to
    a second consumer that mounts the PVC itself — e.g. the GRPO Gym sandbox,
    which re-mounts the environment and dataset directories — must reproduce the
    layout exactly, so it lives here rather than being spelled out per call site.
    """
    return f"jobs/{workspace}/{job_id}"
