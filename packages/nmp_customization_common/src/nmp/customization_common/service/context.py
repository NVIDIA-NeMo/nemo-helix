# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job context for customization container task entrypoints.

Populated from the Job Controller environment variables. Shared by both the
unsloth and automodel backends.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from nmp.common.entities.constants import DEFAULT_WORKSPACE
from nmp.common.jobs.constants import (
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_ENVVAR,
    NEMO_JOB_TASK_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
)
from nmp.customization_common.service.constants import (
    NMP_FILES_URL_ENVVAR,
    NMP_JOBS_URL_ENVVAR,
)
from nmp.customization_common.service.path_utils import (
    get_job_step_config_path_from_env,
    get_job_storage_path_from_env,
)

DEFAULT_JOB_ID = "unknown-job-id"
DEFAULT_ATTEMPT_ID = "attempt-0"
DEFAULT_STEP = "unknown-step"
DEFAULT_TASK = "unknown-task"


# Jobs task names should comply with NAME_PATTERN of EntityCreateInput.name for the Jobs API.
# Generated tasks in k8s don't start with a lowercase letter per NAME_PATTERN, so we normalize
# by adding the prefix when missing.
# In Docker environment core/jobs/src/nmp/core/jobs/controllers/backends/docker.py,
# tasks are prefixed with `task-` by default: task_id = f"task-{uuid.uuid4().hex}"
def _normalize_task_name(task: str) -> str:
    """Ensure task name uses the expected Jobs prefix."""
    if task.startswith("task-"):
        return task
    return f"task-{task}"


@dataclass(frozen=True)
class NMPJobContext:
    """NeMo Platform Job context populated from Job Controller environment variables"""

    workspace: str
    job_id: str
    attempt_id: str
    step: str
    task: str

    # Service URLs
    jobs_url: str | None
    files_url: str | None

    # Storage paths
    storage_path: Path
    config_path: Path

    @property
    def normalized_task(self) -> str:
        """Task normalized for Jobs API compatibility."""
        return _normalize_task_name(self.task)

    @property
    def is_configured(self) -> bool:
        """True only when populated from real Job Controller env vars.

        ``from_env`` fills missing identifiers with non-empty placeholder
        sentinels (``unknown-job-id`` / ``unknown-step`` / ``unknown-task``).
        Callers (e.g. progress reporting) must gate on this rather than a bare
        truthiness check, or they issue failing SDK calls / log spam when
        running outside a real platform job (local ``run`` paths, tests).
        """
        return (
            self.job_id not in ("", DEFAULT_JOB_ID)
            and self.step not in ("", DEFAULT_STEP)
            and self.task not in ("", DEFAULT_TASK)
        )

    @classmethod
    def from_env(cls) -> Self:
        """Create a NMPJobContext from environment variables"""
        return cls(
            workspace=os.environ.get(NEMO_JOB_WORKSPACE_ENVVAR, DEFAULT_WORKSPACE),
            job_id=os.environ.get(NEMO_JOB_ID_ENVVAR, DEFAULT_JOB_ID),
            attempt_id=os.environ.get(NEMO_JOB_ATTEMPT_ID_ENVVAR, DEFAULT_ATTEMPT_ID),
            step=os.environ.get(NEMO_JOB_STEP_ENVVAR, DEFAULT_STEP),
            task=os.environ.get(NEMO_JOB_TASK_ENVVAR, DEFAULT_TASK),
            jobs_url=os.environ.get(NMP_JOBS_URL_ENVVAR),
            files_url=os.environ.get(NMP_FILES_URL_ENVVAR),
            storage_path=get_job_storage_path_from_env(),
            config_path=get_job_step_config_path_from_env(),
        )
