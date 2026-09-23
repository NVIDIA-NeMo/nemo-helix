# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from nhx.common.entities.constants import DEFAULT_WORKSPACE
from nhx.common.jobs.constants import (
    DEFAULT_NEMO_JOB_STEP_CONFIG_FILE_PATH,
    DEFAULT_TASK_STORAGE_PATH,
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_STEP_ENVVAR,
    NEMO_JOB_TASK_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
)
from pydantic import BaseModel, Field

DEFAULT_JOB_ID = "unknown-job-id"
DEFAULT_ATTEMPT_ID = "attempt-0"
DEFAULT_STEP = "unknown-step"
DEFAULT_TASK = "unknown-task"
NHX_MODELS_URL_ENVVAR = "NHX_MODELS_URL"
NHX_FILES_URL_ENVVAR = "NHX_FILES_URL"
NHX_JOBS_URL_ENVVAR = "NHX_JOBS_URL"


@dataclass(frozen=True)
class NHXJobContext:
    """NeMo Helix Job context populated from Job Controller environment variables"""

    workspace: str
    job_id: str
    attempt_id: str
    step: str
    task: str

    # Service URLs
    jobs_url: str | None
    files_url: str | None
    models_url: str | None

    storage_path: Path
    config_path: Path | None

    @classmethod
    def from_env(cls) -> Self:
        """Create a NHXJobContext from environment variables"""
        return cls(
            workspace=os.environ.get(NEMO_JOB_WORKSPACE_ENVVAR, DEFAULT_WORKSPACE),
            job_id=os.environ.get(NEMO_JOB_ID_ENVVAR, DEFAULT_JOB_ID),
            attempt_id=os.environ.get(NEMO_JOB_ATTEMPT_ID_ENVVAR, DEFAULT_ATTEMPT_ID),
            step=os.environ.get(NEMO_JOB_STEP_ENVVAR, DEFAULT_STEP),
            task=os.environ.get(NEMO_JOB_TASK_ENVVAR, DEFAULT_TASK),
            jobs_url=os.environ.get(NHX_JOBS_URL_ENVVAR),
            files_url=os.environ.get(NHX_FILES_URL_ENVVAR),
            models_url=os.environ.get(NHX_MODELS_URL_ENVVAR),
            config_path=Path(
                os.environ.get(NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR, DEFAULT_NEMO_JOB_STEP_CONFIG_FILE_PATH)
            ),
            storage_path=Path(os.environ.get(EPHEMERAL_TASK_STORAGE_PATH_ENVVAR, DEFAULT_TASK_STORAGE_PATH)),
        )


class ModelSpecTaskConfig(BaseModel):
    workspace: str = Field(..., description="Workspace the model entity is in")
    name: str = Field(..., description="The name of the model entity")
