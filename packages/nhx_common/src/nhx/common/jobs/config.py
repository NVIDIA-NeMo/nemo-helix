# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task configuration utilities."""

import json
import os
from pathlib import Path
from typing import TypeVar, overload

from nhx.common.jobs.constants import (
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    TASK_CONFIG_ENVVAR,
)
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def get_job_id() -> str:
    """Get the job ID from environment."""
    return os.environ.get(NEMO_JOB_ID_ENVVAR, "unknown")


def get_workspace() -> str:
    """Get the workspace from environment."""
    return os.environ.get(NEMO_JOB_WORKSPACE_ENVVAR, "default")


@overload
def get_task_config() -> dict: ...


@overload
def get_task_config(model: type[T]) -> T: ...


def _load_config_dict() -> dict:
    """Load config from env var or file.

    Config sources (in order of precedence):
    1. NHX_TASK_CONFIG env var (JSON string) - for CLI usage
    2. NEMO_JOB_STEP_CONFIG_FILE_PATH file - for job containers
    """
    # First try the env var
    config_str = os.environ.get(TASK_CONFIG_ENVVAR)
    if config_str:
        return json.loads(config_str)

    # Then try the config file (set by jobs controller in containers)
    config_file_path = os.environ.get(NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR)
    if config_file_path:
        config_path = Path(config_file_path)
        if config_path.exists():
            return json.loads(config_path.read_text())

    # Default to empty config
    return {}


def get_task_config(model: type[T] | None = None) -> dict | T:
    """Load task configuration from environment or config file.

    Config sources (in order of precedence):
    1. NHX_TASK_CONFIG env var (JSON string) - for CLI usage
    2. NEMO_JOB_STEP_CONFIG_FILE_PATH file - for job containers

    Args:
        model: Optional Pydantic model class to validate and parse the config.
               If not provided, returns a raw dictionary.

    Returns:
        If model is provided: validated Pydantic model instance
        If model is None: raw configuration dictionary

    Raises:
        json.JSONDecodeError: If config JSON is malformed
        pydantic.ValidationError: If model is provided and validation fails
    """
    config_dict = _load_config_dict()

    if model is not None:
        return model.model_validate(config_dict)
    return config_dict
