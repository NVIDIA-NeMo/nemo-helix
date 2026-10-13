# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime helpers for the three step binaries."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from nemo_helix_plugin.jobs.constants import (
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)

logger = logging.getLogger(__name__)


def read_step_config() -> dict[str, Any]:
    path = os.environ.get(NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR)
    if not path:
        raise RuntimeError(
            f"{NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR} is not set; this binary only runs as a platform job step"
        )
    return json.loads(Path(path).read_text())


def job_identity() -> tuple[str, str]:
    workspace = os.environ.get(NEMO_JOB_WORKSPACE_ENVVAR)
    job_id = os.environ.get(NEMO_JOB_ID_ENVVAR)
    if not workspace or not job_id:
        raise RuntimeError(f"{NEMO_JOB_WORKSPACE_ENVVAR} and {NEMO_JOB_ID_ENVVAR} must both be set")
    return workspace, job_id


def work_mount() -> Path:
    mount = os.environ.get(PERSISTENT_JOB_STORAGE_PATH_ENVVAR)
    if not mount:
        raise RuntimeError(
            f"{PERSISTENT_JOB_STORAGE_PATH_ENVVAR} is not set; this step was compiled without a work volume"
        )
    return Path(mount)


def split_fileset_ref(ref: str, default_workspace: str) -> tuple[str, str]:
    if "/" in ref:
        workspace, _, name = ref.partition("/")
        return workspace, name
    return default_workspace, ref
