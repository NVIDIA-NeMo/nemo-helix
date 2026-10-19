# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client_provider import get_nemo_client
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import HelixJobResults
from nemo_helix_plugin.jobs.constants import (
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from pydantic import BaseModel


def run_job_module(job_cls: type[Any], spec_cls: type[BaseModel]) -> int:
    with open(os.environ[NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR], encoding="utf-8") as handle:
        spec = spec_cls.model_validate_json(handle.read())
    client = get_nemo_client(as_service="data-designer")
    ctx = _get_ctx(client)
    result = job_cls().run(spec.model_dump(mode="json"), ctx=ctx, sdk=client)
    exit_code = result.get("exit_code") if isinstance(result, dict) else 1
    return exit_code if isinstance(exit_code, int) else 1


def _get_ctx(client: NemoClient) -> JobContext:
    workspace = os.environ[NEMO_JOB_WORKSPACE_ENVVAR]
    job_name = os.environ[NEMO_JOB_ID_ENVVAR]
    persistent_env = os.environ.get(PERSISTENT_JOB_STORAGE_PATH_ENVVAR)
    storage = StoragePaths(
        ephemeral=Path(os.environ[EPHEMERAL_TASK_STORAGE_PATH_ENVVAR]),
        persistent=Path(persistent_env) if persistent_env else None,
    )
    results = HelixJobResults(workspace=workspace, job_name=job_name, client=client)
    return JobContext(workspace=workspace, job_id=job_name, storage=storage, results=results)


if __name__ == "__main__":
    sys.exit(1)
