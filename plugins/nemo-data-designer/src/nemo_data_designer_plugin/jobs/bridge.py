# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

from nemo_data_designer_plugin.jobs.run import run_step_config
from nemo_data_designer_plugin.jobs.spec import DataDesignerStepConfig
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import PlatformJobResults
from nemo_platform_plugin.jobs.constants import (
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nemo_platform_plugin.sdk_provider import get_platform_sdk


def run() -> int:
    step_config = _get_step_config()
    sdk = get_platform_sdk(as_service="data-designer")
    ctx = _get_ctx(sdk)

    return run_step_config(
        step_config=step_config,
        ctx=ctx,
        sdk=sdk,
    )


def _get_step_config() -> DataDesignerStepConfig:
    with open(os.environ[NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR], "r") as f:
        return DataDesignerStepConfig.model_validate_json(f.read())


def _get_ctx(sdk: NeMoPlatform) -> JobContext:
    workspace = os.environ[NEMO_JOB_WORKSPACE_ENVVAR]
    job_name = os.environ[NEMO_JOB_ID_ENVVAR]

    persistent_env = os.environ.get(PERSISTENT_JOB_STORAGE_PATH_ENVVAR)
    storage = StoragePaths(
        ephemeral=Path(os.environ[EPHEMERAL_TASK_STORAGE_PATH_ENVVAR]),
        persistent=Path(persistent_env) if persistent_env else None,
    )
    results = PlatformJobResults(
        workspace=workspace,
        job_name=job_name,
        client=client_from_platform(sdk, NemoClient),
    )
    return JobContext(
        workspace=workspace,
        job_id=job_name,
        storage=storage,
        results=results,
    )


if __name__ == "__main__":
    sys.exit(run())
