# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.jobs.execution_profiles import JOB_LOGS_ENDPOINT_ENVVAR, JobExecutionProfileConfig
from pydantic import ValidationError


@pytest.mark.parametrize(
    "envvar",
    [
        JOB_LOGS_ENDPOINT_ENVVAR,
        WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    ],
)
def test_job_execution_profile_config_rejects_platform_injected_env_vars(envvar: str) -> None:
    with pytest.raises(ValidationError, match=envvar):
        JobExecutionProfileConfig(env={envvar: "override"})
