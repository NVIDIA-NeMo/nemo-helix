# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.core.jobs.app.schemas import PlatformJobStepSpec
from nmp.core.jobs.app.test_helpers import TestConstants
from nmp.core.jobs.entities import (
    STEP_SPEC_NAME_CONFIG_KEY,
    PlatformJobAttempt,
    get_step_spec_name,
)


def test_get_step_spec_name_prefers_stored_spec_name():
    assert get_step_spec_name({STEP_SPEC_NAME_CONFIG_KEY: "download"}, fallback_name="download-1") == "download"


def test_get_step_spec_name_uses_fallback_without_stored_spec_name():
    assert get_step_spec_name({}, fallback_name="download") == "download"


def test_get_step_spec_name_uses_fallback_for_non_mapping_config():
    config: Any = ["not", "a", "mapping"]

    assert get_step_spec_name(config, fallback_name="download") == "download"


def test_platform_job_attempt_identifies_final_step_spec():
    platform_spec = TestConstants.PLATFORM_SPEC.model_copy(deep=True)
    platform_spec.steps.append(
        PlatformJobStepSpec(name="finalize", executor=TestConstants.TEST_EXECUTOR, config={}),
    )
    attempt = PlatformJobAttempt(
        name="attempt-1",
        workspace=TestConstants.WORKSPACE,
        job="job-1",
        seq=0,
        status=PlatformJobStatus.ACTIVE,
        spec=TestConstants.SPEC_BASIC,
        platform_spec=platform_spec,
    )

    assert attempt.is_final_step_spec("basic") is False
    assert attempt.is_final_step_spec("finalize") is True
