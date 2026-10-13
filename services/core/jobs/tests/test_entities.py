# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.schemas import HelixJobStepSpec
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.entities import (
    STEP_SPEC_NAME_CONFIG_KEY,
    HelixJobAttempt,
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
        HelixJobStepSpec(name="finalize", executor=TestConstants.TEST_EXECUTOR, config={}),
    )
    attempt = HelixJobAttempt(
        name="attempt-1",
        workspace=TestConstants.WORKSPACE,
        job="job-1",
        seq=0,
        status=HelixJobStatus.ACTIVE,
        spec=TestConstants.SPEC_BASIC,
        platform_spec=platform_spec,
    )

    assert attempt.is_final_step_spec("basic") is False
    assert attempt.is_final_step_spec("finalize") is True
