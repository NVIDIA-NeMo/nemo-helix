# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import assert_type

from nemo_helix_plugin.jobs.types import PlatformJobStepWithContext


def test_platform_job_step_with_context_parses_wire_timestamps_to_datetime() -> None:
    step = PlatformJobStepWithContext.model_validate(
        {
            "id": "test-step-id",
            "attempt_id": "test-attempt-id",
            "fileset": "test-fileset",
            "job": "test-job-id",
            "name": "test-step",
            "workspace": "default",
            "created_at": "2026-06-23T19:00:00Z",
            "updated_at": "2026-06-23T19:00:05Z",
        }
    )

    assert_type(step.created_at, datetime)
    assert_type(step.updated_at, datetime)
    assert isinstance(step.created_at, datetime)
    assert isinstance(step.updated_at, datetime)
