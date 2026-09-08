# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for task-image build jobs."""

from __future__ import annotations

import sys

from nemo_platform_plugin.sdk_provider import get_async_task_sdk, get_task_sdk
from nemo_platform_plugin.tasks.dispatcher import run_task
from nemo_scaled_evals_plugin.jobs.task_image_build import TaskImageBuildJob


def main() -> int:
    """Dispatch the task-image build job."""
    return run_task(
        TaskImageBuildJob,
        sdk=get_task_sdk("scaled-evals"),
        async_sdk=get_async_task_sdk("scaled-evals"),
    )


if __name__ == "__main__":
    sys.exit(main())
