# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for task-image build jobs."""

from __future__ import annotations

import logging
import sys

from nemo_platform_plugin.errors import LocalRunError
from nemo_platform_plugin.tasks.dispatcher import exit_code_for, read_step_config
from nemo_platform_plugin.tasks.logging_setup import configure_task_logging
from nemo_scaled_evals_plugin.jobs.task_image_build import TaskImageBuildJob

logger = logging.getLogger(__name__)


def main() -> int:
    """Dispatch the task-image build job."""
    configure_task_logging()
    try:
        config = read_step_config()
        job = TaskImageBuildJob()
    except Exception:
        logger.exception("Failed to prepare task for scaled-evals task-image build")
        return 2

    try:
        return exit_code_for(job.run(config))
    except LocalRunError:
        raise
    except Exception:
        logger.exception("TaskImageBuildJob.run raised")
        return 1


if __name__ == "__main__":
    sys.exit(main())
