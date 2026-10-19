# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for benchmark archive build jobs."""

from __future__ import annotations

import logging
import sys

from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.tasks.dispatcher import exit_code_for, read_step_config
from nemo_helix_plugin.tasks.logging_setup import configure_task_logging
from nemo_scaled_evals_plugin.jobs.benchmark_archive_build import BenchmarkArchiveBuildJob

logger = logging.getLogger(__name__)


def main() -> int:
    """Dispatch the benchmark archive build job."""
    configure_task_logging()
    try:
        config = read_step_config()
        job = BenchmarkArchiveBuildJob()
    except Exception:
        logger.exception("Failed to prepare task for scaled-evals benchmark archive build")
        return 2

    try:
        return exit_code_for(job.run(config))
    except LocalRunError:
        raise
    except Exception:
        logger.exception("BenchmarkArchiveBuildJob.run raised")
        return 1


if __name__ == "__main__":
    sys.exit(main())
