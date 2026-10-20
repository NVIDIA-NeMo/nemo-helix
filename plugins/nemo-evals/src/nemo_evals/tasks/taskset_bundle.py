# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundle Job entrypoint: merge a taskset run's task bundles into one result bundle."""

import sys

from nemo_evals.jobs.taskset_bundle import TasksetBundleJob
from nemo_evals.tasks.runner import run_async_task_main


def main() -> int:
    """Run the step config, a ``TasksetBundleSpec``."""
    return run_async_task_main(TasksetBundleJob, service_name="evals")


if __name__ == "__main__":
    sys.exit(main())
