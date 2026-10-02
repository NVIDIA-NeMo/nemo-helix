# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for staging an evaluator Gym environment."""

from __future__ import annotations

import sys

from nemo_evaluator.jobs.environment_stage import EnvironmentStageJob
from nemo_evaluator.tasks.runner import run_sync_task_main


def main() -> int:
    """Run the stage-environment job inside the CPU tasks container."""
    return run_sync_task_main(EnvironmentStageJob, service_name="evaluator")


if __name__ == "__main__":
    sys.exit(main())
