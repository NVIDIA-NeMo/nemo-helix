# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for evaluator plugin bundle-native jobs."""

from __future__ import annotations

import sys

from nemo_evaluator.jobs.evaluate import AsyncEvaluateJob
from nemo_evaluator.tasks.runner import run_async_task_main


def main() -> int:
    """Build the task SDK and dispatch to the evaluator plugin job."""
    return run_async_task_main(AsyncEvaluateJob, service_name="evaluator")


if __name__ == "__main__":
    sys.exit(main())
