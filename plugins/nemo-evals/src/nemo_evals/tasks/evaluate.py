# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for evals plugin bundle-native jobs."""

from __future__ import annotations

import sys

from nemo_evals.jobs.evaluate import AsyncEvaluateJob
from nemo_evals.tasks.runner import run_async_task_main


def main() -> int:
    """Build the task SDK and dispatch to the evals plugin job."""
    return run_async_task_main(AsyncEvaluateJob, service_name="evals")


if __name__ == "__main__":
    sys.exit(main())
