# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for BEIR retrieval evaluation."""

from __future__ import annotations

import sys

from nemo_evaluator.jobs.retrieve_eval import AsyncRetrieveEvalJob
from nemo_evaluator.tasks.runner import run_async_task_main


def main() -> int:
    """Build task SDK clients and dispatch the retrieval job."""
    return run_async_task_main(AsyncRetrieveEvalJob, service_name="evaluator")


if __name__ == "__main__":
    sys.exit(main())
