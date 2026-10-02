# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for the evaluator plugin's agent-evaluation job."""

from __future__ import annotations

import sys

from nemo_evaluator.jobs.agent_evaluate import AsyncAgentEvalJob
from nemo_evaluator.tasks.runner import run_async_task_main


def main() -> int:
    """Build the task SDK and dispatch to the agent-evaluation job."""
    return run_async_task_main(AsyncAgentEvalJob, service_name="evaluator")


if __name__ == "__main__":
    sys.exit(main())
