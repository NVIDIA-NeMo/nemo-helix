# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for the agent-layer Job."""

from __future__ import annotations

import sys

from nemo_evals.jobs.agent_layer_build import AgentLayerBuildJob
from nemo_evals.tasks.runner import run_async_task_main


def main() -> int:
    """Run the agent-layer Job inside the CPU tasks container."""
    return run_async_task_main(AgentLayerBuildJob, service_name="evals")


if __name__ == "__main__":
    sys.exit(main())
