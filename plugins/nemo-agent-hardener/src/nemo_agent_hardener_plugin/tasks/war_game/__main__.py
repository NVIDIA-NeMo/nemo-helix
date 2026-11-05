# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for the war-game (``python -m nemo_agent_hardener_plugin.tasks.war_game``)."""

from __future__ import annotations

import sys

from nemo_agent_hardener_plugin.jobs.run import AgentHardenerRunJob
from nemo_agent_hardener_plugin.tasks.entrypoint import run_job_task

if __name__ == "__main__":
    sys.exit(run_job_task(AgentHardenerRunJob))
