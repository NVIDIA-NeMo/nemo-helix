# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for benign-suite synthesis (``python -m nemo_agent_hardener_plugin.tasks.synth_benign``)."""

from __future__ import annotations

import sys

from nemo_agent_hardener_plugin.jobs.synth_benign import AgentHardenerSynthBenignJob
from nemo_agent_hardener_plugin.tasks.entrypoint import run_job_task

if __name__ == "__main__":
    sys.exit(run_job_task(AgentHardenerSynthBenignJob))
