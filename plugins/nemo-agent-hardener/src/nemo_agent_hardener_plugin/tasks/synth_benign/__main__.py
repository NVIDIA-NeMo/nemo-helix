# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for benign synthesis (``python -m nemo_agent_hardener_plugin.tasks.synth_benign``).

The executor spawns this module with the ``NEMO_JOB_*`` env populated; it hands off to the framework's
``run_task`` dispatcher, which loads the step config, builds a ``JobContext``, and DI-injects ``ctx``/``sdk``
into :meth:`AgentHardenerSynthBenignJob.run`. Local responsibilities here are only SIGTERM handling and SDK
construction — mirrors :mod:`nemo_agent_hardener_plugin.tasks.war_game`.
"""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from nemo_agent_hardener_plugin.jobs.synth_benign import AgentHardenerSynthBenignJob
from nemo_platform_plugin.sdk_provider import get_task_sdk
from nemo_platform_plugin.tasks.dispatcher import run_task

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, _frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d). Exiting.", signum)
    raise SystemExit(128 + signum)


def main() -> int:
    """Build the on-behalf-of SDK and dispatch to ``run_task``."""
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk("agent-hardener")
    except Exception:
        logger.exception("Failed to build task SDK for agent-hardener")
        return 2
    return run_task(AgentHardenerSynthBenignJob, sdk=sdk)


if __name__ == "__main__":
    sys.exit(main())
