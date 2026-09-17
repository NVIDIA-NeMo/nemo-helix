# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for the war-game (``python -m nemo_agent_hardener_plugin.tasks.war_game``).

The executor spawns this module with the ``NEMO_JOB_*`` env populated; it hands off to the framework's
the module loads the step config, builds a ``JobContext``, and calls
:meth:`AgentHardenerRunJob.run` with its concrete ``ctx``/``sdk`` signature.
"""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from nemo_agent_hardener_plugin.jobs.run import AgentHardenerRunJob
from nemo_platform_plugin.errors import LocalRunError
from nemo_platform_plugin.sdk_provider import get_task_sdk
from nemo_platform_plugin.tasks.dispatcher import build_ctx_from_env, exit_code_for, read_step_config
from nemo_platform_plugin.tasks.logging_setup import configure_task_logging

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, _frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d). Exiting.", signum)
    raise SystemExit(128 + signum)


def main() -> int:
    """Build the on-behalf-of SDK and run the war-game job."""
    configure_task_logging()
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk("agent-hardener")
        ctx = build_ctx_from_env(sdk)
        config = read_step_config()
        job = AgentHardenerRunJob()
    except Exception:
        logger.exception("Failed to prepare task for agent-hardener")
        return 2
    try:
        return exit_code_for(job.run(config, ctx=ctx, sdk=sdk))
    except LocalRunError:
        raise
    except Exception:
        logger.exception("AgentHardenerRunJob.run raised")
        return 1


if __name__ == "__main__":
    sys.exit(main())
