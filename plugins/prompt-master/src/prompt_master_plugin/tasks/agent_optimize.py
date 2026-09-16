# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for the prompt-master agent-optimize job (``python -m prompt_master_plugin.tasks.agent_optimize``)."""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from nemo_platform_plugin.sdk_provider import get_task_sdk
from nemo_platform_plugin.tasks.dispatcher import run_task
from prompt_master_plugin.jobs.optimize import PromptMasterOptimizeJob

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d). Exiting.", signum)
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk("agents")
    except Exception:
        logger.exception("Failed to build task SDK for agents")
        return 2
    return run_task(PromptMasterOptimizeJob, sdk=sdk)


if __name__ == "__main__":
    sys.exit(main())
