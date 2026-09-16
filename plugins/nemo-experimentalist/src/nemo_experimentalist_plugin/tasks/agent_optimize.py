# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for the experimentalist agent-optimize job (``python -m nemo_experimentalist_plugin.tasks.agent_optimize``)."""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from nemo_experimentalist_plugin.jobs.optimize import ExperimentalistOptimizeJob
from nemo_platform_plugin.sdk_provider import get_task_sdk
from nemo_platform_plugin.tasks.dispatcher import run_task

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
    return run_task(ExperimentalistOptimizeJob, sdk=sdk)


if __name__ == "__main__":
    sys.exit(main())
