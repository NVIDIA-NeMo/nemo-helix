# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for the Agents optimize job (``python -m nemo_optimization.tasks.optimize``)."""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from nemo_helix_plugin.sdk_provider import get_task_sdk
from nemo_helix_plugin.tasks.dispatcher import run_task

from nemo_optimization.jobs.optimize import OptimizeJob

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
    return run_task(OptimizeJob, sdk=sdk)


if __name__ == "__main__":
    sys.exit(main())
