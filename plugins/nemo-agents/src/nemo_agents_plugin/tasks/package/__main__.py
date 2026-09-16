# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for ``agents.package`` (``python -m nemo_agents_plugin.tasks.package``).

See :mod:`nemo_agents_plugin.tasks.evaluate_suite` for the shared pattern.
"""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from nemo_agents_plugin.jobs.package_agent import PackageAgentJob
from nemo_helix_plugin.sdk_provider import get_async_task_sdk, get_task_sdk
from nemo_helix_plugin.tasks.dispatcher import run_task

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d).  Exiting.", signum)
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk("agents")
        # The build context is the agent's spec fileset, downloaded by an async
        # helper — without this the job would package agent.yaml alone.
        async_sdk = get_async_task_sdk("agents")
    except Exception:
        logger.exception("Failed to build task SDK for agents")
        return 2
    return run_task(PackageAgentJob, sdk=sdk, async_sdk=async_sdk)


if __name__ == "__main__":
    sys.exit(main())
