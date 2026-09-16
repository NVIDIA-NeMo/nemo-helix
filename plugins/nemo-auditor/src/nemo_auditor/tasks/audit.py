# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for the audit job.

Invoked as ``python -m nemo_auditor.tasks.audit`` inside the nhx-cpu-tasks container.
Builds the task SDK, then dispatches to :class:`~nemo_auditor.jobs.audit.AuditJob`.
The SIGTERM handler installed here is overridden by the one in ``AuditJob.run()``
before the probe loop begins, so partial-result aggregation is handled by the job.
"""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from nemo_auditor.jobs.audit import AuditJob
from nemo_helix_plugin.sdk_provider import get_async_task_sdk, get_task_sdk
from nemo_helix_plugin.tasks.dispatcher import run_task

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d). Exiting.", signum)
    raise SystemExit(0)


def main() -> int:
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk("auditor")
        async_sdk = get_async_task_sdk("auditor")
    except Exception:
        logger.exception("Failed to build task SDK for auditor")
        return 2
    return run_task(AuditJob, sdk=sdk, async_sdk=async_sdk)


if __name__ == "__main__":
    sys.exit(main())
