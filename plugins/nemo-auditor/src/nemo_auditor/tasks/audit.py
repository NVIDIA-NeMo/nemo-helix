# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for the audit job.

Invoked as ``python -m nemo_auditor.tasks.audit`` inside the nmp-cpu-tasks container.
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
from nemo_platform_plugin.errors import LocalRunError
from nemo_platform_plugin.sdk_provider import get_async_task_sdk, get_task_sdk
from nemo_platform_plugin.tasks.dispatcher import build_ctx_from_env, exit_code_for, read_step_config
from nemo_platform_plugin.tasks.logging_setup import configure_task_logging

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d). Exiting.", signum)
    raise SystemExit(0)


def main() -> int:
    configure_task_logging()
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk("auditor")
        async_sdk = get_async_task_sdk("auditor")
        ctx = build_ctx_from_env(sdk)
        config = read_step_config()
        job = AuditJob()
    except Exception:
        logger.exception("Failed to prepare task for auditor")
        return 2
    try:
        return exit_code_for(job.run(config, ctx=ctx, sdk=sdk, async_sdk=async_sdk))
    except LocalRunError:
        raise
    except Exception:
        logger.exception("AuditJob.run raised")
        return 1


if __name__ == "__main__":
    sys.exit(main())
