# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for the scan job.

Invoked as ``python -m garak_plugin.tasks.scan`` inside the nhx-tasks container.
Builds the task clients, then dispatches to :class:`~garak_plugin.jobs.scan.ScanJob`.
The SIGTERM handler installed here is overridden by the one in ``ScanJob.run()``
before the probe loop begins, so partial-result aggregation is handled by the job.
"""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from garak_plugin.jobs.scan import ScanJob
from nemo_helix_plugin.client_provider import get_async_task_nemo_client, get_task_nemo_client
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.tasks.dispatcher import build_ctx_from_env, exit_code_for, read_step_config
from nemo_helix_plugin.tasks.logging_setup import configure_task_logging

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d). Exiting.", signum)
    raise SystemExit(0)


def main() -> int:
    configure_task_logging()
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        client = get_task_nemo_client("garak-plugin")
        async_client = get_async_task_nemo_client("garak-plugin")
        ctx = build_ctx_from_env(client)
        config = read_step_config()
        job = ScanJob()
    except Exception:
        logger.exception("Failed to prepare task for garak_plugin")
        return 2
    try:
        return exit_code_for(job.run(config, ctx=ctx, sdk=client, async_sdk=async_client))
    except LocalRunError:
        raise
    except Exception:
        logger.exception("ScanJob.run raised")
        return 1


if __name__ == "__main__":
    sys.exit(main())
