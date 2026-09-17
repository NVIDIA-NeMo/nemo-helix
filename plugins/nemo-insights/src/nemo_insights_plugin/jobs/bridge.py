# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-process entry point for insights analyzer jobs."""

import logging
import signal
import sys
from types import FrameType

from nemo_insights_plugin.jobs.analyze import AnalyzeJob
from nemo_platform_plugin.errors import LocalRunError
from nemo_platform_plugin.sdk_provider import get_task_sdk
from nemo_platform_plugin.tasks.dispatcher import build_ctx_from_env, exit_code_for, read_step_config
from nemo_platform_plugin.tasks.logging_setup import configure_task_logging

logger = logging.getLogger(__name__)


def _shutdown(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def main() -> int:
    """Run the platform-injected analyzer task config."""
    configure_task_logging()
    signal.signal(signal.SIGTERM, _shutdown)
    try:
        sdk = get_task_sdk("insights")
        ctx = build_ctx_from_env(sdk)
        config = read_step_config()
        job = AnalyzeJob()
    except Exception:
        logger.exception("Failed to prepare AnalyzeJob task")
        return 2
    try:
        return exit_code_for(job.run(config, ctx=ctx, sdk=sdk))
    except LocalRunError:
        raise
    except Exception:
        logger.exception("AnalyzeJob.run raised")
        return 1


if __name__ == "__main__":
    sys.exit(main())
