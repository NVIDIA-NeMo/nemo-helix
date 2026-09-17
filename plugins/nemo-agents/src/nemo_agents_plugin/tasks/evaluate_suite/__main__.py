# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for ``agents.evaluate-suite`` (``python -m nemo_agents_plugin.tasks.evaluate_suite``).

Mirrors :mod:`nemo_agents_plugin.tasks.evaluate`: loads the step config, builds
:class:`~nemo_platform_plugin.job_context.JobContext`, and calls
:meth:`EvaluateSuiteJob.run` with its concrete ``ctx`` signature.

This module is invoked by the platform's host-subprocess executor when a
caller submits an evaluate-suite job (``POST /apis/agents/.../jobs`` →
typed Jobs client create → controller dispatches ``python -m
nemo_agents_plugin.tasks.evaluate_suite``).
"""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteJob
from nemo_platform_plugin.errors import LocalRunError
from nemo_platform_plugin.sdk_provider import get_task_sdk
from nemo_platform_plugin.tasks.dispatcher import build_ctx_from_env, exit_code_for, read_step_config
from nemo_platform_plugin.tasks.logging_setup import configure_task_logging

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d).  Exiting.", signum)
    raise SystemExit(128 + signum)


def main() -> int:
    configure_task_logging()
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk("agents")
        ctx = build_ctx_from_env(sdk)
        config = read_step_config()
        job = EvaluateSuiteJob()
    except Exception:
        logger.exception("Failed to prepare task for agents")
        return 2
    try:
        return exit_code_for(job.run(config, ctx=ctx))
    except LocalRunError:
        raise
    except Exception:
        logger.exception("EvaluateSuiteJob.run raised")
        return 1


if __name__ == "__main__":
    sys.exit(main())
