# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for ``agents.evaluate-suite`` (``python -m nemo_agents_plugin.tasks.evaluate_suite``).

Mirrors :mod:`nemo_agents_plugin.tasks.evaluate`: delegates to
:func:`nemo_helix_plugin.tasks.dispatcher.run_task` so step config loading,
:class:`~nemo_helix_plugin.job_context.JobContext` construction, and
signature-based DI into :meth:`EvaluateSuiteJob.run` are all handled by the
framework.

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
from nemo_helix_plugin.sdk_provider import get_task_sdk
from nemo_helix_plugin.tasks.dispatcher import run_task

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d).  Exiting.", signum)
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk("agents")
    except Exception:
        logger.exception("Failed to build task SDK for agents")
        return 2
    return run_task(EvaluateSuiteJob, sdk=sdk)


if __name__ == "__main__":
    sys.exit(main())
