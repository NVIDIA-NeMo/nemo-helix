# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for ``agents.evaluate`` (``python -m nemo_agents_plugin.tasks.evaluate``).

Delegates to :func:`nemo_helix_plugin.tasks.dispatcher.run_task` so step
config loading, :class:`~nemo_helix_plugin.job_context.JobContext` construction,
and signature-based DI of ``ctx`` / ``sdk`` into
:meth:`EvaluateAgentJob.run` are all handled by the framework.  This
module's only local responsibilities are SIGTERM handling and SDK
construction (the ``"agents"`` service identity is plugin-specific).
"""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from nemo_agents_plugin.jobs.evaluate_agent import EvaluateAgentJob
from nemo_helix_plugin.sdk_provider import get_task_sdk
from nemo_helix_plugin.tasks.dispatcher import run_task

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d).  Exiting.", signum)
    # 128 + signum is the conventional shell exit code for signal-terminated
    # processes; non-zero so the scheduler distinguishes cancellations and
    # timeouts from a clean completion.
    raise SystemExit(128 + signum)


def main() -> int:
    """Build the on-behalf-of SDK and dispatch to ``run_task``.

    SDK construction lives here (not as an inline argument to
    ``run_task``) so failures during ``get_task_sdk`` — missing
    ``NHX_PRINCIPAL``, malformed base URL, network errors building the
    internal-auth client — collapse to the same setup-error exit code
    (``2``) the dispatcher uses for env / step-config setup failures
    rather than crashing with an uncaught exception.
    """
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk("agents")
    except Exception:
        logger.exception("Failed to build task SDK for agents")
        return 2
    return run_task(EvaluateAgentJob, sdk=sdk)


if __name__ == "__main__":
    sys.exit(main())
