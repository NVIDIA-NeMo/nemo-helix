# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ``main`` for the plugin's task modules (``python -m nemo_agent_hardener_plugin.tasks.<job>``).

The executor spawns a task module with the ``NEMO_JOB_*`` env populated; :func:`run_job_task` loads the step
config, builds a ``JobContext`` and calls the job's ``run`` with an on-behalf-of task client.
"""

from __future__ import annotations

import logging
import signal
from types import FrameType

from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.tasks.dispatcher import build_ctx_from_env, exit_code_for, read_step_config
from nemo_helix_plugin.tasks.logging_setup import configure_task_logging

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, _frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d). Exiting.", signum)
    raise SystemExit(128 + signum)


def run_job_task(job_type: type[NemoJob]) -> int:
    """Run one *job_type* step in this process and return its exit code (2 when the task cannot start)."""
    configure_task_logging()
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        client = get_task_nemo_client("agent-hardener")
        ctx = build_ctx_from_env(client)
        config = read_step_config()
        job = job_type()
    except Exception as exc:
        logger.error("Failed to prepare task for agent-hardener: %s", exc)
        return 2
    try:
        return exit_code_for(job.run(config, ctx=ctx, sdk=client))
    except LocalRunError:
        raise
    except Exception as exc:
        logger.error("%s.run failed: %s", job_type.__name__, exc)
        return 1
