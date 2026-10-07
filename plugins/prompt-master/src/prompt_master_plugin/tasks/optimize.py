# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for the prompt-master strategy (``python -m prompt_master_plugin.tasks.optimize``)."""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType

from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.tasks.dispatcher import build_ctx_from_env, exit_code_for, read_step_config
from nemo_helix_plugin.tasks.logging_setup import configure_task_logging
from prompt_master_plugin.jobs.optimize import PromptMasterOptimizeJob

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d). Exiting.", signum)
    raise SystemExit(128 + signum)


def main() -> int:
    configure_task_logging()
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        client = get_task_nemo_client("agents")
        ctx = build_ctx_from_env(client)
        config = read_step_config()
        job = PromptMasterOptimizeJob()
    except Exception:
        logger.exception("Failed to prepare the prompt-master task")
        return 2
    try:
        return exit_code_for(job.run(config, ctx=ctx, sdk=client))
    except LocalRunError:
        raise
    except Exception:
        logger.exception("PromptMasterOptimizeJob.run raised")
        return 1


if __name__ == "__main__":
    sys.exit(main())
