# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for the stub ATIF convert job.

Invoked as ``python -m nemo_customizer.tasks.atif_convert`` inside ``nhx-customizer-tasks``.
"""

from __future__ import annotations

import logging
import sys

from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.tasks.dispatcher import build_ctx_from_env, exit_code_for, read_step_config
from nemo_helix_plugin.tasks.logging_setup import configure_task_logging

from nemo_customizer.jobs.atif_convert import SOURCE, AtifConvertJob

logger = logging.getLogger(__name__)


def main() -> int:
    configure_task_logging()
    try:
        client = get_task_nemo_client(SOURCE)
        ctx = build_ctx_from_env(client)
        config = read_step_config()
        job = AtifConvertJob()
    except Exception:
        logger.exception("Failed to prepare the atif-convert task")
        return 2
    try:
        return exit_code_for(job.run(config, ctx=ctx, sdk=client))
    except LocalRunError:
        raise
    except Exception:
        logger.exception("AtifConvertJob.run raised")
        return 1


if __name__ == "__main__":
    sys.exit(main())
