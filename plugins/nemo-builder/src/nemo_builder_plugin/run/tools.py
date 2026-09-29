# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running crane and cosign: the one subprocess helper, for the steps and the broker alike."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Mapping

from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

#: crane and cosign are given this long each. A registry that stops answering must not hold a
#: process -- or a credential it has materialized -- until some other deadline.
TOOL_TIMEOUT_SECONDS = 30 * 60


def run_tool(args: list[str], *, env: Mapping[str, str] | None = None) -> str:
    """Run crane or cosign, log what it said, and raise on a non-zero exit.

    stderr is logged only on failure: both tools are chatty on success, and a failure is the one
    case where their explanation is the thing a reader needs.

    ``env`` is laid over this process's environment for this call alone. The broker serves several
    requests at once, each with its own registry token, so it cannot set ``DOCKER_CONFIG``
    process-wide the way a step can.
    """
    logger.info("$ %s", sanitize_for_log(" ".join(args)))
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=TOOL_TIMEOUT_SECONDS,
            env={**os.environ, **env} if env else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{args[0]} did not finish within {TOOL_TIMEOUT_SECONDS}s") from exc
    if result.stdout:
        logger.info("%s", sanitize_for_log(result.stdout.strip()))
    if result.returncode != 0:
        logger.error("%s", sanitize_for_log(result.stderr.strip()))
        raise RuntimeError(f"{args[0]} failed with exit {result.returncode}")
    return result.stdout.strip()
