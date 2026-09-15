# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module entry point: ``python -m nemo_profiler_plugin.tasks.profile``."""

import logging
import signal
import sys
from types import FrameType

from nemo_profiler_plugin.tasks.profile.run import run

logger = logging.getLogger(__name__)


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%s). Shutting down gracefully.", signum)
    sys.exit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _shutdown_handler)
    sys.exit(run())
