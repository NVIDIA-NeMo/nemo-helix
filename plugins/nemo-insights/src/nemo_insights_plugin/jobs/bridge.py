# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-process entry point for insights analyzer jobs."""

import signal
import sys
from types import FrameType

from nemo_helix_plugin.sdk_provider import get_task_sdk
from nemo_helix_plugin.tasks.dispatcher import run_task
from nemo_insights_plugin.jobs.analyze import AnalyzeJob


def _shutdown(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def main() -> int:
    """Run the platform-injected analyzer task config."""
    signal.signal(signal.SIGTERM, _shutdown)
    return run_task(AnalyzeJob, sdk=get_task_sdk("insights"))


if __name__ == "__main__":
    sys.exit(main())
