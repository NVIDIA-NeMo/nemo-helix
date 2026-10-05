# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Training task entry point.

Usage:
    python -m nhx.rl.tasks.training

The runner reads the platform Jobs step config (``NEMO_JOB_STEP_CONFIG_FILE_PATH``),
builds a :class:`~nhx.rl.app.jobs.training.schemas.TrainingStepConfig`, and runs
the :class:`~nhx.rl.tasks.training.runner.TrainingRunner`.

In distributed (multi-node) training, all pods run this entry point.
The DistributedContext handles role detection and coordination:
- Rank 0 (coordinator): Runs all phases, reports progress
- Rank > 0 (workers): Participate in training, wait at barriers
"""

import logging
import sys

from .runner import TrainingRunner

logger = logging.getLogger(__name__)


def main() -> int:
    """Execute the training task."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        with TrainingRunner() as runner:
            result = runner.run()
            return 0 if result.success else 1
    except Exception as e:
        logger.exception(f"Training task failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
