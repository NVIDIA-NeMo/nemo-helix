# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Training task entry point.

Usage:
    python -m nhx.automodel.tasks.training

In distributed (multi-node) training, all pods run this entry point.
The DistributedContext handles role detection and coordination:
- Rank 0 (coordinator): Runs all phases, reports progress
- Rank > 0 (workers): Participate in training, wait at barriers
"""

import logging
import sys

from .runner import TrainingRunner

logger = logging.getLogger(__name__)


def run() -> int:
    """Execute training task."""
    try:
        with TrainingRunner() as runner:
            result = runner.run()
            return 0 if result.success else 1
    except Exception as e:
        logger.exception(f"Training task failed: {e}")
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.exit(run())
