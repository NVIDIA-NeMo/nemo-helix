# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch and log plumbing for the hard-negative mining recipe."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

_MINING_SCRIPT = Path(__file__).with_name("mine_hard_negatives.py")
logger = logging.getLogger(__name__)


def run_hard_negative_mining(config_file: Path, nproc_per_node: str | None = None) -> None:
    """Run distributed hard-negative mining with ``config_file`` as the recipe config.

    Rank count is ``GPUS_PER_NODE`` when set, otherwise torchrun's ``gpu`` (visible devices).

    Stdout/stderr are inherited so the jobs log sidecar can scrape miner progress.
    Do not ``capture_output``: that buffers the whole run and hides tqdm/logging
    until the process exits.
    """
    nproc_per_node = nproc_per_node or os.environ.get("GPUS_PER_NODE") or "gpu"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        nproc_per_node,
        str(_MINING_SCRIPT),
        "--config",
        str(config_file),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    logger.info("Launching hard-negative mining: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Hard-negative mining failed: exit {exc.returncode}") from exc
    logger.info("Hard-negative mining subprocess exited 0")
