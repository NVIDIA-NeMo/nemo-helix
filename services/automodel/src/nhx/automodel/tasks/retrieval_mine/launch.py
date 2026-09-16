# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch torch.distributed.run for the hard-negative mining recipe."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_MINING_SCRIPT = Path(__file__).with_name("mine_hard_negatives.py")


def run_hard_negative_mining(config_file: Path, nproc_per_node: str | None = None) -> None:
    """Run distributed hard-negative mining with ``config_file`` as the recipe config.

    Rank count is ``GPUS_PER_NODE`` when set, otherwise torchrun's ``gpu`` (visible devices).
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
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr or result.stdout or f"exit {result.returncode}"
        raise RuntimeError(f"Hard-negative mining failed: {detail}")
