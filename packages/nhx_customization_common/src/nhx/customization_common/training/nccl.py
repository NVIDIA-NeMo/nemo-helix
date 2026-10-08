# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL environment helpers shared by customization training backends."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_IB_SYSFS = Path("/sys/class/infiniband")

# PyTorch's process-group timeout is not applied to a blocking ncclCommInitRank.
# Non-blocking init polls ncclCommGetAsyncError and raises DistBackendError when
# this elapses, which exits the worker so torchrun returns and the pod fails.
# 30 minutes matches the distributed timeout used while a large model is loading
# and peers have not reached the first collective yet.
_NCCL_COMM_INIT_TIMEOUT_SECONDS = 30 * 60


def get_nccl_init_env() -> dict[str, str]:
    """Return env that aborts a multi-node NCCL join instead of hanging in it.

    A failed cross-node bootstrap blocks inside ``ncclCommInitRank``. Torchrun
    waits on that worker forever, so the pod stays Running. Non-blocking init
    turns the same failure into an error once ``TORCH_NCCL_NONBLOCKING_TIMEOUT``
    elapses. Values already present in the environment are left unchanged.
    """
    env: dict[str, str] = {}
    if not os.environ.get("TORCH_NCCL_USE_COMM_NONBLOCKING"):
        env["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
    if not os.environ.get("TORCH_NCCL_NONBLOCKING_TIMEOUT"):
        env["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = str(_NCCL_COMM_INIT_TIMEOUT_SECONDS)
    if env:
        timeout = os.environ.get("TORCH_NCCL_NONBLOCKING_TIMEOUT", str(_NCCL_COMM_INIT_TIMEOUT_SECONDS))
        logger.info("NCCL join fails the job after %ss if nodes cannot connect", timeout)
    return env


def get_multinode_nccl_env() -> dict[str, str]:
    """NCCL environment for a multi-node Automodel or RL job."""
    env = get_nccl_init_env()
    env.update(get_nccl_ib_env())
    return env


def get_nccl_ib_env() -> dict[str, str]:
    """Return NCCL overrides when Mellanox HCAs lack network devices."""
    if os.environ.get("NCCL_IB_HCA") or os.environ.get("NCCL_IB_DISABLE") or not _IB_SYSFS.is_dir():
        return {}

    usable: list[str] = []
    phantom: list[str] = []
    try:
        hcas = sorted(path for path in _IB_SYSFS.iterdir() if path.is_dir())
    except OSError:
        return {}

    for hca in hcas:
        if not hca.name.startswith("mlx"):
            continue
        try:
            has_netdev = any((hca / "device" / "net").iterdir())
        except OSError:
            has_netdev = False
        if has_netdev:
            usable.append(hca.name)
        else:
            phantom.append(hca.name)

    if not phantom:
        return {}

    if not usable:
        logger.info("Disabling NCCL IB because all detected Mellanox HCAs lack network devices")
        return {"NCCL_IB_DISABLE": "1"}

    hca_filter = ",".join(f"={hca}" for hca in usable)
    logger.info("Setting NCCL_IB_HCA=%s (excluded phantom HCAs: %s)", hca_filter, ",".join(phantom))
    return {"NCCL_IB_HCA": hca_filter}
