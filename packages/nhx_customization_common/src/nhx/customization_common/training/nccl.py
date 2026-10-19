# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL environment helpers shared by customization training backends."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_IB_SYSFS = Path("/sys/class/infiniband")


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
