# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF / Intake correlation tags for optimizer evaluations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

# RFC attribute names (see rfc-fabric-backed-agent-optimization.md).
ATIF_EXPERIMENT_ID = "nemo.optimizer.experiment_id"
ATIF_TRIAL_NUMBER = "nemo.optimizer.trial_number"
ATIF_REP = "nemo.optimizer.rep"
ATIF_ROW_ID = "nemo.optimizer.row_id"


def resolve_experiment_id(payload: Mapping[str, Any], *, generate_id: Callable[[], str]) -> str:
    """Return a stable experiment id from payload metadata or generate one."""
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        raw = metadata.get("experiment_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    optimizer = payload.get("optimizer")
    if isinstance(optimizer, Mapping):
        raw = optimizer.get("experiment_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    return generate_id()


def build_atif_trial_tags(
    *,
    experiment_id: str,
    trial_number: int,
    rep: int,
    row_id: str | None = None,
) -> dict[str, str | int]:
    """Build Relay ``AtifConfig.extra`` tags for one optimize trial execution."""
    tags: dict[str, str | int] = {
        ATIF_EXPERIMENT_ID: experiment_id,
        ATIF_TRIAL_NUMBER: trial_number,
        ATIF_REP: rep,
    }
    if row_id:
        tags[ATIF_ROW_ID] = row_id
    return tags
