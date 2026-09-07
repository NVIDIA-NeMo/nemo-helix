# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for shared optimizer ATIF metadata helpers."""

from __future__ import annotations

from nemo_optimization.atif_metadata import (
    ATIF_EXPERIMENT_ID,
    ATIF_REP,
    ATIF_ROW_ID,
    ATIF_TRIAL_NUMBER,
    build_atif_trial_tags,
    resolve_experiment_id,
)

__all__ = [
    "ATIF_EXPERIMENT_ID",
    "ATIF_REP",
    "ATIF_ROW_ID",
    "ATIF_TRIAL_NUMBER",
    "build_atif_trial_tags",
    "resolve_experiment_id",
]
