# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_optimization.atif_metadata import (
    ATIF_EXPERIMENT_ID,
    ATIF_REP,
    ATIF_ROW_ID,
    ATIF_TRIAL_NUMBER,
    build_atif_trial_tags,
    resolve_experiment_id,
)


def test_resolve_experiment_id_from_metadata() -> None:
    payload = {"metadata": {"experiment_id": "exp-from-metadata"}}
    assert resolve_experiment_id(payload, generate_id=lambda: "generated") == "exp-from-metadata"


def test_resolve_experiment_id_from_optimizer() -> None:
    payload = {"optimizer": {"experiment_id": "exp-from-optimizer"}}
    assert resolve_experiment_id(payload, generate_id=lambda: "generated") == "exp-from-optimizer"


def test_resolve_experiment_id_generates_when_missing() -> None:
    assert resolve_experiment_id({}, generate_id=lambda: "optimize-abc") == "optimize-abc"


def test_build_atif_trial_tags() -> None:
    tags = build_atif_trial_tags(experiment_id="exp-1", trial_number=3, rep=1, row_id="row-a")
    assert tags == {
        ATIF_EXPERIMENT_ID: "exp-1",
        ATIF_TRIAL_NUMBER: 3,
        ATIF_REP: 1,
        ATIF_ROW_ID: "row-a",
    }
