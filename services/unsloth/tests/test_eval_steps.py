# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for default Unsloth validation cadence."""

from __future__ import annotations

from nhx.unsloth.tasks.training.backends.unsloth_sft import compute_default_eval_steps


def test_default_eval_steps_once_per_epoch() -> None:
    # commonsense_qa-scale: 9741 samples, effective batch 128 → 77 steps/epoch
    assert (
        compute_default_eval_steps(
            num_train_samples=9741,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=16,
        )
        == 76
    )


def test_default_eval_steps_respects_max_steps_cap() -> None:
    assert (
        compute_default_eval_steps(
            num_train_samples=9741,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=16,
            max_steps=50,
        )
        == 49
    )


def test_default_eval_steps_minimum_one() -> None:
    assert (
        compute_default_eval_steps(
            num_train_samples=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
        )
        == 1
    )
