# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    from scaled_evals.models.evaluations import EvaluationResultSummary
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_boolean_reward_preserves_type_and_has_no_numeric_projection() -> None:
    summary = EvaluationResultSummary(reward=True)

    assert summary.reward is True
    assert summary.legacy_numeric_reward() is None


def test_string_reward_preserves_type_and_has_no_numeric_projection() -> None:
    summary = EvaluationResultSummary(reward="passed")

    assert summary.reward == "passed"
    assert summary.legacy_numeric_reward() is None


def test_numeric_rewards_keep_backward_compatible_projection() -> None:
    integer = EvaluationResultSummary(reward=1)
    floating = EvaluationResultSummary(reward=0.5)

    assert type(integer.reward) is int
    assert integer.legacy_numeric_reward() == 1.0
    assert floating.legacy_numeric_reward() == 0.5
