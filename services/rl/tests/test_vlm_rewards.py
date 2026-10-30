# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

REWARDS = Path(__file__).parents[3] / "scripts" / "grpo-examples" / "servers" / "vlm" / "rewards.py"
SPEC = importlib.util.spec_from_file_location("vlm_rewards", REWARDS)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

NEMO_REWARDS = Path(__file__).parents[4] / "NeMo-RL-nhx-rl" / "nemo_rl" / "environments" / "rewards.py"
CLEVR = [{"name": "format", "weight": 0.2}, {"name": "exact_alnum", "weight": 0.8}]


def _nemo():
    pytest.importorskip("math_verify")
    if not NEMO_REWARDS.is_file():
        pytest.skip("NeMo-RL checkout is not beside nemo-platform")
    spec = importlib.util.spec_from_file_location("nemo_rl_rewards_ref", NEMO_REWARDS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clevr_mix_matches_the_recipe_weights() -> None:
    both = "<think>count the cubes</think><answer>12.</answer>"
    scored = MODULE.score_response(both, "12", CLEVR)
    assert scored["format_reward"] == 1.0
    assert scored["answer_reward"] == 1.0
    assert scored["reward"] == pytest.approx(1.0)
    assert scored["extracted_answer"] == "12."

    wrong = "<think>count</think><answer>no</answer>"
    scored = MODULE.score_response(wrong, "yes", CLEVR)
    assert scored["format_reward"] == 1.0
    assert scored["answer_reward"] == 0.0
    assert scored["reward"] == pytest.approx(0.2)

    answer_only = "<answer>Yes</answer>"
    scored = MODULE.score_response(answer_only, "yes", CLEVR)
    assert scored["format_reward"] == 0.75
    assert scored["answer_reward"] == 1.0
    assert scored["reward"] == pytest.approx(0.95)

    bare = "12"
    scored = MODULE.score_response(bare, "12", CLEVR)
    assert scored["format_reward"] == 0.0
    assert scored["answer_reward"] == 0.0
    assert scored["reward"] == pytest.approx(0.0)
    assert scored["extracted_answer"] is None


def test_unknown_reward_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid reward function"):
        MODULE.score_response("x", "y", [{"name": "bleu", "weight": 1.0}])


def test_bbox_giou_of_the_same_box_is_one() -> None:
    response = "<answer>[0, 0, 1, 1]</answer>"
    score, correct = MODULE.bbox_giou_reward("[0, 0, 1, 1]", response)
    assert score == pytest.approx(1.0)
    assert correct is True


def test_math_expr_and_geo3k_grade_a_matching_answer() -> None:
    pytest.importorskip("math_verify")
    score, correct = MODULE.math_expression_reward("2", "<answer>2</answer>")
    assert score == pytest.approx(1.0)
    assert correct is True
    score, correct = MODULE.geo3k_reward("2", "</think> \\boxed{2}")
    assert score == pytest.approx(1.0)
    assert correct is True


def test_ported_functions_match_nemo_rl() -> None:
    nemo = _nemo()
    pairs = [
        ("format_reward", ("12", "<think>a</think><answer>12</answer>"), {}),
        ("format_reward", ("12", "<answer>12</answer>"), {}),
        ("exact_answer_alphanumeric_reward", ("12", "<answer>12.</answer>"), {}),
        ("exact_answer_alphanumeric_reward", ("yes", "<answer>Yes</answer>"), {}),
        ("exact_answer_alphanumeric_reward", ("yes", "<answer>no</answer>"), {}),
        ("bbox_giou_reward", ("[0, 0, 1, 1]", "<answer>[0, 0, 1, 1]</answer>"), {}),
        ("math_expression_reward", ("2", "<answer>2</answer>"), {}),
        ("geo3k_reward", ("2", "</think> \\boxed{2}"), {}),
    ]
    for name, args, kwargs in pairs:
        assert getattr(MODULE, name)(*args, **kwargs) == getattr(nemo, name)(*args, **kwargs)
