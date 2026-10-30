# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reward functions from NeMo-RL ``nemo_rl.environments.rewards``.

Ported so a Gym resources server can score the same way ``VLMEnvironment`` does.
``combine_reward_functions`` renormalizes weights, so ``0.2`` and ``0.8`` stay
``0.2`` and ``0.8``. ``math_verify`` is imported only by the functions that use it.
"""

from __future__ import annotations

import re
from functools import partial
from typing import Any, Callable

import numpy as np

RewardFn = Callable[..., tuple[float, bool | None]]


def _math_verify():
    from math_verify.errors import TimeoutException  # ty: ignore[unresolved-import]
    from math_verify.metric import math_metric  # ty: ignore[unresolved-import]
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # ty: ignore[unresolved-import]

    return TimeoutException, math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )


def _boxed(value: str) -> str:
    return value if value.startswith("\\boxed{") else "\\boxed{" + value + "}"


def math_expression_reward(ground_truth: str, response: str, tag: str = "answer") -> tuple[float, bool]:
    """Reward when the ``<{tag}>`` answer is the same expression as the ground truth."""
    timeout, math_verify_func = _math_verify()
    match = re.search(rf"<{tag}>([\s\S]*)</{tag}>", response)
    if match:
        answer = match.group(1)
        try:
            score, _ = math_verify_func([_boxed(ground_truth)], [_boxed(answer)])
            return float(score), score > 0.1
        except (Exception, timeout):
            return 0.0, False
    return 0.0, False


def format_reward(
    ground_truth: str,
    response: str,
    think_tag: str = "think",
    answer_tag: str = "answer",
) -> tuple[float, None]:
    """Reward ``<{think_tag}>...</{think_tag}>`` (0.25) and ``<{answer_tag}>...</{answer_tag}>`` (0.75)."""
    reward = 0.0
    if re.search(rf"<{think_tag}>[\s\S]*</{think_tag}>", response):
        reward += 0.25
    if re.search(rf"<{answer_tag}>[\s\S]*</{answer_tag}>", response):
        reward += 0.75
    return reward, None


def exact_answer_alphanumeric_reward(
    ground_truth: str, response: str, answer_tag: str = "answer"
) -> tuple[float, bool]:
    """Reward when the ``<{answer_tag}>`` text matches the ground truth after dropping non-alphanumerics."""
    match = re.search(rf"<{answer_tag}>([\s\S]*)</{answer_tag}>", response)
    if match:
        answer = match.group(1)
        answer_clean = "".join(char for char in answer if char.isalnum()).lower()
        ground_truth_clean = "".join(char for char in ground_truth if char.isalnum()).lower()
        if answer_clean == ground_truth_clean:
            return 1.0, True
    return 0.0, False


def bbox_giou_reward(
    ground_truth: str,
    response: str,
    giou_penalty_thres: float = 10.0,
    answer_tag: str = "answer",
) -> tuple[float, bool]:
    """GIoU of ``[x1, y1, x2, y2]`` boxes inside ``<{answer_tag}>`` tags."""
    match = re.search(rf"<{answer_tag}>([\s\S]*)</{answer_tag}>", response)
    if match:
        answer = match.group(1)
    else:
        return 0.0, False

    try:
        x1g, y1g, x2g, y2g = [float(value) for value in ground_truth.replace("[", "").replace("]", "").split(",")]
        x1r, y1r, x2r, y2r = [float(value) for value in answer.replace("[", "").replace("]", "").split(",")]
    except ValueError:
        return 0.0, False

    area_g = (x2g - x1g) * (y2g - y1g)
    area_r = (x2r - x1r) * (y2r - y1r)
    area_i = max(0.0, min(x2g, x2r) - max(x1g, x1r)) * max(0.0, min(y2g, y2r) - max(y1g, y1r))
    area_u = max(1e-3, area_g + area_r - area_i)
    iou = area_i / area_u
    if iou < giou_penalty_thres:
        area_c = max(1e-3, (max(x2g, x2r) - min(x1g, x1r)) * (max(y2g, y2r) - min(y1g, y1r)))
        giou = iou - (area_c - area_u) / area_c
    else:
        giou = iou
    return giou, giou > 0.5


def geo3k_reward(ground_truth: str, response: str, format_score: float = 0.1) -> tuple[float, bool]:
    """MMPR-Tiny reward: ``</think>`` plus ``\\boxed{}`` format, graded with math_verify."""
    timeout, math_verify_func = _math_verify()
    has_format = bool(re.search(r"</think>.*\\boxed\{.*\}", response, re.DOTALL))
    format_reward_value = 1.0 if has_format else 0.0
    try:
        boxed_token = "\\boxed{"
        boxed_start = response.rfind(boxed_token)
        answer = None
        if boxed_start != -1:
            start = boxed_start + len(boxed_token)
            depth = 1
            for index, char in enumerate(response[start:]):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                if depth == 0:
                    answer = response[start : start + index].strip()
                    break
        if answer is not None:
            score, _ = math_verify_func([_boxed(ground_truth)], [_boxed(answer)])
            is_correct = score > 0.1
            accuracy = 1.0 if is_correct else 0.0
        else:
            accuracy = 0.0
            is_correct = False
    except (Exception, timeout):
        accuracy = 0.0
        is_correct = False
    return (1.0 - format_score) * accuracy + format_score * format_reward_value, is_correct


def combine_reward_functions(
    reward_functions: list[tuple[RewardFn, float]],
) -> Callable[[str, str], tuple[float, bool]]:
    """Weight the functions and renormalize the weights to sum to 1."""
    weights = np.array([weight for _, weight in reward_functions])
    weights = weights / np.sum(weights)

    def combined(ground_truth: str, response: str) -> tuple[float, bool]:
        scores = [reward_func(ground_truth, response) for reward_func, _ in reward_functions]
        rewards = [score[0] for score in scores]
        correctness = [score[1] for score in scores if score[1] is not None]
        return float(np.sum(np.array(rewards) * weights)), all(correctness)

    return combined


REWARD_FUNCTIONS: dict[str, RewardFn] = {
    "format": format_reward,
    "exact_alnum": exact_answer_alphanumeric_reward,
    "math_expr": math_expression_reward,
    "bbox_giou": bbox_giou_reward,
    "geo3k": geo3k_reward,
}

_ANSWER_REWARD_NAMES = {"exact_alnum", "math_expr", "bbox_giou", "geo3k"}


def _answer_text(response: str, tag: str = "answer") -> str | None:
    match = re.search(rf"<{tag}>([\s\S]*)</{tag}>", response)
    if match is None:
        return None
    return match.group(1).strip()


def _as_spec(spec: Any) -> tuple[str, float, dict[str, Any] | None]:
    if isinstance(spec, dict):
        return str(spec["name"]), float(spec["weight"]), spec.get("kwargs")
    return str(spec.name), float(spec.weight), spec.kwargs


def score_response(response: str, expected_answer: str, reward_functions: list[Any]) -> dict[str, Any]:
    """Score one response the way ``VLMVerifyWorker.verify`` scores a pair.

    ``format_reward`` is set when ``format`` is configured. ``answer_reward`` is the
    last configured answer function (``exact_alnum``, ``math_expr``, ``bbox_giou``,
    or ``geo3k``). ``reward`` is the renormalized weighted sum.
    """
    if not reward_functions:
        raise ValueError("No reward functions provided")
    bound: list[tuple[RewardFn, float]] = []
    format_score: float | None = None
    answer_score: float | None = None
    for spec in reward_functions:
        name, weight, kwargs = _as_spec(spec)
        try:
            function = REWARD_FUNCTIONS[name]
        except KeyError as exc:
            raise ValueError(f"Invalid reward function: {name}") from exc
        if kwargs:
            function = partial(function, **kwargs)
        score, _ = function(expected_answer, response)
        if name == "format":
            format_score = float(score)
        elif name in _ANSWER_REWARD_NAMES:
            answer_score = float(score)
        bound.append((function, weight))
    reward, _ = combine_reward_functions(bound)(expected_answer, response)
    return {
        "reward": float(reward),
        "extracted_answer": _answer_text(response),
        "format_reward": format_score,
        "answer_reward": answer_score,
    }
