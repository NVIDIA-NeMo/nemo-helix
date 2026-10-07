# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared execution helpers."""

from __future__ import annotations

from dataclasses import dataclass

from nemo_evaluator_sdk.execution.utils import unique_metric_keys


@dataclass
class _TypedMetric:
    """A metric identified by a plain string type, one of the shapes `metric_type_name` accepts."""

    type: str


def _keys(*types: str) -> list[str]:
    return unique_metric_keys([_TypedMetric(type=t) for t in types])


def test_repeated_types_are_suffixed() -> None:
    assert _keys("a", "b", "a", "b") == ["a", "b", "a_2", "b_2"]


def test_a_type_colliding_with_a_suffixed_key_still_gets_a_unique_key() -> None:
    """A type whose own name ends in `_<n>` must not collide with another type's suffixed form.

    Callers key results by metric key, so a duplicate silently merges two metrics' scores into one
    bucket rather than failing.
    """
    keys = _keys("x", "x", "x_2")

    assert len(set(keys)) == len(keys)
    assert keys == ["x", "x_2", "x_2_2"]
