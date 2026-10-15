# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for score parsing values."""

from __future__ import annotations

import math

import pytest
from nhx_evals_sdk.values import JSONScoreParser, RangeScore
from nhx_evals_sdk.values.scores import ScoreParserJSON


@pytest.mark.parametrize("text", ["3", "[3]"])
def test_json_score_parser_returns_nan_for_non_object_json(text: str) -> None:
    """Valid JSON primitives do not match the JSON object parser contract."""
    parser = ScoreParserJSON(
        score=RangeScore(name="quality", minimum=1, maximum=10, parser=JSONScoreParser(json_path="quality")),
        structured_output={"schema": {"type": "object", "properties": {"quality": {"type": "number"}}}},
    )

    score = parser.parse(text)

    assert math.isnan(score.value)
