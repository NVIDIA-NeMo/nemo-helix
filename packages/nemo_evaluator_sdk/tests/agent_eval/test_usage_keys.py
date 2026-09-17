# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_evaluator_sdk.agent_eval.usage_keys import (
    COMPLETION_TOKEN_KEYS,
    PROMPT_TOKEN_KEYS,
    SEPARATE_CACHE_KEYS,
    USAGE_COUNT_KEYS,
    USAGE_DETAILS_KEYS,
    _first_nonnegative_int,
    _first_usage_details,
)


def test_usage_count_keys_match_warn_loop_order() -> None:
    assert USAGE_COUNT_KEYS == PROMPT_TOKEN_KEYS + COMPLETION_TOKEN_KEYS + SEPARATE_CACHE_KEYS
    assert USAGE_COUNT_KEYS == (
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )
    assert USAGE_DETAILS_KEYS == ("prompt_tokens_details", "input_tokens_details")


@pytest.mark.parametrize(
    ("values", "keys", "expected"),
    [
        ({"prompt_tokens": 8, "input_tokens": 3}, ("prompt_tokens", "input_tokens"), 8),
        ({"input_tokens": 3}, ("prompt_tokens", "input_tokens"), 3),
        ({"prompt_tokens": None, "input_tokens": 3}, ("prompt_tokens", "input_tokens"), 3),
        ({"prompt_tokens": True, "input_tokens": 0}, ("prompt_tokens", "input_tokens"), 0),
        ({"prompt_tokens": False, "input_tokens": 1}, ("prompt_tokens", "input_tokens"), 1),
        ({"prompt_tokens": "8", "input_tokens": 2}, ("prompt_tokens", "input_tokens"), 2),
        ({"prompt_tokens": 1.5, "input_tokens": 2}, ("prompt_tokens", "input_tokens"), 2),
        ({"prompt_tokens": -1, "input_tokens": 2}, ("prompt_tokens", "input_tokens"), 2),
        ({}, ("prompt_tokens", "input_tokens"), None),
        ({"prompt_tokens": None}, ("prompt_tokens",), None),
    ],
)
def test_first_nonnegative_int_skips_unusable_aliases(
    values: dict[str, object],
    keys: tuple[str, ...],
    expected: int | None,
) -> None:
    assert _first_nonnegative_int(values, *keys) == expected


def test_first_usage_details_skips_unusable_preferred_alias() -> None:
    details = {"cached_tokens": 3}

    assert _first_usage_details({"prompt_tokens_details": "invalid", "input_tokens_details": details}) is details
    assert _first_usage_details({"prompt_tokens_details": None}) is None
