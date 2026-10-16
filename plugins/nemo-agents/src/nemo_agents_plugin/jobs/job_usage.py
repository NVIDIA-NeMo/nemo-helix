# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-usage normalization for Agents jobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nemo_helix_plugin.job_usage import JobTokenUsage


def fabric_output_token_usage(output: Any) -> JobTokenUsage | None:
    """Read provider totals from a terminal Fabric output, when present."""
    if not isinstance(output, Mapping):
        return None
    usage = output.get("usage")
    if not isinstance(usage, Mapping):
        return None

    input_tokens = _input_token_count(usage)
    output_tokens = _first_nonnegative_int(usage, "output_tokens", "completion_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    return JobTokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _first_nonnegative_int(values: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in values:
            return _nonnegative_int(values[key])
    return None


def _optional_nonnegative_int(values: Mapping[str, Any], key: str) -> int | None:
    if key not in values:
        return 0
    return _nonnegative_int(values[key])


def _input_token_count(values: Mapping[str, Any]) -> int | None:
    input_tokens = _first_nonnegative_int(values, "input_tokens", "prompt_tokens")
    if input_tokens is None:
        return None
    cache_tokens = [
        _optional_nonnegative_int(values, "cache_read_input_tokens"),
        _optional_nonnegative_int(values, "cache_creation_input_tokens"),
    ]
    if any(token_count is None for token_count in cache_tokens):
        return None
    return input_tokens + sum(token_count for token_count in cache_tokens if token_count is not None)


def _nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None
