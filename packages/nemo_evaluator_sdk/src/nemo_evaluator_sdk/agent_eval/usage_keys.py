# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider usage-key aliases shared by live eval, Gym, and the Intake row adapter.

Owns the Chat Completions / Responses / Anthropic vocabulary those adapters would otherwise
each restate: which keys mean prompt, completion, and cache, and the preference order when a
payload carries more than one name for the same count. Kept stdlib-only so the light
``agent_eval`` path, Gym results, and the evaluator plugin can depend on it without pulling
each other in.

Tuples are preference-ordered alias groups, not unordered sets. Chat Completions names come
first so a mixed-schema payload is read as chat, not Responses.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Chat Completions ``prompt_tokens`` before Responses ``input_tokens``.
PROMPT_TOKEN_KEYS = ("prompt_tokens", "input_tokens")
#: Chat Completions ``completion_tokens`` before Responses ``output_tokens``.
COMPLETION_TOKEN_KEYS = ("completion_tokens", "output_tokens")
CACHE_READ_INPUT_TOKENS_KEY = "cache_read_input_tokens"
CACHE_CREATION_INPUT_TOKENS_KEY = "cache_creation_input_tokens"
SEPARATE_CACHE_KEYS = (CACHE_READ_INPUT_TOKENS_KEY, CACHE_CREATION_INPUT_TOKENS_KEY)
#: Warn-loop order: prompt aliases, then completion aliases, then Anthropic cache fields.
USAGE_COUNT_KEYS = PROMPT_TOKEN_KEYS + COMPLETION_TOKEN_KEYS + SEPARATE_CACHE_KEYS
#: Chat Completions details before Responses details.
USAGE_DETAILS_KEYS = ("prompt_tokens_details", "input_tokens_details")
CACHED_TOKENS_KEY = "cached_tokens"


def _first_usage_details(usage: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the first mapping-shaped usage-details field in preference order."""
    for key in USAGE_DETAILS_KEYS:
        details = usage.get(key)
        if isinstance(details, Mapping):
            return details
    return None


def _first_nonnegative_int(values: Mapping[str, Any], *keys: str) -> int | None:
    """Return the first non-negative int among ``keys``, else ``None``.

    Usage objects alias the same count under different names (``prompt_tokens`` vs
    ``input_tokens``). Scanning aliases in preference order picks the first usable value
    without treating a missing key as zero. ``bool`` is excluded because it is a subclass
    of ``int``; strings, floats, and negatives are skipped so a malformed count is omitted
    rather than coerced.
    """
    for key in keys:
        value = values.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None
