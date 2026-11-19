# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find the JSON object an LLM meant to return in a reply that may wrap it in prose, fences or comments."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from typing import Any

import json_repair

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def top_level_objects(text: str) -> list[str]:
    """Every balanced top-level ``{...}`` span; braces inside JSON strings don't count."""
    spans: list[str] = []
    depth = 0
    start = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"' and depth > 0:
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : index + 1])
    return spans


def _parse(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    try:
        return json_repair.loads(candidate)
    except (ValueError, RecursionError):
        return None


def _candidates(text: str) -> Iterator[str]:
    stripped = _THINK.sub("", text)
    yield from reversed([match.group(1) for match in _FENCE.finditer(stripped)])
    yield from sorted(top_level_objects(stripped), key=len, reverse=True)


def parse_error(text: str) -> str:
    """Why the largest candidate in *text* is not the expected JSON object, for logs and the retry prompt."""
    candidates = list(_candidates(text))
    if not candidates:
        return "no {...} object in the reply"
    largest = max(candidates, key=len)
    try:
        json.loads(largest)
    except json.JSONDecodeError as error:
        return f"{error.msg} at line {error.lineno} column {error.colno} of a {len(largest)}-character object"
    return "no object with the expected keys"


def json_object_in(text: str, *, keys: Sequence[str] = ()) -> dict[str, Any] | None:
    """The first candidate that parses (strictly, then repaired) to an object holding one of *keys*.

    Candidates are fenced blocks, last first, then top-level brace spans, largest first. Requiring an expected key
    keeps a stray ``{{ column }}`` in prose from passing as the answer.
    """
    for candidate in _candidates(text):
        parsed = _parse(candidate)
        if isinstance(parsed, dict) and parsed and (not keys or any(key in parsed for key in keys)):
            return parsed
    return None
