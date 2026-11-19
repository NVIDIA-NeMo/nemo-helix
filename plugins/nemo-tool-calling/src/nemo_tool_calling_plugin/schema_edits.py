# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply JSON Pointer edits from the analyzer to a published tool input schema."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaEdit:
    tool: str
    path: str
    value: Any


def pointer_segments(path: str) -> list[str]:
    """RFC 6901 pointer segments; ``""`` is the whole document."""
    if path == "":
        return []
    if not path.startswith("/"):
        raise ValueError("a JSON Pointer must be empty or start with '/'")
    return [segment.replace("~1", "/").replace("~0", "~") for segment in path[1:].split("/")]


def _child(node: Any, segment: str) -> Any:
    if isinstance(node, dict):
        if segment not in node:
            raise KeyError(segment)
        return node[segment]
    if isinstance(node, list) and segment.isdigit() and int(segment) < len(node):
        return node[int(segment)]
    raise KeyError(segment)


def _apply(schema: Any, edit: SchemaEdit) -> Any:
    segments = pointer_segments(edit.path)
    if not segments:
        return copy.deepcopy(edit.value)
    parent = schema
    for depth, segment in enumerate(segments[:-1]):
        try:
            parent = _child(parent, segment)
        except KeyError:
            raise ValueError(f"'/{'/'.join(segments[: depth + 1])}' does not exist") from None
    last = segments[-1]
    if isinstance(parent, dict):
        parent[last] = copy.deepcopy(edit.value)
    elif isinstance(parent, list) and last == "-":
        parent.append(copy.deepcopy(edit.value))
    elif isinstance(parent, list) and last.isdigit() and int(last) < len(parent):
        parent[int(last)] = copy.deepcopy(edit.value)
    else:
        raise ValueError(f"cannot set {last!r} on a {type(parent).__name__}")
    return schema


def apply_edits(schema: dict[str, Any], edits: Sequence[SchemaEdit]) -> tuple[dict[str, Any], list[str]]:
    """A patched copy of *schema* and one message per edit that could not be applied."""
    patched: Any = copy.deepcopy(schema)
    skipped: list[str] = []
    for edit in edits:
        try:
            updated = _apply(patched, edit)
        except ValueError as error:
            skipped.append(f"{edit.path or '<root>'}: {error}")
            continue
        if not isinstance(updated, dict):
            skipped.append(f"{edit.path or '<root>'}: the input schema must stay an object")
            continue
        patched = updated
    return patched, skipped
