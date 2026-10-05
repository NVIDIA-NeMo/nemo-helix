#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown extraction helpers for audit-spec documents."""

from __future__ import annotations

import re
from pathlib import Path

BEGIN_MARKER = "<!-- BEGIN:nemo-eval-author-audit:v1 -->"
END_MARKER = "<!-- END:nemo-eval-author-audit:v1 -->"

_BEGIN_MARKER_RE = re.compile(rf"(?m)^[ \t]*{re.escape(BEGIN_MARKER)}[ \t]*$")
_END_MARKER_RE = re.compile(rf"(?m)^[ \t]*{re.escape(END_MARKER)}[ \t]*$")
_YAML_BLOCK_RE = re.compile(
    r"^[ \t]*```(?:yaml|yml)[ \t]*\n(?P<body>.*?)\n[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


class AuditMarkdownError(ValueError):
    """Raised when the audit-spec block cannot be extracted from Markdown."""


def extract_schema_block(path: Path) -> str:
    """Return the YAML block between the audit markers in *path*."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AuditMarkdownError(f"could not read {path}: {exc}") from exc

    starts = [match.end() for match in _BEGIN_MARKER_RE.finditer(text)]
    ends = [match.start() for match in _END_MARKER_RE.finditer(text)]
    if len(starts) != 1 or len(ends) != 1:
        raise AuditMarkdownError(f"{path} must contain exactly one {BEGIN_MARKER!r} and one {END_MARKER!r}")
    start = starts[0]
    end = ends[0]
    if start >= end:
        raise AuditMarkdownError(f"{path} has audit markers in the wrong order")

    matches = list(_YAML_BLOCK_RE.finditer(text[start:end]))
    if len(matches) != 1:
        raise AuditMarkdownError("audit marker block must contain one fenced yaml block")
    return matches[0].group("body")
