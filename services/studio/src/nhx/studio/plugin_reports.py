# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Studio plugin load-error report endpoint: validation, log formatting, throttling."""

from __future__ import annotations

import re
import time
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

WINDOW_SECONDS = 60.0
MAX_REPORTS_PER_WINDOW = 20
MAX_REPORTS_PER_PLUGIN_PER_WINDOW = 5
# Largest valid report is ~31 KB; anything bigger is not a real report.
MAX_BODY_BYTES = 64 * 1024

# Schema limits. web/packages/studio/src/plugins/reportLoadFailures.ts clamps to the same numbers; keep in sync.
MAX_ISSUES = 10
MAX_SUBJECT = 200
MAX_MESSAGE = 500
MAX_STEPS = 8
MAX_STEP = 300

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# C0, DEL and C1 control characters, plus Unicode characters that split lines in log viewers
# (U+2028/2029) or reorder/hide terminal text (bidi controls, zero-width and format characters).
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202e\u2066-\u2069\ufeff]")


class PluginLoadIssue(BaseModel):
    # A pattern, not an enum: the TypeScript `IssueCode` plus 'PLUGIN_LOAD_FAILED'
    # (web/packages/studio/src/plugins/types.ts) must not become a 422 when a code is added.
    code: str = Field(pattern=r"^PLUGIN_[A-Z_]+$", max_length=64)
    subject: str = Field(max_length=MAX_SUBJECT)
    message: str = Field(max_length=MAX_MESSAGE)
    steps: list[Annotated[str, Field(max_length=MAX_STEP)]] = Field(default_factory=list, max_length=MAX_STEPS)


class PluginLoadErrorReport(BaseModel):
    plugin: str = Field(pattern=r"^[a-z][a-z0-9-]+$", max_length=64)
    issues: list[PluginLoadIssue] = Field(min_length=1, max_length=MAX_ISSUES)


def is_loopback_origin(origin: str) -> bool:
    """True when an Origin header names a loopback host (`null` and unparseable values are not)."""
    try:
        return urlsplit(origin).hostname in LOOPBACK_HOSTS
    except ValueError:
        return False


def _clean(text: str) -> str:
    return _CONTROL_CHARS.sub(" ", text)


def format_report(report: PluginLoadErrorReport) -> str:
    """Render a report as one single-line log message (` | ` delimited); every field is control-char free."""
    count = len(report.issues)
    parts = [f"Studio plugin '{report.plugin}' failed to load ({count} issue{'s' if count != 1 else ''})"]
    for issue in report.issues:
        parts.append(f"[{issue.code}] {_clean(issue.message)}")
        parts.extend(f"{i}. {_clean(step)}" for i, step in enumerate(issue.steps, 1))
    return " | ".join(parts)


class ReportThrottle:
    """Rolling-window dedupe, per-plugin cap and global rate limit.

    In-memory per process; fine for the single-process platform.
    """

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._accepted: dict[str, list[float]] = {}

    def check(self, plugin: str, rendered: str) -> Literal["log", "duplicate", "limited"]:
        """Classify a report by its rendered log text.

        Order: duplicates first (free), then the per-plugin cap, then the global cap.
        """
        now = time.monotonic()
        cutoff = now - WINDOW_SECONDS
        self._seen = {k: t for k, t in self._seen.items() if t > cutoff}
        self._accepted = {p: live for p, ts in self._accepted.items() if (live := [t for t in ts if t > cutoff])}

        if rendered in self._seen:
            return "duplicate"
        if len(self._accepted.get(plugin, ())) >= MAX_REPORTS_PER_PLUGIN_PER_WINDOW:
            return "limited"
        if sum(len(ts) for ts in self._accepted.values()) >= MAX_REPORTS_PER_WINDOW:
            return "limited"
        self._seen[rendered] = now
        self._accepted.setdefault(plugin, []).append(now)
        return "log"
