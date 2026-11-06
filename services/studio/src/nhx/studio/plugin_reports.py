# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Studio plugin load-error report endpoint: validation, log formatting, throttling."""

from __future__ import annotations

import re
import time
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

WINDOW_SECONDS = 60.0
MAX_REPORTS_PER_WINDOW = 20

# C0, DEL and C1 control characters, plus Unicode characters that split lines in log viewers
# (U+2028/2029) or reorder/hide terminal text (bidi controls, zero-width and format characters).
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202e\u2066-\u2069\ufeff]")


class PluginIssueCode(str, Enum):
    MANIFEST_MISSING = "PLUGIN_MANIFEST_MISSING"
    SPECIFIER_UNMAPPED = "PLUGIN_SPECIFIER_UNMAPPED"
    SINGLETON_BUNDLED = "PLUGIN_SINGLETON_BUNDLED"
    CONTRACT_UNSUPPORTED = "PLUGIN_CONTRACT_UNSUPPORTED"
    VERSION_INCOMPATIBLE = "PLUGIN_VERSION_INCOMPATIBLE"
    LOAD_FAILED = "PLUGIN_LOAD_FAILED"


class PluginLoadIssue(BaseModel):
    code: PluginIssueCode
    subject: str = Field(max_length=200)
    message: str = Field(max_length=500)
    steps: list[Annotated[str, Field(max_length=300)]] = Field(default_factory=list, max_length=8)


class PluginLoadErrorReport(BaseModel):
    plugin: str = Field(pattern=r"^[a-z][a-z0-9-]+$", max_length=64)
    issues: list[PluginLoadIssue] = Field(min_length=1, max_length=10)


def _clean(text: str) -> str:
    return _CONTROL_CHARS.sub(" ", text)


def format_report(report: PluginLoadErrorReport) -> str:
    """Render a report as one single-line log message (` | ` delimited); every field is control-char free."""
    count = len(report.issues)
    parts = [f"Studio plugin '{_clean(report.plugin)}' failed to load ({count} issue{'s' if count != 1 else ''})"]
    for issue in report.issues:
        parts.append(f"[{issue.code.value}] {_clean(issue.message)}")
        parts.extend(f"{i}. {_clean(step)}" for i, step in enumerate(issue.steps, 1))
    return " | ".join(parts)


class ReportThrottle:
    """Rolling-window dedupe plus global rate limit.

    In-memory per process; fine for the single-process platform.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[str, frozenset[tuple[str, str]]], float] = {}
        self._accepted: list[float] = []

    def check(self, report: PluginLoadErrorReport) -> Literal["log", "duplicate", "limited"]:
        """Classify a report: duplicates are checked first so they never consume rate-limit budget."""
        now = time.monotonic()
        cutoff = now - WINDOW_SECONDS
        self._seen = {k: t for k, t in self._seen.items() if t > cutoff}
        self._accepted = [t for t in self._accepted if t > cutoff]

        key = (report.plugin, frozenset((i.code.value, i.subject) for i in report.issues))
        if key in self._seen:
            return "duplicate"
        if len(self._accepted) >= MAX_REPORTS_PER_WINDOW:
            return "limited"
        self._seen[key] = now
        self._accepted.append(now)
        return "log"
