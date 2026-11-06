# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Studio plugin load-error report endpoint: validation, log formatting, throttling."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Annotated

from fastapi import status
from pydantic import BaseModel, Field
from starlette.responses import Response

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0
MAX_REPORTS_PER_WINDOW = 20

# C0, DEL and C1 control characters (includes \n, \r, \x1b).
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


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
    """Render a report as one multi-line log message; every field is control-char free."""
    lines = [f"Studio plugin '{_clean(report.plugin)}' reported {len(report.issues)} load issue(s)"]
    for issue in report.issues:
        lines.append(f"  [{issue.code.value}] {_clean(issue.message)}")
        lines.extend(f"    {i}. {_clean(step)}" for i, step in enumerate(issue.steps, 1))
    return "\n".join(lines)


class ReportThrottle:
    """Rolling-window dedupe plus global rate limit.

    In-memory per process; fine for the single-process platform.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[str, frozenset[tuple[str, str]]], float] = {}
        self._accepted: list[float] = []

    def check(self, report: PluginLoadErrorReport) -> int | None:
        """Return None to log, or the HTTP status to answer with instead."""
        now = time.monotonic()
        cutoff = now - WINDOW_SECONDS
        self._seen = {k: t for k, t in self._seen.items() if t > cutoff}
        self._accepted = [t for t in self._accepted if t > cutoff]

        key = (report.plugin, frozenset((i.code.value, i.subject) for i in report.issues))
        if key in self._seen:
            return status.HTTP_204_NO_CONTENT
        if len(self._accepted) >= MAX_REPORTS_PER_WINDOW:
            return status.HTTP_429_TOO_MANY_REQUESTS
        self._seen[key] = now
        self._accepted.append(now)
        return None


def build_report_handler(is_enabled: Callable[[], bool]) -> Callable[[PluginLoadErrorReport], Awaitable[Response]]:
    """Create the POST handler; throttle state lives in this closure (per service instance)."""
    throttle = ReportThrottle()

    async def report_plugin_load_errors(report: PluginLoadErrorReport) -> Response:
        if not is_enabled():
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        rejected = throttle.check(report)
        if rejected is None:
            logger.warning("%s", format_report(report))
            rejected = status.HTTP_204_NO_CONTENT
        return Response(status_code=rejected)

    return report_plugin_load_errors
