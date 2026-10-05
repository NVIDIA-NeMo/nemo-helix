# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True, slots=True)
class JobStatusEvent:
    kind: Literal["status"]
    job_name: str
    status: str
    status_details: Mapping[str, object]
    terminal: bool
    successful: bool | None
    error_details: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class JobLogEvent:
    kind: Literal["log"]
    job_name: str
    timestamp: datetime | None
    step_id: str | None
    task_id: str | None
    message: str


@dataclass(frozen=True, slots=True)
class JobWarningEvent:
    kind: Literal["warning"]
    job_name: str
    message: str


JobWatchEvent = JobStatusEvent | JobLogEvent | JobWarningEvent


class JobWatchTimeoutError(TimeoutError):
    """Raised when a job watch exceeds its timeout."""
