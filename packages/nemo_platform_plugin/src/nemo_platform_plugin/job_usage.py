# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-facing token-usage reporting for platform jobs."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from nemo_platform_plugin.jobs.client import JobsClient
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

logger = logging.getLogger(__name__)


class JobTokenUsage(BaseModel):
    """Cumulative model-token totals for the current job attempt."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_a_total(self) -> JobTokenUsage:
        """Reject empty reports, which cannot add information to job status."""
        if self.input_tokens is None and self.output_tokens is None:
            raise ValueError("at least one token total is required")
        return self


class JobUsageReporter(ABC):
    """Sync task-facing API for replacing cumulative job token totals.

    Calls set whole-attempt totals rather than deltas. The Jobs status-details
    endpoint uses last-write-wins merging and cannot atomically increment
    counters, so producers must aggregate locally and report once before the
    task exits.
    """

    def report_totals(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Validate and publish cumulative totals for the current attempt."""
        self._report(JobTokenUsage(input_tokens=input_tokens, output_tokens=output_tokens))

    @abstractmethod
    def _report(self, usage: JobTokenUsage) -> None: ...


class LocalJobUsageReporter(JobUsageReporter):
    """In-memory reporter used by local job execution and unit tests."""

    def __init__(self) -> None:
        self.latest: JobTokenUsage | None = None

    def _report(self, usage: JobTokenUsage) -> None:
        self.latest = usage


class PlatformJobUsageReporter(JobUsageReporter):
    """Best-effort reporter backed by the Jobs status-details endpoint."""

    def __init__(self, *, job_name: str, workspace: str, jobs_client: JobsClient) -> None:
        self._job_name = job_name
        self._workspace = workspace
        self._jobs_client = jobs_client

    def _report(self, usage: JobTokenUsage) -> None:
        try:
            self._jobs_client.update_status_details(
                self._job_name,
                workspace=self._workspace,
                body=usage.model_dump(exclude_none=True),
            )
        except Exception:
            # Usage reporting is observability, not job correctness. Preserve
            # the job result while retaining a diagnostic in task logs.
            logger.warning(
                "Failed to report token usage for job %r in workspace %r",
                self._job_name,
                self._workspace,
                exc_info=True,
            )


__all__ = [
    "JobTokenUsage",
    "JobUsageReporter",
    "LocalJobUsageReporter",
    "PlatformJobUsageReporter",
]
