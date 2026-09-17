# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progress reporting for the file_io container task."""

import logging
from typing import Any, Protocol

from nemo_platform_plugin.client.errors import NemoHTTPError
from nemo_platform_plugin.jobs.client import JobsClient
from nemo_platform_plugin.jobs.types import PlatformJobTaskUpdate
from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.customization_common.schemas.file_io import ProgressReportError
from nmp.customization_common.service.context import NMPJobContext
from nmp.customization_common.tasks.file_io_utils import sdk_error_handler

logger = logging.getLogger(__name__)


class ProgressReporter(Protocol):
    """Interface for reporting task progress."""

    def update_progress(
        self,
        status: PlatformJobStatus,
        status_details: dict[str, Any] | None = None,
        error_details: dict[str, Any] | None = None,
        error_stack: str | None = None,
    ) -> None:
        """Update task progress."""


class NoOpProgressReporter:
    """Progress reporter that does nothing. Used when Jobs service is not configured."""

    def update_progress(
        self,
        status: PlatformJobStatus,
        status_details: dict[str, Any] | None = None,
        error_details: dict[str, Any] | None = None,
        error_stack: str | None = None,
    ) -> None:
        """No-op: silently ignore progress updates."""


class JobsServiceProgressReporter:
    """Reports progress to the Jobs service via SDK."""

    def __init__(self, jobs: JobsClient, workspace: str, job_id: str, step_name: str, task_id: str):
        self.jobs = jobs
        self.workspace = workspace
        self.job_id = job_id
        self.step_name = step_name
        self.task_id = task_id

    def update_progress(
        self,
        status: PlatformJobStatus,
        status_details: dict[str, object] | None = None,
        error_details: dict[str, object] | None = None,
        error_stack: str | None = None,
    ) -> None:
        """Update task progress via SDK."""
        try:
            with sdk_error_handler(
                ProgressReportError,
                f"update progress for task: {self.task_id}, job: {self.job_id}, step: {self.step_name}",
                passthrough=(NemoHTTPError,),
            ):
                # Only set fields that have values (mirrors the SDK's ``omit``
                # sentinels — ``exclude_unset`` keeps them off the wire).
                task_update: dict[str, Any] = {"status": status}
                if status_details:
                    task_update["status_details"] = status_details
                if error_details:
                    task_update["error_details"] = error_details
                if error_stack:
                    task_update["error_stack"] = error_stack

                self.jobs.update_job_step_task(
                    name=self.task_id,
                    workspace=self.workspace,
                    job=self.job_id,
                    step=self.step_name,
                    body=PlatformJobTaskUpdate(**task_update),
                )
                logger.debug(f"Progress updated: {status} - {status_details}")
        except Exception as e:
            logger.warning(
                f"Failed to report progress for task {self.task_id}, job {self.job_id}, step {self.step_name}: {e}",
            )

    @staticmethod
    def create_progress_reporter(jobs: JobsClient, job_ctx: NMPJobContext) -> ProgressReporter:
        """Build a JobsServiceProgressReporter when jobs_url is set, else NoOpProgressReporter."""
        if job_ctx.jobs_url:
            logger.info(f"Progress reporting enabled: {job_ctx.jobs_url}")
            return JobsServiceProgressReporter(
                jobs, job_ctx.workspace, job_ctx.job_id, job_ctx.step, job_ctx.normalized_task
            )
        logger.info("Progress reporting disabled: jobs_url not configured")
        return NoOpProgressReporter()
