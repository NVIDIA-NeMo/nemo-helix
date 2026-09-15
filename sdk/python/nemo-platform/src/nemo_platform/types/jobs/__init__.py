# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.jobs.spec import (
    PlatformJobSpec as PlatformJobSpec,
    PlatformJobStepSpec as PlatformJobStepSpec,
)
from nemo_platform_plugin.jobs.types import (
    JobLogsQueryParams as JobLogsQueryParams,
    ListJobsQueryParams as ListJobsQueryParams,
    PlatformJobResponse as PlatformJobResponse,
    ListStepsQueryParams as ListStepsQueryParams,
    PlatformJobSortField as PlatformJobSortField,
    PlatformJobTaskUpdate as PlatformJobTaskUpdate,
    JobStatusDetailsUpdate as JobStatusDetailsUpdate,
    PlatformJobLogSortField as PlatformJobLogSortField,
    PlatformJobStepResponse as PlatformJobStepResponse,
    PlatformJobTaskResponse as PlatformJobTaskResponse,
    CreatePlatformJobRequest as CreatePlatformJobRequest,
    PlatformJobListSortField as PlatformJobListSortField,
    ListJobResultsQueryParams as ListJobResultsQueryParams,
    PlatformJobStepWithContext as PlatformJobStepWithContext,
    PlatformJobAttemptSortField as PlatformJobAttemptSortField,
    PlatformJobListTaskResponse as PlatformJobListTaskResponse,
    PlatformJobStatusUpdateRequest as PlatformJobStatusUpdateRequest,
    PlatformJobStatusDetailsUpdateRequest as PlatformJobStatusDetailsUpdateRequest,
)
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobLog as PlatformJobLog,
    FileStorageType as FileStorageType,
    PlatformJobStatus as PlatformJobStatus,
    PlatformJobLogPage as PlatformJobLogPage,
    PlatformJobResultResponse as PlatformJobResultResponse,
    PlatformJobStatusResponse as PlatformJobStatusResponse,
    PlatformJobListResultResponse as PlatformJobListResultResponse,
    PlatformJobStepStatusResponse as PlatformJobStepStatusResponse,
    PlatformJobTaskStatusResponse as PlatformJobTaskStatusResponse,
    PlatformJobResultCreateRequest as PlatformJobResultCreateRequest,
)

PlatformJobStep = PlatformJobStepResponse
