# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
from typing import Any

from nemo_helix_plugin.schema import Value
from pydantic import BaseModel, Field


class FileStorageType(str, Enum):
    FILESET = "fileset"


class PlatformJobResultCreateRequest(BaseModel):
    artifact_url: str
    artifact_storage_type: FileStorageType


class PlatformJobResultResponse(BaseModel):
    name: str
    job: str
    workspace: str
    project: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    artifact_url: str
    artifact_storage_type: FileStorageType
    download_url: str | None = None


class PlatformJobListResultResponse(Value):
    data: list[PlatformJobResultResponse]


class PlatformJobStatus(str, Enum):
    """Enumeration of possible job statuses.

    This enum represents the various states a job can be in during its lifecycle,
    from creation to a terminal state.
    """

    CREATED = "created"
    PENDING = "pending"
    ACTIVE = "active"
    CANCELLED = "cancelled"
    CANCELLING = "cancelling"
    ERROR = "error"
    COMPLETED = "completed"
    PAUSED = "paused"
    PAUSING = "pausing"
    RESUMING = "resuming"

    @staticmethod
    def terminals() -> list["PlatformJobStatus"]:
        return [PlatformJobStatus.COMPLETED, PlatformJobStatus.ERROR, PlatformJobStatus.CANCELLED]

    @staticmethod
    def non_terminals() -> list["PlatformJobStatus"]:
        return [
            PlatformJobStatus.CREATED,
            PlatformJobStatus.PENDING,
            PlatformJobStatus.ACTIVE,
            PlatformJobStatus.CANCELLING,
            PlatformJobStatus.PAUSING,
            PlatformJobStatus.PAUSED,
            PlatformJobStatus.RESUMING,
        ]

    def is_terminal(self) -> bool:
        return self in PlatformJobStatus.terminals()

    def can_transition_to(self, new_status: "PlatformJobStatus") -> bool:
        """Validate if a status transition is valid."""

        if self == new_status:
            return True

        # Define valid transitions
        valid_transitions = {
            PlatformJobStatus.CREATED: {
                PlatformJobStatus.PENDING,
                PlatformJobStatus.ACTIVE,
                PlatformJobStatus.CANCELLING,
                PlatformJobStatus.CANCELLED,
                PlatformJobStatus.PAUSING,
                PlatformJobStatus.PAUSED,
                PlatformJobStatus.ERROR,
            },
            PlatformJobStatus.PENDING: {
                PlatformJobStatus.ACTIVE,
                PlatformJobStatus.CANCELLING,
                PlatformJobStatus.CANCELLED,
                PlatformJobStatus.ERROR,
                PlatformJobStatus.PAUSING,
                PlatformJobStatus.PAUSED,
                PlatformJobStatus.COMPLETED,
            },
            PlatformJobStatus.ACTIVE: {
                PlatformJobStatus.COMPLETED,
                PlatformJobStatus.ERROR,
                PlatformJobStatus.CANCELLING,
                PlatformJobStatus.CANCELLED,
                PlatformJobStatus.PAUSING,
                PlatformJobStatus.PAUSED,
            },
            PlatformJobStatus.PAUSED: {
                PlatformJobStatus.RESUMING,
                PlatformJobStatus.ACTIVE,
                PlatformJobStatus.CANCELLING,
            },
            PlatformJobStatus.CANCELLING: {PlatformJobStatus.CANCELLED, PlatformJobStatus.ERROR},
            PlatformJobStatus.PAUSING: {PlatformJobStatus.PAUSED, PlatformJobStatus.ERROR},
            PlatformJobStatus.RESUMING: {
                PlatformJobStatus.PENDING,
                PlatformJobStatus.ACTIVE,
                PlatformJobStatus.ERROR,
                PlatformJobStatus.CANCELLING,
            },
            PlatformJobStatus.COMPLETED: set(),
            PlatformJobStatus.CANCELLED: set(),
            PlatformJobStatus.ERROR: set(),
        }

        return new_status in valid_transitions.get(self, set())


class PlatformJobTaskStatusResponse(BaseModel):
    id: str
    name: str
    status: PlatformJobStatus
    status_details: dict[str, Any]
    error_details: dict[str, Any] | None
    error_stack: str | None
    created_at: datetime
    updated_at: datetime


class PlatformJobStepStatusResponse(BaseModel):
    id: str
    name: str
    status: PlatformJobStatus
    status_details: dict[str, Any]
    error_details: dict[str, Any] | None
    tasks: list[PlatformJobTaskStatusResponse]
    created_at: datetime
    updated_at: datetime


class PlatformJobStatusResponse(BaseModel):
    id: str
    name: str
    status: PlatformJobStatus
    status_details: dict[str, Any]
    error_details: dict[str, Any] | None
    steps: list[PlatformJobStepStatusResponse]
    created_at: datetime
    updated_at: datetime


class PlatformJobLog(BaseModel):
    timestamp: datetime
    job: str
    job_step: str
    job_task: str
    message: str


class PlatformJobLogPage(BaseModel):
    data: list[PlatformJobLog]
    total: int
    next_page: str | None
    prev_page: str | None
