# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
from typing import Any

from nemo_helix_plugin.schema import Value
from pydantic import BaseModel, Field


class FileStorageType(str, Enum):
    FILESET = "fileset"


class HelixJobResultCreateRequest(BaseModel):
    artifact_url: str
    artifact_storage_type: FileStorageType


class HelixJobResultResponse(BaseModel):
    name: str
    job: str
    workspace: str
    project: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    artifact_url: str
    artifact_storage_type: FileStorageType
    download_url: str | None = None


class HelixJobListResultResponse(Value):
    data: list[HelixJobResultResponse]


class HelixJobStatus(str, Enum):
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
    def terminals() -> list["HelixJobStatus"]:
        return [HelixJobStatus.COMPLETED, HelixJobStatus.ERROR, HelixJobStatus.CANCELLED]

    @staticmethod
    def non_terminals() -> list["HelixJobStatus"]:
        return [
            HelixJobStatus.CREATED,
            HelixJobStatus.PENDING,
            HelixJobStatus.ACTIVE,
            HelixJobStatus.CANCELLING,
            HelixJobStatus.PAUSING,
            HelixJobStatus.PAUSED,
            HelixJobStatus.RESUMING,
        ]

    def is_terminal(self) -> bool:
        return self in HelixJobStatus.terminals()

    def can_transition_to(self, new_status: "HelixJobStatus") -> bool:
        """Validate if a status transition is valid."""

        if self == new_status:
            return True

        # Define valid transitions
        valid_transitions = {
            HelixJobStatus.CREATED: {
                HelixJobStatus.PENDING,
                HelixJobStatus.ACTIVE,
                HelixJobStatus.CANCELLING,
                HelixJobStatus.CANCELLED,
                HelixJobStatus.PAUSING,
                HelixJobStatus.PAUSED,
                HelixJobStatus.ERROR,
            },
            HelixJobStatus.PENDING: {
                HelixJobStatus.ACTIVE,
                HelixJobStatus.CANCELLING,
                HelixJobStatus.CANCELLED,
                HelixJobStatus.ERROR,
                HelixJobStatus.PAUSING,
                HelixJobStatus.PAUSED,
                HelixJobStatus.COMPLETED,
            },
            HelixJobStatus.ACTIVE: {
                HelixJobStatus.COMPLETED,
                HelixJobStatus.ERROR,
                HelixJobStatus.CANCELLING,
                HelixJobStatus.CANCELLED,
                HelixJobStatus.PAUSING,
                HelixJobStatus.PAUSED,
            },
            HelixJobStatus.PAUSED: {
                HelixJobStatus.RESUMING,
                HelixJobStatus.ACTIVE,
                HelixJobStatus.CANCELLING,
            },
            HelixJobStatus.CANCELLING: {HelixJobStatus.CANCELLED, HelixJobStatus.ERROR},
            HelixJobStatus.PAUSING: {HelixJobStatus.PAUSED, HelixJobStatus.ERROR},
            HelixJobStatus.RESUMING: {
                HelixJobStatus.PENDING,
                HelixJobStatus.ACTIVE,
                HelixJobStatus.ERROR,
                HelixJobStatus.CANCELLING,
            },
            HelixJobStatus.COMPLETED: set(),
            HelixJobStatus.CANCELLED: set(),
            HelixJobStatus.ERROR: set(),
        }

        return new_status in valid_transitions.get(self, set())


class HelixJobTaskStatusResponse(BaseModel):
    id: str
    name: str
    status: HelixJobStatus
    status_details: dict[str, Any]
    error_details: dict[str, Any] | None
    error_stack: str | None
    created_at: datetime
    updated_at: datetime


class HelixJobStepStatusResponse(BaseModel):
    id: str
    name: str
    status: HelixJobStatus
    status_details: dict[str, Any]
    error_details: dict[str, Any] | None
    tasks: list[HelixJobTaskStatusResponse]
    created_at: datetime
    updated_at: datetime


class HelixJobStatusResponse(BaseModel):
    id: str
    name: str
    status: HelixJobStatus
    status_details: dict[str, Any]
    error_details: dict[str, Any] | None
    steps: list[HelixJobStepStatusResponse]
    created_at: datetime
    updated_at: datetime


class HelixJobLog(BaseModel):
    timestamp: datetime
    job: str
    job_step: str
    job_task: str
    message: str


class HelixJobLogPage(BaseModel):
    data: list[HelixJobLog]
    total: int
    next_page: str | None
    prev_page: str | None
