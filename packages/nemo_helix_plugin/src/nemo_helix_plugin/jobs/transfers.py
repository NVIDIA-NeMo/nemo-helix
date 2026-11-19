# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step Fileset configuration and Task transfer progress."""

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TransferOutcome = Literal["success", "failure"]
TransferPhase = Literal["pulling", "ready", "running", "finished", "uploaded", "pull_failed", "upload_failed"]
TransferAction = Literal["wait", "run", "stop", "upload_success", "upload_failure", "done", "abort"]


class FilesetPull(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fileset_id: str = Field(min_length=1)
    directory: str = Field(min_length=1)
    path: str = ""

    @field_validator("directory")
    @classmethod
    def local_directory(cls, value: str) -> str:
        if not value.startswith("/") or "\x00" in value or ".." in value.split("/"):
            raise ValueError("directory must be an absolute path without parent traversal")
        return value

    @field_validator("path")
    @classmethod
    def remote_directory(cls, value: str) -> str:
        if value.startswith("/") or "\x00" in value or "\\" in value or ".." in value.split("/"):
            raise ValueError("path must be relative without parent traversal")
        normalized = str(PurePosixPath(value))
        return "" if normalized == "." else normalized


class FilesetPush(FilesetPull):
    when: Literal["success", "failure", "always"] = "always"


class StepFilesets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pull: list[FilesetPull] = Field(default_factory=list)
    push: list[FilesetPush] = Field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.pull or self.push)


class FilesetUploadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fileset_id: str
    success: bool
    error: str = ""


class TaskTransferProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: TransferPhase
    runtime_id: str = ""
    attempts: dict[str, int] = Field(default_factory=dict)
    exit_code: int | None = None
    stopped: bool = False
    leader: bool = False
    error: str = ""
    uploads: list[FilesetUploadResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def completion_requires_exit(self):
        if self.phase in {"finished", "uploaded", "upload_failed"} and self.exit_code is None:
            raise ValueError("completed workload requires exit_code")
        return self


class StepTransferState(BaseModel):
    phase: Literal["staging", "workload", "publishing", "done", "aborted"] = "staging"
    outcome: TransferOutcome | None = None
    lost: bool = False
    commit_attempts: dict[str, int] = Field(default_factory=dict)
    retry_at: dict[str, datetime] = Field(default_factory=dict)
    commits: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
