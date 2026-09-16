# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for the customization file_io task configuration.

Shared by both the unsloth and automodel backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

FILESET_PROTOCOL = "fileset://"


class TaskStatus(StrEnum):
    """Status of a file I/O task."""

    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class TaskPhase(StrEnum):
    """Phase of a file I/O task."""

    DOWNLOADING = "downloading"
    UPLOADING = "uploading"
    COMPLETED = "completed"


class FileSetRef(BaseModel):
    """Reference to a FileSet."""

    # workspace is optional because at compile time, the workspace is not known.
    # None tells the file_io task to use the job's workspace from the NHXJobContext.
    workspace: Optional[str] = None
    name: str

    def __str__(self) -> str:
        if self.workspace is None:
            return self.name
        return f"{self.workspace}/{self.name}"

    def __repr__(self) -> str:
        return f"FileSetRef(workspace={self.workspace}, name={self.name})"

    @classmethod
    def _parse_string_parts(cls, ref: str) -> tuple[Optional[str], str] | None:
        """Parse a FileSet reference string into a tuple of workspace and name."""
        if len(ref) == 0:
            return None
        if ref.startswith(FILESET_PROTOCOL):
            ref = ref[len(FILESET_PROTOCOL) :]
        parts = ref.split("/", 1)
        if len(parts) == 1:
            return None, parts[0]
        # split("/", 1) yields at most 2 parts, so this is the 2-part case.
        return parts[0], parts[1]

    @classmethod
    def extract_name(cls, ref: str) -> str:
        """Extract the fileset/entity name from a reference string.

        Supports:
        - workspace/name
        - name
        - fileset://workspace/name (legacy, stripped)
        """
        return cls.model_validate(ref).name

    @model_validator(mode="before")
    @classmethod
    def _convert_string_input(cls, v: object) -> object:
        """Convert a FileSet reference string into a dict of workspace and name.

        This makes it possible to create a FileSetRef from a string directly.
        """
        if isinstance(v, str):
            result = cls._parse_string_parts(v)
            if result is None:
                raise ValueError(f"Invalid FileSet reference: {v!r}. Expected format: 'workspace/name' or 'name'.")
            workspace, name = result
            return {"workspace": workspace, "name": name}
        return v


class DownloadItem(BaseModel):
    """Configures a single download: fileset -> local path.

    Note: dest is an absolute path where files will be downloaded.
    This path should be under the job's shared storage (e.g., /var/run/scratch/job/model).
    """

    src: FileSetRef = Field(
        description=(
            "FileSet reference for the source files. "
            "Accepts 'workspace/name' or 'name' (job workspace used when omitted)."
        ),
    )
    dest: str = Field(
        default=".",
        description="Absolute destination path for downloaded files (e.g., '/var/run/scratch/job/model').",
    )


class UploadItem(BaseModel):
    """Configures a single upload: local path -> fileset."""

    src: str = Field(
        description="Absolute source path for files to upload (e.g., '/var/run/scratch/job/output_model').",
    )
    dest: FileSetRef = Field(
        description=(
            "FileSet reference for the destination. "
            "Accepts 'workspace/name' or 'name' (job workspace used when omitted)."
        ),
    )
    metadata: Optional[dict] = Field(
        default=None,
        description=(
            "Optional metadata to set on the created fileset (e.g., tool_calling config "
            "propagated from the source model entity)."
        ),
    )


class FileIOTaskConfig(BaseModel):
    """Configuration for the file_io task.

    Used when running ``python -m nhx.<backend>.tasks.file_io``.
    """

    download: list[DownloadItem] = Field(default_factory=list, description="List of FileSets to download.")
    upload: list[UploadItem] = Field(default_factory=list, description="List of files to upload to FileSets.")


class TaskCompilationError(Exception):
    """Error compiling a task configuration."""


class FileDownloadError(Exception):
    """Error downloading files from the Files service."""


class FileUploadError(Exception):
    """Error uploading files to the Files service."""


class ProgressReportError(Exception):
    """Error reporting progress to the Jobs service."""


class PathTraversalError(ValueError):
    """Error when a path attempts to escape the allowed base directory.

    This is a security error raised when user-provided paths like '../..' would
    result in file operations outside the designated storage directory.
    """


@dataclass
class FileStats:
    """Statistics for a file operation."""

    total_bytes: int = 0
    failed_files: int = 0


@dataclass
class DownloadStats(FileStats):
    """Statistics for a download operation."""

    files_downloaded: int = 0


@dataclass
class UploadStats(FileStats):
    """Statistics for an upload operation."""

    files_uploaded: int = 0
