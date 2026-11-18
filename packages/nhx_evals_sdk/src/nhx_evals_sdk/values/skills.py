# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact input for experimental skill evaluation metrics."""

from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SkillSnapshot(BaseModel):
    """Experimental grader input supplied by a skill artifact adapter.

    The digest identifies the original package. The adapter provides its entry document
    and complete file inventory independently of any agent-modified workspace.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    directory_name: str
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    document: str | None
    files: list[str]

    @field_validator("files")
    @classmethod
    def validate_files(cls, files: list[str]) -> list[str]:
        for name in files:
            path = PurePosixPath(name)
            if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != name or "\\" in name:
                raise ValueError("skill inventory must contain canonical package-relative file paths")
        return files
