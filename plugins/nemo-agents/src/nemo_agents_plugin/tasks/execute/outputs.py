# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select bounded job outputs without walking an agent's scratch workspace."""

import os
import stat
from contextlib import ExitStack
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentOutputFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Exact POSIX path relative to the agent workspace; no glob patterns.")
    max_bytes: int = Field(gt=0, strict=True, description="Maximum number of bytes exported for this file.")
    required: bool = False

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if "\\" in value or "\x00" in value or any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("Output path must be a relative path without empty, '.' or '..' components")
        return value


def select_output_files(
    workspace: Path, destination: Path, files: list[AgentOutputFile], *, require_files: bool
) -> None:
    """Copy only declared regular files; neither traversal nor symlinks are followed."""
    for output in files:
        with ExitStack() as stack:
            try:
                directory = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                stack.callback(os.close, directory)
                parts = output.path.split("/")
                for part in parts[:-1]:
                    directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                    stack.callback(os.close, directory)
                source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            except FileNotFoundError:
                if output.required and require_files:
                    raise ValueError(f"Required output file is missing: {output.path}") from None
                continue
            except OSError as error:
                raise ValueError(f"Cannot safely open output file: {output.path}") from error
            stack.callback(os.close, source)
            info = os.fstat(source)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"Output must be a regular file: {output.path}")
            if info.st_size > output.max_bytes:
                raise ValueError(f"Output exceeds {output.max_bytes} bytes: {output.path}")
            stream = stack.enter_context(os.fdopen(source, "rb", closefd=False))
            target = destination / output.path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as saved:
                remaining = output.max_bytes
                while chunk := stream.read(min(64 * 1024, remaining + 1)):
                    if len(chunk) > remaining:
                        raise ValueError(f"Output exceeds {output.max_bytes} bytes: {output.path}")
                    saved.write(chunk)
                    remaining -= len(chunk)
