# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from scaled_evals.models.benchmark_archives import BenchmarkArchiveMember


class BenchmarkArchiveRequest(BaseModel):
    force: bool = False


class BenchmarkArchiveResponse(BaseModel):
    benchmark_run_id: str
    status: Literal["missing", "queued", "building", "ready", "failed"]
    format: Literal["harbor-job-tar.gz"] = "harbor-job-tar.gz"
    generation: str | None = None
    requested_at: datetime | None = None
    built_at: datetime | None = None
    size_bytes: int | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error: str | None = None
    partial: bool | None = None
    members: list[BenchmarkArchiveMember] = Field(default_factory=list)
    download: str | None = None
