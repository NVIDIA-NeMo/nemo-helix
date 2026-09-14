# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen public identities of the executions included in a benchmark export."""

from datetime import datetime

from pydantic import BaseModel


class BenchmarkArchiveMember(BaseModel):
    id: str
    task_id: str
    task_revision: int
    task_slug: str | None = None
    task_name: str | None = None
    status: str
    current_execution: int
    archive_object_key: str
    archive_built_at: datetime
    archive_size_bytes: int
