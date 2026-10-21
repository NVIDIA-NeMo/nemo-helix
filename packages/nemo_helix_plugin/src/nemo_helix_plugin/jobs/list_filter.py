# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The base list filter for job collections, kept apart from the FastAPI route factory.

``NemoJob.jobs_list_filter`` is typed with it, so importing ``nemo_helix_plugin.job`` must not pull
in FastAPI.
"""

from typing import Any

from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.schema import DatetimeFilter, Filter, StringFilter
from pydantic import Field


class BaseJobsListFilter(Filter):
    created_at: DatetimeFilter | None = Field(
        default=None, description="Jobs created at 'gte' datetime or 'lte' datetime."
    )
    name: StringFilter | str | None = Field(default=None, description="Name of the job.")
    workspace: str | None = Field(default=None, description="Workspace of the job.")
    project: str | None = Field(default=None, description="Project containing the job.")
    status: HelixJobStatus | None = Field(default=None, description="The current status.")
    spec: dict[str, Any] | None = Field(
        default=None, description="Filter on a path within the job's spec, e.g. `spec.target.format`."
    )
    updated_at: DatetimeFilter | None = Field(
        default=None, description="Jobs updated at 'gte' datetime or 'lte' datetime."
    )

    @classmethod
    def _get_entity_namespace_map(cls) -> dict[str, str]:
        """Declare ``spec`` as a namespace so ``spec.<path>`` passes field validation."""
        return {"spec": "data.spec"}
