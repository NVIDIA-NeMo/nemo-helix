# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for plugin job collections.

A plugin job collection lives at ``/apis/{api}/v2/workspaces/{workspace}/{collection}``, where
``api`` is the plugin's ``nemo.jobs`` entry-point prefix and ``collection`` is
:func:`~nemo_helix_plugin.job.job_collection_path_for` the job class (``jobs/create``,
``automodel/jobs``, ...). One endpoint therefore serves every plugin's jobs.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import get
from nemo_helix_plugin.client.types import Paginated
from nemo_helix_plugin.plugin_jobs.types import ListPluginJobsQueryParams, PluginJob


@get("/apis/{api}/v2/workspaces/{workspace}/{collection:path}")
@abstractmethod
def list_plugin_jobs(
    *,
    api: str,
    collection: str,
    workspace: str | None = None,
    query_params: ListPluginJobsQueryParams | None = None,
) -> Paginated[PluginJob]: ...
