# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for Compass plugin.

Mounted on :class:`~nemo_helix_plugin.client.client.NemoClient` as ``client.compass`` via
the ``nemo.sdk`` entry-point in :file:`pyproject.toml`. Exposes:

- ``client.compass.analysis_configs.{enable,disable,list_configs,get,update}``
  — :class:`~nemo_compass_plugin.entities.AnalysisConfig` CRUD/control for
  periodic analysis opt-in state.
- ``client.compass.analysis_runs.{create,list_runs,get,wait}`` —
  :class:`~nemo_compass_plugin.entities.AnalysisRun` submission and read-back
  for on-demand Compass runs, each backed by an ``agents.execute`` job.
- ``client.compass.analysis_run_statuses.{list_statuses,get,update}``
  — :class:`~nemo_compass_plugin.entities.AnalysisRunStatus` state written by
  analyzer jobs.
- ``client.compass.insights.{create,list_insights,get,update,delete}`` —
  :class:`~nemo_compass_plugin.entities.Insight` CRUD against the FastAPI
  routes mounted under ``/apis/compass/v2/workspaces/{workspace}/``.

Modeled on ``nemo_garak.sdk`` — same shape, same hand-written CRUD-only
resource pattern.
"""

from nemo_compass_plugin.client import AsyncCompassClient, CompassClient
from nemo_compass_plugin.sdk_resources.analysis_configs import (
    _AnalysisConfigResource,
    _AnalysisRunStatusResource,
    _AsyncAnalysisConfigResource,
    _AsyncAnalysisRunStatusResource,
)
from nemo_compass_plugin.sdk_resources.analysis_runs import (
    _AnalysisRunResource,
    _AsyncAnalysisRunResource,
)
from nemo_compass_plugin.sdk_resources.insights import (
    _AsyncInsightResource,
    _InsightResource,
)
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.sdk import NemoPluginSDKResources


class CompassPluginResource:
    """Sync SDK namespace mounted as ``client.compass``."""

    def __init__(self, platform: NemoClient) -> None:
        self._platform = platform
        self._client = CompassClient.from_client(platform)
        self._insights: _InsightResource | None = None
        self._analysis_configs: _AnalysisConfigResource | None = None
        self._analysis_runs: _AnalysisRunResource | None = None
        self._analysis_run_statuses: _AnalysisRunStatusResource | None = None

    @property
    def insights(self) -> _InsightResource:
        if self._insights is None:
            self._insights = _InsightResource(self)
        return self._insights

    @property
    def analysis_configs(self) -> _AnalysisConfigResource:
        if self._analysis_configs is None:
            self._analysis_configs = _AnalysisConfigResource(self)
        return self._analysis_configs

    @property
    def analysis_runs(self) -> _AnalysisRunResource:
        if self._analysis_runs is None:
            self._analysis_runs = _AnalysisRunResource(self)
        return self._analysis_runs

    @property
    def analysis_run_statuses(self) -> _AnalysisRunStatusResource:
        if self._analysis_run_statuses is None:
            self._analysis_run_statuses = _AnalysisRunStatusResource(self)
        return self._analysis_run_statuses


class AsyncCompassPluginResource:
    """Async SDK namespace mounted as ``client.compass``."""

    def __init__(self, platform: AsyncNemoClient) -> None:
        self._platform = platform
        self._client = AsyncCompassClient.from_client(platform)
        self._insights: _AsyncInsightResource | None = None
        self._analysis_configs: _AsyncAnalysisConfigResource | None = None
        self._analysis_runs: _AsyncAnalysisRunResource | None = None
        self._analysis_run_statuses: _AsyncAnalysisRunStatusResource | None = None

    @property
    def insights(self) -> _AsyncInsightResource:
        if self._insights is None:
            self._insights = _AsyncInsightResource(self)
        return self._insights

    @property
    def analysis_configs(self) -> _AsyncAnalysisConfigResource:
        if self._analysis_configs is None:
            self._analysis_configs = _AsyncAnalysisConfigResource(self)
        return self._analysis_configs

    @property
    def analysis_runs(self) -> _AsyncAnalysisRunResource:
        if self._analysis_runs is None:
            self._analysis_runs = _AsyncAnalysisRunResource(self)
        return self._analysis_runs

    @property
    def analysis_run_statuses(self) -> _AsyncAnalysisRunStatusResource:
        if self._analysis_run_statuses is None:
            self._analysis_run_statuses = _AsyncAnalysisRunStatusResource(self)
        return self._analysis_run_statuses


compass_sdk_resources = NemoPluginSDKResources(
    sync_resource=CompassPluginResource,
    async_resource=AsyncCompassPluginResource,
)
