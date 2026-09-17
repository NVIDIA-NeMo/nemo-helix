# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for the insights plugin.

Mounted on :class:`~nemo_platform.NeMoPlatform` as ``client.insights`` via
the ``nemo.sdk`` entry-point in :file:`pyproject.toml`. Exposes:

- ``client.insights.analysis_configs.{enable,disable,list_configs,get,update}``
  — :class:`~nemo_insights_plugin.entities.AnalysisConfig` CRUD/control for
  periodic analysis opt-in state.
- ``client.insights.analysis_runs.{create,list_runs,get,wait}`` —
  :class:`~nemo_insights_plugin.entities.AnalysisRun` submission and read-back
  for on-demand Analyst runs, each backed by an ``agents.execute`` job.
- ``client.insights.analysis_run_statuses.{list_statuses,get,update}``
  — :class:`~nemo_insights_plugin.entities.AnalysisRunStatus` state written by
  analyzer jobs.
- ``client.insights.insights.{create,list_insights,get,update,delete}`` —
  :class:`~nemo_insights_plugin.entities.Insight` CRUD against the FastAPI
  routes mounted under ``/apis/insights/v2/workspaces/{workspace}/``.

Modeled on ``nemo_auditor.sdk`` — same shape, same hand-written CRUD-only
resource pattern. No Stainless codegen.
"""

from nemo_insights_plugin.client import AsyncInsightsClient, InsightsClient
from nemo_insights_plugin.sdk_resources.analysis_configs import (
    _AnalysisConfigResource,
    _AnalysisRunStatusResource,
    _AsyncAnalysisConfigResource,
    _AsyncAnalysisRunStatusResource,
)
from nemo_insights_plugin.sdk_resources.analysis_runs import (
    _AnalysisRunResource,
    _AsyncAnalysisRunResource,
)
from nemo_insights_plugin.sdk_resources.insights import (
    _AsyncInsightResource,
    _InsightResource,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.sdk import NemoPluginSDKResources


class InsightsPluginResource:
    """Sync SDK namespace mounted as ``client.insights``."""

    def __init__(self, platform: NeMoPlatform) -> None:
        self._platform = platform
        self._client = client_from_platform(platform, InsightsClient)
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


class AsyncInsightsPluginResource:
    """Async SDK namespace mounted as ``client.insights``."""

    def __init__(self, platform: AsyncNeMoPlatform) -> None:
        self._platform = platform
        self._client = client_from_platform(platform, AsyncInsightsClient)
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


insights_sdk_resources = NemoPluginSDKResources(
    sync_resource=InsightsPluginResource,
    async_resource=AsyncInsightsPluginResource,
)
