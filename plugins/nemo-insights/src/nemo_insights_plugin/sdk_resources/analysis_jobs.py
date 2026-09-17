# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for the Insights analysis-jobs typed client."""

from nemo_insights_plugin.client import AsyncInsightsClient, InsightsClient
from nemo_insights_plugin.types import AnalysisJob, CreateAnalysisJobRequest, ListAnalysisJobsQueryParams

AnalysisJobsClient = InsightsClient
AsyncAnalysisJobsClient = AsyncInsightsClient

__all__ = [
    "AnalysisJob",
    "AnalysisJobsClient",
    "AsyncAnalysisJobsClient",
    "CreateAnalysisJobRequest",
    "ListAnalysisJobsQueryParams",
]
