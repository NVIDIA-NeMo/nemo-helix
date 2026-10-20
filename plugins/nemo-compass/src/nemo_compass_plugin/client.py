# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for Compass plugin API."""

from __future__ import annotations

from nemo_compass_plugin import endpoints
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method


class _CompassMethods:
    create_insight = method(endpoints.create_insight)
    list_insights = method(endpoints.list_insights)
    get_insight = method(endpoints.get_insight)
    update_insight = method(endpoints.update_insight)
    delete_insight = method(endpoints.delete_insight)

    enable_analysis_config = method(endpoints.enable_analysis_config)
    disable_analysis_config = method(endpoints.disable_analysis_config)
    list_analysis_configs = method(endpoints.list_analysis_configs)
    get_analysis_config = method(endpoints.get_analysis_config)
    update_analysis_config = method(endpoints.update_analysis_config)

    list_analysis_run_statuses = method(endpoints.list_analysis_run_statuses)
    get_analysis_run_status = method(endpoints.get_analysis_run_status)
    update_analysis_run_status = method(endpoints.update_analysis_run_status)

    create_analysis_run = method(endpoints.create_analysis_run)
    list_analysis_runs = method(endpoints.list_analysis_runs)
    get_analysis_run = method(endpoints.get_analysis_run)


class CompassClient(_CompassMethods, NemoClient):
    """Sync client for Compass plugin API."""


class AsyncCompassClient(_CompassMethods, AsyncNemoClient):
    """Async client for Compass plugin API."""
