# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for plugin job collections."""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.plugin_jobs import endpoints


class _PluginJobsMethods:
    list_plugin_jobs = method(endpoints.list_plugin_jobs)


class PluginJobsClient(_PluginJobsMethods, NemoClient):
    """Sync client for any plugin's job collections."""


class AsyncPluginJobsClient(_PluginJobsMethods, AsyncNemoClient):
    """Async client for any plugin's job collections."""
