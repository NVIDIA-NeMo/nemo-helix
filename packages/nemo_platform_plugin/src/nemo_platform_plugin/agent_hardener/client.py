# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Agent Hardener service.

Wraps the endpoint functions from ``agent_hardener.endpoints`` as direct methods
using the ``method()`` descriptor, following the files/models pattern.
"""

from nemo_platform_plugin.agent_hardener import endpoints
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method


class _AgentHardenerMethods:
    get_manifest = method(endpoints.get_manifest)
    list_manifests = method(endpoints.list_manifests)
    create_manifest = method(endpoints.create_manifest)
    update_manifest = method(endpoints.update_manifest)
    refresh_manifest = method(endpoints.refresh_manifest)
    delete_manifest = method(endpoints.delete_manifest)
    get_run = method(endpoints.get_run)
    list_runs = method(endpoints.list_runs)
    create_run = method(endpoints.create_run)
    delete_run = method(endpoints.delete_run)
    validate_model = method(endpoints.validate_model)


class AgentHardenerClient(_AgentHardenerMethods, NemoClient):
    """Sync client for the Agent Hardener service API."""


class AsyncAgentHardenerClient(_AgentHardenerMethods, AsyncNemoClient):
    """Async client for the Agent Hardener service API."""
