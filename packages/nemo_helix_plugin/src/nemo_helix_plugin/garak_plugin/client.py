# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Garak Plugin service.

Wraps the endpoint functions from ``garak_plugin.endpoints`` as direct methods
using the ``method()`` descriptor, following the files/models pattern.
"""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.garak_plugin import endpoints


class _GarakPluginMethods:
    get_scan_config = method(endpoints.get_scan_config)
    list_scan_configs = method(endpoints.list_scan_configs)
    create_scan_config = method(endpoints.create_scan_config)
    update_scan_config = method(endpoints.update_scan_config)
    delete_scan_config = method(endpoints.delete_scan_config)
    get_scan_target = method(endpoints.get_scan_target)
    list_scan_targets = method(endpoints.list_scan_targets)
    create_scan_target = method(endpoints.create_scan_target)
    update_scan_target = method(endpoints.update_scan_target)
    delete_scan_target = method(endpoints.delete_scan_target)
    submit_scan = method(endpoints.submit_scan)
    list_scan_jobs = method(endpoints.list_scan_jobs)
    get_scan_job = method(endpoints.get_scan_job)


class GarakPluginClient(_GarakPluginMethods, NemoClient):
    """Sync client for the Garak Plugin service API."""


class AsyncGarakPluginClient(_GarakPluginMethods, AsyncNemoClient):
    """Async client for the Garak Plugin service API."""
