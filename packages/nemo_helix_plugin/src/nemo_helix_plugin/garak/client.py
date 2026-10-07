# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Garak service.

Wraps the endpoint functions from ``garak.endpoints`` as direct methods
using the ``method()`` descriptor, following the files/models pattern.
"""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.garak import endpoints


class _GarakMethods:
    get_audit_config = method(endpoints.get_audit_config)
    list_audit_configs = method(endpoints.list_audit_configs)
    create_audit_config = method(endpoints.create_audit_config)
    update_audit_config = method(endpoints.update_audit_config)
    delete_audit_config = method(endpoints.delete_audit_config)
    get_audit_target = method(endpoints.get_audit_target)
    list_audit_targets = method(endpoints.list_audit_targets)
    create_audit_target = method(endpoints.create_audit_target)
    update_audit_target = method(endpoints.update_audit_target)
    delete_audit_target = method(endpoints.delete_audit_target)
    submit_audit = method(endpoints.submit_audit)
    list_audit_jobs = method(endpoints.list_audit_jobs)
    get_audit_job = method(endpoints.get_audit_job)


class GarakClient(_GarakMethods, NemoClient):
    """Sync client for the Garak service API."""


class AsyncGarakClient(_GarakMethods, AsyncNemoClient):
    """Async client for the Garak service API."""
