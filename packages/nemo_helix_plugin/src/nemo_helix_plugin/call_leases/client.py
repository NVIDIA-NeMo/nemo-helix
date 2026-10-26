# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed sync and async clients for Inference Gateway call leases."""

from nemo_helix_plugin.call_leases import endpoints
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method


class _CallLeasesMethods:
    create_call_lease = method(endpoints.create_call_lease)
    end_call_lease = method(endpoints.end_call_lease)
    get_call_lease = method(endpoints.get_call_lease)
    list_call_leases = method(endpoints.list_call_leases)


class CallLeasesClient(_CallLeasesMethods, NemoClient):
    """Sync client for Inference Gateway call leases."""


class AsyncCallLeasesClient(_CallLeasesMethods, AsyncNemoClient):
    """Async client for Inference Gateway call leases."""
