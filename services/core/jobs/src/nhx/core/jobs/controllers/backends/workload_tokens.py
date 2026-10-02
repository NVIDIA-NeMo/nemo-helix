# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for controller-managed workload identity subject tokens."""

from __future__ import annotations

import nhx.common.auth.workload_identity as _workload_identity
from nemo_helix_plugin.client.adapter import SyncHelixClient, client_from_platform
from nemo_helix_plugin.entities.client import EntitiesClient
from nhx.common.auth import SyncWorkloadDelegationStore
from nhx.common.entities import SyncEntityClient

WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS = _workload_identity.WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS
build_token_archive = _workload_identity.build_token_archive
workload_delegation_expires_at = _workload_identity.workload_delegation_expires_at


def create_authenticated_workload_delegation_store(nhx_sdk: SyncHelixClient) -> SyncWorkloadDelegationStore:
    """Create a jobs service-scoped workload delegation store from a platform SDK."""
    entity_client = SyncEntityClient(client_from_platform(nhx_sdk, EntitiesClient)).as_service("jobs", internal=True)
    return SyncWorkloadDelegationStore(entity_client)
