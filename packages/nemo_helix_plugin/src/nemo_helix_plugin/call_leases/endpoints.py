# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for Inference Gateway call leases."""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.call_leases.types import (
    CallLease,
    CreateCallLeaseRequest,
    EndCallLeaseRequest,
    ListCallLeasesQueryParams,
)
from nemo_helix_plugin.client.endpoint import get, patch, post
from nemo_helix_plugin.client.types import Paginated

_CALL_LEASES = "/apis/inference-gateway/v2/workspaces/{workspace}/call-leases"


@post(_CALL_LEASES)
@abstractmethod
def create_call_lease(*, workspace: str | None = None, body: CreateCallLeaseRequest) -> CallLease: ...


@patch(_CALL_LEASES + "/{name}")
@abstractmethod
def end_call_lease(*, workspace: str | None = None, name: str, body: EndCallLeaseRequest) -> CallLease: ...


@get(_CALL_LEASES + "/{name}")
@abstractmethod
def get_call_lease(*, workspace: str | None = None, name: str) -> CallLease: ...


@get(_CALL_LEASES)
@abstractmethod
def list_call_leases(
    *, workspace: str | None = None, query_params: ListCallLeasesQueryParams | None = None
) -> Paginated[CallLease]: ...
