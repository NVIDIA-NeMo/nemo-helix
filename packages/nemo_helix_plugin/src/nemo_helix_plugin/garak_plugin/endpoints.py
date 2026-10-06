# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Garak Plugin service.

Single source of truth for the HTTP contract. Replaces the Stainless-generated
garak_plugin resource from ``garak_plugin.sdk``.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import delete, get, post, put
from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.garak_plugin.types import (
    CreateScanConfigRequest,
    CreateScanTargetRequest,
    ListScanConfigsQueryParams,
    ListScanJobsQueryParams,
    ListScanTargetsQueryParams,
    ScanConfig,
    ScanJobResponse,
    ScanTarget,
    SubmitScanRequest,
    UpdateScanConfigRequest,
    UpdateScanTargetRequest,
)

_CONFIGS = "/apis/garak-plugin/v2/workspaces/{workspace}/configs"
_TARGETS = "/apis/garak-plugin/v2/workspaces/{workspace}/targets"
_JOBS = "/apis/garak-plugin/v2/workspaces/{workspace}/jobs/scan"


# ---------------------------------------------------------------------------
# Config CRUD
# ---------------------------------------------------------------------------


@get(f"{_CONFIGS}/{{name}}")
@abstractmethod
def get_scan_config(*, workspace: str | None = None, name: str) -> ScanConfig: ...


@get(_CONFIGS)
@abstractmethod
def list_scan_configs(
    *, workspace: str | None = None, query_params: ListScanConfigsQueryParams | None = None
) -> Paginated[ScanConfig]: ...


def _get_scan_config_on_conflict(body: CreateScanConfigRequest, workspace: str | None) -> PreparedRequest[ScanConfig]:
    return get_scan_config(name=body.name, workspace=workspace)


@post(_CONFIGS, get_on_conflict=_get_scan_config_on_conflict)
@abstractmethod
def create_scan_config(
    *, workspace: str | None = None, body: CreateScanConfigRequest, exist_ok: bool = False
) -> ScanConfig: ...


@put(f"{_CONFIGS}/{{name}}")
@abstractmethod
def update_scan_config(*, workspace: str | None = None, name: str, body: UpdateScanConfigRequest) -> ScanConfig: ...


@delete(f"{_CONFIGS}/{{name}}")
@abstractmethod
def delete_scan_config(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Target CRUD
# ---------------------------------------------------------------------------


@get(f"{_TARGETS}/{{name}}")
@abstractmethod
def get_scan_target(*, workspace: str | None = None, name: str) -> ScanTarget: ...


@get(_TARGETS)
@abstractmethod
def list_scan_targets(
    *, workspace: str | None = None, query_params: ListScanTargetsQueryParams | None = None
) -> Paginated[ScanTarget]: ...


def _get_scan_target_on_conflict(body: CreateScanTargetRequest, workspace: str | None) -> PreparedRequest[ScanTarget]:
    return get_scan_target(name=body.name, workspace=workspace)


@post(_TARGETS, get_on_conflict=_get_scan_target_on_conflict)
@abstractmethod
def create_scan_target(
    *, workspace: str | None = None, body: CreateScanTargetRequest, exist_ok: bool = False
) -> ScanTarget: ...


@put(f"{_TARGETS}/{{name}}")
@abstractmethod
def update_scan_target(*, workspace: str | None = None, name: str, body: UpdateScanTargetRequest) -> ScanTarget: ...


@delete(f"{_TARGETS}/{{name}}")
@abstractmethod
def delete_scan_target(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Scan job submission and retrieval
# ---------------------------------------------------------------------------


@post(_JOBS)
@abstractmethod
def submit_scan(*, workspace: str | None = None, body: SubmitScanRequest) -> ScanJobResponse: ...


@get(_JOBS)
@abstractmethod
def list_scan_jobs(
    *, workspace: str | None = None, query_params: ListScanJobsQueryParams | None = None
) -> Paginated[ScanJobResponse]: ...


@get(f"{_JOBS}/{{name}}")
@abstractmethod
def get_scan_job(*, workspace: str | None = None, name: str) -> ScanJobResponse: ...
