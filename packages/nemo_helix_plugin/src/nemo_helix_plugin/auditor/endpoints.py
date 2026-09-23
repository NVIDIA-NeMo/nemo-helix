# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Auditor service.

Single source of truth for the HTTP contract. Replaces the Stainless-generated
auditor resource from ``nemo_auditor.sdk``.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.auditor.types import (
    AuditConfig,
    AuditJobResponse,
    AuditTarget,
    CreateAuditConfigRequest,
    CreateAuditTargetRequest,
    ListAuditConfigsQueryParams,
    ListAuditJobsQueryParams,
    ListAuditTargetsQueryParams,
    SubmitAuditRequest,
    UpdateAuditConfigRequest,
    UpdateAuditTargetRequest,
)
from nemo_helix_plugin.client.endpoint import delete, get, post, put
from nemo_helix_plugin.client.types import Paginated, PreparedRequest

_CONFIGS = "/apis/auditor/v2/workspaces/{workspace}/configs"
_TARGETS = "/apis/auditor/v2/workspaces/{workspace}/targets"
_JOBS = "/apis/auditor/v2/workspaces/{workspace}/jobs/audit"


# ---------------------------------------------------------------------------
# Config CRUD
# ---------------------------------------------------------------------------


@get(f"{_CONFIGS}/{{name}}")
@abstractmethod
def get_audit_config(*, workspace: str | None = None, name: str) -> AuditConfig: ...


@get(_CONFIGS)
@abstractmethod
def list_audit_configs(
    *, workspace: str | None = None, query_params: ListAuditConfigsQueryParams | None = None
) -> Paginated[AuditConfig]: ...


def _get_audit_config_on_conflict(
    body: CreateAuditConfigRequest, workspace: str | None
) -> PreparedRequest[AuditConfig]:
    return get_audit_config(name=body.name, workspace=workspace)


@post(_CONFIGS, get_on_conflict=_get_audit_config_on_conflict)
@abstractmethod
def create_audit_config(
    *, workspace: str | None = None, body: CreateAuditConfigRequest, exist_ok: bool = False
) -> AuditConfig: ...


@put(f"{_CONFIGS}/{{name}}")
@abstractmethod
def update_audit_config(*, workspace: str | None = None, name: str, body: UpdateAuditConfigRequest) -> AuditConfig: ...


@delete(f"{_CONFIGS}/{{name}}")
@abstractmethod
def delete_audit_config(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Target CRUD
# ---------------------------------------------------------------------------


@get(f"{_TARGETS}/{{name}}")
@abstractmethod
def get_audit_target(*, workspace: str | None = None, name: str) -> AuditTarget: ...


@get(_TARGETS)
@abstractmethod
def list_audit_targets(
    *, workspace: str | None = None, query_params: ListAuditTargetsQueryParams | None = None
) -> Paginated[AuditTarget]: ...


def _get_audit_target_on_conflict(
    body: CreateAuditTargetRequest, workspace: str | None
) -> PreparedRequest[AuditTarget]:
    return get_audit_target(name=body.name, workspace=workspace)


@post(_TARGETS, get_on_conflict=_get_audit_target_on_conflict)
@abstractmethod
def create_audit_target(
    *, workspace: str | None = None, body: CreateAuditTargetRequest, exist_ok: bool = False
) -> AuditTarget: ...


@put(f"{_TARGETS}/{{name}}")
@abstractmethod
def update_audit_target(*, workspace: str | None = None, name: str, body: UpdateAuditTargetRequest) -> AuditTarget: ...


@delete(f"{_TARGETS}/{{name}}")
@abstractmethod
def delete_audit_target(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Audit job submission and retrieval
# ---------------------------------------------------------------------------


@post(_JOBS)
@abstractmethod
def submit_audit(*, workspace: str | None = None, body: SubmitAuditRequest) -> AuditJobResponse: ...


@get(_JOBS)
@abstractmethod
def list_audit_jobs(
    *, workspace: str | None = None, query_params: ListAuditJobsQueryParams | None = None
) -> Paginated[AuditJobResponse]: ...


@get(f"{_JOBS}/{{name}}")
@abstractmethod
def get_audit_job(*, workspace: str | None = None, name: str) -> AuditJobResponse: ...
