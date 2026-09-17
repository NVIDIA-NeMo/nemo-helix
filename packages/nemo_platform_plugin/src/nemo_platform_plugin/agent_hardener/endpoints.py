# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Agent Hardener service.

Single source of truth for the HTTP contract. Replaces the Stainless-generated
agent_hardener resource from ``nemo_agent_hardener_plugin.sdk``.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_platform_plugin.agent_hardener.types import (
    AgentHardenerManifest,
    AgentHardenerRun,
    CreateManifestRequest,
    CreateRunRequest,
    ListManifestsQueryParams,
    ListRunsQueryParams,
    UpdateManifestRequest,
    ValidateModelRequest,
    ValidateModelResponse,
)
from nemo_platform_plugin.client.endpoint import delete, get, patch, post
from nemo_platform_plugin.client.types import Paginated, PreparedRequest

_MANIFESTS = "/apis/agent-hardener/v2/workspaces/{workspace}/manifests"
_RUNS = "/apis/agent-hardener/v2/workspaces/{workspace}/runs"


# ---------------------------------------------------------------------------
# Manifest CRUD
# ---------------------------------------------------------------------------


@get(f"{_MANIFESTS}/{{name}}")
@abstractmethod
def get_manifest(*, workspace: str | None = None, name: str) -> AgentHardenerManifest: ...


@get(_MANIFESTS)
@abstractmethod
def list_manifests(
    *, workspace: str | None = None, query_params: ListManifestsQueryParams | None = None
) -> Paginated[AgentHardenerManifest]: ...


def _get_manifest_on_conflict(
    body: CreateManifestRequest, workspace: str | None
) -> PreparedRequest[AgentHardenerManifest]:
    return get_manifest(name=body.agent, workspace=workspace)


@post(_MANIFESTS, get_on_conflict=_get_manifest_on_conflict)
@abstractmethod
def create_manifest(
    *, workspace: str | None = None, body: CreateManifestRequest, exist_ok: bool = False
) -> AgentHardenerManifest: ...


@patch(f"{_MANIFESTS}/{{name}}")
@abstractmethod
def update_manifest(
    *, workspace: str | None = None, name: str, body: UpdateManifestRequest
) -> AgentHardenerManifest: ...


@post(f"{_MANIFESTS}/{{name}}/refresh")
@abstractmethod
def refresh_manifest(*, workspace: str | None = None, name: str) -> AgentHardenerManifest: ...


@delete(f"{_MANIFESTS}/{{name}}")
@abstractmethod
def delete_manifest(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------


@get(f"{_RUNS}/{{name}}")
@abstractmethod
def get_run(*, workspace: str | None = None, name: str) -> AgentHardenerRun: ...


@get(_RUNS)
@abstractmethod
def list_runs(
    *, workspace: str | None = None, query_params: ListRunsQueryParams | None = None
) -> Paginated[AgentHardenerRun]: ...


@post(_RUNS)
@abstractmethod
def create_run(*, workspace: str | None = None, body: CreateRunRequest) -> AgentHardenerRun: ...


@delete(f"{_RUNS}/{{name}}")
@abstractmethod
def delete_run(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Model config validation
# ---------------------------------------------------------------------------


@post("/apis/agent-hardener/v2/workspaces/{workspace}/model-config/validate")
@abstractmethod
def validate_model(*, workspace: str | None = None, body: ValidateModelRequest) -> ValidateModelResponse: ...
