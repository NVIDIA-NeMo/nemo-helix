# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Guardrails service.

Single source of truth for the HTTP contract. Replaces the Stainless-generated
``nemo_helix.resources.guardrail`` resource.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import delete, get, patch, post
from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.entities.types import DeleteResponse
from nemo_helix_plugin.guardrail.types import (
    CreateGuardrailConfigRequest,
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    GuardrailConfig,
    ListGuardrailConfigsQueryParams,
    UpdateGuardrailConfigRequest,
)

_CHECKS = "/apis/guardrails/v2/workspaces/{workspace}/checks"
_CONFIGS = "/apis/guardrails/v2/workspaces/{workspace}/configs"


@get(f"{_CONFIGS}/{{name}}")
@abstractmethod
def get_guardrail_config(*, workspace: str | None = None, name: str) -> GuardrailConfig: ...


@get(_CONFIGS)
@abstractmethod
def list_guardrail_configs(
    *, workspace: str | None = None, query_params: ListGuardrailConfigsQueryParams | None = None
) -> Paginated[GuardrailConfig]: ...


def _get_guardrail_config_on_conflict(
    body: CreateGuardrailConfigRequest, workspace: str | None
) -> PreparedRequest[GuardrailConfig]:
    """Build the retrieve request replayed when ``create_guardrail_config(exist_ok=True)`` 409s."""
    return get_guardrail_config(name=body.name, workspace=workspace)


@post(_CONFIGS, get_on_conflict=_get_guardrail_config_on_conflict)
@abstractmethod
def create_guardrail_config(
    *, workspace: str | None = None, body: CreateGuardrailConfigRequest, exist_ok: bool = False
) -> GuardrailConfig: ...


@patch(f"{_CONFIGS}/{{name}}")
@abstractmethod
def update_guardrail_config(
    *, workspace: str | None = None, name: str, body: UpdateGuardrailConfigRequest
) -> GuardrailConfig: ...


@delete(f"{_CONFIGS}/{{name}}")
@abstractmethod
def delete_guardrail_config(*, workspace: str | None = None, name: str) -> DeleteResponse: ...


@post(_CHECKS)
@abstractmethod
def check_guardrail(*, workspace: str | None = None, body: GuardrailCheckRequest) -> GuardrailCheckResponse: ...
