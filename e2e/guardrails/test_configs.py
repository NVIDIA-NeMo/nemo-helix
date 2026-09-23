# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for Guardrails config CRUD endpoints.

These tests exercise the typed Guardrail client against the real platform
subprocess for the GuardrailConfig lifecycle. Service integration tests cover
the full matrix of validation errors; this file keeps e2e coverage focused on
the user-facing create, retrieve, list, update, and delete workflow.
"""

from typing import Any

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.guardrail.types import CreateGuardrailConfigRequest, UpdateGuardrailConfigRequest

from e2e.guardrails.utils import (
    CONTENT_SAFETY_INPUT_FLOW,
    RailType,
    content_safety_config,
    guardrail_client_from_sdk,
    unique_name,
)


def _config_data(workspace: str, *, rail_types: tuple[RailType, ...] = ("input",)) -> dict[str, Any]:
    return content_safety_config(
        content_safety_model_ref=f"{workspace}/{unique_name('cs-model')}",
        rail_types=rail_types,
        streaming=False,
    )


def test_guardrail_config_create_and_retrieve(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("crud-config")
    client = guardrail_client_from_sdk(sdk)

    created = client.create_guardrail_config(
        workspace=workspace,
        body=CreateGuardrailConfigRequest(
            name=name,
            description="Initial CRUD config",
            data=_config_data(workspace),
        ),
    ).data()

    assert created.name == name
    assert created.workspace == workspace
    assert created.description == "Initial CRUD config"
    assert created.id
    assert created.created_at is not None
    assert created.updated_at is not None

    retrieved = client.get_guardrail_config(workspace=workspace, name=name).data()
    assert retrieved.id == created.id
    assert retrieved.name == name
    assert retrieved.data is not None
    assert retrieved.data.rails is not None
    assert retrieved.data.rails.input is not None
    assert retrieved.data.rails.input.flows == [CONTENT_SAFETY_INPUT_FLOW]


def test_guardrail_config_create_and_list(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("list-config")
    client = guardrail_client_from_sdk(sdk)
    created = client.create_guardrail_config(
        workspace=workspace,
        body=CreateGuardrailConfigRequest(
            name=name,
            description="List CRUD config",
            data=_config_data(workspace),
        ),
    ).data()

    listed_names = {
        config.name
        for config in client.list_guardrail_configs(workspace=workspace, query_params={"page_size": 100}).items()
    }
    assert name in listed_names

    listed_config = next(
        config
        for config in client.list_guardrail_configs(workspace=workspace, query_params={"page_size": 100}).items()
        if config.name == name
    )
    assert listed_config.id == created.id
    assert listed_config.created_at is not None
    assert listed_config.updated_at is not None


def test_guardrail_config_update(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("update-config")
    client = guardrail_client_from_sdk(sdk)
    created = client.create_guardrail_config(
        workspace=workspace,
        body=CreateGuardrailConfigRequest(
            name=name,
            description="Initial update config",
            data=_config_data(workspace),
        ),
    ).data()

    updated = client.update_guardrail_config(
        workspace=workspace,
        name=name,
        body=UpdateGuardrailConfigRequest(
            description="Updated CRUD config",
            data=_config_data(workspace, rail_types=("input", "output")),
        ),
    ).data()

    assert updated.id == created.id
    assert updated.created_at == created.created_at
    assert updated.description == "Updated CRUD config"
    assert updated.data is not None
    assert updated.data.rails is not None
    assert updated.data.rails.output is not None


def test_guardrail_config_delete(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("delete-config")
    client = guardrail_client_from_sdk(sdk)
    client.create_guardrail_config(
        workspace=workspace,
        body=CreateGuardrailConfigRequest(
            name=name,
            description="Delete CRUD config",
            data=_config_data(workspace),
        ),
    )

    listed_names = {
        config.name
        for config in client.list_guardrail_configs(workspace=workspace, query_params={"page_size": 100}).items()
    }
    assert name in listed_names

    client.delete_guardrail_config(workspace=workspace, name=name)

    with pytest.raises(NotFoundError):
        client.get_guardrail_config(workspace=workspace, name=name)


def test_guardrail_config_create_duplicate_name_returns_conflict(
    sdk: NeMoHelix,
    workspace: str,
) -> None:
    name = unique_name("duplicate-config")
    config_data = _config_data(workspace)
    client = guardrail_client_from_sdk(sdk)

    client.create_guardrail_config(
        workspace=workspace,
        body=CreateGuardrailConfigRequest(
            name=name,
            description="Duplicate config",
            data=config_data,
        ),
    )

    with pytest.raises(ConflictError):
        client.create_guardrail_config(
            workspace=workspace,
            body=CreateGuardrailConfigRequest(
                name=name,
                description="Duplicate config",
                data=config_data,
            ),
        )
