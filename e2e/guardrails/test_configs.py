# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for Guardrails config CRUD endpoints.

These tests exercise the generated SDK against the real platform subprocess for
the GuardrailConfig lifecycle. Service integration tests cover the full matrix
of validation errors; this file keeps e2e coverage focused on the user-facing
create, retrieve, list, update, and delete workflow.
"""

import nemo_helix
import pytest
from nemo_helix import NeMoHelix

from e2e.guardrails.utils import CONTENT_SAFETY_INPUT_FLOW, RailType, content_safety_config, unique_name


def _config_data(workspace: str, *, rail_types: tuple[RailType, ...] = ("input",)) -> dict:
    return content_safety_config(
        content_safety_model_ref=f"{workspace}/{unique_name('cs-model')}",
        rail_types=rail_types,
        streaming=False,
    )


def test_guardrail_config_create_and_retrieve(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("crud-config")

    created = sdk.guardrail.configs.create(
        workspace=workspace,
        name=name,
        description="Initial CRUD config",
        data=_config_data(workspace),
    )

    assert created.name == name
    assert created.workspace == workspace
    assert created.description == "Initial CRUD config"
    assert created.id
    assert created.entity_id == created.id  # `entity_id` is an alias for `id`, not a `workspace/name` ref.
    assert created.created_at is not None
    assert created.updated_at is not None

    retrieved = sdk.guardrail.configs.retrieve(workspace=workspace, name=name)
    assert retrieved.id == created.id
    assert retrieved.name == name
    assert retrieved.data is not None
    assert retrieved.data.rails is not None
    assert retrieved.data.rails.input is not None
    assert retrieved.data.rails.input.flows == [CONTENT_SAFETY_INPUT_FLOW]


def test_guardrail_config_create_and_list(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("list-config")
    created = sdk.guardrail.configs.create(
        workspace=workspace,
        name=name,
        description="List CRUD config",
        data=_config_data(workspace),
    )

    listed_names = {config.name for config in sdk.guardrail.configs.list(workspace=workspace, page_size=100)}
    assert name in listed_names

    listed_config = next(
        config for config in sdk.guardrail.configs.list(workspace=workspace, page_size=100) if config.name == name
    )
    assert listed_config.id == created.id
    assert listed_config.created_at is not None
    assert listed_config.updated_at is not None


def test_guardrail_config_update(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("update-config")
    created = sdk.guardrail.configs.create(
        workspace=workspace,
        name=name,
        description="Initial update config",
        data=_config_data(workspace),
    )

    updated = sdk.guardrail.configs.update(
        name,
        workspace=workspace,
        description="Updated CRUD config",
        data=_config_data(workspace, rail_types=("input", "output")),
    )

    assert updated.id == created.id
    assert updated.created_at == created.created_at
    assert updated.description == "Updated CRUD config"
    assert updated.data is not None
    assert updated.data.rails is not None
    assert updated.data.rails.output is not None


def test_guardrail_config_delete(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("delete-config")
    sdk.guardrail.configs.create(
        workspace=workspace,
        name=name,
        description="Delete CRUD config",
        data=_config_data(workspace),
    )

    listed_names = {config.name for config in sdk.guardrail.configs.list(workspace=workspace, page_size=100)}
    assert name in listed_names

    sdk.guardrail.configs.delete(workspace=workspace, name=name)

    with pytest.raises(nemo_helix.NotFoundError):
        sdk.guardrail.configs.retrieve(workspace=workspace, name=name)


def test_guardrail_config_create_duplicate_name_returns_conflict(
    sdk: NeMoHelix,
    workspace: str,
) -> None:
    name = unique_name("duplicate-config")
    config_data = _config_data(workspace)

    sdk.guardrail.configs.create(
        workspace=workspace,
        name=name,
        description="Duplicate config",
        data=config_data,
    )

    with pytest.raises(nemo_helix.ConflictError):
        sdk.guardrail.configs.create(
            workspace=workspace,
            name=name,
            description="Duplicate config",
            data=config_data,
        )
