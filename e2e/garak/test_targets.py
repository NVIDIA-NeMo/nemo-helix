# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for AuditTarget CRUD and filtering."""

import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.garak.client import GarakClient
from nemo_helix_plugin.garak.types import CreateAuditTargetRequest, UpdateAuditTargetRequest

from e2e.garak.utils import minimal_audit_target, unique_name


def _list_raw(garak: GarakClient, workspace: str, garak_url: str, **params) -> dict:
    resp = garak._client.get(
        f"{garak_url}/v2/workspaces/{workspace}/targets",
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


def test_target_create_and_get(garak: GarakClient, workspace: str) -> None:
    name = unique_name("tgt-cg")
    body = minimal_audit_target(description="create-and-get target", type="nim", model="meta/llama-3.1-8b-instruct")

    created = garak.create_audit_target(workspace=workspace, body=CreateAuditTargetRequest(name=name, **body)).data()

    assert created.name == name
    assert created.workspace == workspace
    assert created.description == "create-and-get target"
    assert created.type == "nim"
    assert created.model == "meta/llama-3.1-8b-instruct"

    retrieved = garak.get_audit_target(workspace=workspace, name=name).data()
    assert retrieved.name == name
    assert retrieved.type == "nim"
    assert retrieved.model == "meta/llama-3.1-8b-instruct"


def test_target_list_contains_created(garak: GarakClient, workspace: str) -> None:
    name = unique_name("tgt-list")

    garak.create_audit_target(
        workspace=workspace, body=CreateAuditTargetRequest(name=name, **minimal_audit_target())
    ).data()

    page = garak.list_audit_targets(workspace=workspace, query_params={"page_size": 100})
    names = [item.name for item in page.items()]
    assert name in names


def test_target_update(garak: GarakClient, workspace: str) -> None:
    name = unique_name("tgt-upd")

    garak.create_audit_target(
        workspace=workspace,
        body=CreateAuditTargetRequest(name=name, **minimal_audit_target(description="original", model="gpt-4o-mini")),
    ).data()

    updated = garak.update_audit_target(
        workspace=workspace,
        name=name,
        body=UpdateAuditTargetRequest(**minimal_audit_target(description="updated", model="gpt-4o")),
    ).data()

    assert updated.name == name
    assert updated.description == "updated"
    assert updated.model == "gpt-4o"


def test_target_delete(garak: GarakClient, workspace: str) -> None:
    name = unique_name("tgt-del")
    garak.create_audit_target(
        workspace=workspace, body=CreateAuditTargetRequest(name=name, **minimal_audit_target())
    ).data()

    garak.delete_audit_target(workspace=workspace, name=name)

    with pytest.raises(NotFoundError):
        garak.get_audit_target(workspace=workspace, name=name)


def test_target_filter_by_type(garak: GarakClient, workspace: str, garak_url: str) -> None:
    nim_name = unique_name("tgt-nim")
    openai_name = unique_name("tgt-oai")

    garak.create_audit_target(
        workspace=workspace,
        body=CreateAuditTargetRequest(
            name=nim_name, **minimal_audit_target(type="nim", model="meta/llama-3.1-8b-instruct")
        ),
    ).data()
    garak.create_audit_target(
        workspace=workspace,
        body=CreateAuditTargetRequest(name=openai_name, **minimal_audit_target(type="openai", model="gpt-4o-mini")),
    ).data()

    result = _list_raw(garak, workspace, garak_url, **{"filter[type]": "nim"})
    names = [item["name"] for item in result["data"]]

    assert nim_name in names
    assert openai_name not in names
