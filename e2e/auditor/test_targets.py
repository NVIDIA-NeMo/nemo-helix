# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for AuditTarget CRUD and filtering."""

import httpx
import pytest
from nemo_helix import NeMoHelix

from e2e.auditor.utils import minimal_audit_target, unique_name


def _list_raw(sdk: NeMoHelix, workspace: str, auditor_url: str, **params) -> dict:
    resp = sdk.auditor._http_client.get(
        f"{auditor_url}/v2/workspaces/{workspace}/targets",
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


def test_target_create_and_get(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("tgt-cg")
    body = minimal_audit_target(description="create-and-get target", type="nim", model="meta/llama-3.1-8b-instruct")

    created = sdk.auditor.targets.create(workspace=workspace, name=name, **body)

    assert created.name == name
    assert created.workspace == workspace
    assert created.description == "create-and-get target"
    assert created.type == "nim"
    assert created.model == "meta/llama-3.1-8b-instruct"

    # Note: id/created_at are not populated by the auditor SDK (raw httpx + model_validate
    # cannot set the private _id PrivateAttr). Use name for cross-call identity checks.
    retrieved = sdk.auditor.targets.get(workspace=workspace, name=name)
    assert retrieved.name == name
    assert retrieved.type == "nim"
    assert retrieved.model == "meta/llama-3.1-8b-instruct"


def test_target_list_contains_created(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("tgt-list")

    sdk.auditor.targets.create(workspace=workspace, name=name, **minimal_audit_target())

    page = sdk.auditor.targets.list(workspace=workspace, page_size=100)
    names = [item["name"] for item in page["data"]]
    assert name in names


def test_target_update(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("tgt-upd")

    sdk.auditor.targets.create(
        workspace=workspace,
        name=name,
        **minimal_audit_target(description="original", model="gpt-4o-mini"),
    )

    updated = sdk.auditor.targets.update(
        workspace=workspace,
        name=name,
        **minimal_audit_target(description="updated", model="gpt-4o"),
    )

    assert updated.name == name
    assert updated.description == "updated"
    assert updated.model == "gpt-4o"


def test_target_delete(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("tgt-del")
    sdk.auditor.targets.create(workspace=workspace, name=name, **minimal_audit_target())

    sdk.auditor.targets.delete(workspace=workspace, name=name)

    # Auditor SDK uses raw httpx, so errors surface as httpx.HTTPStatusError.
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        sdk.auditor.targets.get(workspace=workspace, name=name)
    assert exc_info.value.response.status_code == 404


def test_target_filter_by_type(sdk: NeMoHelix, workspace: str, auditor_url: str) -> None:
    nim_name = unique_name("tgt-nim")
    openai_name = unique_name("tgt-oai")

    sdk.auditor.targets.create(
        workspace=workspace,
        name=nim_name,
        **minimal_audit_target(type="nim", model="meta/llama-3.1-8b-instruct"),
    )
    sdk.auditor.targets.create(
        workspace=workspace,
        name=openai_name,
        **minimal_audit_target(type="openai", model="gpt-4o-mini"),
    )

    result = _list_raw(sdk, workspace, auditor_url, **{"filter[type]": "nim"})
    names = [item["name"] for item in result["data"]]

    assert nim_name in names
    assert openai_name not in names
